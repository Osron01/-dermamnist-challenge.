"""DermaMNIST loading, normalisation and augmentation.

Rules from the brief this file enforces:
  * training uses the official `train` split only,
  * model selection uses `val`,
  * `test` is only loaded by `scripts/final_test.py` (once, at the end),
  * the model input is (N, 3, 28, 28), normalised with the starter's MEAN/STD.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src.utils import REPO_ROOT, seed_worker

# MEAN and STD copied unchanged from the starter notebook day4_challenge_starter.ipynb
# (channel mean / spread of the training images, R G B). They must stay exactly
# like this, because the grading script normalises the test images with them.
MEAN: tuple[float, float, float] | None = (0.7631, 0.5381, 0.5614)
STD: tuple[float, float, float] | None = (0.1366, 0.1543, 0.1692)

NUM_CLASSES = 7
CLASS_NAMES = [
    "actinic keratoses / intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]
CLASS_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DATA_DIR = REPO_ROOT / "data"


def get_mean_std() -> tuple[tuple[float, ...], tuple[float, ...]]:
    if MEAN is None or STD is None:
        raise RuntimeError(
            "MEAN/STD are not set. Copy them from the starter notebook into "
            "src/data.py (see README, step 1)."
        )
    return MEAN, STD


class RandomRot90:
    """Rotate a PIL image by 0, 90, 180 or 270 degrees (lesions have no 'up')."""

    _ops = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]

    def __call__(self, img: Image.Image) -> Image.Image:
        op = self._ops[int(torch.randint(0, 4, (1,)))]
        return img if op is None else img.transpose(op)


def build_transforms(aug_cfg: dict | None, mean, std, train: bool) -> T.Compose:
    """PIL image -> normalised float tensor (3, 28, 28). Augmentation only if train."""
    ops: list = []
    if train and aug_cfg:
        if aug_cfg.get("trivialaugment"):
            ops.append(T.TrivialAugmentWide())
        if aug_cfg.get("flip"):
            ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        if aug_cfg.get("rot90"):
            ops.append(RandomRot90())
        jitter = aug_cfg.get("color_jitter", 0.0)
        if jitter:
            ops.append(T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class FakeDerma(Dataset):
    """Random 28x28 images with DermaMNIST-like class shares, for offline smoke tests."""

    SHARES = np.array([228, 359, 769, 80, 779, 4693, 99], dtype=float)

    def __init__(self, n: int, transform, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.imgs = rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8)
        labels = rng.choice(NUM_CLASSES, size=n, p=self.SHARES / self.SHARES.sum())
        labels[:NUM_CLASSES] = np.arange(NUM_CLASSES)  # every class present at least once
        self.labels = labels.reshape(-1, 1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.imgs[i])), self.labels[i]


def load_split(split: str, transform, fake: bool = False):
    """One DermaMNIST split as a torch Dataset (28x28, RGB)."""
    if fake:
        n = {"train": 512, "val": 128, "test": 128}[split]
        return FakeDerma(n, transform, seed={"train": 0, "val": 1, "test": 2}[split])
    from medmnist import DermaMNIST  # imported lazily so tests run without the data

    DATA_DIR.mkdir(exist_ok=True)
    return DermaMNIST(split=split, transform=transform, download=True, size=28, root=str(DATA_DIR))


def class_counts(dataset) -> np.ndarray:
    labels = np.asarray(dataset.labels).reshape(-1)
    return np.bincount(labels, minlength=NUM_CLASSES)


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.as_tensor(np.asarray(ys).reshape(-1), dtype=torch.long)


def get_loaders(cfg: dict, generator: torch.Generator, fake: bool = False):
    """Train and val loaders plus the training class counts."""
    if fake:
        mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    else:
        mean, std = get_mean_std()
    data_cfg = cfg.get("data", {})
    train_ds = load_split("train", build_transforms(cfg.get("augment"), mean, std, train=True), fake)
    val_ds = load_split("val", build_transforms(None, mean, std, train=False), fake)
    counts = class_counts(train_ds)

    bs, workers = data_cfg.get("batch_size", 128), data_cfg.get("num_workers", 2)
    common = dict(num_workers=workers, worker_init_fn=seed_worker, collate_fn=_collate,
                  pin_memory=torch.cuda.is_available())
    if data_cfg.get("sampler", "random") == "class_balanced":
        labels = np.asarray(train_ds.labels).reshape(-1)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(labels), replacement=True, generator=generator)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, **common)
    return train_loader, val_loader, counts
