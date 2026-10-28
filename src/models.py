"""Model zoo. Every model takes (N, 3, 28, 28) normalised images and returns (N, 7) logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BaselineCNN(nn.Module):
    """B0: the starter notebook's baseline CNN, unchanged (16 071 parameters)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.head = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


def _conv_block(c_in: int, c_out: int, pool: bool) -> nn.Sequential:
    layers = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ImprovedCNN(nn.Module):
    """E5: BatchNorm after every conv, two convs per block, dropout before the head."""

    def __init__(self, channels=(32, 64, 128, 256), dropout: float = 0.3, num_classes: int = NUM_CLASSES):
        super().__init__()
        blocks, c_in = [], 3
        for i, c in enumerate(channels):
            blocks.append(_conv_block(c_in, c, pool=i < len(channels) - 1))  # 28 -> 14 -> 7 -> 3
            c_in = c
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(torch.flatten(self.features(x), 1)))


class ResNetUp(nn.Module):
    """E7: torchvision ResNet with the 28x28 -> `size` upsampling INSIDE the model,
    so the exported model.pt still accepts (N, 3, 28, 28) as the grader expects.

    If `renorm_imagenet`, the input is mapped from DermaMNIST normalisation to
    ImageNet normalisation, which is what pretrained weights were trained on.
    """

    def __init__(self, arch: str = "resnet18", pretrained: bool = True, size: int = 112,
                 data_mean=None, data_std=None, renorm_imagenet: bool = True,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        import torchvision

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        self.size = size
        self.renorm = bool(renorm_imagenet and data_mean is not None)
        dm = torch.tensor(data_mean if data_mean is not None else (0.0, 0.0, 0.0)).view(1, 3, 1, 1)
        ds = torch.tensor(data_std if data_std is not None else (1.0, 1.0, 1.0)).view(1, 3, 1, 1)
        self.register_buffer("data_mean", dm)
        self.register_buffer("data_std", ds)
        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.renorm:
            x = (x * self.data_std + self.data_mean - self.in_mean) / self.in_std
        if self.size != 28:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.net(x)


def build_model(cfg: dict, data_mean=None, data_std=None) -> nn.Module:
    m = cfg["model"]
    name = m["name"]
    if name == "baseline_cnn":
        return BaselineCNN()
    if name == "improved_cnn":
        return ImprovedCNN(channels=tuple(m.get("channels", (32, 64, 128, 256))), dropout=m.get("dropout", 0.3))
    if name == "resnet_up":
        return ResNetUp(arch=m.get("arch", "resnet18"), pretrained=m.get("pretrained", True),
                        size=m.get("size", 112), data_mean=data_mean, data_std=data_std,
                        renorm_imagenet=m.get("renorm_imagenet", True))
    raise ValueError(f"unknown model name: {name}")
