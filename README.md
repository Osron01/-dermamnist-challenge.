# DermaMNIST skin-lesion classification challenge

Deep Learning exam project: classify 28×28 dermatoscopy images into 7 lesion
types, maximise **balanced accuracy** on the official test split, and compare
every improvement fairly against the starter baseline (≥3 seeds, mean ± std,
95 % bootstrap interval of the difference).

Team: _team name_ — _members_

## Repository layout

```
configs/            one YAML per variant; each inherits baseline.yaml and changes one thing
src/
  data.py           DermaMNIST loading, MEAN/STD normalisation, augmentation
  models.py         baseline CNN, improved CNN, ResNet with upsampling inside the model
  losses.py         cross-entropy, class-weighted CE, logit-adjusted loss, focal loss
  sam.py            Sharpness-Aware Minimisation optimiser
  train.py          train(config, seed) -> results/<variant>/seed<k>/
  evaluate.py       balanced accuracy, macro-F1, per-class sensitivity, bootstrap CI
  export.py         TorchScript export of the final model + load-and-compare check
  utils.py          seeding, config inheritance, device, git hash
scripts/
  run_seeds.py      train one config with seeds 0, 1, 2
  summarize.py      results table (mean ± std, Δ vs baseline with 95 % CI) from saved logits
  describe_data.py  class counts and example-image grid (Phase A)
  final_test.py     test-split evaluation, run once at the very end
tests/              fast checks on random images (no download needed)
notebooks/          the starter notebook and exploration notebooks
app/                Gradio app (from app_template.py)
report/             IEEE LaTeX report
results/            saved val logits and metrics per run (committed), summary tables
```

## Setup

Python 3.11 or 3.12 recommended.

```bash
git clone https://github.com/<user>/dermamnist-challenge.git
cd dermamnist-challenge
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # should pass in under a minute
```

GPU: `pip install torch` gives a CPU build on Windows/macOS. For an NVIDIA GPU,
install torch first with the command from https://pytorch.org/get-started/locally/,
then `pip install -r requirements.txt`. On Colab, torch is preinstalled.

Weights & Biases (optional, recommended): `wandb login` once, then add `--wandb` to any run.

### Step 1 — copy three things from the starter notebook

Before the first real run, open `notebooks/day4_challenge_starter.ipynb` and copy:

1. **MEAN and STD** into `src/data.py` (the code refuses to run until they are set)
2. **The baseline CNN** into `BaselineCNN` in `src/models.py`
3. **Epochs, learning rate, batch size, optimiser** into `configs/baseline.yaml`

Then check that the baseline run reproduces the notebook's numbers.

## Reproduce

```bash
python scripts/describe_data.py                        # class counts + example grid
python scripts/run_seeds.py configs/baseline.yaml      # B0, seeds 0 1 2
python scripts/run_seeds.py configs/e1_class_weights.yaml
# ... one line per config in configs/
python scripts/summarize.py                            # results/summary.md and .csv

python -m src.export --config configs/<final>.yaml --seeds 0 1 2 --out model.pt
python -m src.export --config configs/baseline.yaml --seeds 0 1 2 --out baseline.pt
python scripts/final_test.py --models baseline.pt model.pt --i-have-frozen-the-model
```

Single run: `python -m src.train --config configs/e3_flip_rot.yaml --seed 0 --wandb`

### On Colab

```python
!git clone https://github.com/<user>/dermamnist-challenge.git
%cd dermamnist-challenge
!pip install -q medmnist wandb
!python scripts/run_seeds.py configs/e7_resnet18.yaml --wandb
```

Copy `results/` back to the repo (or to Google Drive) after each session.

## Rules this code follows

- Training uses the official **train** split only; models and settings are chosen on **val**.
- The **test** split is loaded only by `scripts/final_test.py`, once, after the final model is frozen.
- Model input is (N, 3, 28, 28), normalised with the starter's MEAN/STD; output is (N, 7) logits.
- Every variant changes exactly one thing relative to its parent config and runs with ≥3 seeds.

## Data and licence

DermaMNIST (MedMNIST v2, Yang et al., *Scientific Data* 2023), derived from HAM10000,
licence CC BY-NC 4.0. The data is downloaded to `data/` (git-ignored) on first use.

## AI use

_To be completed: which AI tools were used, for what, and how the output was checked._
