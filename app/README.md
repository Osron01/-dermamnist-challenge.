# App

`app.py` is the course's `app_template.py` (kept unchanged next to it) with four extras for our model:
what the model sees (the 28 x 28 input, enlarged), all 7 class probabilities, a malignancy flag
(P(akiec) + P(bcc) + P(mel) at or above a threshold chosen on val) and an example gallery, plus a
model-card tab. MEAN/STD, the 28 x 28 resize and the top-3 output are exactly as in the template.

## Run it locally

```bash
python -m src.export --config configs/s2_la_resnet18_fliprot.yaml --seeds 0 1 2 --out model.pt   # if model.pt is missing
python scripts/app_assets.py     # threshold, gallery images, model-card numbers (val only) -> app/
pip install -r app/requirements.txt
python app/app.py                # prints a local and a public (share) link
```

## Hugging Face Spaces

New Space, SDK **Gradio**, then upload `app.py`, `app_config.json`, `examples/`, `requirements.txt`
and `model.pt` (128 MB; the Space's web upload stores it with Git LFS). The Space starts the app
by itself. Free Spaces sleep when unused: open the link a few minutes before the presentation.

Example images: DermaMNIST validation images (MedMNIST v2, CC BY-NC 4.0), the first one of each class.
