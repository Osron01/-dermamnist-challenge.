"""Skin-lesion classifier app, built from the course's app_template.py.

Upload an image -> the model's top-3 classes with probabilities (the template),
plus four extras for this model:
  * what the model sees: the 28 x 28 input, enlarged, next to the upload
  * all 7 class probabilities, not only the top 3
  * a malignancy flag when P(akiec) + P(bcc) + P(mel) reaches a threshold chosen on val
  * an example gallery (one val image per class) and a model-card tab

Local:                 python app/app.py               (model.pt in app/ or in the repo root)
Colab:                 !pip install -q gradio   then run this file (model.pt next to it)
Hugging Face Spaces:   upload app.py, app_config.json, examples/, model.pt and
                       requirements.txt (= app/requirements.txt) - the Space starts by itself.
Do not change MEAN / STD: they must be the numbers of the starter notebook.
"""
import json
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

MEAN = [0.7631, 0.5381, 0.5614]
STD = [0.1366, 0.1543, 0.1692]
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]
SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = [0, 1, 4]          # actinic keratoses / intraepithelial carcinoma, basal cell carcinoma, melanoma

HERE = Path(__file__).resolve().parent
MODEL_PATH = next(p for p in (HERE / "model.pt", HERE.parent / "model.pt") if p.exists())
CONFIG = json.loads((HERE / "app_config.json").read_text())
THRESHOLD = CONFIG["malignancy_flag"]["threshold"]

model = torch.jit.load(str(MODEL_PATH), map_location="cpu")   # our exported model
model.eval()
prepare = v2.Compose([
    v2.ToImage(),                                  # PIL image -> tensor (3, H, W)
    v2.Resize((28, 28), antialias=True),           # the size the model was trained on
    v2.ToDtype(torch.float32, scale=True),         # 0...255 -> 0...1
    v2.Normalize(MEAN, STD),                       # exactly like in the starter notebook
])
DISCLAIMER = ("**Research demo, not a medical device.** Student project on 28 x 28 images; "
              "see a dermatologist for any concern about your skin.")


def classify(image):
    """PIL image -> top-3 dict (template), all-7 dict, malignancy banner, 28 x 28 view."""
    if image is None:
        return None, None, "", None
    image = image.convert("RGB")
    x = prepare(image).unsqueeze(0)                          # (1, 3, 28, 28)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top = probs.topk(3)
    top3 = {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}
    all7 = {CLASSES[i]: float(p) for i, p in enumerate(probs)}

    p_mal = float(probs[MALIGNANT].sum())
    if p_mal >= THRESHOLD:
        flag = (f"### ⚠ Possibly malignant: {p_mal:.0%}\n"
                f"Actinic keratoses, basal cell carcinoma and melanoma together reach {p_mal:.0%} "
                f"(warning threshold {THRESHOLD:.0%}). Please have this lesion checked by a dermatologist.")
    else:
        flag = (f"### No malignancy warning: {p_mal:.0%}\n"
                f"The three malignant classes together stay below the warning threshold of {THRESHOLD:.0%}. "
                f"This does not rule anything out.")
    seen = image.resize((28, 28), Image.BILINEAR).resize((224, 224), Image.NEAREST)
    return top3, all7, flag, seen


def model_card() -> str:
    v = CONFIG["val_metrics"]
    f = CONFIG["malignancy_flag"]
    rows = "\n".join(f"| {CLASSES[i]} ({s}) | {v['sensitivity_per_class'][s]:.2f} |" for i, s in enumerate(SHORT))
    return f"""
### Model
Three ImageNet-pretrained ResNet-18 networks (seeds 0, 1, 2), trained with a logit-adjusted loss and random
flips / 90° rotations. The 28 x 28 input is upsampled to 112 x 112 inside the model; the three networks'
logits are averaged.

### Training data
DermaMNIST (MedMNIST v2, Yang et al. 2023), from the HAM10000 dermatoscopy images; 7 007 training images,
67 % of them melanocytic nevi. Licence CC BY-NC 4.0 — non-commercial use with attribution.

### How well it does (validation images, chosen and measured on val)
Balanced accuracy **{v['balanced_accuracy']:.3f}**, macro-F1 **{v['macro_f1']:.3f}**.

| class | sensitivity |
| --- | --- |
{rows}

### Known weak spots
- Every image is shrunk to 28 x 28: fine texture and borders are lost.
- Actinic keratoses, dermatofibroma and benign keratosis are found only about half of the time.
- The loss pushes the model towards rare classes, so harmless moles are sometimes called something else.
- The malignancy warning (threshold {f['threshold']:.0%}) catches {f['val_sensitivity']:.0%} of the malignant
  validation images, but also fires on {f['val_flag_rate_benign']:.0%} of the benign ones.
- Trained on dermatoscope images only: phone photos of skin look different and give unreliable answers.
"""


with gr.Blocks(title="Skin lesion classifier (DermaMNIST challenge)") as demo:
    gr.Markdown("# Skin lesion classifier (DermaMNIST challenge)\n"
                "Upload a dermatoscope image of a skin lesion. Team *Am besten*, Deep Learning, SRH.")
    gr.Markdown(DISCLAIMER)
    with gr.Tab("Classify"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="skin lesion image")
                seen = gr.Image(type="pil", label="what the model sees (28 x 28, enlarged)", interactive=False)
            with gr.Column():
                top3 = gr.Label(num_top_classes=3, label="prediction")
                flag = gr.Markdown()
                all7 = gr.Label(num_top_classes=7, label="all 7 classes")
        gr.Examples(examples=[[str(HERE / "examples" / f"{s}.png")] for s in SHORT], inputs=image,
                    label="examples: one validation image per class, in the order " + ", ".join(SHORT))
        image.change(classify, inputs=image, outputs=[top3, all7, flag, seen])
    with gr.Tab("Model card"):
        gr.Markdown(model_card())

if __name__ == "__main__":
    demo.launch(share=True)   # share=True: public link (needed in Colab); on Hugging Face Spaces it is ignored
