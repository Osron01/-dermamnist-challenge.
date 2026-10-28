"""Challenge app template: upload a skin-lesion image -> top-3 classes with probabilities.

Colab:                 !pip install -q gradio   then run this file's code in a cell (model.pt next to it)
                       -> demo.launch(share=True) prints a public link you can open on your phone
Hugging Face Spaces:   new Space (SDK: Gradio), upload app.py (= this file), model.pt and requirements.txt
                       (= requirements_app.txt) - the Space starts the app by itself.
Do not change MEAN / STD: they must be the numbers of the starter notebook.
"""
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

MEAN = [0.7631, 0.5381, 0.5614]
STD = [0.1366, 0.1543, 0.1692]
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]

model = torch.jit.load("/home/annika/Dokumente/Daten/Promotion/03_Lehrauftrag/SRH/Lectures/challenge/baseline/model.pt", map_location="cpu")   # your exported model
model.eval()
prepare = v2.Compose([
    v2.ToImage(),                                  # PIL image -> tensor (3, H, W)
    v2.Resize((28, 28), antialias=True),           # the size the model was trained on
    v2.ToDtype(torch.float32, scale=True),         # 0...255 -> 0...1
    v2.Normalize(MEAN, STD),                       # exactly like in the starter notebook
])


def classify(image):
    """PIL image -> {class name: probability} for the 3 most likely classes."""
    x = prepare(image.convert("RGB")).unsqueeze(0)          # (1, 3, 28, 28)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top = probs.topk(3)
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="skin lesion image"),
    outputs=gr.Label(num_top_classes=3, label="prediction"),
    title="Skin lesion classifier (DermaMNIST challenge)",
    description="Upload an image of a skin lesion. Student project - not a medical device!",
)

if __name__ == "__main__":
    demo.launch(share=True)   # share=True: public link (needed in Colab); on Hugging Face Spaces it is ignored
