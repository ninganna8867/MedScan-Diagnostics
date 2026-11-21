
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Allow importing model.py from src
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import create_model

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_DIR / "models" / "experiment_config2.pth"

CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model once when application starts
model = create_model(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("MedScan model loaded successfully!")


def predict_xray(image):
    if image is None:
        return {"Error": 1.0}

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    results = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return results


demo = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=gr.Label(
        num_top_classes=3,
        label="Model Prediction Probabilities"
    ),
    title="MedScan Diagnostics",
    description=(
        "Educational chest X-ray classification demonstration. "
        "Upload an image to view model-predicted class probabilities. "
        "Not intended for clinical diagnosis."
    )
)

if __name__ == "__main__":
    demo.launch(show_error=True)
