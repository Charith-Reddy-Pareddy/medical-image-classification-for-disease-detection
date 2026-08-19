import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import IMAGE_SIZE, MODEL_DIR
from src.data.dataset import get_transforms
from src.interpret.gradcam import compute_gradcam
from src.models import MODEL_REGISTRY

st.set_page_config(page_title="Chest X-Ray Pneumonia Screen", layout="centered")
st.title("Chest X-Ray Pneumonia Screen")
st.caption(
    "Research demo, not a diagnostic tool. Trained on a single-institution pediatric dataset -- "
    "see the project README for measured domain-shift and shortcut-learning failure modes before "
    "reading anything into the prediction below."
)


@st.cache_resource
def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_REGISTRY[model_name]()
    ckpt_path = MODEL_DIR / f"{model_name}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


available_models = [name for name in MODEL_REGISTRY if (MODEL_DIR / f"{name}.pt").exists()]
if not available_models:
    st.error(f"No trained checkpoints found in {MODEL_DIR}. Run scripts/train.py first.")
    st.stop()

model_name = st.selectbox("Model", available_models)
uploaded = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    model, device = load_model(model_name)
    image = Image.open(uploaded).convert("RGB")

    transform = get_transforms(IMAGE_SIZE, train=False)
    tensor = transform(image).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor.unsqueeze(0)).squeeze(1)).item()
    predicted_class = int(prob >= 0.5)
    label = "PNEUMONIA" if predicted_class == 1 else "NORMAL"

    cam = compute_gradcam(model, tensor, category=predicted_class)
    display_img = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(display_img, cam, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Input (resized)", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    st.metric("Prediction", label, f"{prob:.1%} pneumonia probability")
