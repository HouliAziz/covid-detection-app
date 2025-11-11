import streamlit as st
from PIL import Image
import torch
import numpy as np
import h5py
import json
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Page config
st.set_page_config(
    page_title="COVID-19 Detection System",
    page_icon="hospital",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .prediction-box {padding: 2rem; border-radius: 10px; margin: 1rem 0; text-align: center;}
    .covid {background-color: #ffebee; border-left: 5px solid #f44336;}
    .normal {background-color: #e8f5e9; border-left: 5px solid #4caf50;}
    .viral-pneumonia {background-color: #fff3e0; border-left: 5px solid #ff9800;}
    .lung-opacity {background-color: #e3f2fd; border-left: 5px solid #2196f3;}
    .confidence-bar {height: 20px; background-color: #f0f0f0; border-radius: 10px; margin: 10px 0;}
    .confidence-fill {height: 100%; border-radius: 10px; text-align: center; color: white; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Class order
CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']


@st.cache_resource
def load_model_and_processor():
    try:
        processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=4,
            ignore_mismatched_sizes=True
        )
        with h5py.File("model/covid_model.h5", 'r') as f:
            state_dict = {k: torch.tensor(v[:])
                          for k, v in f['state_dict'].items()}
            model.load_state_dict(state_dict)
            id2label = json.loads(f['config/id2label'][()].decode('utf-8'))
            label2id = json.loads(f['config/label2id'][()].decode('utf-8'))
            model.config.id2label = {int(k): v for k, v in id2label.items()}
            model.config.label2id = label2id
        model.eval()
        st.success("Model loaded successfully!")
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure model/covid_model.h5 exists")
        return None, None

# FIXED INDENTATION HERE


def apply_mask(image_pil, mask_pil=None):
    if mask_pil is None:
        mask_pil = image_pil.convert("L")
    if mask_pil.size != image_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, Image.BILINEAR)

    img_np = np.array(image_pil.convert("L")).astype(np.float32)
    mask_np = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
    masked = img_np * mask_np
    masked_rgb = np.stack([masked] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(masked_rgb)


def predict_image(image_pil, mask_pil, model, processor):
    masked_img = apply_mask(image_pil, mask_pil)
    inputs = processor(images=masked_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()[0], masked_img


# Load model
model, processor = load_model_and_processor()


def main():
    st.markdown('<h1 class="main-header">COVID-19 Detection System</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("About")
        st.info("Vision Transformer (ViT) + Lung masks\nTrained on 21k+ X-rays")
        st.header("Warning")
        st.warning("Educational use only")
        st.header("Model")
        st.write("• ViT-Base\n• 92.2% accuracy\n• Uses lung masks")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload X-Ray")
        uploaded_img = st.file_uploader("Image", type=['png', 'jpg', 'jpeg'])
        uploaded_mask = st.file_uploader("Lung mask (optional)", type=[
                                         'png', 'jpg', 'jpeg'], key="mask")
        if uploaded_img:
            image = Image.open(uploaded_img)
            st.image(image, caption="Original X-ray", use_column_width=True)

    with col2:
        st.subheader("Results")
        if uploaded_img and model:
            with st.spinner("Analyzing..."):
                mask_pil = Image.open(uploaded_mask) if uploaded_mask else None
                probs, masked_img = predict_image(
                    image, mask_pil, model, processor)
                pred_idx = np.argmax(probs)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100

                st.image(masked_img, caption="Input to model", width=200)

                color = {'COVID': 'covid', 'Normal': 'normal',
                         'Viral Pneumonia': 'viral-pneumonia', 'Lung_Opacity': 'lung-opacity'}[pred_class]
                emoji = {'COVID': 'virus', 'Normal': 'check mark',
                         'Viral Pneumonia': 'lungs', 'Lung_Opacity': 'magnifying glass'}[pred_class]

                st.markdown(f"""
                    <div class="prediction-box {color}">
                        <h2>{emoji} {pred_class}</h2>
                        <p><strong>Confidence: {confidence:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("All Scores")
                for i, name in enumerate(CLASS_NAMES):
                    p = probs[i] * 100
                    col = {'COVID': '#f44336', 'Normal': '#4caf50',
                           'Viral Pneumonia': '#ff9800', 'Lung_Opacity': '#2196f3'}[name]
                    st.write(f"**{name}:** {p:.2f}%")
                    st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {p}%; background-color: {col};">
                                {p:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.subheader("Recommendation")
                if pred_class == "COVID":
                    st.error("URGENT: Isolate + PCR + Contact doctor NOW")
                elif pred_class == "Viral Pneumonia":
                    st.warning("See doctor soon")
                elif pred_class == "Lung_Opacity":
                    st.warning("Follow-up imaging recommended")
                else:
                    st.success("No acute findings detected")
        elif uploaded_img:
            st.error("Model failed to load")
        else:
            st.info("Upload an X-ray image to start")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Made with Streamlit + PyTorch ViT | Educational Only</p>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
