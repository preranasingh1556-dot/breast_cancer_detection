"""
Breast Cancer Detection - Streamlit Web App
Run with: streamlit run app/app.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from gradcam import GradCAM
import streamlit as st
from torchvision import models, transforms
from PIL 
import Image
import json
import google.generativeai as genai
import os 
from dotenv import load_dotenv
load_dotenv() 
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model", "breast_cancer_model.pth")
CLASS_NAMES  = ["Non-Cancerous (IDC Negative)", "Cancerous (IDC Positive)"]
IMG_SIZE     = 50
DISPLAY_SIZE = 224          # resize for display / Grad-CAM overlay
DEVICE       = torch.device("cpu")
# ───────────────────────────────────────────────────────────────────────────────


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def get_ai_explanation(pred_label, confidence, class_probs):
    try:
        model = genai.GenerativeModel("gemini-3.5-flash")
        prompt = f"""
        You are an expert in histopathology and breast cancer detection.
        A model has predicted that a histopathology image is **{pred_label}** with a confidence of {confidence:.1f}%.
        The class probabilities are:
        - Non-Cancerous: {class_probs[0]*100:.1f}%
        - Cancerous: {class_probs[1]*100:.1f}%
    
    Write your response as a single, well-structured flowing paragraph (do NOT use bullet points, numbered lists, or headers). Explain what this prediction means, the implications for the patient, and any relevant caution - in plain English, under 250 words.
        """

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 500}
        )
        return response.text
    except Exception as e:
        return f"AI explanation unavailable right now. ({e})"


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def show_gradcam_figure(original_img, overlay, heatmap):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0e1117")

    images  = [original_img, heatmap, overlay]
    titles  = ["Original Image", "Grad-CAM Heatmap", "Overlay (Explainability)"]
    borders = ["#4a90d9", "#e74c3c", "#2ecc71"]

    for ax, img, title, color in zip(axes, images, titles, borders):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─── Page Layout ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Breast Cancer Detection",
        page_icon="🩺",
        layout="wide",
    )

    # Header
    st.markdown("""
        <h1 style='text-align:center; color:#4a90d9;'>
            🩺 Breast Cancer Detection with Grad-CAM Explainability
        </h1>
        <p style='text-align:center; color:#aaa;'>
            Upload a histopathology patch image to detect
            <b>Invasive Ductal Carcinoma (IDC)</b> and visualize
            which regions influenced the prediction.
        </p>
        <hr style='border-color:#333;'>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 2], gap="large")

    with col_upload:
        st.subheader("Upload Image")
        uploaded = st.file_uploader(
            "Choose a histopathology image (.jpg / .png)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            st.markdown("---")
            st.markdown("**What is IDC?**")
            st.info(
                "Invasive Ductal Carcinoma (IDC) is the most common "
                "subtype of breast cancer (~80% of cases). Early detection "
                "significantly improves survival rates."
            )

    if uploaded:
        with col_result:
            # Check model exists
            if not os.path.exists(MODEL_PATH):
                st.error(
                    "Trained model not found at `app/model/breast_cancer_model.pth`.\n\n"
                    "Please run `python train.py` first to train the model."
                )
                return

            with st.spinner("Analyzing image..."):
                model   = load_model()
                gradcam = GradCAM(model, target_layer=model.layer2[-1])

                input_tensor = preprocess(image)
                cam, class_idx, probs = gradcam.generate(input_tensor)

                # Prepare display image
                display_img = np.array(image.resize((DISPLAY_SIZE, DISPLAY_SIZE)))

                cam_resized = cv2.resize(cam, (display_img.shape[1], display_img.shape[0]))
                overlay = gradcam.overlay_on_image(cam_resized, display_img)
                heatmap = cam_resized

            # ── Prediction ──────────────────────────────────────────────────
            st.subheader("Prediction Result")

            pred_label = CLASS_NAMES[class_idx]
            confidence = probs[class_idx] * 100

            if class_idx == 1:
                st.error(f"**{pred_label}**  —  Confidence: {confidence:.1f}%")
            else:
                st.success(f"**{pred_label}**  —  Confidence: {confidence:.1f}%")

            # Probability bars
            st.markdown("**Class Probabilities:**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Non-Cancerous", f"{probs[0]*100:.1f}%")
                st.progress(float(probs[0]))
            with c2:
                st.metric("Cancerous", f"{probs[1]*100:.1f}%")
                st.progress(float(probs[1]))

            st.markdown("---")

            # ── Grad-CAM ────────────────────────────────────────────────────
            st.subheader("Grad-CAM Visualization")
            st.caption(
                "Red/warm regions = high influence on prediction  |  "
                "Blue/cool regions = low influence"
            )
            show_gradcam_figure(display_img, overlay, heatmap)

            st.markdown("---")

            # ── AI Explanation ──────────────────────────────────────────────
            st.subheader("AI-Generated Explanation")
            with st.spinner("Generating explanation..."):
                explanation = get_ai_explanation(pred_label, confidence, probs)
            
            st.markdown(f"""
    <div style='
        background-color:#1a1c24;
        border-left:4px solid #4a90d9;
        border-radius:8px;
        padding:20px 24px;
        line-height:1.7;
        color:#dddddd;
        font-size:15px;
        text-align:justify;
        margin-top:8px;
    '>
        {explanation}
    </div>
""", unsafe_allow_html=True)

            # ── Metrics ──────────────────────────────────────────────────────
            st.subheader("Model Performance")
            metrics = load_metrics()
            if metrics:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
                m2.metric("Precision", f"{metrics.get('precision', 0):.1f}%")
                m3.metric("Recall", f"{metrics.get('recall', 0):.1f}%")
                m4.metric("F1_Score", f"{metrics.get('f1', 0):.1f}%")

            else:
             with col_result:
                   st.markdown("""
                <div style='
                    display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    height:300px; border:2px dashed #333; border-radius:12px;
                    color:#555; font-size:16px;
                '>
                    <p>Upload an image on the left to see results here</p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
