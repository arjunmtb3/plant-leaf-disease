import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "plant_disease.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Plant Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide",
)


st.markdown("""
<style>
    
/* Fix upload button and file information contrast */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    background-color: #e8f5ec !important;
    color: #14532d !important;
    border: 1px solid #a7cdb1 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button svg {
    fill: #14532d !important;
    color: #14532d !important;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    opacity: 1;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #37483d !important;
}

[data-testid="stFileUploader"] section button:hover {
    background-color: #d4eadb !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌿 Plant Leaf Disease Classification</h1>
  <p>Upload a leaf image and use a trained CNN model to predict its category.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error("The trained model files are missing.")
    st.markdown(
        "Place `plant_disease.keras` and `class_names.json` inside the "
        "`models/` folder. If you have not trained the model yet, prepare "
        "the dataset and run `python train.py` first."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("📤 Upload Leaf Image")
    st.write("Supported formats: JPG, JPEG, PNG")
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.error("This file could not be opened as an image.")
            st.stop()
    else:
        st.info("Upload a clear photo of a single plant leaf to get started.")
        st.markdown("**Starter classes**")
        st.write("• Tomato healthy")
        st.write("• Tomato early blight")
        st.write("• Tomato late blight")

    predict_clicked = st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

with right:
    st.subheader("🧪 Prediction Result")
    if uploaded_file is None:
        st.markdown(
            '<div class="result"><h3>Waiting for an image</h3>'
            '<p class="muted">Your predicted class and model scores will appear here.</p></div>',
            unsafe_allow_html=True,
        )
    elif predict_clicked:
        try:
            resized = image.resize(IMG_SIZE)
            array = np.asarray(resized, dtype=np.float32)
            array = np.expand_dims(array, axis=0)

            scores = model.predict(array, verbose=0)[0]
            if len(scores) != len(class_names):
                st.error(
                    "The number of model outputs does not match class_names.json. "
                    "Check that both files came from the same training run."
                )
                st.stop()

            
            predicted_index = int(np.argmax(scores))
            predicted_class = class_names[predicted_index]
            confidence = float(scores[predicted_index])

            if confidence < 0.60:
                st.warning(
                    "Low-confidence prediction. Please upload a clear "
                    "image of a single plant leaf. The result may be unreliable."
                )
            else:
                st.success(
                    f"Predicted class: **{predicted_class.replace('_', ' ')}**"
                )

            st.metric("Model confidence", f"{confidence * 100:.2f}%")
            st.progress(max(0.0, min(1.0, confidence)))

            st.markdown("#### Class-wise scores")
            ranked = sorted(
                zip(class_names, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            for name, score in ranked:
                st.write(
                    f"**{name.replace('_', ' ')}** — {float(score) * 100:.2f}%"
                )
                st.progress(max(0.0, min(1.0, float(score))))

            st.caption(
                "Model scores are not guaranteed probabilities of correctness. "
                "This model may misclassify unrelated images or unfamiliar "
                "field conditions."
            )


st.divider()
st.caption("Plant Leaf Disease Classification • B.Tech CSE ML Lab Project")
