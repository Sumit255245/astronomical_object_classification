# SDSS — Streamlit Web App
# Run with: streamlit run app.py

import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    bundle = joblib.load("models/model.joblib")
    return bundle["model"], bundle["scaler"], bundle["label_encoder"]

model, scaler, le = load_model()

# Page config
st.set_page_config(page_title="SDSS Star Classifier", page_icon="🔭", layout="centered")

st.title("🔭 Astronomical Object Classifier")
st.markdown("Enter the spectral measurements of an object to classify it as a **Star**, **Galaxy**, or **Quasar**.")
st.divider()

# Input sliders
col1, col2 = st.columns(2)

with col1:
    st.subheader("Photometric Bands")
    u = st.number_input("u band (ultraviolet)", 10.0, 35.0, 18.0, step=0.1)
    g = st.number_input("g band (green)",       10.0, 35.0, 17.8, step=0.1)
    r = st.number_input("r band (red)",         10.0, 35.0, 17.6, step=0.1)
    i = st.number_input("i band (near-IR)",     10.0, 35.0, 17.5, step=0.1)
    z = st.number_input("z band (infrared)",    10.0, 35.0, 17.4, step=0.1)

with col2:
    st.subheader("Spectral")
    redshift = st.number_input("Redshift", 0.0, 6.0, 0.0001, step=0.0001, format="%.4f")

st.divider()

# Predict
if st.button("🔍 Classify Object", use_container_width=True):
    # Build feature array (same order as Step 3)
    features = np.array([[u, g, r, i, z, redshift,
                          u-g, g-r, r-i, i-z]])
    scaled = scaler.transform(features)
    pred   = le.inverse_transform(model.predict(scaled))[0]
    proba  = model.predict_proba(scaled)[0]

    # Result display
    CLASS_EMOJI = {"GALAXY": "🌌", "STAR": "🌟", "QSO": "⚡"}
    CLASS_DESC  = {
        "GALAXY": "A system of millions or billions of stars.",
        "STAR"  : "A luminous ball of plasma held together by gravity.",
        "QSO"   : "A quasi-stellar object — an extremely luminous galaxy core.",
    }

    st.success(f"### {CLASS_EMOJI[pred]} Prediction: **{pred}**")
    st.markdown(f"*{CLASS_DESC[pred]}*")

    # Confidence bars
    st.markdown("#### Confidence")
    for cls, prob in zip(le.classes_, proba):
        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")