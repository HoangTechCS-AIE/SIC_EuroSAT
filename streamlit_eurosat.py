# streamlit_eurosat.py
import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="EuroSAT Classifier 🌍", layout="centered")

st.title("🛰️ Dự đoán ảnh vệ tinh EuroSAT")

uploaded_file = st.file_uploader("📤 Tải ảnh vệ tinh RGB", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh đã tải", use_column_width=True)

    img = Image.open(uploaded_file).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    if st.button("✨ Dự đoán"):
        res = requests.post("http://localhost:8000/predict", json={"image_base64": img_base64})
        if res.ok:
            data = res.json()
            st.success(f"📌 Kết quả: **{data['label']}** (class {data['index']})")
        else:
            st.error("❌ Lỗi từ API:")
            st.text(res.text)
