import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🪖 Deteksi Pengendara Helm & Non-Helm")
st.markdown("Upload gambar – model akan mendeteksi pengendara yang menggunakan helm dan tidak.")

# 🌟 Path model berdasarkan direktori tempat app.py berada
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "best.pt")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di `{MODEL_PATH}`. Pastikan best.pt ada di folder app/models/")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    # Confidence threshold slider
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Semakin rendah, semakin banyak objek yang dideteksi (termasuk false positive)."
    )

    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    with st.spinner("Mendeteksi..."):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path)

        results = model(tmp_path, conf=conf_threshold)   # pakai threshold dari slider
        os.unlink(tmp_path)

    result = results[0]
    annotated_img = result.plot()  # numpy BGR

    # Konversi BGR ke RGB untuk Streamlit
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(annotated_rgb, use_container_width=True, channels="RGB")

    boxes = result.boxes
    if boxes is not None and len(boxes.cls) > 0:
        total = len(boxes.cls)
        class_ids = boxes.cls.int().tolist()
        confs = boxes.conf.float().tolist()
        with_helmet = class_ids.count(0)
        without_helmet = class_ids.count(1)

        st.success(f"**Total pengendara terdeteksi: {total}**")
        col_a, col_b = st.columns(2)
        col_a.metric("✅ Dengan Helm", with_helmet)
        col_b.metric("❌ Tanpa Helm", without_helmet)

        # Tampilkan detail confidence setiap deteksi
        st.markdown("### Detail Confidence Setiap Deteksi")
        for i, (cls_id, conf) in enumerate(zip(class_ids, confs), 1):
            cls_name = "Dengan Helm" if cls_id == 0 else "Tanpa Helm"
            st.write(f"{i}. {cls_name}: {conf:.2f}")
    else:
        st.warning("Tidak ada objek terdeteksi. Coba turunkan confidence threshold di slider.")

    if boxes is not None and len(boxes.conf) > 0:
        avg_conf = float(boxes.conf.mean())
        st.caption(f"Rata-rata confidence: {avg_conf:.2f}")