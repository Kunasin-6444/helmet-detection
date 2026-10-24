"""UI owner: customize this file. Inference lives in detector.py."""

import hashlib
from io import BytesIO
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from detector import load_model, predict_frame, process_video


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(page_title="Helmet Check", page_icon="🪖", layout="wide")


def get_model():
    # Session-local model avoids concurrent calls to a shared mutable predictor.
    stamp = (MODEL_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_size)
    if st.session_state.get("model_stamp") != stamp:
        st.session_state.model = load_model(MODEL_PATH)
        st.session_state.model_stamp = stamp
    return st.session_state.model


st.title("🪖 Helmet Check")
st.write("ตรวจจับการสวมหมวกนิรภัยจากภาพและวิดีโอ")
st.caption("ต้นแบบสำหรับมินิโปรเจกต์ • YOLO11n • ภาพและวิดีโอประมวลผลในเครื่อง")

with st.sidebar:
    st.header("ตั้งค่าการตรวจจับ")
    confidence = st.slider("ค่าความมั่นใจขั้นต่ำ", 0.05, 0.95, 0.25, 0.05)
    st.caption("ค่าต่ำอาจตรวจพบมากขึ้น แต่มีโอกาสตรวจผิดเพิ่มขึ้น")
    st.divider()
    model_ready = MODEL_PATH.is_file()
    if model_ready:
        st.success("พบไฟล์โมเดลแล้ว")
    else:
        st.warning("ยังไม่ได้เพิ่มโมเดล")
        st.code("models/best.pt")
    st.markdown("🟢 **Helmet:** สวมหมวก  \n🔴 **No helmet:** ไม่สวมหมวก")
    st.caption("จำนวนเป็นสิ่งที่ตรวจพบต่อเฟรม ไม่ใช่จำนวนพนักงานไม่ซ้ำทั้งคลิป")

uploaded = st.file_uploader(
    "เลือกภาพหรือวิดีโอ", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    help="ไฟล์ไม่เกิน 100 MB • วิดีโอไม่เกิน 60 วินาที",
)
st.caption("วิดีโอผลลัพธ์ไม่มีเสียง และลดด้านยาวเหลือไม่เกิน 1,280 พิกเซล")

if uploaded is None:
    st.session_state.pop("output", None)
    st.info("เริ่มต้นโดยอัปโหลดภาพหรือคลิปสั้น แล้วกดเริ่มตรวจจับ")
    with st.expander("วิธีเริ่มต้นสำหรับผู้พัฒนา"):
        st.markdown("1. วางโมเดลที่ `models/best.pt`\n2. ติดตั้ง `requirements.txt`\n"
                    "3. อัปโหลดภาพ JPG/PNG หรือคลิปสั้น\n4. เริ่มตรวจจับและดาวน์โหลดผล")
    st.stop()

payload = uploaded.getvalue()
is_image = Path(uploaded.name).suffix.lower() in {".jpg", ".jpeg", ".png"}
stamp = (MODEL_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_size) if model_ready else None
job_key = (hashlib.sha256(payload).hexdigest(), uploaded.name, confidence, stamp)
if st.session_state.get("job_key") != job_key:
    st.session_state.pop("output", None)
    st.session_state.job_key = job_key

left, right = st.columns(2)
image = None
with left:
    st.subheader("ไฟล์ต้นฉบับ")
    if is_image:
        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(payload))).convert("RGB")
            st.image(image, use_container_width=True)
        except Exception:
            st.error("อ่านภาพไม่ได้ กรุณาเลือกไฟล์ JPG หรือ PNG ที่สมบูรณ์")
            st.stop()
    else:
        st.video(payload, format=uploaded.type or "video/mp4")
        st.caption("หากเบราว์เซอร์เล่นต้นฉบับไม่ได้ ให้ลอง MP4 แบบ H.264")

if st.button("เริ่มตรวจจับ", type="primary", disabled=not model_ready):
    st.session_state.pop("output", None)
    try:
        with st.spinner("กำลังประมวลผล…"):
            model = get_model()
            if is_image:
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                annotated, counts = predict_frame(model, frame, confidence)
                ok, encoded = cv2.imencode(".png", annotated)
                if not ok:
                    raise RuntimeError("บันทึกภาพผลลัพธ์ไม่ได้")
                st.session_state.output = {"data": encoded.tobytes(), "counts": counts}
            else:
                bar = st.progress(0, text="กำลังอ่านวิดีโอ")

                def update_progress(done, total):
                    fraction = min(done / total, 1.0) if total > 0 else 0.0
                    bar.progress(fraction, text=f"ประมวลผลแล้ว {done} เฟรม")

                with tempfile.TemporaryDirectory(prefix="helmet_") as folder:
                    source = Path(folder) / ("input" + Path(uploaded.name).suffix.lower())
                    destination = Path(folder) / "result.mp4"
                    source.write_bytes(payload)
                    info = process_video(model, source, destination, confidence, update_progress)
                    st.session_state.output = {"data": destination.read_bytes(), "info": info}
                bar.empty()
    except Exception as exc:
        st.error(f"ประมวลผลไม่สำเร็จ: {exc}")

with right:
    st.subheader("ผลการตรวจจับ")
    output = st.session_state.get("output")
    if output:
        if is_image:
            st.image(output["data"], use_container_width=True)
            a, b = st.columns(2)
            a.metric("สวมหมวก (จุด)", output["counts"]["helmet"])
            b.metric("ไม่สวมหมวก (จุด)", output["counts"]["head"])
            if sum(output["counts"].values()) == 0:
                st.info("ไม่พบหัวหรือหมวกตามเกณฑ์นี้ ไม่ได้หมายความว่าทุกคนสวมหมวก")
        else:
            st.video(output["data"])
            st.caption(f"ประมวลผล {output['info']['frames']} เฟรม • จำนวนแสดงบนแต่ละเฟรม")
        suffix, mime = ("png", "image/png") if is_image else ("mp4", "video/mp4")
        st.download_button("ดาวน์โหลดผลลัพธ์", output["data"],
                           file_name=f"{Path(uploaded.name).stem}_detected.{suffix}", mime=mime)
    else:
        st.info("ผลลัพธ์จะแสดงที่นี่หลังประมวลผล")

st.caption("ผลอาจคลาดเคลื่อนเมื่อศีรษะถูกบัง อยู่ไกล หรือแสงไม่เพียงพอ • ไม่ตรวจการรัดสายหมวก")
