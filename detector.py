"""Inference helpers, independent of Streamlit. Frames use OpenCV BGR order."""

from pathlib import Path
import math
import subprocess

import cv2


COLORS = {"helmet": (75, 170, 35), "head": (50, 50, 220)}
LABELS = {"helmet": "Helmet", "head": "No helmet"}
MAX_VIDEO_SECONDS = 60


def validate_names(names):
    values = names.values() if isinstance(names, dict) else names
    if set(values) != {"head", "helmet"}:
        raise ValueError("โมเดลต้องมีสองคลาส head และ helmet กรุณาตรวจ best.pt")


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("ไม่พบโมเดล กรุณาวางไฟล์ที่ models/best.pt")
    from ultralytics import YOLO

    model = YOLO(str(path))
    validate_names(model.names)
    return model


def annotate(frame, result):
    """Use class names, not fixed IDs; return per-frame counts, never people IDs."""
    validate_names(result.names)
    output = frame.copy()
    counts = {"helmet": 0, "head": 0}
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        counts[name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = COLORS[name]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{LABELS[name]} {float(box.conf.item()):.2f}"
        cv2.putText(output, text, (max(0, x1), max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    summary = f"Helmet: {counts['helmet']} | No helmet: {counts['head']}"
    cv2.putText(output, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 4)
    cv2.putText(output, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    return output, counts


def predict_frame(model, frame, confidence):
    result = model.predict(source=frame, conf=confidence, imgsz=640, verbose=False)[0]
    return annotate(frame, result)


def process_video(model, source, destination, confidence, progress=None):
    """Create a silent H.264 MP4 for browser playback; process every decoded frame."""
    import imageio_ffmpeg

    capture = cv2.VideoCapture(str(source))
    writer = None
    intermediate = Path(destination).with_name("intermediate.mp4")
    processed = 0
    try:
        if not capture.isOpened():
            raise ValueError("เปิดวิดีโอไม่ได้ ลองใช้ไฟล์ MP4 ที่เล่นได้ตามปกติ")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("อ่าน FPS ของวิดีโอไม่ได้ กรุณาแปลงเป็น MP4 ก่อน")
        if total > fps * MAX_VIDEO_SECONDS + 1:
            raise ValueError("ต้นแบบรองรับคลิปยาวไม่เกิน 60 วินาที กรุณาตัดคลิปก่อน")

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            processed += 1
            if processed > fps * MAX_VIDEO_SECONDS + 1:
                raise ValueError("ต้นแบบรองรับคลิปยาวไม่เกิน 60 วินาที")
            # Bound output resolution for a small local demo. Preserve aspect ratio.
            h, w = frame.shape[:2]
            scale = min(1.0, 1280 / max(h, w))
            size = (max(2, int(w * scale) // 2 * 2),
                    max(2, int(h * scale) // 2 * 2))
            frame = cv2.resize(frame, size)
            if writer is None:
                writer = cv2.VideoWriter(str(intermediate),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("สร้างไฟล์วิดีโอไม่ได้")
            annotated, _ = predict_frame(model, frame, confidence)
            writer.write(annotated)
            if progress and (processed == 1 or processed % 10 == 0):
                progress(processed, total)
        if processed == 0:
            raise ValueError("วิดีโอไม่มีเฟรมที่อ่านได้")
        if total > 0 and processed < total - 1:
            raise ValueError("อ่านวิดีโอไม่ครบ ไฟล์อาจเสียหาย กรุณาลองแปลงเป็น MP4")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    # MP4V from OpenCV often cannot play in browsers. Re-encode to H.264.
    conversion = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(intermediate),
         "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
        capture_output=True, text=True, timeout=180,
    )
    if conversion.returncode != 0:
        raise RuntimeError("แปลงวิดีโอเป็น H.264 ไม่สำเร็จ: " + conversion.stderr[-600:])
    if progress:
        progress(processed, processed)
    return {"frames": processed, "fps": fps, "seconds": processed / fps}
