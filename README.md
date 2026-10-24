# Helmet Check — โครงมินิโปรเจกต์งานคู่

แอป Streamlit สำหรับอัปโหลดภาพหรือวิดีโอ ตรวจจับ `head` / `helmet` ด้วยโมเดล YOLO11n ที่ฝึกแล้ว และดาวน์โหลดผลลัพธ์
ใช้ในเครื่อง ไม่ต้องใช้ Roboflow API key และไม่ต้องฝึกใหม่

## เริ่มใช้งานบน Windows

ใช้ Python 3.11 หรือ 3.12 ทั้งเจ้าของโมเดลและเพื่อนใช้ขั้นตอนเดียวกันได้ เปิด Terminal ในโฟลเดอร์ที่มี `requirements.txt` และ `app.py` แล้วเลือกคำสั่งตาม shell ที่ใช้

### Git Bash

สำหรับเครื่องใหม่ สร้าง virtual environment ก่อน (ถ้ามี `.venv` แล้วให้ข้ามบรรทัดนี้):

```bash
python -m venv .venv
```

ติดตั้งไลบรารีทั้งหมด แล้วเปิดแอป:

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m streamlit run app.py
```

หากเพื่อนต้องการทำเฉพาะ UI ก่อน ใช้คำสั่งนี้แทนการติดตั้ง requirements ทั้งหมด:

```bash
./.venv/Scripts/python.exe -m pip install -r requirements-ui.txt
./.venv/Scripts/python.exe -m streamlit run app.py
```

เมื่อพร้อมใช้โมเดล ให้ติดตั้ง `requirements.txt` เพิ่มและวางไฟล์ที่ `models/best.pt`
ครั้งถัดไปเปิดแอปด้วยคำสั่ง `streamlit run` ด้านบนได้เลย ไม่ต้องสร้าง `.venv` หรือติดตั้งซ้ำ เว้นแต่มีการแก้ requirements
แต่ละคนสร้าง `.venv` ในเครื่องตัวเอง ไม่ต้องส่งโฟลเดอร์นี้ให้กัน

### PowerShell

รันทีละบรรทัด โดยข้ามบรรทัดสร้าง `.venv` หากมีอยู่แล้ว:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

ไม่ต้อง activate environment จึงไม่ติดข้อจำกัด ExecutionPolicy ของ PowerShell
เปิด URL ที่ terminal แสดง โดยทั่วไปคือ http://localhost:8501
หยุดแอปด้วย Ctrl+C ใน terminal

ก่อนตรวจจับ ให้วางโมเดลจาก Colab ที่ `models/best.pt` (ชื่อไฟล์ตรงตามนี้)
หากต้องการทำ UI ก่อนโดยยังไม่ติดตั้ง YOLO ใช้ `requirements-ui.txt` แทนได้
หน้าแอปยังเปิดได้เมื่อไม่มีโมเดล โดยปุ่มตรวจจับจะปิดไว้

เวอร์ชัน Ultralytics ใน `requirements.txt` อ้างอิง log การฝึกที่ส่งมาคือ `8.4.184`
หาก pip หาเวอร์ชันนี้ไม่พบ ให้ตรวจ `ultralytics.__version__` ใน Colab อีกครั้งและปรับไฟล์ให้ตรงก่อนส่งต่อ
ยังไม่รวมไฟล์โมเดลในโครงนี้ ต้องส่ง `best.pt` ให้เพื่อนแยกต่างหาก

## โครงสร้างและเจ้าของงาน

```text
app.py                  # เพื่อน: หน้า UI, upload, preview, download
detector.py             # คุณ: load_model, predict_frame, process_video
models/best.pt          # คุณ: นำไฟล์จาก Colab มาวาง (ไม่อยู่ใน Git)
.streamlit/config.toml  # เพื่อน: สีธีม และขนาดไฟล์อัปโหลด
requirements.txt        # ไลบรารีทั้งหมดสำหรับใช้งานจริง
requirements-ui.txt     # ไลบรารีสำหรับทำ UI ก่อนมีโมเดล
tests/verify_pipeline.py # ตรวจ pipeline ด้วยโมเดลจำลอง ไม่ใช่ผลวัดความแม่นยำ
HANDOFF.md              # รายการงานต่อและเกณฑ์ทดสอบร่วมกัน
```

## สิ่งที่โครงนี้เตรียมไว้

- อัปโหลด JPG, JPEG, PNG, MP4, MOV, AVI ขนาดไม่เกิน 100 MB
- ปรับ confidence threshold (ค่าเริ่มต้น 0.25)
- อ่านชื่อคลาสจากโมเดล ไม่ผูกความหมายกับเลขคลาสแบบตายตัว
- วาด `Helmet` สีเขียวและ `No helmet` สีแดง พร้อม confidence
- แสดงและดาวน์โหลดภาพ PNG หรือวิดีโอ H.264 MP4
- เก็บผลใน session เพื่อกดดาวน์โหลดแล้วไม่ต้องประมวลผลใหม่
- เมื่อเปลี่ยนไฟล์, confidence หรือไฟล์โมเดล จะล้างผลเก่า
- ใช้ไฟล์ชั่วคราวที่ถูกล้างหลังประมวลผลวิดีโอ

กรอบในไฟล์ใช้ภาษาอังกฤษเพื่อให้ OpenCV แสดงข้อความได้โดยไม่ต้องติดตั้งฟอนต์ไทยเพิ่ม
หน้า UI เป็นภาษาไทย เพื่อนสามารถเพิ่มฟอนต์ไทยในส่วนวาดกรอบภายหลังได้

วิดีโอจำกัดไม่เกิน 60 วินาที ลดด้านยาวไม่เกิน 1,280 พิกเซล และไม่มีเสียงในผลลัพธ์
ประมวลผลครบทุกเฟรมที่อ่านได้ โดยจำนวนในคลิปเป็นจำนวนต่อเฟรม ไม่ใช่คนไม่ซ้ำทั้งคลิป
แอปใช้ imageio-ffmpeg เพื่อแปลง MP4V จาก OpenCV เป็น H.264 สำหรับเบราว์เซอร์
MOV/AVI บาง codec อาจเล่นตัวอย่างในเบราว์เซอร์ไม่ได้ ให้ลองแปลงเป็น H.264 MP4

ไม่มีการระบุตัวบุคคล การติดตาม ID หรือการตรวจว่ารัดสายหมวกถูกต้อง
ไม่พบหมวกไม่ได้แปลว่าไม่สวมหมวก: สถานะไม่สวมหมวกมาจากการตรวจคลาส `head`
หากตรวจไม่พบอะไร ระบบจะแจ้งว่าไม่พบตามเกณฑ์ ไม่สรุปว่าปลอดภัย

## ผลชุด Test จาก Colab ที่ผู้ใช้รายงาน

706 ภาพ / 2,641 วัตถุ: Precision 0.9402, Recall 0.9412, mAP50 0.9739, mAP50–95 0.6784
ชื่อคลาส `{0: 'head', 1: 'helmet'}`
ผลนี้เป็นการประเมิน dataset ไม่ใช่ความแม่นยำที่รับรองสำหรับวิดีโอใหม่หรือสถานที่จริง

## ตรวจโครงโดยไม่ใช้ best.pt

Git Bash:

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -p "verify_*.py" -v
```

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "verify_*.py" -v
```

ทดสอบด้วยผลตรวจจำลองและวิดีโอสังเคราะห์ เพื่อดู mapping, การบันทึกภาพ/วิดีโอ และการแจ้งข้อผิดพลาด
ต้องทดสอบกับโมเดลจริงเพิ่มเติมหลังวาง `best.pt`

## แหล่งอ้างอิง

- Dataset: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/5
- YOLO11: https://docs.ultralytics.com/models/yolo11/
- Streamlit: https://docs.streamlit.io/get-started/installation
