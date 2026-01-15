from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []

    for box in results[0].boxes:
        detections.append({
            "cls": int(box.cls),
            "conf": float(box.conf),
            "xyxy": box.xyxy.tolist()
        })

    return {"detections": detections}

