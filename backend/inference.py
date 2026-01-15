from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # 軽量モデル

img = cv2.imread("test.jpg")
results = model(img)

annotated = results[0].plot()
cv2.imwrite("result.jpg", annotated)
