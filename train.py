# Author: G N Paneendra
# Date: 18 August 2025

from ultralytics import YOLO
import datetime

t1=datetime.datetime.now()
print(f"\nStarting time: {t1.strftime('%H:%M:%S')}\n")

model = YOLO('yolo11n')

model.train(
    data='/burst_detection_using_yolo/data.yaml',
    epochs=200,
    imgsz=640,
    device=0
)

t2 = datetime.now()
print(f"\nEnding time: {t2.strftime('%H:%M:%S')}")

tf = t2 - t1
print(f"\nTotal time: {tf}\n")

# Weights saved location will be displayed in the end
