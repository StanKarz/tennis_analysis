# import os
# print(os.getcwd())
from ultralytics import YOLO

model = YOLO("models/yolov8_trained.pt")
model.track('input/input_video.mp4', conf=0.2, save=True)
