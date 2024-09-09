# import os
# print(os.getcwd())
from ultralytics import YOLO

model = YOLO("models/yolo8_trained.pt")
model.predict('input/vid_1.mp4', conf=0.2, save=True)
