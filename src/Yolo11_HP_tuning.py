from ultralytics import YOLO
import torch
import cv2 
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

seg_models = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"]
obb_models = ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"]

device = torch.device("cuda:0")

train_args = {
    "data": "datasets\Exponat_img\data.yaml",
    "epochs": 30,          
    "batch": 8,            # 8/16
    "imgsz": 640,          # 640 Pretrained
}

def tuning(model_list, train_args):
    for modelname in model_list:
        model = YOLO(modelname)
        model.to(device)
        model.tune(
            **train_args,
            iterations=30,
            optimizer="AdamW",
            plots=False,
            save=False,
            val=False
            )

tuning(obb_models, train_args)