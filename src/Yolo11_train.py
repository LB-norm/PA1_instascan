from ultralytics import YOLO
import torch
import cv2 
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0")

seg_models = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"]
obb_models = ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"]
seg_hp_8_dict = {
    "yolo11n-seg.pt": r"best_HP\seg_8_Batch\tune\best_hyperparameters.yaml",
    "yolo11s-seg.pt": r"best_HP\seg_8_Batch\tune2\best_hyperparameters.yaml", 
    "yolo11m-seg.pt": r"best_HP\seg_8_Batch\tune3\best_hyperparameters.yaml", 
    "yolo11l-seg.pt": r"best_HP\seg_8_Batch\tune4\best_hyperparameters.yaml", 
    "yolo11x-seg.pt": r"best_HP\seg_8_Batch\tune5\best_hyperparameters.yaml"
}

seg_hp_16_dict = {
    "yolo11n-seg.pt": r"best_HP\seg_16_Batch\tune\best_hyperparameters.yaml",
    "yolo11s-seg.pt": r"best_HP\seg_16_Batch\tune2\best_hyperparameters.yaml", 
    "yolo11m-seg.pt": r"best_HP\seg_16_Batch\tune3\best_hyperparameters.yaml", 
    "yolo11l-seg.pt": r"best_HP\seg_16_Batch\tune4\best_hyperparameters.yaml", 
    "yolo11x-seg.pt": r"best_HP\seg_16_Batch\tune5\best_hyperparameters.yaml"
}

obb_hp_8_dict = {
    "yolo11n-obb.pt": r"best_HP\obb_8_Batch\tune\best_hyperparameters.yaml",
    "yolo11s-obb.pt": r"best_HP\obb_8_Batch\tune2\best_hyperparameters.yaml", 
    "yolo11m-obb.pt": r"best_HP\obb_8_Batch\tune3\best_hyperparameters.yaml", 
    "yolo11l-obb.pt": r"best_HP\obb_8_Batch\tune4\best_hyperparameters.yaml", 
    "yolo11x-obb.pt": r"best_HP\obb_8_Batch\tune5\best_hyperparameters.yaml"
}

obb_hp_16_dict = {
    "yolo11n-obb.pt": r"best_HP\obb_16_Batch\tune\best_hyperparameters.yaml",
    "yolo11s-obb.pt": r"best_HP\obb_16_Batch\tune2\best_hyperparameters.yaml", 
    "yolo11m-obb.pt": r"best_HP\obb_16_Batch\tune3\best_hyperparameters.yaml", 
    "yolo11l-obb.pt": r"best_HP\obb_16_Batch\tune4\best_hyperparameters.yaml", 
    "yolo11x-obb.pt": r"best_HP\obb_16_Batch\tune5\best_hyperparameters.yaml"
}

def training_multiple_models(model_dict, project_name):
    for model_path, best_hyp in model_dict.items():
        print(f"Loading model {model_path} with hyperparameters from {best_hyp}")
        model_name = model_path.rstrip(".pt")
        print(model_name)
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Train the model
        train_results = model.train(
            data="C:\Python Projects\datasets\Exponat_img\data.yaml",  # path to dataset YAML
            cfg = best_hyp,
            optimizer="AdamW",
            epochs=50,  # number of training epochs
            imgsz=640,  # training image size
            batch=16,
            device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
            workers=0,
            project=project_name,
            name= model_name,
            single_cls=True #only 1 class
        )

training_multiple_models(obb_hp_16_dict, "OBB_B16")





