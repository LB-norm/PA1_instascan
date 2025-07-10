from ultralytics import YOLO
import torch
import cv2 
import numpy as np
import os
import pandas as pd


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0")

model_folder = r"Trained_models"
def get_yolo_model_paths(project_folder):
    """
    Retrieve the paths of trained YOLO models named 'best.pt'.

    Args:
        project_folder (str): Path to the main project folder.

    Returns:
        dict: A dictionary with keys as 'Batch folder name + model name'
              and values as the path to the corresponding 'best.pt' file.
    """
    model_paths = {}

    # Iterate over the batch folders
    for batch_folder in os.listdir(project_folder):
        batch_folder_path = os.path.join(project_folder, batch_folder)

        # Ensure it is a directory
        if os.path.isdir(batch_folder_path):

            # Iterate over model folders within the batch folder
            for model_folder in os.listdir(batch_folder_path):
                model_folder_path = os.path.join(batch_folder_path, model_folder)

                # Ensure it is a directory
                if os.path.isdir(model_folder_path):

                    # Construct the full path to 'best.pt'
                    best_model_path = os.path.join(model_folder_path, "weights", 'best.pt')

                    # Check if 'best.pt' exists
                    if os.path.isfile(best_model_path):
                        # Combine the batch folder name and model name as the key
                        key = f"{batch_folder}_{model_folder}"
                        model_paths[key] = best_model_path

    return model_paths


model_paths = get_yolo_model_paths(model_folder)
print(len(model_paths))
print(model_paths)

for model_name, model in model_paths.items():
    print(f"Model {model_name} at {model}")
    model = YOLO(model)
    project_path = os.path.join("Prediction_results", model_name)

    val_results = model.predict(
        source=r"datasets\Exponat_img\test\images",
        imgsz=640,
        device="0",
        save=True,
        save_txt=True,
        project=project_path,
        )
    speed_list = []
    for result in val_results:
        print(result)
        speed_list.append(result.speed)
    speed_df = pd.DataFrame(speed_list)
    speed_file = os.path.join(project_path, "prediction_speed.xlsx")
    speed_df.to_excel(speed_file, index=False)

