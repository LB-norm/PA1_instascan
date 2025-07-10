from ultralytics import YOLO
import torch
import cv2 
import numpy as np
import os
import pandas as pd
import visualization
from shapely.geometry import Polygon

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0")

model_folder = r"Trained_models"
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def plot_polygons(pred_polygon, gt_polygon, title="Polygon Comparison"):
    """
    Plot the predicted polygon and ground truth polygon side by side.

    Parameters:
        pred_polygon (list): Predicted polygon as a list of (x, y) tuples.
        gt_polygon (list): Ground truth polygon as a list of (x, y) tuples.
        title (str): Title of the plot.
    """
    # Convert polygons to format compatible with Matplotlib
    pred_x, pred_y = zip(*pred_polygon)
    gt_x, gt_y = zip(*gt_polygon)

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # Plot Predicted Polygon
    axs[0].fill(pred_x, pred_y, alpha=0.5, label="Predicted Polygon", color="blue")
    axs[0].plot(pred_x + (pred_x[0],), pred_y + (pred_y[0],), 'r--')  # Closing line
    axs[0].set_title("Predicted Polygon")
    axs[0].legend()
    axs[0].axis("equal")

    # Plot Ground Truth Polygon
    axs[1].fill(gt_x, gt_y, alpha=0.5, label="Ground Truth Polygon", color="green")
    axs[1].plot(gt_x + (gt_x[0],), gt_y + (gt_y[0],), 'r--')  # Closing line
    axs[1].set_title("Ground Truth Polygon")
    axs[1].legend()
    axs[1].axis("equal")

    plt.show()

def load_single_polygon_from_txt(file_path):
    """
    Load a single polygon from a .txt file.

    Parameters:
        file_path (str): Path to the .txt file containing a single polygon.

    Returns:
        list: A polygon represented as a list of (x, y) tuples.
    """
    with open(file_path, 'r') as file:
        line = file.read().strip()
        data = line.split()
        # Ignore the first value (class label, always 0)
        coords = list(map(float, data[1:]))
        # Convert to list of (x, y) tuples
        polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return polygon

def calculate_polygon_iou(pred_coords, gt_coords):
    """
    Calculate IoU for two polygons.

    Parameters:
        pred_coords (list): List of (x, y) tuples representing the predicted polygon.
        gt_coords (list): List of (x, y) tuples representing the ground truth polygon.

    Returns:
        float: IoU score (0 to 1).
    """
    pred_polygon = Polygon(pred_coords)
    print(pred_polygon)
    gt_polygon = Polygon(gt_coords)

    if not pred_polygon.is_valid or not gt_polygon.is_valid:
        print(f"Pred Polygon valid: {pred_polygon.is_valid}")
        print(f"Pred Polygon: {pred_polygon}")
        pred_polygon = pred_polygon.buffer(0)
        print(f"Pred Polygon valid after buffer: {pred_polygon.is_valid}")

    if not gt_polygon.is_valid:
        print(f"GT Polygon valid: {gt_polygon.is_valid}")
        print(f"GT Polygon: {gt_polygon}")

    intersection = pred_polygon.intersection(gt_polygon).area
    union = pred_polygon.union(gt_polygon).area

    if union == 0:
        return 0.0

    return intersection / union

# def evaluate_segmentation(predictions, ground_truths, iou_threshold=0.5):
#     tp, fp, fn = 0, 0, 0
#     used_gt = set()

#     for pred in predictions:
#         best_iou = 0
#         best_gt = None

#         for i, gt in enumerate(ground_truths):
#             if i in used_gt:
#                 continue
#             iou = calculate_polygon_iou(pred["coords"], gt["coords"])
#             if iou > best_iou:
#                 best_iou = iou
#                 best_gt = i

#         if best_iou >= iou_threshold:
#             tp += 1
#             used_gt.add(best_gt)
#         else:
#             fp += 1

#     fn = len(ground_truths) - len(used_gt)
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#     return {"precision": precision, "recall": recall, "TP": tp, "FP": fp, "FN": fn}

# Load polygons from text files
ground_truth_path = r"datasets\Exponat_img\test\labels"
normed_ground_truth_path = r"C:\Python_Projects\PA1 SKALA YOLO\datasets\Exponat_img\test\normed_labels"

Segmentation_B8_yolo11l_seg = r"Prediction_results\Segmentation\Segmentation_B8_yolo11l-seg\predict\labels"
Segmentation_B8_yolo11m_seg = r"Prediction_results\Segmentation\Segmentation_B8_yolo11m-seg\predict\labels"
Segmentation_B8_yolo11n_seg = r"Prediction_results\Segmentation\Segmentation_B8_yolo11n-seg\predict\labels"
Segmentation_B8_yolo11s_seg = r"Prediction_results\Segmentation\Segmentation_B8_yolo11s-seg\predict\labels"
Segmentation_B8_yolo11x_seg = r"Prediction_results\Segmentation\Segmentation_B8_yolo11x-seg\predict\labels"

Segmentation_B16_yolo11l_seg = r"Prediction_results\Segmentation\Segmentation_B16_yolo11l-seg\predict\labels"
Segmentation_B16_yolo11m_seg = r"Prediction_results\Segmentation\Segmentation_B16_yolo11m-seg\predict\labels"
Segmentation_B16_yolo11n_seg = r"Prediction_results\Segmentation\Segmentation_B16_yolo11n-seg\predict\labels"
Segmentation_B16_yolo11s_seg = r"Prediction_results\Segmentation\Segmentation_B16_yolo11s-seg\predict\labels"
Segmentation_B16_yolo11x_seg = r"Prediction_results\Segmentation\Segmentation_B16_yolo11x-seg\predict\labels"

Normed_Segmentation_B8_yolo11l_seg = r"Prediction_results\Segmentation_normed\Segmentation_B8_yolo11l-seg\predict\labels"
Normed_Segmentation_B8_yolo11m_seg = r"Prediction_results\Segmentation_normed\Segmentation_B8_yolo11m-seg\predict\labels"
Normed_Segmentation_B8_yolo11n_seg = r"Prediction_results\Segmentation_normed\Segmentation_B8_yolo11n-seg\predict\labels"
Normed_Segmentation_B8_yolo11s_seg = r"Prediction_results\Segmentation_normed\Segmentation_B8_yolo11s-seg\predict\labels"
Normed_Segmentation_B8_yolo11x_seg = r"Prediction_results\Segmentation_normed\Segmentation_B8_yolo11x-seg\predict\labels"

Normed_Segmentation_B16_yolo11l_seg = r"Prediction_results\Segmentation_normed\Segmentation_B16_yolo11l-seg\predict\labels"
Normed_Segmentation_B16_yolo11m_seg = r"Prediction_results\Segmentation_normed\Segmentation_B16_yolo11m-seg\predict\labels"
Normed_Segmentation_B16_yolo11n_seg = r"Prediction_results\Segmentation_normed\Segmentation_B16_yolo11n-seg\predict\labels"
Normed_Segmentation_B16_yolo11s_seg = r"Prediction_results\Segmentation_normed\Segmentation_B16_yolo11s-seg\predict\labels"
Normed_Segmentation_B16_yolo11x_seg = r"Prediction_results\Segmentation_normed\Segmentation_B16_yolo11x-seg\predict\labels"

OBB_B8_yolo11l_obb = r"Prediction_results\OBB\OBB_B8_yolo11l-obb\predict\labels"
OBB_B8_yolo11m_obb = r"Prediction_results\OBB\OBB_B8_yolo11m-obb\predict\labels"
OBB_B8_yolo11n_obb = r"Prediction_results\OBB\OBB_B8_yolo11n-obb\predict\labels"
OBB_B8_yolo11s_obb = r"Prediction_results\OBB\OBB_B8_yolo11s-obb\predict\labels"
OBB_B8_yolo11x_obb = r"Prediction_results\OBB\OBB_B8_yolo11x-obb\predict\labels"

OBB_B16_yolo11l_obb = r"Prediction_results\OBB\OBB_B16_yolo11l-obb\predict\labels"
OBB_B16_yolo11m_obb = r"Prediction_results\OBB\OBB_B16_yolo11m-obb\predict\labels"
OBB_B16_yolo11n_obb = r"Prediction_results\OBB\OBB_B16_yolo11n-obb\predict\labels"
OBB_B16_yolo11s_obb = r"Prediction_results\OBB\OBB_B16_yolo11s-obb\predict\labels"
OBB_B16_yolo11x_obb = r"Prediction_results\OBB\OBB_B16_yolo11x-obb\predict\labels"

rule_based_pred = r"Prediction_results\Rule_based_predictions"
#Get label files
ground_truth_files = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)])
normed_ground_truth_files = sorted([os.path.join(normed_ground_truth_path, f) for f in os.listdir(normed_ground_truth_path)])

Segmentation_B8_yolo11l_seg_files = sorted([os.path.join(Segmentation_B8_yolo11l_seg, f) for f in os.listdir(Segmentation_B8_yolo11l_seg)])
Segmentation_B8_yolo11m_seg_files = sorted([os.path.join(Segmentation_B8_yolo11m_seg, f) for f in os.listdir(Segmentation_B8_yolo11m_seg)])
Segmentation_B8_yolo11n_seg_files = sorted([os.path.join(Segmentation_B8_yolo11n_seg, f) for f in os.listdir(Segmentation_B8_yolo11n_seg)])
Segmentation_B8_yolo11s_seg_files = sorted([os.path.join(Segmentation_B8_yolo11s_seg, f) for f in os.listdir(Segmentation_B8_yolo11s_seg)])
Segmentation_B8_yolo11x_seg_files = sorted([os.path.join(Segmentation_B8_yolo11x_seg, f) for f in os.listdir(Segmentation_B8_yolo11x_seg)])

Segmentation_B16_yolo11l_seg_files = sorted([os.path.join(Segmentation_B16_yolo11l_seg, f) for f in os.listdir(Segmentation_B16_yolo11l_seg)])
Segmentation_B16_yolo11m_seg_files = sorted([os.path.join(Segmentation_B16_yolo11m_seg, f) for f in os.listdir(Segmentation_B16_yolo11m_seg)])
Segmentation_B16_yolo11n_seg_files = sorted([os.path.join(Segmentation_B16_yolo11n_seg, f) for f in os.listdir(Segmentation_B16_yolo11n_seg)])
Segmentation_B16_yolo11s_seg_files = sorted([os.path.join(Segmentation_B16_yolo11s_seg, f) for f in os.listdir(Segmentation_B16_yolo11s_seg)])
Segmentation_B16_yolo11x_seg_files = sorted([os.path.join(Segmentation_B16_yolo11x_seg, f) for f in os.listdir(Segmentation_B16_yolo11x_seg)])

Normed_Segmentation_B8_yolo11l_seg_files = sorted([os.path.join(Normed_Segmentation_B8_yolo11l_seg, f) for f in os.listdir(Normed_Segmentation_B8_yolo11l_seg)])
Normed_Segmentation_B8_yolo11m_seg_files = sorted([os.path.join(Normed_Segmentation_B8_yolo11m_seg, f) for f in os.listdir(Normed_Segmentation_B8_yolo11m_seg)])
Normed_Segmentation_B8_yolo11n_seg_files = sorted([os.path.join(Normed_Segmentation_B8_yolo11n_seg, f) for f in os.listdir(Normed_Segmentation_B8_yolo11n_seg)])
Normed_Segmentation_B8_yolo11s_seg_files = sorted([os.path.join(Normed_Segmentation_B8_yolo11s_seg, f) for f in os.listdir(Normed_Segmentation_B8_yolo11s_seg)])
Normed_Segmentation_B8_yolo11x_seg_files = sorted([os.path.join(Normed_Segmentation_B8_yolo11x_seg, f) for f in os.listdir(Normed_Segmentation_B8_yolo11x_seg)])

Normed_Segmentation_B16_yolo11l_seg_files = sorted([os.path.join(Normed_Segmentation_B16_yolo11l_seg, f) for f in os.listdir(Normed_Segmentation_B16_yolo11l_seg)])
Normed_Segmentation_B16_yolo11m_seg_files = sorted([os.path.join(Normed_Segmentation_B16_yolo11m_seg, f) for f in os.listdir(Normed_Segmentation_B16_yolo11m_seg)])
Normed_Segmentation_B16_yolo11n_seg_files = sorted([os.path.join(Normed_Segmentation_B16_yolo11n_seg, f) for f in os.listdir(Normed_Segmentation_B16_yolo11n_seg)])
Normed_Segmentation_B16_yolo11s_seg_files = sorted([os.path.join(Normed_Segmentation_B16_yolo11s_seg, f) for f in os.listdir(Normed_Segmentation_B16_yolo11s_seg)])
Normed_Segmentation_B16_yolo11x_seg_files = sorted([os.path.join(Normed_Segmentation_B16_yolo11x_seg, f) for f in os.listdir(Normed_Segmentation_B16_yolo11x_seg)])

OBB_B8_yolo11l_obb_files = sorted([os.path.join(OBB_B8_yolo11l_obb, f) for f in os.listdir(OBB_B8_yolo11l_obb)])
OBB_B8_yolo11m_obb_files = sorted([os.path.join(OBB_B8_yolo11m_obb, f) for f in os.listdir(OBB_B8_yolo11m_obb)])
OBB_B8_yolo11n_obb_files = sorted([os.path.join(OBB_B8_yolo11n_obb, f) for f in os.listdir(OBB_B8_yolo11n_obb)])
OBB_B8_yolo11s_obb_files = sorted([os.path.join(OBB_B8_yolo11s_obb, f) for f in os.listdir(OBB_B8_yolo11s_obb)])
OBB_B8_yolo11x_obb_files = sorted([os.path.join(OBB_B8_yolo11x_obb, f) for f in os.listdir(OBB_B8_yolo11x_obb)])

OBB_B16_yolo11l_obb_files = sorted([os.path.join(OBB_B16_yolo11l_obb, f) for f in os.listdir(OBB_B16_yolo11l_obb)])
OBB_B16_yolo11m_obb_files = sorted([os.path.join(OBB_B16_yolo11m_obb, f) for f in os.listdir(OBB_B16_yolo11m_obb)])
OBB_B16_yolo11n_obb_files = sorted([os.path.join(OBB_B16_yolo11n_obb, f) for f in os.listdir(OBB_B16_yolo11n_obb)])
OBB_B16_yolo11s_obb_files = sorted([os.path.join(OBB_B16_yolo11s_obb, f) for f in os.listdir(OBB_B16_yolo11s_obb)])
OBB_B16_yolo11x_obb_files = sorted([os.path.join(OBB_B16_yolo11x_obb, f) for f in os.listdir(OBB_B16_yolo11x_obb)])

rule_based_pred_files = sorted([os.path.join(rule_based_pred, f) for f in os.listdir(rule_based_pred)])
# def calculate_IoU(pred_files, ground_truth_files):
#     iou_list = []
#     for pred_file, gt_file in zip(pred_files, ground_truth_files):
#         gt_polygon = load_single_polygon_from_txt(gt_file)
#         pred_polygon = load_single_polygon_from_txt(pred_file)
#         # plot_polygons(pred_polygon, gt_polygon, title="Debugging IoU Calculations")
#         # Calculate IoU
#         iou = calculate_polygon_iou(pred_polygon, gt_polygon)
#         print(f"IoU for {os.path.basename(pred_file)} vs {os.path.basename(gt_file)}: {iou:.2f}")
#         iou_list.append(iou)
#     return iou_list

def calculate_IoU(pred_files, ground_truth_files):
    # Create a lookup dictionary for prediction files based on filename
    pred_dict = {os.path.basename(f): f for f in pred_files}
    iou_list = []
    
    for gt_file in ground_truth_files:
        basename = os.path.basename(gt_file)
        if basename not in pred_dict:
            print(f"No prediction found for {basename}. IoU set to 0.")
            iou_list.append(0)  # Assign IoU 0 for missing predictions
        else:
            pred_file = pred_dict[basename]
            gt_polygon = load_single_polygon_from_txt(gt_file)
            pred_polygon = load_single_polygon_from_txt(pred_file)
            # Optionally plot polygons for debugging:
            # plot_polygons(pred_polygon, gt_polygon, title="Debugging IoU Calculations")
            iou = calculate_polygon_iou(pred_polygon, gt_polygon)
            print(f"IoU for {basename}: {iou:.2f}")
            iou_list.append(iou)
    
    return iou_list

if __name__ == "__main__":
    """ Without normalizing to rectangles"""
    Segmentation_B8_yolo11l_seg_iou = calculate_IoU(Segmentation_B8_yolo11l_seg_files, ground_truth_files)
    Segmentation_B8_yolo11m_seg_iou = calculate_IoU(Segmentation_B8_yolo11m_seg_files, ground_truth_files)
    Segmentation_B8_yolo11n_seg_iou = calculate_IoU(Segmentation_B8_yolo11n_seg_files, ground_truth_files)
    Segmentation_B8_yolo11s_seg_iou = calculate_IoU(Segmentation_B8_yolo11s_seg_files, ground_truth_files)
    Segmentation_B8_yolo11x_seg_iou = calculate_IoU(Segmentation_B8_yolo11x_seg_files, ground_truth_files)

    Segmentation_B16_yolo11l_seg_iou = calculate_IoU(Segmentation_B16_yolo11l_seg_files, ground_truth_files)
    Segmentation_B16_yolo11m_seg_iou = calculate_IoU(Segmentation_B16_yolo11m_seg_files, ground_truth_files)
    Segmentation_B16_yolo11n_seg_iou = calculate_IoU(Segmentation_B16_yolo11n_seg_files, ground_truth_files)
    Segmentation_B16_yolo11s_seg_iou = calculate_IoU(Segmentation_B16_yolo11s_seg_files, ground_truth_files)
    Segmentation_B16_yolo11x_seg_iou = calculate_IoU(Segmentation_B16_yolo11x_seg_files, ground_truth_files)

    OBB_B8_yolo11l_obb_iou = calculate_IoU(OBB_B8_yolo11l_obb_files, ground_truth_files)
    OBB_B8_yolo11m_obb_iou = calculate_IoU(OBB_B8_yolo11m_obb_files, ground_truth_files)
    OBB_B8_yolo11n_obb_iou = calculate_IoU(OBB_B8_yolo11n_obb_files, ground_truth_files)
    OBB_B8_yolo11s_obb_iou = calculate_IoU(OBB_B8_yolo11s_obb_files, ground_truth_files)
    OBB_B8_yolo11x_obb_iou = calculate_IoU(OBB_B8_yolo11x_obb_files, ground_truth_files)

    OBB_B16_yolo11l_obb_iou = calculate_IoU(OBB_B16_yolo11l_obb_files, ground_truth_files)
    OBB_B16_yolo11m_obb_iou = calculate_IoU(OBB_B16_yolo11m_obb_files, ground_truth_files)
    OBB_B16_yolo11n_obb_iou = calculate_IoU(OBB_B16_yolo11n_obb_files, ground_truth_files)
    OBB_B16_yolo11s_obb_iou = calculate_IoU(OBB_B16_yolo11s_obb_files, ground_truth_files)
    OBB_B16_yolo11x_obb_iou = calculate_IoU(OBB_B16_yolo11x_obb_files, ground_truth_files)

    Rule_based_iou = calculate_IoU(rule_based_pred_files, ground_truth_files)

    IoU_result_dict = {
        "Segmentation_B8_yolo11l": Segmentation_B8_yolo11l_seg_iou,
        "Segmentation_B8_yolo11m": Segmentation_B8_yolo11m_seg_iou,
        "Segmentation_B8_yolo11n": Segmentation_B8_yolo11n_seg_iou,
        "Segmentation_B8_yolo11s": Segmentation_B8_yolo11s_seg_iou,
        "Segmentation_B8_yolo11x": Segmentation_B8_yolo11x_seg_iou,
        "Segmentation_B16_yolo11l": Segmentation_B16_yolo11l_seg_iou,
        "Segmentation_B16_yolo11m": Segmentation_B16_yolo11m_seg_iou,
        "Segmentation_B16_yolo11n": Segmentation_B16_yolo11n_seg_iou,
        "Segmentation_B16_yolo11s": Segmentation_B16_yolo11s_seg_iou,
        "Segmentation_B16_yolo11x": Segmentation_B16_yolo11x_seg_iou,
        "OBB_B8_yolo11l": OBB_B8_yolo11l_obb_iou,
        "OBB_B8_yolo11m": OBB_B8_yolo11m_obb_iou,
        "OBB_B8_yolo11n": OBB_B8_yolo11n_obb_iou,
        "OBB_B8_yolo11s": OBB_B8_yolo11s_obb_iou,
        "OBB_B8_yolo11x": OBB_B8_yolo11x_obb_iou,
        "OBB_B16_yolo11l": OBB_B16_yolo11l_obb_iou,
        "OBB_B16_yolo11m": OBB_B16_yolo11m_obb_iou,
        "OBB_B16_yolo11n": OBB_B16_yolo11n_obb_iou,
        "OBB_B16_yolo11s": OBB_B16_yolo11s_obb_iou,
        "OBB_B16_yolo11x": OBB_B16_yolo11x_obb_iou,
        "Rule_based": Rule_based_iou
    }
    print(IoU_result_dict)
    print(len(Segmentation_B8_yolo11l_seg_iou))
    print(sum(Segmentation_B8_yolo11l_seg_iou)/len(Segmentation_B8_yolo11l_seg_iou))
    visualization.plot_iou_averages(IoU_result_dict)
    visualization.evaluate_and_plot_model_predictions(IoU_result_dict, 0.9)

    # """ Output and Seg_predictions normalized to rectangles"""
    # Segmentation_B8_yolo11l_seg_iou = calculate_IoU(Segmentation_B8_yolo11l_seg_files, normed_ground_truth_files)
    # Segmentation_B8_yolo11m_seg_iou = calculate_IoU(Segmentation_B8_yolo11m_seg_files, normed_ground_truth_files)
    # Segmentation_B8_yolo11n_seg_iou = calculate_IoU(Segmentation_B8_yolo11n_seg_files, normed_ground_truth_files)
    # Segmentation_B8_yolo11s_seg_iou = calculate_IoU(Segmentation_B8_yolo11s_seg_files, normed_ground_truth_files)
    # Segmentation_B8_yolo11x_seg_iou = calculate_IoU(Segmentation_B8_yolo11x_seg_files, normed_ground_truth_files)

    # Segmentation_B16_yolo11l_seg_iou = calculate_IoU(Segmentation_B16_yolo11l_seg_files, normed_ground_truth_files)
    # Segmentation_B16_yolo11m_seg_iou = calculate_IoU(Segmentation_B16_yolo11m_seg_files, normed_ground_truth_files)
    # Segmentation_B16_yolo11n_seg_iou = calculate_IoU(Segmentation_B16_yolo11n_seg_files, normed_ground_truth_files)
    # Segmentation_B16_yolo11s_seg_iou = calculate_IoU(Segmentation_B16_yolo11s_seg_files, normed_ground_truth_files)
    # Segmentation_B16_yolo11x_seg_iou = calculate_IoU(Segmentation_B16_yolo11x_seg_files, normed_ground_truth_files)

    # Normed_Segmentation_B8_yolo11l_seg_iou = calculate_IoU(Normed_Segmentation_B8_yolo11l_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B8_yolo11m_seg_iou = calculate_IoU(Normed_Segmentation_B8_yolo11m_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B8_yolo11n_seg_iou = calculate_IoU(Normed_Segmentation_B8_yolo11n_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B8_yolo11s_seg_iou = calculate_IoU(Normed_Segmentation_B8_yolo11s_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B8_yolo11x_seg_iou = calculate_IoU(Normed_Segmentation_B8_yolo11x_seg_files, normed_ground_truth_files)

    # Normed_Segmentation_B16_yolo11l_seg_iou = calculate_IoU(Normed_Segmentation_B16_yolo11l_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B16_yolo11m_seg_iou = calculate_IoU(Normed_Segmentation_B16_yolo11m_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B16_yolo11n_seg_iou = calculate_IoU(Normed_Segmentation_B16_yolo11n_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B16_yolo11s_seg_iou = calculate_IoU(Normed_Segmentation_B16_yolo11s_seg_files, normed_ground_truth_files)
    # Normed_Segmentation_B16_yolo11x_seg_iou = calculate_IoU(Normed_Segmentation_B16_yolo11x_seg_files, normed_ground_truth_files)

    # OBB_B8_yolo11l_obb_iou = calculate_IoU(OBB_B8_yolo11l_obb_files, normed_ground_truth_files)
    # OBB_B8_yolo11m_obb_iou = calculate_IoU(OBB_B8_yolo11m_obb_files, normed_ground_truth_files)
    # OBB_B8_yolo11n_obb_iou = calculate_IoU(OBB_B8_yolo11n_obb_files, normed_ground_truth_files)
    # OBB_B8_yolo11s_obb_iou = calculate_IoU(OBB_B8_yolo11s_obb_files, normed_ground_truth_files)
    # OBB_B8_yolo11x_obb_iou = calculate_IoU(OBB_B8_yolo11x_obb_files, normed_ground_truth_files)

    # OBB_B16_yolo11l_obb_iou = calculate_IoU(OBB_B16_yolo11l_obb_files, normed_ground_truth_files)
    # OBB_B16_yolo11m_obb_iou = calculate_IoU(OBB_B16_yolo11m_obb_files, normed_ground_truth_files)
    # OBB_B16_yolo11n_obb_iou = calculate_IoU(OBB_B16_yolo11n_obb_files, normed_ground_truth_files)
    # OBB_B16_yolo11s_obb_iou = calculate_IoU(OBB_B16_yolo11s_obb_files, normed_ground_truth_files)
    # OBB_B16_yolo11x_obb_iou = calculate_IoU(OBB_B16_yolo11x_obb_files, normed_ground_truth_files)

    # Rule_based_iou = calculate_IoU(rule_based_pred_files, normed_ground_truth_files)

    # IoU_result_dict = {
    #     "Normed_Segmentation_B8_yolo11l": Normed_Segmentation_B8_yolo11l_seg_iou,
    #     "Normed_Segmentation_B8_yolo11m": Normed_Segmentation_B8_yolo11m_seg_iou,
    #     "Normed_Segmentation_B8_yolo11n": Normed_Segmentation_B8_yolo11n_seg_iou,
    #     "Normed_Segmentation_B8_yolo11s": Normed_Segmentation_B8_yolo11s_seg_iou,
    #     "Normed_Segmentation_B8_yolo11x": Normed_Segmentation_B8_yolo11x_seg_iou,
    #     "Normed_Segmentation_B16_yolo11l": Normed_Segmentation_B16_yolo11l_seg_iou,
    #     "Normed_Segmentation_B16_yolo11m": Normed_Segmentation_B16_yolo11m_seg_iou,
    #     "Normed_Segmentation_B16_yolo11n": Normed_Segmentation_B16_yolo11n_seg_iou,
    #     "Normed_Segmentation_B16_yolo11s": Normed_Segmentation_B16_yolo11s_seg_iou,
    #     "Normed_Segmentation_B16_yolo11x": Normed_Segmentation_B16_yolo11x_seg_iou,
    #     "OBB_B8_yolo11l": OBB_B8_yolo11l_obb_iou,
    #     "OBB_B8_yolo11m": OBB_B8_yolo11m_obb_iou,
    #     "OBB_B8_yolo11n": OBB_B8_yolo11n_obb_iou,
    #     "OBB_B8_yolo11s": OBB_B8_yolo11s_obb_iou,
    #     "OBB_B8_yolo11x": OBB_B8_yolo11x_obb_iou,
    #     "OBB_B16_yolo11l": OBB_B16_yolo11l_obb_iou,
    #     "OBB_B16_yolo11m": OBB_B16_yolo11m_obb_iou,
    #     "OBB_B16_yolo11n": OBB_B16_yolo11n_obb_iou,
    #     "OBB_B16_yolo11s": OBB_B16_yolo11s_obb_iou,
    #     "OBB_B16_yolo11x": OBB_B16_yolo11x_obb_iou,
    #     "Rule_based": Rule_based_iou
    # }
    # print(IoU_result_dict)
    # print(len(Segmentation_B8_yolo11l_seg_iou))
    # print(sum(Segmentation_B8_yolo11l_seg_iou)/len(Segmentation_B8_yolo11l_seg_iou))
    # visualization.plot_iou_averages(IoU_result_dict)
    # visualization.evaluate_and_plot_model_predictions(IoU_result_dict, 0.9)

    # Jetzt noch vergleichen wie viele nicht detektiert wurden (bzw. sehr schlecht?) 
    # Dann nocheinmal IoU aber nur für detektierte Rechtecke

#(Evt. eigene Metrik basteln? Text höher gewichtet?)