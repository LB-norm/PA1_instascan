import os
import glob
import cv2
import numpy as np
from PIL import Image

def order_rect_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points of a rectangle in a consistent order: top-left, top-right, bottom-right, bottom-left.
    """
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def convert_polygons_to_rectangles(input_dir: str, output_dir: str) -> None:
    """
    Convert YOLO instance-segmentation polygon labels into minimum-area rectangles.

    Each input .txt file in input_dir should contain lines like:
      0 x1,y1 x2,y2 x3,y3 ...

    The output files in output_dir will use space-separated rectangle corners:
      0 x1 y1 x2 y2 x3 y3 x4 y4
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))

    for in_path in txt_files:
        lines_out = []
        with open(in_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_idx = parts[0]
                raw = ' '.join(parts[1:]).replace(',', ' ')
                nums = raw.split()
                coords = list(map(float, nums))
                pts = np.array(coords, dtype=np.float32).reshape(-1, 2)

                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
                ordered = order_rect_points(box)

                flat = ordered.reshape(-1)
                coords_str = ' '.join(f"{c:.6f}" for c in flat)
                new_line = f"{class_idx} {coords_str}"
                lines_out.append(new_line)

        basename = os.path.basename(in_path)
        out_path = os.path.join(output_dir, basename)
        with open(out_path, 'w') as f:
            f.write("\n".join(lines_out))


def resize_images_in_folder(folder_path, size=(300, 200), suffix="_trimmed"):
    """
    Resize all images in folder_path to the given size and save them
    alongside the originals with `suffix` added to the filename.
    
    :param folder_path: Path to the folder containing images.
    :param size: Tuple (width, height) to resize to. Default is (400, 300).
    :param suffix: String to append to filenames before the extension.
    """
    # Supported image extensions
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

    # Figure out the right resampling attribute
    try:
        resample_filter = Image.Resampling.LANCZOS  # Pillow ≥ 9.1.0
    except AttributeError:
        resample_filter = Image.LANCZOS            # Older Pillow

    for fname in os.listdir(folder_path):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue

        src_path = os.path.join(folder_path, fname)
        dst_fname = f"{base}{suffix}{ext}"
        dst_path = os.path.join(folder_path, dst_fname)

        try:
            with Image.open(src_path) as img:
                resized = img.resize(size, resample=resample_filter)
                resized.save(dst_path)
                print(f"Saved resized image to {dst_fname}")
        except Exception as e:
            print(f"Skipping {fname}: {e}")


if __name__ == '__main__':

    input_path1 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B8_yolo11l-seg\predict\labels"
    input_path2 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B8_yolo11m-seg\predict\labels"
    input_path3 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B8_yolo11n-seg\predict\labels"
    input_path4 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B8_yolo11s-seg\predict\labels"
    input_path5 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B8_yolo11x-seg\predict\labels"

    input_path6 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B16_yolo11l-seg\predict\labels"
    input_path7 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B16_yolo11m-seg\predict\labels"
    input_path8 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B16_yolo11n-seg\predict\labels"
    input_path9 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B16_yolo11s-seg\predict\labels"
    input_path10 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation\Segmentation_B16_yolo11x-seg\predict\labels"

    output_path1 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B8_yolo11l-seg\predict\labels"
    output_path2 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B8_yolo11m-seg\predict\labels"
    output_path3 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B8_yolo11n-seg\predict\labels"
    output_path4 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B8_yolo11s-seg\predict\labels"
    output_path5 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B8_yolo11x-seg\predict\labels"

    output_path6 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B16_yolo11l-seg\predict\labels"
    output_path7 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B16_yolo11m-seg\predict\labels"
    output_path8 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B16_yolo11n-seg\predict\labels"
    output_path9 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B16_yolo11s-seg\predict\labels"
    output_path10 = r"C:\Python_Projects\PA1 SKALA YOLO\Prediction_results\Segmentation_normed\Segmentation_B16_yolo11x-seg\predict\labels"

    ground_truth_input = r"C:\Python_Projects\PA1 SKALA YOLO\datasets\Exponat_img\test\labels"
    ground_truth_output = r"C:\Python_Projects\PA1 SKALA YOLO\datasets\Exponat_img\test\normed_labels"
    # convert_polygons_to_rectangles(ground_truth_input, ground_truth_output)
    img_folder = r"C:\Users\brink\Desktop\Uni\Master IT\PA1\Prediction Vergleich\OBB_B16_s"
    resize_images_in_folder(img_folder, size=(200, 150), suffix="_trimmed")
