import cv2 
import os
import numpy as np
from PIL import Image

def draw_polygon(image_path, annotation):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    coords = annotation[1:]  # Skip the class label
    points = np.array([(float(coords[i]) * w, float(coords[i+1]) * h) for i in range(0, len(coords), 2)], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Test", 1000, 1000)
    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

