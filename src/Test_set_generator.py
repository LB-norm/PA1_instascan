import os
import random
import shutil

# Set paths to your dataset
dataset_images_path = r"C:\Python_Projects\Yard Lense\datasets\YL_dataset\images"
dataset_annotations_path = r"C:\Python_Projects\Yard Lense\datasets\YL_dataset\labels"

# Paths for test data output
test_images_path = r"C:\Python_Projects\Yard Lense\datasets\YL_dataset\test\images"
test_annotations_path = r"C:\Python_Projects\Yard Lense\datasets\YL_dataset\test\labels"

# Number of test samples to select
num_test_samples = 70

# Create output directories if they don't exist
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_annotations_path, exist_ok=True)

# Get all image and annotation filenames
image_files = sorted([f for f in os.listdir(dataset_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
annotation_files = sorted([f for f in os.listdir(dataset_annotations_path) if f.endswith('.txt')])

# Ensure each image has a corresponding annotation file
image_annotation_pairs = [
    (img, img.replace(os.path.splitext(img)[1], ".txt"))
    for img in image_files
    if img.replace(os.path.splitext(img)[1], ".txt") in annotation_files
]

# Randomly select 50 pairs
if len(image_annotation_pairs) < num_test_samples:
    raise ValueError("Not enough image-annotation pairs to select from!")
test_pairs = random.sample(image_annotation_pairs, num_test_samples)

# Copy the selected pairs to the test folder
for image_file, annotation_file in test_pairs:
    # Full paths for source files
    src_image_path = os.path.join(dataset_images_path, image_file)
    src_annotation_path = os.path.join(dataset_annotations_path, annotation_file)

    # Full paths for destination files
    dest_image_path = os.path.join(test_images_path, image_file)
    dest_annotation_path = os.path.join(test_annotations_path, annotation_file)

    # Copy files
    shutil.move(src_image_path, dest_image_path)
    shutil.move(src_annotation_path, dest_annotation_path)

print(f"Successfully created test set with {num_test_samples} samples.")

