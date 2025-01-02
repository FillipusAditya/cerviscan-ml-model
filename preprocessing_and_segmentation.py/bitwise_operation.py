"""
bitwise_operation.py
This script performs bitwise operations to apply a mask to the original image.
Input: Images and masks.
Output: Masked images saved to the output directory.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

# Folder paths for original images, masks, and output
original_image_folder = "#"
mask_folder = "#"
output_folder_path = "#"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Array of input image files
image_files = glob.glob(os.path.join(original_image_folder, "*"))
mask_files = glob.glob(os.path.join(mask_folder, "*"))

# Total number of images
for image_file in tqdm(image_files, desc="Bitwise Segmentation", unit="file", ncols=100):
    # Load the original image
    original_image = cv2.imread(image_file)
    mask_image = None

    # Match the mask file with the current image file
    for mask_file in mask_files:
        if os.path.basename(image_file) == os.path.basename(mask_file):
            mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            break

    # Resize the mask image to match the dimensions of the original image
    mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # Convert the grayscale mask to a 3-channel image
    mask_image_3channel = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    # Perform bitwise AND operation
    dest_and = cv2.bitwise_and(original_image, mask_image_3channel)

    # Construct the output file path
    filename = os.path.basename(image_file)
    output_path = os.path.join(output_folder_path, filename)

    # Save the masked image
    cv2.imwrite(output_path, dest_and)
    