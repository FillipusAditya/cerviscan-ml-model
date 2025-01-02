"""
grayscale_filter.py
This script converts RGB images to grayscale.
Input: RGB images.
Output: Grayscale images saved to the output directory.
"""
import cv2
import os
import glob
from tqdm import tqdm

# Folder paths for input and output images
folder_path = "#"
output_folder_path = "#"

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Array of input image files
image_files = glob.glob(os.path.join(folder_path, '*'))

# Total number of images
for image_file in tqdm(image_files, desc="Processing images", unit="file", ncols=100):
    # Read the image
    image = cv2.imread(image_file)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Construct the output file path
    filename = os.path.basename(image_file)
    output_path = os.path.join(output_folder_path, filename)

    # Save the grayscale image
    cv2.imwrite(output_path, gray_image)