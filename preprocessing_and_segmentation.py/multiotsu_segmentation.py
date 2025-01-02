import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from skimage.filters import threshold_multiotsu

import glob
import os
from tqdm import tqdm

# Image Folder Path
folder_path = "#"
output_folder_path = "#"

# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Array Image Files
image_files = glob.glob(os.path.join(folder_path, "*"))

# Total number of images
total_images = len(image_files)

for image_file in tqdm(image_files, desc="Multiotsu Segmentation", unit="file", ncols=100):
    
    # Read Image
    img = io.imread(image_file)

    # Compute multi-Otsu thresholds
    threshold = threshold_multiotsu(img, classes=5)

    # Digitize (segment) the image based on the thresholds
    regions = np.digitize(img, bins=threshold)

    # Convert regions to uint8 explicitly to avoid the warning
    output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
    output[output < np.unique(output)[-1]] = 0
    output[output >= np.unique(output)[-1]] = 1

    # Construct the output path
    filename = os.path.basename(image_file)
    output_path = os.path.join(output_folder_path, filename)
    
    plt.imsave(output_path, output, cmap="gray")
