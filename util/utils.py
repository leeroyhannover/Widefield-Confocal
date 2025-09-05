import os
import re
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from skimage import io
from natsort import natsorted
from PIL import Image


# -------------------------------
# Visualization utilities
# -------------------------------
def subShow(IMG1, IMG2):
    """Display two images side by side in grayscale."""
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(IMG1, cmap='gray')
    plt.axis("off")
    plt.title("Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(IMG2, cmap='gray')
    plt.axis("off")
    plt.title("Image 2")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Image normalization utilities
# -------------------------------
def normIMG(image):
    """Normalize image values to the range [0, 1]."""
    img_min, img_max = np.min(image), np.max(image)
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - img_min) / (img_max - img_min)
    return scaled.astype(np.float32)


def hist_correct(img, h=0.998, l=0.05):
    """Histogram correction by clipping and rescaling intensities."""
    flat_image = img.flatten()
    hist, bins = np.histogram(flat_image, bins=100, range=(0, 1))

    # Cumulative distribution function (CDF)
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]

    # Find lower and upper percentile bounds
    lower_bound = bins[np.searchsorted(cdf_normalized, l)]
    upper_bound = bins[np.searchsorted(cdf_normalized, h)]

    clipped = np.clip(img, lower_bound, upper_bound)
    rescaled = (clipped - lower_bound) / max((upper_bound - lower_bound), 1e-6)
    return rescaled.astype(np.float32)


# -------------------------------
# Image reading utilities
# -------------------------------
def readTIF(image_path, image_list):
    """Read a list of TIFF images, normalize, and histogram correct them."""
    images = []
    for fname in image_list:
        print("Reading:", fname)
        im = io.imread(os.path.join(image_path, fname))
        im = normIMG(im)
        im = hist_correct(im)
        images.append(im)
    return np.asarray(images, dtype=np.float32)


def list_subfolders_with_string(path, search_string='ZStep'):
    """Return a list of subfolders containing the given search string."""
    folder_list = natsorted(os.listdir(path))
    return [
        folder for folder in folder_list
        if search_string in folder and os.path.isdir(os.path.join(path, folder))
    ]


# -------------------------------
# Load and stack images
# -------------------------------

def load_and_stack_images(base_dir, folder_regex=r'ZStep_(\d+)', file_extension='.tif'):
    """
    Load and stack image data from a specified directory into a 4D array (z, stack_num, x, y).

    Args:
        base_dir (str): Root directory of the image data.
        folder_regex (str): Regex pattern to match subfolder names. Default is 'ZStep_(\d+)'.
        file_extension (str): File extension of the image files. Default is '.tif'.

    Returns:
        numpy.ndarray: A 4D array with shape (z, stack_num, x, y).
    """
    # Regex pattern for matching folder names
    folder_pattern = re.compile(folder_regex)

    # Collect all folders matching the pattern
    folders = []
    for folder_name in os.listdir(base_dir):
        match = folder_pattern.match(folder_name)
        if match and os.path.isdir(os.path.join(base_dir, folder_name)):
            x_value = int(match.group(1))  # Extract numeric x
            folders.append((x_value, folder_name))

    # Sort folders by extracted x value
    folders.sort(key=lambda item: item[0])

    if not folders:
        raise ValueError("No folders matching the pattern were found.")

    # Read and stack all image files with the specified extension in each folder
    stacks_by_z = []

    # Get image file list from the first folder
    first_folder_path = os.path.join(base_dir, folders[0][1])
    tif_files = natsorted([f for f in os.listdir(first_folder_path) if f.endswith(file_extension)])

    # Read images layer by layer (by z)
    for z_idx in range(len(tif_files)):
        z_stack = []  # Store all images for the current z-layer

        for x_value, folder_name in folders:
            folder_path = os.path.join(base_dir, folder_name)
            tif_files_in_folder = natsorted([f for f in os.listdir(folder_path) if f.endswith(file_extension)])

            if z_idx < len(tif_files_in_folder):
                img_path = os.path.join(folder_path, tif_files_in_folder[z_idx])

                # Read image and convert to NumPy array
                img = Image.open(img_path)
                img_array = np.array(img, dtype=np.float32)  # Read as float32 for normalization
                # img_array = normIMG(img_array)  # Normalize image to range 0-1
                
                z_stack.append(img_array)
                

        # Stack current z-layer images into shape (stack_num, x, y)
        if z_stack:
            # stacks_by_z.append(temp)
            temp = normIMG(z_stack)
            stacks_by_z.append(np.stack(temp, axis=0))

    # Stack into final 4D array (z, stack_num, x, y)
    if stacks_by_z:
        final_array = np.stack(stacks_by_z, axis=0)
        print(f"Final stacked image shape: {final_array.shape}")
        return final_array
    else:
        raise ValueError("No valid image files were found.")

