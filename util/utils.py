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
    Load and stack images from subfolders into a 4D NumPy array (z, stack_num, x, y).

    Args:
        base_dir (str): Root directory containing subfolders.
        folder_regex (str): Regex pattern to match subfolder names. Default: 'ZStep_(\\d+)'.
        file_extension (str): File extension to filter images. Default: '.tif'.

    Returns:
        numpy.ndarray: 4D array of shape (z, stack_num, x, y).
    """
    folder_pattern = re.compile(folder_regex)

    # Collect valid folders with numeric indices
    folders = []
    for folder_name in os.listdir(base_dir):
        match = folder_pattern.match(folder_name)
        if match and os.path.isdir(os.path.join(base_dir, folder_name)):
            folders.append((int(match.group(1)), folder_name))

    folders.sort(key=lambda x: x[0])
    if not folders:
        raise ValueError("No valid subfolders found.")

    # Use the first folder to determine z-depth
    first_folder = os.path.join(base_dir, folders[0][1])
    tif_files = natsorted([f for f in os.listdir(first_folder) if f.endswith(file_extension)])
    if not tif_files:
        raise ValueError("No image files found in the first folder.")

    stacks_by_z = []

    # Iterate over z-layers
    for z_idx in range(len(tif_files)):
        z_stack = []

        for _, folder_name in folders:
            folder_path = os.path.join(base_dir, folder_name)
            tif_files_in_folder = natsorted([f for f in os.listdir(folder_path) if f.endswith(file_extension)])

            if z_idx < len(tif_files_in_folder):
                img_path = os.path.join(folder_path, tif_files_in_folder[z_idx])
                img = Image.open(img_path)
                img_array = np.array(img, dtype=np.float32)
                z_stack.append(img_array)

        if z_stack:
            z_stack = np.stack([normIMG(img) for img in z_stack], axis=0)
            stacks_by_z.append(z_stack)

    if not stacks_by_z:
        raise ValueError("No images were loaded.")

    final_array = np.stack(stacks_by_z, axis=0)
    print(f"Final stacked shape: {final_array.shape}")
    return final_array
