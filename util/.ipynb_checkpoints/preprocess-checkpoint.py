import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from joblib import Parallel, delayed

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import regionprops
from scipy import ndimage as ndi


# -------------------------------
# Utility functions
# -------------------------------
def normIMG(image):
    """Normalize image to the range [0, 1]."""
    img_min, img_max = np.min(image), np.max(image)
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - img_min) / (img_max - img_min)
    return scaled.astype(np.float32)


# -------------------------------
# Image registration
# -------------------------------
def register_image_stack(temp_c_slice, temp_w_slice, temp_w, in_place=False, n_jobs=-1):
    """
    Perform drift correction by aligning temp_w with reference temp_c.

    Args:
        temp_c_slice (np.ndarray): Reference image (2D).
        temp_w_slice (np.ndarray): Image to align (2D).
        temp_w (np.ndarray): Image stack to be aligned (shape: [batch, height, width]).
        in_place (bool): If True, modifies the original array. Default: False.
        n_jobs (int): Number of parallel jobs. -1 uses all CPU cores.

    Returns:
        np.ndarray: Registered image stack with the same shape as temp_w.
        np.ndarray: The computed shift vector (dy, dx).
    """
    shift_vector, _, _ = phase_cross_correlation(temp_c_slice, temp_w_slice, upsample_factor=10)
    print(f"Computed shift: {shift_vector}")

    batch, _, _ = temp_w.shape

    registered_stack = temp_w if in_place else np.empty_like(temp_w)

    # Parallelized shifting
    results = Parallel(n_jobs=n_jobs)(
        delayed(shift)(temp_w[i], shift=shift_vector, mode="constant") for i in range(batch)
    )

    for i in range(batch):
        registered_stack[i] = results[i]

    return registered_stack, shift_vector


# -------------------------------
# Nuclei detection
# -------------------------------
def detect_nuclei_positions(image, sigma=2, min_size=50):
    """
    Detect nuclei centroids in a microscopy image.

    Args:
        image (np.ndarray): 2D input image (range: 0-1).
        sigma (float): Gaussian blur sigma for denoising. Default: 2.
        min_size (int): Minimum size of detected nuclei to keep. Default: 50.

    Returns:
        list: List of (x, y) centroid coordinates.
        np.ndarray: Labeled segmentation mask.
    """
    image = normIMG(image)

    # Step 1: Denoising
    blurred = gaussian(image, sigma)

    # Step 2: Thresholding
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh

    # Step 3: Morphological cleanup
    binary = remove_small_objects(binary, min_size=min_size)

    # Step 4: Watershed segmentation
    distance = ndi.distance_transform_edt(binary)
    markers = ndi.label(binary)[0]
    labels = watershed(-distance, markers, mask=binary)

    # Step 5: Extract centroids
    centroids = [(int(r.centroid[1]), int(r.centroid[0])) for r in regionprops(labels)]

    return centroids, labels


def validate_centroids(image, centroids, window_size=20):
    """
    Validate centroids by ensuring they are not too close to image boundaries.

    Args:
        image (np.ndarray): 2D image.
        centroids (list): List of (x, y) centroid coordinates.
        window_size (int): Cropping window size. Default: 20.

    Returns:
        list: Valid centroids inside the image boundaries.
    """
    height, width = image.shape
    half_window = window_size // 2
    valid = []

    for x, y in centroids:
        if (half_window <= x < width - half_window and
                half_window <= y < height - half_window):
            valid.append((x, y))

    return valid


def crop_stack_windows(image_stack, centroids, window_size=20):
    """
    Crop square windows around centroids across an image stack.

    Args:
        image_stack (np.ndarray): 3D array (stack_size, height, width).
        centroids (list): List of valid (x, y) centroid coordinates.
        window_size (int): Window size (assumes square). Default: 20.

    Returns:
        np.ndarray: Cropped stacks with shape
                    (num_centroids, stack_size, window_size, window_size).
    """
    stack_size, height, width = image_stack.shape
    half_window = window_size // 2
    cropped_stacks = []

    for x, y in centroids:
        if (half_window <= x < width - half_window and
                half_window <= y < height - half_window):
            windows = image_stack[:, y - half_window:y + half_window, x - half_window:x + half_window]
            cropped_stacks.append(windows)

    return np.array(cropped_stacks, dtype=np.float32)


# split the train/val/test
def split_data(data, ratios=[0.8, 0.1, 0.1]):
    """
    Splits a 4D NumPy array along the first dimension based on given ratios.

    Args:
        data (np.ndarray): Input array of shape [num, 64, 64, 64].
        ratios (list): List of split ratios (e.g., [0.8, 0.1, 0.1]).

    Returns:
        tuple: Three NumPy arrays (train, val, test).
    """
    num_samples = data.shape[0]  # Get the first dimension size

    # Compute split indices
    train_end = int(num_samples * ratios[0])
    val_end = train_end + int(num_samples * ratios[1])

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

