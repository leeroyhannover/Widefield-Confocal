from util.utils import *  
from util.preprocess import *

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------
C_PATH = 'xxx/Confocal-20x/'
W_PATH = 'xxx/Widefield-20x/'
OUT_PROCESSED = 'xxx/processed/'
OUT_ROOT = 'xxx/'
PLOT_EXAMPLES = True            # set False to skip plotting
NUCLEI_WINDOW = 128             # crop window for nuclei validation & extraction
REG_REF_IDX_C = 45              # reference slice index for confocal during registration
REG_REF_IDX_W = 19              # reference slice index for widefield during registration
CROP_Z_RANGE = slice(25, 65)    # post-crop z-range for confocal windows (as in your code)
indices_filtered = set()        # provide indices to skip; left empty by default

os.makedirs(OUT_PROCESSED, exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, 'val'), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, 'test'), exist_ok=True)

# -------------------------------
# 1) Load raw stacks
# -------------------------------
all_c_img = load_and_stack_images(C_PATH)  # expected shape: [z, stack, x, y] or similar per your util
all_w_img = load_and_stack_images(W_PATH)
print("Loaded shapes:", all_c_img.shape, all_w_img.shape)

# Basic sanity check that z-depth matches
if all_c_img.shape[0] != all_w_img.shape[0]:
    raise ValueError(f"Z depth mismatch: confocal z={all_c_img.shape[0]}, widefield z={all_w_img.shape[0]}")

# -------------------------------
# 2) Register widefield to confocal (per z-slab)
# -------------------------------
all_w_img_reg = []
for i in tqdm(range(all_w_img.shape[0]), desc="Registering stacks (per z)"):
    temp_w, temp_c = all_w_img[i], all_c_img[i]  # shapes: [stack, h, w]

    # Defensive checks on indices
    if REG_REF_IDX_W >= temp_w.shape[0] or REG_REF_IDX_C >= temp_c.shape[0]:
        raise IndexError(
            f"Reference index out of range for z={i}: "
            f"widefield has {temp_w.shape[0]} slices, confocal has {temp_c.shape[0]} slices. "
            f"Requested W={REG_REF_IDX_W}, C={REG_REF_IDX_C}"
        )

    temp_w_slice = normIMG(temp_w[REG_REF_IDX_W])
    temp_c_slice = normIMG(temp_c[REG_REF_IDX_C])

    # in_place=True modifies temp_w directly; we still append the returned view for clarity
    reg_w_stack, applied_shift = register_image_stack(temp_c_slice, temp_w_slice, temp_w, in_place=True)
    all_w_img_reg.append(reg_w_stack)

all_w_img_reg = np.asarray(all_w_img_reg)
print("Registered W shape:", all_w_img_reg.shape)

# Alias to keep the rest of your original variable names/logic working
all_w = all_w_img_reg
all_c = all_c_img

# -------------------------------
# 3) Detect nuclei on a chosen slice (here: slice 35)
# -------------------------------
all_pos = []
target_slice_for_detection = 35

if target_slice_for_detection >= all_w.shape[1]:
    raise IndexError(
        f"Detection slice {target_slice_for_detection} exceeds stack depth {all_w.shape[1]}"
    )

for i in tqdm(range(all_w.shape[0]), desc="Detecting nuclei"):
    temp = all_w[i, target_slice_for_detection]  # 2D image
    centroids, labels = detect_nuclei_positions(temp, sigma=2)
    valid_centroids = validate_centroids(temp, centroids, window_size=NUCLEI_WINDOW)
    valid_centroids = np.asarray(valid_centroids, dtype=np.int32)

    if PLOT_EXAMPLES:
        plt.figure(figsize=(4, 4))
        plt.imshow(temp, cmap='gray')
        if valid_centroids.size > 0:
            plt.scatter(valid_centroids[:, 0], valid_centroids[:, 1], s=5, label='centroids')
            plt.legend()
        plt.title(f"Validated Nuclei Centroids (stack {i})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"z={i}: {valid_centroids.shape[0]} centroids")
    all_pos.append(valid_centroids)

all_pos = np.asarray(all_pos, dtype=object)  # ragged per z is possible; keep as object array
print("all_pos length (per z):", len(all_pos))

# -------------------------------
# 4) Crop windows around detected positions across stacks
# -------------------------------
all_w_pos, all_c_pos = [], []

for i in tqdm(range(len(all_pos)), desc="Cropping windows"):
    if i in indices_filtered:
        print(f"Skipping unqualified stack z={i}")
        continue

    temp_pos = all_pos[i]
    if temp_pos is None or len(temp_pos) == 0:
        print(f"No centroids for z={i}, skipping.")
        continue

    cropped_w_windows = crop_stack_windows(all_w[i], temp_pos, window_size=NUCLEI_WINDOW)
    cropped_c_windows = crop_stack_windows(all_c[i], temp_pos, window_size=NUCLEI_WINDOW)

    # Defensive: ensure both crops have same number of items
    if cropped_w_windows.shape[0] != cropped_c_windows.shape[0]:
        raise ValueError(
            f"Crop count mismatch at z={i}: W={cropped_w_windows.shape[0]} vs C={cropped_c_windows.shape[0]}"
        )

    all_w_pos.append(cropped_w_windows.astype(np.float32))
    all_c_pos.append(cropped_c_windows.astype(np.float32))
    print(f"Cropped z={i}: {temp_pos.shape}")

if not all_w_pos:
    raise RuntimeError("No windows were cropped. Check detection/cropping parameters.")

# Concatenate along the centroid dimension
all_w_pos = np.concatenate(all_w_pos, axis=0)
all_c_pos = np.concatenate(all_c_pos, axis=0)[:, CROP_Z_RANGE]  # keep your original z-subsampling for confocal

print("Final cropped shapes -> W:", all_w_pos.shape, " C:", all_c_pos.shape)

# -------------------------------
# 5) Save the aggregated patch dataset
# -------------------------------
np.savez(os.path.join(OUT_PROCESSED, 'all_patch_128.npz'), w=all_w_pos, c=all_c_pos)

# -------------------------------
# 6) Split train/val/test and save
# -------------------------------
train_w, val_w, test_w = split_data(all_w_pos)
train_c, val_c, test_c = split_data(all_c_pos)

print(f"Train shape: W{train_w.shape}, C{train_c.shape}")
print(f"Validation shape: W{val_w.shape}, C{val_c.shape}")
print(f"Test shape: W{test_w.shape}, C{test_c.shape}")

np.savez(os.path.join(OUT_ROOT, 'train', 'train.npz'), w=train_w, c=train_c)
np.savez(os.path.join(OUT_ROOT, 'val', 'val.npz'), w=val_w, c=val_c)
np.savez(os.path.join(OUT_ROOT, 'test', 'test.npz'), w=test_w, c=test_c)
