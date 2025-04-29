import numpy as np
from skimage.measure import label
from PIL import Image
import cv2

# Mean shift filtering for grayscale images
def mean_shift_filter_grey(I, w, h_r, epsilon=1e-5, max_iter=100):
    H, W = I.shape
    F = np.zeros_like(I, dtype=float)
    for i in range(H):
        for j in range(W):
            I_current = I[i, j]
            for _ in range(max_iter):
                k_min = max(0, i - w)
                k_max = min(H, i + w + 1)
                l_min = max(0, j - w)
                l_max = min(W, j + w + 1)
                window = I[k_min:k_max, l_min:l_max]
                mask = np.abs(window - I_current) < h_r
                if np.sum(mask) == 0:
                    break
                I_mean = np.mean(window[mask])
                if np.abs(I_mean - I_current) < epsilon:
                    break
                I_current = I_mean
            F[i, j] = I_current
    return F

# Segmentation for grayscale images
def segment_grey(I, w, h_r, epsilon=1e-5, max_iter=100, round_decimals=1):
    F = mean_shift_filter_grey(I, w, h_r, epsilon, max_iter)
    F_rounded = np.round(F, decimals=round_decimals)
    unique_values, inverse_indices = np.unique(F_rounded, return_inverse=True)
    index_img = inverse_indices.reshape(I.shape)
    labels = label(index_img)
    return labels, len(unique_values)

# Mean shift filtering for color images (RGB)
def mean_shift_filter_color(I, w, h_r, epsilon=1e-5, max_iter=100):
    H, W, _ = I.shape
    F = np.zeros_like(I, dtype=float)
    for i in range(H):
        for j in range(W):
            C_current = I[i, j].copy()
            for _ in range(max_iter):
                k_min = max(0, i - w)
                k_max = min(H, i + w + 1)
                l_min = max(0, j - w)
                l_max = min(W, j + w + 1)
                window = I[k_min:k_max, l_min:l_max]
                distances = np.sqrt(np.sum((window - C_current) ** 2, axis=2))
                mask = distances < h_r
                if np.sum(mask) == 0:
                    break
                C_mean = np.mean(window[mask], axis=0)
                if np.sqrt(np.sum((C_mean - C_current) ** 2)) < epsilon:
                    break
                C_current = C_mean
            F[i, j] = C_current
    return F

# Segmentation for color images (RGB)
def segment_color(I, w, h_r, epsilon=1e-5, max_iter=100, round_decimals=1):
    F = mean_shift_filter_color(I, w, h_r, epsilon, max_iter)
    F_rounded = np.round(F, decimals=round_decimals)
    H, W, _ = F.shape
    F_flat = F_rounded.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(F_flat, axis=0, return_inverse=True)
    index_img = inverse_indices.reshape(H, W)
    labels = label(index_img)
    return labels, len(unique_colors)


# Example usage
def run_mean_shift_segmentation(img_path, w, h_r, round_decimals=1):
    epsilon=1e-5, max_iter=100
    original = cv2.imread(img_path)
    img = Image.open(img_path)
    if img.mode == 'L':  # Grayscale
        I = np.array(img, dtype=float) / 255.0
        labels, num_segments = segment_grey(I, w, h_r, epsilon, max_iter, round_decimals)
    else:  # RGB
        I = np.array(img.convert('RGB'), dtype=float) / 255.0
        labels, num_segments = segment_color(I, w, h_r, epsilon, max_iter, round_decimals)

    print(f"Number of segments: {num_segments}")
    return original, labels