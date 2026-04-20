import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(BASE_DIR, "images", "runway.png") 
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q10")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Manual Bilateral Filter
# -----------------------------
def bilateral_filter_manual(img, diameter, sigma_s, sigma_r):
    img = img.astype(np.float32)
    h, w = img.shape
    half = diameter // 2

    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            wp_total = 0
            filtered_pixel = 0

            for k in range(-half, half + 1):
                for l in range(-half, half + 1):

                    ni = i + k
                    nj = j + l

                    if 0 <= ni < h and 0 <= nj < w:

                        # Spatial Gaussian
                        gs = np.exp(-(k**2 + l**2) / (2 * sigma_s**2))

                        # Range Gaussian
                        gr = np.exp(-(img[ni, nj] - img[i, j])**2 / (2 * sigma_r**2))

                        w_p = gs * gr

                        filtered_pixel += img[ni, nj] * w_p
                        wp_total += w_p

            output[i, j] = filtered_pixel / wp_total

    return np.clip(output, 0, 255).astype(np.uint8)

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread(IMG_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# (b) Gaussian Blur
# -----------------------------
gaussian = cv2.GaussianBlur(gray, (5, 5), 2)

# -----------------------------
# (c) OpenCV Bilateral Filter
# -----------------------------
bilateral_cv = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=2)

# -----------------------------
# (d) Manual Bilateral Filter
# -----------------------------
bilateral_manual = bilateral_filter_manual(gray, diameter=5, sigma_s=2, sigma_r=50)

# -----------------------------
# Save Outputs
# -----------------------------
cv2.imwrite(os.path.join(OUT_DIR, "original.png"), gray)
cv2.imwrite(os.path.join(OUT_DIR, "gaussian.png"), gaussian)
cv2.imwrite(os.path.join(OUT_DIR, "bilateral_cv.png"), bilateral_cv)
cv2.imwrite(os.path.join(OUT_DIR, "bilateral_manual.png"), bilateral_manual)

# -----------------------------
# Combined Figure (IMPORTANT FOR REPORT)
# -----------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(bilateral_cv, cmap='gray')
plt.title("Bilateral (OpenCV)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(bilateral_manual, cmap='gray')
plt.title("Bilateral (Manual)")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparison.png"))
plt.close()