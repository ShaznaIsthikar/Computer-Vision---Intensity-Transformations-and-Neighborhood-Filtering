import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(BASE_DIR, "images", "Figure4.png") 
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q8")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

# -----------------------------
# (a) Gaussian Smoothing
# -----------------------------
gaussian = cv2.GaussianBlur(img, (5, 5), 1)

# -----------------------------
# (b) Median Filtering
# -----------------------------
median = cv2.medianBlur(img, 5)

# -----------------------------
# Save Outputs
# -----------------------------
cv2.imwrite(os.path.join(OUT_DIR, "original.png"), img)
cv2.imwrite(os.path.join(OUT_DIR, "gaussian.png"), gaussian)
cv2.imwrite(os.path.join(OUT_DIR, "median.png"), median)

# -----------------------------
# Combined Visualization (IMPORTANT for report)
# -----------------------------
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(median, cmap='gray')
plt.title("Median Filter")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparison.png"))
plt.close()