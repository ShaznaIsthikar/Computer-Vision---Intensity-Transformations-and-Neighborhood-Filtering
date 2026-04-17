import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths (as per your structure)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(BASE_DIR, "images", "runway.png")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q3")

os.makedirs(OUT_DIR, exist_ok=True)

def histogram_equalization(img):
    # Flatten image
    flat = img.flatten()

    # Step 1: Histogram
    hist = np.bincount(flat, minlength=256)

    # Step 2: Normalize histogram (PDF)
    pdf = hist / np.sum(hist)

    # Step 3: Cumulative Distribution Function (CDF)
    cdf = np.cumsum(pdf)

    # Step 4: Mapping (scale to [0,255])
    cdf_scaled = np.round(cdf * 255).astype(np.uint8)

    # Step 5: Map original pixels
    equalized = cdf_scaled[flat]

    # Reshape back to image
    equalized_img = equalized.reshape(img.shape)

    return equalized_img, hist, cdf

# Read image (grayscale)
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

# Apply custom histogram equalization
eq_img, hist, cdf = histogram_equalization(img)

# Save outputs
cv2.imwrite(os.path.join(OUT_DIR, "original.png"), img)
cv2.imwrite(os.path.join(OUT_DIR, "equalized.png"), eq_img)

# Plot histograms
plt.figure()
plt.hist(img.ravel(), bins=256)
plt.title("Original Histogram")
plt.savefig(os.path.join(OUT_DIR, "hist_original.png"))

plt.figure()
plt.hist(eq_img.ravel(), bins=256)
plt.title("Equalized Histogram")
plt.savefig(os.path.join(OUT_DIR, "hist_equalized.png"))

print("Q3 completed. Outputs saved in:", OUT_DIR)