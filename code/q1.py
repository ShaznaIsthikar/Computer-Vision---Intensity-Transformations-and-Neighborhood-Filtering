import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

IMG_PATH = os.path.join(BASE_DIR, "images", "runway.png")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q1")

os.makedirs(OUT_DIR, exist_ok=True)

# Load image (grayscale)
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Check path!")
img_norm = img / 255.0  # normalize to [0,1]

# -------------------------
# Gamma Correction Function
# -------------------------
def gamma_correction(image, gamma):
    return np.power(image, gamma)

# -------------------------
# Contrast Stretching
# -------------------------
def contrast_stretch(image, r1=0.2, r2=0.8):
    stretched = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = image[i, j]
            if r < r1:
                s = 0
            elif r <= r2:
                s = (r - r1) / (r2 - r1)
            else:
                s = 1
            stretched[i, j] = s

    return stretched

# Apply transformations
gamma_05 = gamma_correction(img_norm, 0.5)
gamma_2 = gamma_correction(img_norm, 2)
contrast = contrast_stretch(img_norm)

# Convert back to 0–255
gamma_05_img = (gamma_05 * 255).astype(np.uint8)
gamma_2_img = (gamma_2 * 255).astype(np.uint8)
contrast_img = (contrast * 255).astype(np.uint8)

# Save images
cv2.imwrite(os.path.join(OUT_DIR, "gamma_0.5.png"), gamma_05_img)
cv2.imwrite(os.path.join(OUT_DIR, "gamma_2.png"), gamma_2_img)
cv2.imwrite(os.path.join(OUT_DIR, "contrast_stretch.png"), contrast_img)

# -------------------------
# Plot histograms
# -------------------------
def plot_hist(image, title, filename):
    plt.figure()
    plt.hist(image.ravel(), bins=256)
    plt.title(title)
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

plot_hist(img, "Original Histogram", "hist_original.png")
plot_hist(gamma_05_img, "Gamma 0.5 Histogram", "hist_gamma_05.png")
plot_hist(gamma_2_img, "Gamma 2 Histogram", "hist_gamma_2.png")
plot_hist(contrast_img, "Contrast Stretch Histogram", "hist_contrast.png")

print(f"✅ Outputs saved in: {OUT_DIR}")