import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

IMG_PATH = os.path.join(BASE_DIR, "images", "figure2.jpg")  # update if needed
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q2")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 2. Load Image
# -----------------------------
img_bgr = cv2.imread(IMG_PATH)

if img_bgr is None:
    raise FileNotFoundError(f"Image not found at {IMG_PATH}")

# Convert once
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(img_lab)

L_norm = L / 255.0

# Save original histogram once
plt.figure()
plt.hist(L.flatten(), bins=256)
plt.title("Original L Histogram")
plt.savefig(os.path.join(OUT_DIR, "hist_original.png"))
plt.close()

# Save original image
cv2.imwrite(os.path.join(OUT_DIR, "original.png"), img_bgr)

# -----------------------------
# 3. Gamma Values Loop
# -----------------------------
gamma_values = [0.5, 2.0]

for gamma in gamma_values:
    # Apply gamma
    L_gamma = np.power(L_norm, gamma)
    L_new = np.uint8(L_gamma * 255)

    # Merge and convert back
    img_lab_corrected = cv2.merge((L_new, a, b))
    img_corrected_bgr = cv2.cvtColor(img_lab_corrected, cv2.COLOR_LAB2BGR)

    # Save image
    out_img_path = os.path.join(OUT_DIR, f"gamma_{gamma}.png")
    cv2.imwrite(out_img_path, img_corrected_bgr)

    # Save histogram
    plt.figure()
    plt.hist(L_new.flatten(), bins=256)
    plt.title(f"Gamma Histogram (γ={gamma})")
    plt.savefig(os.path.join(OUT_DIR, f"hist_gamma_{gamma}.png"))
    plt.close()

print("✅ Processing complete for gamma values:", gamma_values)