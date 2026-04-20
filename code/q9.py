import cv2
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(BASE_DIR, "images", "q9_input.png") 
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q9")
os.makedirs(OUT_DIR, exist_ok=True)
# Load image
img = cv2.imread(IMG_PATH)

# Sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
# Apply sharpening
sharpened = cv2.filter2D(img, -1, kernel)
# Combine images side by side
combined = np.hstack((img, sharpened))
# Save result
cv2.imwrite(os.path.join(OUT_DIR, "q9_combined.png"),combined)

