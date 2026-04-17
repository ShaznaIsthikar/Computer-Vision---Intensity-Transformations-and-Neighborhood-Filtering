import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(BASE_DIR, "images", "girl.png")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q4")

os.makedirs(OUT_DIR, exist_ok=True)

print(IMG_PATH)
print(os.path.exists(IMG_PATH))

# Load image
img = cv2.imread(IMG_PATH)
print("Image object:", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite(os.path.join(OUT_DIR, "gray.png"), gray)

# Otsu thresholding
threshold_value, binary_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu Threshold Value:", threshold_value)

cv2.imwrite(os.path.join(OUT_DIR, "binary_mask.png"), binary_mask)

foreground = cv2.bitwise_and(gray, gray, mask=binary_mask)

cv2.imwrite(os.path.join(OUT_DIR, "foreground.png"), foreground)