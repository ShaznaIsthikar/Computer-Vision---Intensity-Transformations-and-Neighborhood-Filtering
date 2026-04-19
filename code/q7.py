import cv2
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_SMALL = os.path.join(BASE_DIR, "images", "im01small.png")
IMG_LARGE = os.path.join(BASE_DIR, "images", "im01.png")

OUT_DIR = os.path.join(BASE_DIR, "outputs", "q7")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Nearest Neighbor Zoom
# -----------------------------
def zoom_nearest(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    zoomed = np.zeros((new_h, new_w, 3), dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            x = int(i / scale)
            y = int(j / scale)
            zoomed[i, j] = img[x, y]

    return zoomed

# -----------------------------
# Bilinear Interpolation Zoom
# -----------------------------
def zoom_bilinear(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    zoomed = np.zeros((new_h, new_w, 3), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            x = i / scale
            y = j / scale

            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)

            dx, dy = x - x1, y - y1

            top = (1 - dy) * img[x1, y1] + dy * img[x1, y2]
            bottom = (1 - dy) * img[x2, y1] + dy * img[x2, y2]

            zoomed[i, j] = (1 - dx) * top + dx * bottom

    return zoomed.astype(np.uint8)

# -----------------------------
# SSD Calculation
# -----------------------------
def normalized_ssd(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    ssd = np.sum(diff ** 2)
    return ssd / (img1.shape[0] * img1.shape[1])

# -----------------------------
# Main
# -----------------------------
small = cv2.imread(IMG_SMALL)
large = cv2.imread(IMG_LARGE)

scale = large.shape[0] / small.shape[0]

nn_img = zoom_nearest(small, scale)
bl_img = zoom_bilinear(small, scale)

# Save outputs
cv2.imwrite(os.path.join(OUT_DIR, "nearest.png"), nn_img)
cv2.imwrite(os.path.join(OUT_DIR, "bilinear.png"), bl_img)

# Resize to exact match (safety)
nn_img = cv2.resize(nn_img, (large.shape[1], large.shape[0]))
bl_img = cv2.resize(bl_img, (large.shape[1], large.shape[0]))

# Compute SSD
ssd_nn = normalized_ssd(nn_img, large)
ssd_bl = normalized_ssd(bl_img, large)

print("Normalized SSD (Nearest Neighbor):", ssd_nn)
print("Normalized SSD (Bilinear):", ssd_bl)

