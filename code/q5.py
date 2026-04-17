import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q5")

os.makedirs(OUT_DIR, exist_ok=True)

# (a) Compute a normalized 5×5 Gaussian kernel (σ = 2)

import numpy as np

def gaussian_kernel(size=5, sigma=2):
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for x in range(-k, k+1):
        for y in range(-k, k+1):
            kernel[x+k, y+k] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= (2 * np.pi * sigma**2)
    kernel /= kernel.sum()  # Normalize

    return kernel

kernel_5x5 = gaussian_kernel(5, 2)
print(kernel_5x5)
np.savetxt(os.path.join(OUT_DIR, "kernel_5x5_sigma2.txt"), kernel_5x5, fmt="%.6f")

# (b) Visualize a 51×51 Gaussian kernel as a 3D surface

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kernel_51 = gaussian_kernel(51, 2)

x = np.arange(51)
y = np.arange(51)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, kernel_51)

ax.set_title("3D Gaussian Kernel (51x51, σ=2)")

# Save figure
plt.savefig(os.path.join(OUT_DIR, "gaussian_3d_51x51.png"))
plt.close()

# (c) Apply Gaussian smoothing using manually computed kernel

import cv2

def apply_gaussian_manual(image, kernel):
    return cv2.filter2D(image, -1, kernel)

img = cv2.imread("images/runway.png", cv2.IMREAD_GRAYSCALE)

kernel = gaussian_kernel(5, 2)
smoothed_manual = apply_gaussian_manual(img, kernel)

cv2.imwrite("outputs/q5/manual_gaussian.png", smoothed_manual)

# (d) OpenCV GaussianBlur (save output image)

smoothed_cv = cv2.GaussianBlur(img, (5, 5), 2)

cv2.imwrite(os.path.join(OUT_DIR, "opencv_gaussian.png"), smoothed_cv)