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
plt.show()