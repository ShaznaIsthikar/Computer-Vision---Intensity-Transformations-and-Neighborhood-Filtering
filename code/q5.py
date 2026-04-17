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