import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, pi, exp, diff, simplify

BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
IMG_PATH = os.path.join(BASE_DIR, "images", "girl.png")  # change if needed
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q6")

os.makedirs(OUT_DIR, exist_ok=True)

def verify_derivative_of_gaussian():
    x, y, sigma = symbols('x y sigma', positive=True, real=True)

    G = (1 / (2 * pi * sigma**2)) * exp(-(x**2 + y**2) / (2 * sigma**2))

    dG_dx = diff(G, x)
    dG_dy = diff(G, y)

    rhs_x = -(x / sigma**2) * G
    rhs_y = -(y / sigma**2) * G

    check_x = simplify(dG_dx - rhs_x)
    check_y = simplify(dG_dy - rhs_y)

    print("\n=== Part (a): Symbolic Verification ===")
    print("G(x, y) =", G)
    print("dG/dx   =", dG_dx)
    print("dG/dy   =", dG_dy)
    print("simplify(dG/dx - (-(x/sigma^2)G)) =", check_x)
    print("simplify(dG/dy - (-(y/sigma^2)G)) =", check_y)

    with open(os.path.join(OUT_DIR, "part_a_verification.txt"), "w", encoding="utf-8") as f:
        f.write("Derivative of Gaussian symbolic verification\n\n")
        f.write(f"G(x,y) = {G}\n\n")
        f.write(f"dG/dx = {dG_dx}\n\n")
        f.write(f"dG/dy = {dG_dy}\n\n")
        f.write(f"simplify(dG/dx - (-(x/sigma^2)G)) = {check_x}\n")
        f.write(f"simplify(dG/dy - (-(y/sigma^2)G)) = {check_y}\n")

    print("Saved symbolic result to:", os.path.join(OUT_DIR, "part_a_verification.txt"))


verify_derivative_of_gaussian()

import numpy as np

def gaussian_derivative_kernels(size=5, sigma=2):
    k = size // 2
    x = np.arange(-k, k+1)
    y = np.arange(-k, k+1)
    X, Y = np.meshgrid(x, y)

    # Gaussian
    G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Derivatives
    Gx = -(X / sigma**2) * G
    Gy = -(Y / sigma**2) * G

    # Normalize (sum of absolute values = 1)
    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))

    return Gx, Gy

Gx, Gy = gaussian_derivative_kernels()
print("Gx:\n", Gx)
print("Gy:\n", Gy)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_derivative(size=51, sigma=2):
    k = size // 2
    x = np.arange(-k, k+1)
    y = np.arange(-k, k+1)
    X, Y = np.meshgrid(x, y)

    G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    Gx = -(X / sigma**2) * G

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Gx)
    ax.set_title("Derivative of Gaussian (X-direction)")
    plt.show()

    save_path = os.path.join(OUT_DIR, "dog_3d_surface.png")
    plt.savefig(save_path, dpi=200)


visualize_derivative()

import cv2

img = cv2.imread("images/runway.png", cv2.IMREAD_GRAYSCALE)

Gx, Gy = gaussian_derivative_kernels()

grad_x = cv2.filter2D(img, -1, Gx)
grad_y = cv2.filter2D(img, -1, Gy)

# Gradient magnitude
grad_mag = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite(os.path.join(OUT_DIR, "grad_x.png"), grad_x)
cv2.imwrite(os.path.join(OUT_DIR, "grad_y.png"), grad_y)
cv2.imwrite(os.path.join(OUT_DIR, "grad_mag.png"), grad_mag)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite(os.path.join(OUT_DIR, "sobel_x.png"), sobel_x)
cv2.imwrite(os.path.join(OUT_DIR, "sobel_y.png"), sobel_y)
cv2.imwrite(os.path.join(OUT_DIR, "sobel_mag.png"), sobel_mag)

