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