#!/usr/bin/env python3

"""
lic.py — Convolutional line integration (LIC) to visualize 2D vector fields.

Requires: numpy, Pillow
"""

import math
import numpy as np
from PIL import Image


# ---------------------------
# Utilities: kernels & sampler
# ---------------------------

def _hann_kernel(n_half: int) -> np.ndarray:
    """Symmetric Hann kernel of length 2*n_half+1, normalized to sum=1."""
    if n_half <= 0:
        return np.array([1.0], dtype=np.float32)
    L = 2 * n_half + 1
    k = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(L) / (L - 1)))
    k = k.astype(np.float32)
    k /= k.sum()
    return k

def _box_kernel(n_half: int) -> np.ndarray:
    L = 2 * n_half + 1
    k = np.ones(L, dtype=np.float32)
    k /= k.sum()
    return k


def _bilinear_sample_gray(img: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample a single-channel image at (x,y) with clamp-to-edge."""
    h, w = img.shape
    if x < 0.0: x = 0.0
    if y < 0.0: y = 0.0
    if x > w - 1.0: x = w - 1.0
    if y > h - 1.0: y = h - 1.0

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    v0 = v00 * (1 - fx) + v10 * fx
    v1 = v01 * (1 - fx) + v11 * fx
    return v0 * (1 - fy) + v1 * fy


def _bilinear_sample_vec(vec: np.ndarray, x: float, y: float) -> tuple:
    """Bilinear sample a 2‑channel vector field at (x,y) with clamp-to-edge."""
    h, w, _ = vec.shape
    if x < 0.0: x = 0.0
    if y < 0.0: y = 0.0
    if x > w - 1.0: x = w - 1.0
    if y > h - 1.0: y = h - 1.0

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0

    # Corners
    v00x, v00y = vec[y0, x0, 0], vec[y0, x0, 1]
    v10x, v10y = vec[y0, x1, 0], vec[y0, x1, 1]
    v01x, v01y = vec[y1, x0, 0], vec[y1, x0, 1]
    v11x, v11y = vec[y1, x1, 0], vec[y1, x1, 1]

    v0x = v00x * (1 - fx) + v10x * fx
    v0y = v00y * (1 - fx) + v10y * fx
    v1x = v01x * (1 - fx) + v11x * fx
    v1y = v01y * (1 - fx) + v11y * fx

    vx = v0x * (1 - fy) + v1x * fy
    vy = v0y * (1 - fy) + v1y * fy
    return vx, vy


# ---------------------------
# Core LIC implementation
# ---------------------------

def lic(
    vec: np.ndarray,
    noise: np.ndarray,
    kernel_length_px: float = 20.0,
    step_px: float = 1.0,
    kernel: str = "hann",
    oriented: bool = False,
) -> np.ndarray:
    """
    Compute a grayscale LIC image.

    Args:
        vec: (H, W, 2) vector field. Can be unnormalized; we normalize internally.
        noise: (H, W) float32 in [0,1]; if None, white noise is generated.
        kernel_length_px: total convolution length in pixels (approx).
        step_px: step size along streamline in pixels.
        kernel: 'hann' or 'box' weighting along the streamline.
        oriented: if True, flips sign for backward direction (oriented LIC / OLIC).

    Returns:
        (H, W) float32 image in [0,1].
    """
    assert vec.ndim == 3 and vec.shape[2] == 2, "vec must be HxWx2"
    H, W, _ = vec.shape
    if noise is None:
        noise = np.random.RandomState(0).rand(H, W).astype(np.float32)
    else:
        assert noise.shape == (H, W)
        noise = noise.astype(np.float32, copy=False)

    # Normalize the vector field to unit direction (avoid division by 0).
    vf = vec.astype(np.float32)
    mag = np.sqrt(vf[..., 0] * vf[..., 0] + vf[..., 1] * vf[..., 1]) + 1e-8
    vf[..., 0] /= mag
    vf[..., 1] /= mag

    # Determine half-length (number of steps each side of the center).
    n_half = max(1, int(0.5 * kernel_length_px / max(1e-6, step_px)))

    if kernel.lower().startswith("han"):
        k = _hann_kernel(n_half)
    else:
        k = _box_kernel(n_half)

    # Center index in kernel
    k_center = n_half

    out = np.zeros((H, W), dtype=np.float32)
    _lic_accumulate(out, vf, noise, k, k_center, step_px, oriented)
    # Normalize to [0,1]
    out -= out.min()
    m = out.max()
    if m > 0:
        out /= m
    return out


def _lic_accumulate(out, vf, noise, k, k_center, step_px, oriented):
    H, W = out.shape
    n_half = k_center

    for y0 in range(H):
        for x0 in range(W):
            # Start at pixel center
            x = x0 + 0.5
            y = y0 + 0.5

            acc = k[k_center] * _bilinear_sample_gray(noise, x, y)
            wsum = k[k_center]

            # Forward (+) and backward (-) directions
            for sign in (-1.0, +1.0):
                xs, ys = x, y
                vx, vy = _bilinear_sample_vec(vf, xs, ys)

                for i in range(1, n_half + 1):
                    # If the field is degenerate, stop integrating this side.
                    vlen = math.hypot(vx, vy)
                    if vlen < 1e-6:
                        break
                    # Unit direction already, but be robust:
                    ux = vx / vlen
                    uy = vy / vlen

                    xs += sign * ux * step_px
                    ys += sign * uy * step_px

                    val = _bilinear_sample_gray(noise, xs, ys)
                    w = k[k_center + (i if sign > 0 else -i)]  # symmetric kernel

                    # Oriented LIC flips sign of backward samples to preserve direction
                    if oriented and sign < 0:
                        val = 1.0 - val  # simple phase flip (OLIC-style)

                    acc += w * val
                    wsum += w

                    # Update direction field at new position
                    vx, vy = _bilinear_sample_vec(vf, xs, ys)

            out[y0, x0] = acc / (wsum + 1e-12)


# ---------------------------
# Optional: direction-color overlay
# ---------------------------

def colorize_direction(vec: np.ndarray, lic_gray: np.ndarray, sat: float = 0.85) -> np.ndarray:
    """
    Map flow direction to hue and use LIC intensity as value for an HSV->RGB overlay.
    """
    H, W, _ = vec.shape
    vx = vec[..., 0].astype(np.float32)
    vy = vec[..., 1].astype(np.float32)
    ang = np.arctan2(vy, vx)  # [-pi, pi]
    hue = (ang + np.pi) / (2 * np.pi)  # [0,1)

    # HSV to RGB
    h = (hue * 6.0).astype(np.float32)
    c = sat * 1.0  # value will be lic_gray later
    x = c * (1 - np.abs((h % 2) - 1))

    z = np.zeros_like(h)
    r = np.where((0 <= h) & (h < 1), c, np.where((1 <= h) & (h < 2), x,
        np.where((2 <= h) & (h < 3), z, np.where((3 <= h) & (h < 4), z,
        np.where((4 <= h) & (h < 5), x, c)))))
    g = np.where((0 <= h) & (h < 1), x, np.where((1 <= h) & (h < 2), c,
        np.where((2 <= h) & (h < 3), c, np.where((3 <= h) & (h < 4), x,
        np.where((4 <= h) & (h < 5), z, z)))))
    b = np.where((0 <= h) & (h < 1), z, np.where((1 <= h) & (h < 2), z,
        np.where((2 <= h) & (h < 3), x, np.where((3 <= h) & (h < 4), c,
        np.where((4 <= h) & (h < 5), c, x)))))

    # Multiply by LIC value as V (value channel)
    V = np.clip(lic_gray, 0.0, 1.0).astype(np.float32)
    rgb = np.stack([r * V, g * V, b * V], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------
# Demo / CLI
# ---------------------------

def _make_swirl(H=512, W=512) -> np.ndarray:
    """A simple divergence-free swirl centered in the image."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    X = (x - cx) / (0.5 * min(H, W))
    Y = (y - cy) / (0.5 * min(H, W))
    # Rotational field: v = (-Y, X) scaled by radius falloff
    R2 = X * X + Y * Y + 1e-6
    vx = -Y / np.sqrt(R2)
    vy =  X / np.sqrt(R2)
    return np.stack([vx, vy], axis=-1).astype(np.float32)


def _save_gray(img: np.ndarray, path: str):
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(path)


def _save_rgb(img: np.ndarray, path: str):
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="RGB").save(path)


if __name__ == "__main__":
    # Build a demo field and render LIC
    vf = _make_swirl(720, 720)

    lic_img = lic(
        vf,
        noise=None,                 # auto-generate white noise
        kernel_length_px=40.0,      # longer = smoother streaks
        step_px=1.0,                # integration step in pixels
        kernel="hann",              # 'box' also supported
        oriented=False,             # True -> oriented LIC (OLIC-like)
    )

    _save_gray(lic_img, "lic_grayscale.png")

    # Direction-colored overlay (HSV hue = flow direction)
    rgb = colorize_direction(vf, lic_img, sat=0.85)
    _save_rgb(rgb, "lic_color.png")

    print("Wrote lic_grayscale.png and lic_color.png")

'''
How to plug in your own data.

# Assume U, V are your numpy arrays (H×W) in physical units
vec = np.stack([U, V], axis=-1).astype(np.float32)

# Optional: map physical grid spacing into pixels before calling lic()
# e.g., if dx != dy, scale components so a 1-pixel step is roughly isotropic
vec[..., 0] *= (dx / min(dx, dy))
vec[..., 1] *= (dy / min(dx, dy))

img = lic(vec, kernel_length_px=30, step_px=0.8, kernel="hann")
'''
