"""
utils.py – Image helpers, matplotlib visualizations, and synthetic test images.

All routines operate on plain NumPy arrays; no OpenCV or scikit-image is used.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# IMAGE CONVERSION HELPERS
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to float64 grayscale.

    Uses ITU-R BT.601 luminance weights for colour images:
        Y = 0.2989·R + 0.5870·G + 0.1140·B
    """
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.shape[2] == 4:               # RGBA → drop alpha
        image = image[..., :3]
    return (0.2989 * image[..., 0].astype(np.float64)
            + 0.5870 * image[..., 1].astype(np.float64)
            + 0.1140 * image[..., 2].astype(np.float64))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = float(image.min()), float(image.max())
    if mx == mn:
        return np.zeros_like(image, dtype=np.float64)
    return (image.astype(np.float64) - mn) / (mx - mn)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 255] uint8 via min-max normalization."""
    return (normalize_image(image) * 255).astype(np.uint8)


def _extract_rc(keypoints: list) -> Tuple[list, list]:
    """
    Safely extract (row, col) from keypoints that may be:
      - (row, col)           — 2-tuple
      - (row, col, strength) — 3-tuple
    Returns two lists: rows, cols.
    """
    rows, cols = [], []
    for kp in keypoints:
        rows.append(kp[0])
        cols.append(kp[1])
    return rows, cols


# MATPLOTLIB VISUALIZATION

def fig_keypoints(
    image: np.ndarray,
    keypoints: list,
    title: str = "Keypoints",
    color: str = "lime",
    max_kp: int = 500,
) -> plt.Figure:
    """
    Return a Figure with *keypoints* scattered over the image.

    Accepts keypoints as (row, col) or (row, col, strength).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    gray = to_grayscale(image)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255 if gray.max() > 1 else 1)

    shown = keypoints[:max_kp]
    if shown:
        ys, xs = _extract_rc(shown)
        ax.scatter(
            xs, ys,
            s=10, c=color,
            linewidths=0.4, edgecolors="black",
            zorder=5,
        )
    ax.set_title(f"{title}\n({len(shown)} shown / {len(keypoints)} total)",
                 fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_response_map(
    response: np.ndarray,
    title: str = "Response Map",
    cmap: str = "hot",
) -> plt.Figure:
    """Return a Figure showing *response* as a colour-mapped heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(response, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_matches(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    matches: List[Tuple[int, int, float]],
    title: str = "Feature Matches",
    max_matches: int = 50,
) -> plt.Figure:
    """
    Return a Figure with matched keypoints connected by coloured lines.

    Accepts keypoints as (row, col) or (row, col, strength).
    """
    shown = matches[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)

    g1 = normalize_image(to_grayscale(img1))
    g2 = normalize_image(to_grayscale(img2))

    canvas = np.zeros((H, w1 + w2), dtype=np.float64)
    canvas[:h1, :w1]      = g1
    canvas[:h2, w1:w1+w2] = g2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)

    cmap_lines = plt.get_cmap("hsv", max(len(shown), 1))
    for idx, (i, j, _score) in enumerate(shown):
        r1, c1 = kp1[i][0], kp1[i][1]   # safe: works for 2-tuple or 3-tuple
        r2, c2 = kp2[j][0], kp2[j][1]
        c2_shifted = c2 + w1
        col = cmap_lines(idx / max(len(shown), 1))
        ax.plot([c1, c2_shifted], [r1, r2], "-", color=col,
                linewidth=0.9, alpha=0.75)
        ax.plot(c1,         r1, "o", color=col, markersize=4)
        ax.plot(c2_shifted, r2, "o", color=col, markersize=4)

    ax.axvline(x=w1, color="white", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_title(f"{title}  ({len(shown)} matches shown / {len(matches)} total)",
                 fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_descriptor_heatmaps(
    descriptors: np.ndarray,
    n: int = 5,
) -> plt.Figure:
    """
    Visualise up to *n* 128-dim SIFT descriptors as 16×8 heatmaps.
    """
    n = min(n, len(descriptors))
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No descriptors to display",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        d_vis = descriptors[i].reshape(16, 8)
        im = ax.imshow(d_vis, cmap="viridis", aspect="auto")
        ax.set_title(f"KP #{i + 1}", fontsize=9)
        ax.set_xlabel("Orientation bin", fontsize=8)
        if i == 0:
            ax.set_ylabel("Spatial cell (4×4 grid)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)

    fig.suptitle(
        "SIFT Descriptor Heatmaps — 128-dim reshaped to 16×8\n"
        "(rows = 4×4 sub-cells, cols = 8 orientation bins)",
        fontsize=9,
    )
    plt.tight_layout()
    return fig


# SYNTHETIC TEST IMAGES

def create_sample_images() -> Dict[str, np.ndarray]:
    """
    Generate a set of synthetic images for quick testing.
    """
    samples: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)

    # 1. Checkerboard
    size, sq = 256, 32
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(size // sq):
        for j in range(size // sq):
            if (i + j) % 2 == 0:
                board[i * sq:(i + 1) * sq, j * sq:(j + 1) * sq] = 255
    samples["Checkerboard"] = np.stack([board] * 3, axis=-1)

    # 2. Checkerboard (shifted)
    shifted = np.roll(board, shift=(16, 16), axis=(0, 1))
    samples["Checkerboard (Shifted)"] = np.stack([shifted] * 3, axis=-1)

    # 3. Geometric shapes
    geo = np.full((300, 400, 3), 200, dtype=np.uint8)
    geo[40:130,  40:160] = [30,  100, 210]
    geo[40:130, 230:350] = [210,  50,  50]
    y, x = np.ogrid[:300, :400]
    geo[(y - 210) ** 2 + (x -  100) ** 2 <= 55 ** 2] = [30, 180, 80]
    geo[(y - 210) ** 2 + (x -  300) ** 2 <= 55 ** 2] = [220, 200, 50]
    for row in range(80):
        hw = row // 2
        geo[160 + row, 190 - hw: 210 + hw] = [240, 240, 240]
    samples["Geometric Shapes"] = geo

    # 4. Building pattern
    bld = np.full((256, 256, 3), 160, dtype=np.uint8)
    for r in range(10, 246, 36):
        for c in range(10, 246, 28):
            bld[r:r + 22, c:c + 18] = [70, 100, 145]
    samples["Building Pattern"] = bld

    # 5. Noisy gradient
    grad = np.tile(np.linspace(0, 255, 256, dtype=np.float32), (256, 1))
    noise = rng.integers(-25, 25, (256, 256), dtype=np.int16)
    grad = np.clip(grad.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    samples["Noisy Gradient"] = np.stack([grad] * 3, axis=-1)

    return samples