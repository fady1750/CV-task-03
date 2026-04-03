"""
filters.py – Core computer-vision algorithms, implemented from scratch.

All mathematical operations use plain NumPy – no OpenCV, scikit-image,
or scipy functions are called for the algorithm logic.

Contents
--------
Primitives
    _make_gaussian_kernel   – build a 2-D Gaussian kernel
    _convolve2d             – 2-D convolution via sliding-window view
    _sobel_gradients        – Ix, Iy using Sobel kernels
    _nms_2d                 – non-maximum suppression on a response map

Feature Detectors
    HarrisDetector          – Harris (R = det − k·tr²) and λ- (min eigenvalue)

Feature Descriptors
    SIFTDescriptor          – 128-dim SIFT descriptor (uses Harris keypoints)

Feature Matchers
    SSDMatcher              – Sum of Squared Differences + Lowe ratio test
    NCCMatcher              – Normalised Cross-Correlation
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#  LOW-LEVEL PRIMITIVES  (pure NumPy, no built-in CV functions)

def _make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a square, normalised 2-D Gaussian kernel.

    Parameters
    ----------
    size  : side length (should be odd).
    sigma : standard deviation in pixels.

    Returns
    -------
    kernel : float64 ndarray of shape (size, size), sum = 1.
    """
    k = size // 2
    y, x = np.mgrid[-k: k + 1, -k: k + 1]          # integer grids
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float64)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D discrete convolution (correlation) using NumPy's sliding_window_view.

    Equivalent to ``scipy.ndimage.convolve(image, kernel, mode='reflect')``
    but implemented manually with NumPy primitives only.

    Complexity: O(H · W · kh · kw) via vectorised einsum.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded  = np.pad(
        image.astype(np.float64),
        ((ph, ph), (pw, pw)),
        mode="reflect",
    )
    # windows shape: (H, W, kh, kw)
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel.astype(np.float64))


def _sobel_gradients(
    gray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial gradients using the Sobel operator (built from scratch).

    Kernels
    -------
    Kx = [[-1, 0, 1],      Ky = [[-1, -2, -1],
           [-2, 0, 2],             [ 0,  0,  0],
           [-1, 0, 1]]             [ 1,  2,  1]]

    Returns
    -------
    Ix : horizontal gradient (∂I/∂x)
    Iy : vertical   gradient (∂I/∂y)
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)
    return _convolve2d(gray, Kx), _convolve2d(gray, Ky)


def _nms_2d(response: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Non-maximum suppression: a pixel is kept iff it is the maximum
    within its (window × window) neighbourhood.

    Returns
    -------
    mask : boolean ndarray of the same shape as *response*.
    """
    half   = window // 2
    padded = np.pad(response, half, mode="constant", constant_values=0.0)
    # local_max[i,j] = maximum in the window centred at (i,j)
    local_max = sliding_window_view(padded, (window, window)).max(axis=(-2, -1))
    return (response == local_max) & (response > 0.0)

