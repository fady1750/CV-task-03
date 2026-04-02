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
from typing import List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base  import FeatureDetector, FeatureDescriptor, FeatureMatcher
from .utils import to_grayscale


# ═══════════════════════════════════════════════════════════════
#  LOW-LEVEL PRIMITIVES  (pure NumPy, no built-in CV functions)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  HARRIS CORNER DETECTOR
# ═══════════════════════════════════════════════════════════════

class HarrisDetector(FeatureDetector):
    """
    Harris / Shi-Tomasi corner detector — implemented entirely from scratch.

    Two response formulations are supported:

    ``'harris'``
        Classic Harris:  R = det(M) − k · trace(M)²
        where M is the 2×2 structure tensor.

    ``'lambda_minus'``
        Shi-Tomasi / λ₋ method:  R = λ_min(M)
        Computed analytically as:
            R = (A+C)/2 − √( ((A−C)/2)² + B² )
        with A = Σ Ix², C = Σ Iy², B = Σ IxIy.

    Parameters
    ----------
    method          : 'harris' or 'lambda_minus'
    k               : Harris sensitivity constant  (default 0.04)
    sigma           : σ for the Gaussian averaging of the structure tensor
    window_size     : size of the Gaussian kernel  (must be odd, default 5)
    threshold_ratio : fraction of max(R) used as the detection threshold
    nms_window      : side length of the NMS neighbourhood window
    """

    def __init__(
        self,
        method: str           = "harris",
        k: float              = 0.04,
        sigma: float          = 1.0,
        window_size: int      = 5,
        threshold_ratio: float = 0.01,
        nms_window: int       = 7,
    ) -> None:
        if method not in ("harris", "lambda_minus"):
            raise ValueError("method must be 'harris' or 'lambda_minus'")
        self.method          = method
        self.k               = k
        self.sigma           = sigma
        self.window_size     = window_size
        self.threshold_ratio = threshold_ratio
        self.nms_window      = nms_window

    # ── public API ────────────────────────────────────────────
    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Detect corners in *image*.

        Pipeline
        --------
        1. Convert to grayscale float64.
        2. Compute Ix, Iy via Sobel.
        3. Build structure-tensor elements: Ixx, Iyy, Ixy.
        4. Gaussian-smooth each element.
        5. Compute corner response R.
        6. Threshold at threshold_ratio × max(R).
        7. Non-maximum suppression.
        8. Return keypoints + response map.
        """
        gray = to_grayscale(image)

        # Step 2 – Gradients
        Ix, Iy = _sobel_gradients(gray)

        # Step 3 – Structure tensor elements
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Step 4 – Gaussian smoothing (spatial averaging over a neighbourhood)
        g   = _make_gaussian_kernel(self.window_size, self.sigma)
        Sxx = _convolve2d(Ixx, g)
        Syy = _convolve2d(Iyy, g)
        Sxy = _convolve2d(Ixy, g)

        # Step 5 – Corner response
        det   = Sxx * Syy - Sxy ** 2
        trace = Sxx + Syy

        if self.method == "harris":
            R = det - self.k * (trace ** 2)
        else:                                      # lambda_minus
            half_diff = (Sxx - Syy) / 2.0
            R = (Sxx + Syy) / 2.0 - np.sqrt(half_diff ** 2 + Sxy ** 2)

        # Step 6 – Threshold
        thresh = self.threshold_ratio * float(R.max())
        R_pos  = np.where(R > thresh, R, 0.0)

        # Step 7 – NMS
        mask      = _nms_2d(R_pos, self.nms_window)
        rows, cols = np.where(mask)
        keypoints  = list(zip(rows.tolist(), cols.tolist()))

        return keypoints, R


# ═══════════════════════════════════════════════════════════════
#  SIFT DESCRIPTOR  (128-dim, from scratch)
# ═══════════════════════════════════════════════════════════════

class SIFTDescriptor(FeatureDescriptor):
    """
    Scale-Invariant Feature Transform descriptor — 128 dimensions.

    Implemented entirely from scratch (no OpenCV SIFT).
    Keypoints are supplied externally (typically from HarrisDetector).

    Descriptor pipeline per keypoint
    ─────────────────────────────────
    1. Extract a 16×16 gradient-magnitude / orientation patch.
    2. Build a 36-bin orientation histogram over the full patch;
       the dominant bin defines the *canonical orientation*.
    3. Subtract the canonical orientation from all patch orientations
       → rotation-invariant relative orientations.
    4. Divide the patch into a 4×4 grid of 4×4 sub-cells.
       For each sub-cell build a magnitude-weighted 8-bin orientation
       histogram → 16 histograms × 8 bins = 128-dim vector.
    5. L2-normalise → clip at 0.2 → re-normalise (standard SIFT).

    Parameters
    ----------
    max_keypoints : maximum number of keypoints to describe (the strongest
                    ones, ranked by gradient magnitude, are chosen).
    """

    PATCH_SIZE = 16   # total patch side
    NUM_CELLS  = 4    # number of cells per side  (4×4 grid)
    NUM_BINS   = 8    # orientation bins per cell
    DESC_DIM   = NUM_CELLS * NUM_CELLS * NUM_BINS   # = 128

    def __init__(self, max_keypoints: int = 300) -> None:
        self.max_keypoints = max_keypoints

    # ── public API ────────────────────────────────────────────
    def describe(
        self,
        image: np.ndarray,
        keypoints: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute 128-dim SIFT descriptors for the given *keypoints*.

        Returns
        -------
        valid_kps   : keypoints for which a descriptor was computed.
        descriptors : float64 ndarray of shape (N, 128).
        """
        gray     = to_grayscale(image)
        Ix, Iy   = _sobel_gradients(gray)
        mag      = np.sqrt(Ix ** 2 + Iy ** 2)
        ori      = np.arctan2(Iy, Ix)             # ∈ [−π, π]

        half    = self.PATCH_SIZE // 2
        cell_sz = self.PATCH_SIZE // self.NUM_CELLS   # = 4

        # Select strongest keypoints by gradient magnitude at their location
        strengths  = [float(mag[int(kp[0]), int(kp[1])]) for kp in keypoints]
        order      = np.argsort(strengths)[::-1]
        kps_sorted = [keypoints[i] for i in order][: self.max_keypoints]

        valid_kps:   List[Tuple[int, int]] = []
        descriptors: List[np.ndarray]      = []

        for kp in kps_sorted:
            r, c = int(kp[0]), int(kp[1])

            # Skip keypoints too close to the border
            if (r - half < 0 or r + half >= gray.shape[0]
                    or c - half < 0 or c + half >= gray.shape[1]):
                continue

            # Extract 16×16 patch
            pmag = mag[r - half: r + half, c - half: c + half]   # (16,16)
            pori = ori[r - half: r + half, c - half: c + half]   # (16,16)

            # ── Step 2: dominant orientation (36-bin histogram) ──
            bins36 = (
                ((pori.ravel() + np.pi) / (2.0 * np.pi)) * 36
            ).astype(int) % 36
            hist36  = np.bincount(bins36, weights=pmag.ravel(), minlength=36)
            dom_ori = (np.argmax(hist36) / 36.0) * 2.0 * np.pi - np.pi

            # ── Step 3: relative orientations ───────────────────
            rel_ori = pori - dom_ori

            # ── Step 4: 4×4 spatial grid of 8-bin histograms ────
            desc = np.zeros(self.DESC_DIM, dtype=np.float64)
            for ci in range(self.NUM_CELLS):
                for cj in range(self.NUM_CELLS):
                    rs, re = ci * cell_sz, (ci + 1) * cell_sz
                    cs, ce = cj * cell_sz, (cj + 1) * cell_sz

                    cell_mag = pmag[rs:re, cs:ce].ravel()       # 16 values
                    cell_ori = rel_ori[rs:re, cs:ce].ravel()

                    b = (
                        ((cell_ori + np.pi) / (2.0 * np.pi)) * self.NUM_BINS
                    ).astype(int) % self.NUM_BINS

                    start = (ci * self.NUM_CELLS + cj) * self.NUM_BINS
                    desc[start: start + self.NUM_BINS] = np.bincount(
                        b, weights=cell_mag, minlength=self.NUM_BINS
                    )

            # ── Step 5: L2-normalise → clip → re-normalise ──────
            n = np.linalg.norm(desc)
            if n > 1e-6:
                desc /= n
            np.clip(desc, 0.0, 0.2, out=desc)
            n2 = np.linalg.norm(desc)
            if n2 > 1e-6:
                desc /= n2

            valid_kps.append(kp)
            descriptors.append(desc)

        arr = (np.array(descriptors, dtype=np.float64)
               if descriptors else np.zeros((0, self.DESC_DIM), dtype=np.float64))
        return valid_kps, arr


# ═══════════════════════════════════════════════════════════════
#  SSD MATCHER
# ═══════════════════════════════════════════════════════════════

class SSDMatcher(FeatureMatcher):
    """
    Nearest-neighbour matcher using **Sum of Squared Differences** (SSD).

    For each descriptor *d1* in *desc1*:
    1. Compute SSD to every descriptor in *desc2* (vectorised NumPy).
    2. Find the two nearest neighbours (best, second_best).
    3. Apply **Lowe's ratio test**: accept iff
           SSD(best) / SSD(second_best) < ratio_threshold.
       This rejects ambiguous matches where two candidates are similarly
       close, producing fewer but more reliable correspondences.

    Parameters
    ----------
    ratio_threshold : Lowe ratio threshold (default 0.75). Lower = stricter.
    """

    def __init__(self, ratio_threshold: float = 0.75) -> None:
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """
        Match descriptors and return accepted pairs sorted by SSD score.

        Returns
        -------
        matches : list of (idx1, idx2, ssd_score), sorted ascending.
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        matches: List[Tuple[int, int, float]] = []

        for i, d1 in enumerate(desc1):
            # Vectorised SSD: broadcast subtraction over all of desc2
            diff = desc2 - d1                          # (N2, D)
            ssd  = (diff * diff).sum(axis=1)            # (N2,)

            if len(ssd) < 2:
                best = int(np.argmin(ssd))
                matches.append((i, best, float(ssd[best])))
                continue

            # Find indices of the two smallest values (partial sort)
            two_idx   = np.argpartition(ssd, 2)[:2]
            sorted2   = two_idx[np.argsort(ssd[two_idx])]
            best, sec = int(sorted2[0]), int(sorted2[1])

            # Lowe ratio test
            if ssd[sec] > 1e-10 and ssd[best] / ssd[sec] < self.ratio_threshold:
                matches.append((i, best, float(ssd[best])))

        matches.sort(key=lambda m: m[2])   # ascending: smaller SSD is better
        return matches


# ═══════════════════════════════════════════════════════════════
#  NCC MATCHER
# ═══════════════════════════════════════════════════════════════

class NCCMatcher(FeatureMatcher):
    """
    Nearest-neighbour matcher using **Normalised Cross-Correlation** (NCC).

    For descriptor vectors *d1* and *d2*:

        NCC(d1, d2) = (d1 − μ₁) · (d2 − μ₂)
                      ─────────────────────────
                      ‖d1 − μ₁‖ · ‖d2 − μ₂‖

    where μᵢ = mean(dᵢ).  NCC ∈ [−1, 1]; a value near +1 means
    the descriptors are very similar.

    The full NCC matrix between *desc1* and *desc2* is computed in a
    single vectorised pass, making this O(N1 · N2 · D) with no Python
    loops over descriptor pairs.

    Parameters
    ----------
    threshold : minimum NCC value to accept a match (default 0.7).
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    # ── Internal ──────────────────────────────────────────────
    @staticmethod
    def _ncc_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """
        Compute the full (N1 × N2) NCC matrix in one vectorised pass.

        Each entry ncc[i, j] = NCC(desc1[i], desc2[j]).
        """
        d1c   = desc1 - desc1.mean(axis=1, keepdims=True)   # (N1, D)
        d2c   = desc2 - desc2.mean(axis=1, keepdims=True)   # (N2, D)
        dot   = d1c @ d2c.T                                  # (N1, N2)
        n1    = np.linalg.norm(d1c, axis=1, keepdims=True)  # (N1,  1)
        n2    = np.linalg.norm(d2c, axis=1, keepdims=True)  # (N2,  1)
        denom = n1 * n2.T                                    # (N1, N2)
        return np.where(denom > 1e-10, dot / denom, 0.0)

    # ── public API ────────────────────────────────────────────
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """
        Match descriptors and return accepted pairs sorted by NCC score.

        Returns
        -------
        matches : list of (idx1, idx2, ncc_score), sorted descending.
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        ncc_mat = self._ncc_matrix(desc1, desc2)   # (N1, N2)

        matches: List[Tuple[int, int, float]] = []
        for i in range(len(desc1)):
            best  = int(np.argmax(ncc_mat[i]))
            score = float(ncc_mat[i, best])
            if score >= self.threshold:
                matches.append((i, best, score))

        matches.sort(key=lambda m: -m[2])   # descending: higher NCC is better
        return matches
