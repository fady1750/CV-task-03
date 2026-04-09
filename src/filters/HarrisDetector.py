from typing import List, Tuple
import numpy as np

from .baseFilters import _convolve2d, _make_gaussian_kernel, _nms_2d, _sobel_gradients
from ..base import FeatureDetector
from ..utils import to_grayscale


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
        method: str = "harris",
        k: float = 0.04,
        sigma: float = 1.0,
        window_size: int = 5,
        threshold_ratio: float = 0.01,
        nms_window: int = 7,
    ) -> None:
        if method not in ("harris", "lambda_minus"):
            raise ValueError("method must be 'harris' or 'lambda_minus'")
        self.method = method
        self.k = k
        self.sigma = sigma
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio
        self.nms_window = nms_window

    # public API 
    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
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

        Returns
        -------
        keypoints : List of (row, col, response) - response is R value
        response_map : Full R matrix (for visualisation)
        """
        gray = to_grayscale(image)

        # Step 2 – Gradients
        Ix, Iy = _sobel_gradients(gray)

        # Step 3 – Structure tensor elements
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Step 4 – Gaussian smoothing (spatial averaging over a neighbourhood)
        g = _make_gaussian_kernel(self.window_size, self.sigma)
        Sxx = _convolve2d(Ixx, g)
        Syy = _convolve2d(Iyy, g)
        Sxy = _convolve2d(Ixy, g)

        # Step 5 – Corner response
        det = Sxx * Syy - Sxy ** 2
        trace = Sxx + Syy

        if self.method == "harris":
            R = det - self.k * (trace ** 2)
        else:  # lambda_minus
            half_diff = (Sxx - Syy) / 2.0
            R = (Sxx + Syy) / 2.0 - np.sqrt(half_diff ** 2 + Sxy ** 2)

        # Step 6 – Threshold
        thresh = self.threshold_ratio * float(R.max())
        R_pos = np.where(R > thresh, R, 0.0)

        # Step 7 – NMS
        mask = _nms_2d(R_pos, self.nms_window)
        rows, cols = np.where(mask)

        # Include the response strength for each keypoint
        keypoints = [(int(r), int(c), float(R_pos[r, c])) for r, c in zip(rows, cols)]

        return keypoints, R