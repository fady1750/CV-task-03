from typing import List, Tuple

import numpy as np
from ..base  import FeatureDescriptor
from ..utils import to_grayscale
from .baseFilters import  _sobel_gradients



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