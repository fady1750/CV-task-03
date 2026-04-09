from typing import List, Tuple

import numpy as np
from ..base import FeatureDescriptor
from ..utils import to_grayscale
from .baseFilters import _sobel_gradients, _make_gaussian_kernel


class SIFTDescriptor(FeatureDescriptor):
    """
    Scale-Invariant Feature Transform descriptor — 128 dimensions.
    """
    
    PATCH_SIZE = 16  # total patch side
    NUM_CELLS = 4
    NUM_BINS = 8
    DESC_DIM = NUM_CELLS * NUM_CELLS * NUM_BINS

    def __init__(self, max_keypoints: int = 300) -> None:
        self.max_keypoints = max_keypoints

    def describe(
        self,
        image: np.ndarray,
        keypoints: List[Tuple[int, int, float]],
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        gray = to_grayscale(image)
        Ix, Iy = _sobel_gradients(gray)
        mag = np.sqrt(Ix ** 2 + Iy ** 2)
        ori = np.arctan2(Iy, Ix)

        half = self.PATCH_SIZE // 2
        cell_sz = self.PATCH_SIZE // self.NUM_CELLS

        kps_sorted = sorted(keypoints, key=lambda x: x[2], reverse=True)[:self.max_keypoints]

        valid_kps = []
        descriptors = []

        # Create Gaussian weighting for the patch
        # Create a 16x16 Gaussian window (not a kernel for convolution)
        y, x = np.mgrid[-half:half, -half:half]
        sigma = self.PATCH_SIZE / 2
        gaussian_window = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_window = gaussian_window / gaussian_window.max()  # normalize

        for kp in kps_sorted:
            r, c, strength = kp

            if (r - half < 0 or r + half >= gray.shape[0] or
                c - half < 0 or c + half >= gray.shape[1]):
                continue

            # Extract 16×16 patch
            pmag = mag[r - half: r + half, c - half: c + half]
            pori = ori[r - half: r + half, c - half: c + half]

            # Apply Gaussian weighting (both are 16x16 now)
            pmag = pmag * gaussian_window

            # ── Step 2: dominant orientation (36-bin histogram) ──
            bins36 = (
                ((pori.ravel() + np.pi) / (2.0 * np.pi)) * 36
            ).astype(int) % 36
            
            hist36 = np.bincount(bins36, weights=pmag.ravel(), minlength=36)
            
            # Smooth histogram
            hist36 = np.convolve(hist36, np.ones(3)/3, mode='same')
            
            dom_ori = (np.argmax(hist36) / 36.0) * 2.0 * np.pi - np.pi

            # ── Step 3: relative orientations ──
            rel_ori = pori - dom_ori

            # ── Step 4: 4×4 spatial grid of 8-bin histograms ──
            desc = np.zeros(self.DESC_DIM, dtype=np.float64)
            
            for ci in range(self.NUM_CELLS):
                for cj in range(self.NUM_CELLS):
                    rs, re = ci * cell_sz, (ci + 1) * cell_sz
                    cs, ce = cj * cell_sz, (cj + 1) * cell_sz

                    cell_mag = pmag[rs:re, cs:ce].ravel()
                    cell_ori = rel_ori[rs:re, cs:ce].ravel()

                    b = (
                        ((cell_ori + np.pi) / (2.0 * np.pi)) * self.NUM_BINS
                    ).astype(int) % self.NUM_BINS

                    start = (ci * self.NUM_CELLS + cj) * self.NUM_BINS
                    desc[start: start + self.NUM_BINS] = np.bincount(
                        b, weights=cell_mag, minlength=self.NUM_BINS
                    )

            # ── Step 5: L2-normalise → clip → re-normalise ──
            n = np.linalg.norm(desc)
            if n > 1e-6:
                desc /= n
            np.clip(desc, 0.0, 0.2, out=desc)
            n2 = np.linalg.norm(desc)
            if n2 > 1e-6:
                desc /= n2

            valid_kps.append((r, c))
            descriptors.append(desc)

        arr = (np.array(descriptors, dtype=np.float64)
               if descriptors else np.zeros((0, self.DESC_DIM), dtype=np.float64))
        return valid_kps, arr