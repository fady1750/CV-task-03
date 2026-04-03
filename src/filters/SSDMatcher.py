from typing import List, Tuple

import numpy as np
from ..base  import FeatureMatcher



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