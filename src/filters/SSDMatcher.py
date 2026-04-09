from typing import List, Tuple
import numpy as np
from ..base import FeatureMatcher


class SSDMatcher(FeatureMatcher):

    def __init__(self, ratio_threshold: float = 0.75) -> None:
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[Tuple[int, int, float]]:

        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # ── Forward pass: for each point in desc1, find best match in desc2 ──
        forward = {}
        for i, d1 in enumerate(desc1):
            diff = desc2 - d1                           # (N2, D)
            ssd  = (diff * diff).sum(axis=1)            # (N2,)

            if len(ssd) < 2:
                # Only one candidate — accept it directly, no ratio test possible
                best = int(np.argmin(ssd))
                forward[i] = (best, float(ssd[best]))
                continue

            # Get indices of the two smallest SSD values
            two_idx = np.argpartition(ssd, 2)[:2]
            sorted2 = two_idx[np.argsort(ssd[two_idx])]
            best, sec = int(sorted2[0]), int(sorted2[1])

            # Lowe's ratio test: best match must win by a clear margin
            if ssd[sec] > 1e-10 and ssd[best] / ssd[sec] < self.ratio_threshold:
                forward[i] = (best, float(ssd[best]))

        # ── Backward pass: for each point in desc2, find best match in desc1 ──
        backward = {}
        for j, d2 in enumerate(desc2):
            diff = desc1 - d2
            ssd  = (diff * diff).sum(axis=1)

            if len(ssd) < 2:
                best = int(np.argmin(ssd))
                backward[j] = best
                continue

            two_idx = np.argpartition(ssd, 2)[:2]
            sorted2 = two_idx[np.argsort(ssd[two_idx])]
            best, sec = int(sorted2[0]), int(sorted2[1])

            if ssd[sec] > 1e-10 and ssd[best] / ssd[sec] < self.ratio_threshold:
                backward[j] = best

        # ── Cross-check: only keep matches that are mutual in both directions ──
        matches: List[Tuple[int, int, float]] = []
        for i, (j, score) in forward.items():
            if backward.get(j) == i:        # mutual match confirmed
                matches.append((i, j, score))

        matches.sort(key=lambda m: m[2])    # ascending: smaller SSD is better
        return matches