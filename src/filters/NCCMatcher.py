from typing import List, Tuple
import numpy as np
from ..base import FeatureMatcher


class NCCMatcher(FeatureMatcher):

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    @staticmethod
    def _ncc_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        # Subtract mean from each descriptor to remove lighting bias
        d1c = desc1 - desc1.mean(axis=1, keepdims=True)    # (N1, D)
        d2c = desc2 - desc2.mean(axis=1, keepdims=True)    # (N2, D)

        # Dot product between all pairs at once — shape (N1, N2)
        dot = d1c @ d2c.T

        # Compute L2 norm for each descriptor
        n1 = np.linalg.norm(d1c, axis=1, keepdims=True)   # (N1, 1)
        n2 = np.linalg.norm(d2c, axis=1, keepdims=True)   # (N2, 1)

        # Denominator: product of norms for every pair
        denom = n1 * n2.T                                  # (N1, N2)

        # Avoid division by zero — return 0 when norm is too small
        return np.where(denom > 1e-10, dot / denom, 0.0)

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[Tuple[int, int, float]]:

        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Compute full NCC matrix — shape (N1, N2)
        ncc_mat = self._ncc_matrix(desc1, desc2)

        # Forward: best match in desc2 for each point in desc1
        forward  = np.argmax(ncc_mat, axis=1)   # (N1,)

        # Backward: best match in desc1 for each point in desc2
        backward = np.argmax(ncc_mat, axis=0)   # (N2,)

        matches: List[Tuple[int, int, float]] = []
        for i in range(len(desc1)):
            j     = int(forward[i])
            score = float(ncc_mat[i, j])

            # Accept only if match is mutual AND score is above threshold
            if backward[j] == i and score >= self.threshold:
                matches.append((i, j, score))

        matches.sort(key=lambda m: -m[2])   # descending: higher NCC is better
        return matches