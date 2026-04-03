from typing import List, Tuple

import numpy as np
from ..base  import FeatureMatcher



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