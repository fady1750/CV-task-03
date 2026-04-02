from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class FeatureDetector(ABC):
    """
    Abstract base class for all feature / corner detectors.

    Subclasses must implement :meth:`detect`.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Detect interest points in *image*.

        Parameters
        ----------
        image : np.ndarray
            Input image – grayscale (H, W) or colour (H, W, 3).

        Returns
        -------
        keypoints : list of (row, col) integer tuples
        response  : np.ndarray of shape (H, W)
            The raw scalar response map (e.g. Harris R or λ-).
        """


class FeatureDescriptor(ABC):
    """
    Abstract base class for local feature descriptors.

    Subclasses must implement :meth:`describe`.
    """

    @abstractmethod
    def describe(
        self,
        image: np.ndarray,
        keypoints: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute a descriptor vector for each keypoint.

        Parameters
        ----------
        image     : np.ndarray  – input image (grayscale or colour).
        keypoints : list of (row, col) tuples.

        Returns
        -------
        valid_keypoints : list of (row, col) tuples
            Subset of *keypoints* for which a descriptor could be computed
            (border keypoints are typically dropped).
        descriptors : np.ndarray of shape (N, D)
            One descriptor row per valid keypoint.
        """


class FeatureMatcher(ABC):
    """
    Abstract base class for feature matchers.

    Subclasses must implement :meth:`match`.
    """

    @abstractmethod
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """
        Match descriptors from two images.

        Parameters
        ----------
        desc1 : np.ndarray  shape (N1, D)
        desc2 : np.ndarray  shape (N2, D)

        Returns
        -------
        matches : list of (idx1, idx2, score)
            Sorted by score (ascending for distance-based, descending for
            similarity-based).
        """
