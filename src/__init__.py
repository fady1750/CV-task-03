"""
src – Computer Vision Feature Detection & Matching
===================================================

Public API
----------
Detectors
    HarrisDetector   – Harris (R = det − k·tr²) and λ₋ (min eigenvalue)

Descriptors
    SIFTDescriptor   – 128-dim SIFT descriptor (pure NumPy, from scratch)

Matchers
    SSDMatcher       – Sum of Squared Differences + Lowe ratio test
    NCCMatcher       – Normalised Cross-Correlation

Utilities
    to_grayscale, normalize_image, to_uint8
    fig_keypoints, fig_response_map, fig_matches, fig_descriptor_heatmaps
    create_sample_images
"""

from .base import FeatureDetector, FeatureDescriptor, FeatureMatcher
from .filters import HarrisDetector, SIFTDescriptor, SSDMatcher, NCCMatcher
from .utils import (
    to_grayscale,
    normalize_image,
    to_uint8,
    fig_keypoints,
    fig_response_map,
    fig_matches,
    fig_descriptor_heatmaps,
    create_sample_images,
)

__all__ = [
    # Abstract bases
    "FeatureDetector",
    "FeatureDescriptor",
    "FeatureMatcher",
    # Concrete implementations
    "HarrisDetector",
    "SIFTDescriptor",
    "SSDMatcher",
    "NCCMatcher",
    # Utilities
    "to_grayscale",
    "normalize_image",
    "to_uint8",
    "fig_keypoints",
    "fig_response_map",
    "fig_matches",
    "fig_descriptor_heatmaps",
    "create_sample_images",
]
