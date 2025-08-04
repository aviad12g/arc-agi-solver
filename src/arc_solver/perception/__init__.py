"""Perception layer for ARC-AGI solver.

This module handles the conversion of raw pixel grids into structured
representations including blob detection, feature extraction, and
invariant computation.
"""

from .blob_labeling import BlobLabeler, create_blob_labeler
from .symmetry import (
    BitboardSymmetryDetector, SymmetryType, create_symmetry_detector, get_d4_group_elements
)
from .features import (
    OrbitSignatureComputer, SpectralFeatureComputer, PersistentHomologyComputer, 
    ZernikeMomentComputer, BlobFeatures, create_orbit_signature_computer, 
    create_spectral_feature_computer, create_persistence_computer, create_zernike_computer
)

__all__ = [
    'BlobLabeler',
    'create_blob_labeler',
    'BitboardSymmetryDetector',
    'SymmetryType',
    'create_symmetry_detector',
    'get_d4_group_elements',
    'OrbitSignatureComputer',
    'SpectralFeatureComputer',
    'PersistentHomologyComputer',
    'ZernikeMomentComputer',
    'BlobFeatures',
    'create_orbit_signature_computer',
    'create_spectral_feature_computer',
    'create_persistence_computer',
    'create_zernike_computer'
]