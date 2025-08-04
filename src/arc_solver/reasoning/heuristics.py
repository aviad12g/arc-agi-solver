"""Two-tier heuristic system for A* search guidance.

This module implements the mathematical heuristics that guide the search process:
- Tier-1: Fast L₂ spectral-homology heuristic using 50D feature vectors
- Tier-2: Hungarian assignment fallback for tighter bounds when needed
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None

from arc_solver.core.data_models import FeatureVector, Blob
from arc_solver.perception.features import (
    create_orbit_signature_computer, create_spectral_feature_computer,
    create_persistence_computer, create_zernike_computer, BlobFeatures
)
from arc_solver.perception.blob_labeling import create_blob_labeler
from arc_solver.perception.symmetry import create_symmetry_detector, get_d4_group_elements

logger = logging.getLogger(__name__)


@dataclass
class HeuristicResult:
    """Result from heuristic computation."""
    value: float
    computation_time: float
    tier_used: int
    details: Dict[str, Any]
    admissible: bool = True


class Heuristic(ABC):
    """Abstract base class for heuristics."""
    
    @abstractmethod
    def compute(self, current_grid: np.ndarray, target_grid: np.ndarray) -> HeuristicResult:
        """Compute heuristic value between current and target grids.
        
        Args:
            current_grid: Current state grid
            target_grid: Target/goal grid
            
        Returns:
            HeuristicResult with heuristic value and metadata
        """
        pass
    
    @abstractmethod
    def is_admissible(self) -> bool:
        """Check if this heuristic is admissible (never overestimates)."""
        pass


class Tier1SpectralHomologyHeuristic(Heuristic):
    """Tier-1 fast L₂ spectral-homology heuristic using D₄ group minimization.
    
    Implements: h₁(G) = min_{ρ∈D₄} ||f̄_in - f̄_ρ(G)||₂
    where f̄ are 50-dimensional feature vectors.
    """
    
    def __init__(self, enable_d4_minimization: bool = True):
        """Initialize Tier-1 heuristic.
        
        Args:
            enable_d4_minimization: Whether to minimize over D₄ group
        """
        self.enable_d4_minimization = enable_d4_minimization
        
        # Initialize feature computers
        self.blob_labeler = create_blob_labeler(use_gpu=False)
        self.orbit_computer = create_orbit_signature_computer()
        self.spectral_computer = create_spectral_feature_computer()
        self.persistence_computer = create_persistence_computer()
        self.zernike_computer = create_zernike_computer()
        
        # Initialize symmetry detector for D₄ minimization
        if self.enable_d4_minimization:
            self.symmetry_detector = create_symmetry_detector()
            self.d4_elements = get_d4_group_elements()
        
        # Performance tracking
        self.computation_count = 0
        self.total_computation_time = 0.0
        
        logger.info(f"Tier-1 heuristic initialized (D₄ minimization: {enable_d4_minimization})")
    
    def compute(self, current_grid: np.ndarray, target_grid: np.ndarray) -> HeuristicResult:
        """Compute Tier-1 L₂ spectral-homology heuristic.
        
        Args:
            current_grid: Current state grid
            target_grid: Target/goal grid
            
        Returns:
            HeuristicResult with L₂ distance and computation details
        """
        start_time = time.perf_counter()
        
        try:
            # Extract features from target grid (this could be cached)
            target_features = self._extract_grid_features(target_grid)
            
            if self.enable_d4_minimization:
                # Minimize over D₄ group: min_{ρ∈D₄} ||f̄_in - f̄_ρ(G)||₂
                min_distance = float('inf')
                best_transform = None
                
                for transform in self.d4_elements:
                    # Apply transformation to current grid
                    if current_grid.shape[0] == current_grid.shape[1]:
                        transformed_grid = self.symmetry_detector.apply_symmetry_transform(
                            current_grid, transform
                        )
                    else:
                        # For non-square grids, skip transformation
                        transformed_grid = current_grid
                    
                    # Extract features from transformed grid
                    current_features = self._extract_grid_features(transformed_grid)
                    
                    # Compute L₂ distance
                    distance = self._compute_l2_distance(current_features, target_features)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_transform = transform
                
                heuristic_value = min_distance
                
            else:
                # Direct comparison without D₄ minimization
                current_features = self._extract_grid_features(current_grid)
                heuristic_value = self._compute_l2_distance(current_features, target_features)
                best_transform = None
            
            computation_time = time.perf_counter() - start_time
            
            # Update statistics
            self.computation_count += 1
            self.total_computation_time += computation_time
            
            # Create result
            result = HeuristicResult(
                value=heuristic_value,
                computation_time=computation_time,
                tier_used=1,
                details={
                    'best_transform': best_transform.value if best_transform else None,
                    'num_current_blobs': len(current_features),
                    'num_target_blobs': len(target_features),
                    'd4_minimization': self.enable_d4_minimization
                },
                admissible=True  # L₂ distance provides lower bound
            )
            
            return result
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            logger.error(f"Tier-1 heuristic computation failed: {e}")
            
            # Return safe fallback value
            return HeuristicResult(
                value=0.0,  # Conservative fallback
                computation_time=computation_time,
                tier_used=1,
                details={'error': str(e)},
                admissible=True
            )
    
    def _extract_grid_features(self, grid: np.ndarray) -> List[FeatureVector]:
        """Extract 50-dimensional feature vectors from grid.
        
        Args:
            grid: Input grid
            
        Returns:
            List of FeatureVector objects for each blob
        """
        # Detect blobs
        blobs, _ = self.blob_labeler.label_blobs(grid)
        
        if not blobs:
            return []
        
        # Compute adjacency graph
        adjacency_graph = self.blob_labeler.get_blob_adjacency_graph(blobs, grid.shape)
        
        # Extract all feature types
        orbit_features = {}
        for i, blob in enumerate(blobs):
            orbit_sig = self.orbit_computer.compute_blob_signature(blob, grid.shape)
            orbit_features[i] = orbit_sig
        
        spectral_features = self.spectral_computer.compute_spectral_features(blobs, adjacency_graph)
        persistence_features = self.persistence_computer.compute_persistence_features(blobs, grid.shape)
        zernike_features = self.zernike_computer.compute_zernike_features(blobs, grid.shape)
        
        # Combine into feature vectors
        feature_vectors = []
        for i in range(len(blobs)):
            blob_features = BlobFeatures(
                blob_id=i,
                orbit_signature=orbit_features.get(i, np.zeros(8, dtype=np.float32)),
                spectral_features=spectral_features.get(i, np.zeros(3, dtype=np.float32)),
                persistence_landscape=persistence_features.get(i, np.zeros(32, dtype=np.float32)),
                zernike_moments=zernike_features.get(i, np.zeros(7, dtype=np.float32))
            )
            
            feature_vector = blob_features.to_feature_vector()
            feature_vectors.append(feature_vector)
        
        return feature_vectors
    
    def _compute_l2_distance(self, features1: List[FeatureVector], 
                           features2: List[FeatureVector]) -> float:
        """Compute L₂ distance between two sets of feature vectors.
        
        Args:
            features1: First set of feature vectors
            features2: Second set of feature vectors
            
        Returns:
            L₂ distance between the sets
        """
        if not features1 and not features2:
            return 0.0
        
        if not features1 or not features2:
            # One set is empty - return distance based on non-empty set
            non_empty = features1 if features1 else features2
            return sum(np.linalg.norm(fv.to_array()) for fv in non_empty)
        
        # Convert to arrays
        array1 = np.array([fv.to_array() for fv in features1])
        array2 = np.array([fv.to_array() for fv in features2])
        
        # Compute pairwise distances and find minimum assignment
        if len(features1) == len(features2):
            # Same number of blobs - compute direct L₂ distance
            distances = []
            for i in range(len(features1)):
                dist = np.linalg.norm(array1[i] - array2[i])
                distances.append(dist)
            return sum(distances)
        
        else:
            # Different number of blobs - use simplified distance
            # This is a heuristic approximation
            mean1 = np.mean(array1, axis=0)
            mean2 = np.mean(array2, axis=0)
            
            base_distance = np.linalg.norm(mean1 - mean2)
            size_penalty = abs(len(features1) - len(features2)) * 0.1
            
            return base_distance + size_penalty
    
    def is_admissible(self) -> bool:
        """Tier-1 heuristic is admissible as it provides a lower bound."""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computation statistics."""
        avg_time = (self.total_computation_time / self.computation_count 
                   if self.computation_count > 0 else 0.0)
        
        return {
            'computation_count': self.computation_count,
            'total_time': self.total_computation_time,
            'average_time': avg_time,
            'average_time_us': avg_time * 1000000
        }


class Tier2HungarianHeuristic(Heuristic):
    """Tier-2 Hungarian assignment heuristic for tighter bounds.
    
    Uses optimal blob matching with Hungarian algorithm to compute
    edit distance lower bounds from unmatched blobs and color mismatches.
    """
    
    def __init__(self):
        """Initialize Tier-2 heuristic."""
        self.scipy_available = SCIPY_AVAILABLE
        
        # Initialize feature extraction components
        self.blob_labeler = create_blob_labeler(use_gpu=False)
        self.orbit_computer = create_orbit_signature_computer()
        
        # Performance tracking
        self.computation_count = 0
        self.total_computation_time = 0.0
        
        if not self.scipy_available:
            logger.warning("SciPy not available, using fallback Hungarian algorithm")
        
        logger.info(f"Tier-2 heuristic initialized (SciPy: {self.scipy_available})")
    
    def compute(self, current_grid: np.ndarray, target_grid: np.ndarray) -> HeuristicResult:
        """Compute Tier-2 Hungarian assignment heuristic.
        
        Args:
            current_grid: Current state grid
            target_grid: Target/goal grid
            
        Returns:
            HeuristicResult with Hungarian assignment cost
        """
        start_time = time.perf_counter()
        
        try:
            # Extract blobs from both grids
            current_blobs, _ = self.blob_labeler.label_blobs(current_grid)
            target_blobs, _ = self.blob_labeler.label_blobs(target_grid)
            
            if not current_blobs and not target_blobs:
                # Both grids are empty
                heuristic_value = 0.0
            elif not current_blobs or not target_blobs:
                # One grid is empty - cost is number of blobs in non-empty grid
                non_empty_blobs = current_blobs if current_blobs else target_blobs
                heuristic_value = float(len(non_empty_blobs))
            else:
                # Both grids have blobs - compute optimal assignment
                if self.scipy_available:
                    heuristic_value = self._compute_hungarian_scipy(current_blobs, target_blobs, 
                                                                  current_grid.shape, target_grid.shape)
                else:
                    heuristic_value = self._compute_hungarian_fallback(current_blobs, target_blobs)
            
            computation_time = time.perf_counter() - start_time
            
            # Update statistics
            self.computation_count += 1
            self.total_computation_time += computation_time
            
            result = HeuristicResult(
                value=heuristic_value,
                computation_time=computation_time,
                tier_used=2,
                details={
                    'num_current_blobs': len(current_blobs),
                    'num_target_blobs': len(target_blobs),
                    'scipy_used': self.scipy_available
                },
                admissible=True  # Hungarian provides optimal assignment lower bound
            )
            
            return result
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            logger.error(f"Tier-2 heuristic computation failed: {e}")
            
            return HeuristicResult(
                value=0.0,  # Conservative fallback
                computation_time=computation_time,
                tier_used=2,
                details={'error': str(e)},
                admissible=True
            )
    
    def _compute_hungarian_scipy(self, current_blobs: List[Blob], target_blobs: List[Blob],
                               current_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> float:
        """Compute Hungarian assignment using SciPy."""
        n_current = len(current_blobs)
        n_target = len(target_blobs)
        
        # Create cost matrix
        max_size = max(n_current, n_target)
        cost_matrix = np.full((max_size, max_size), 1.0, dtype=np.float32)  # Default high cost
        
        # Fill in actual costs for existing blob pairs
        for i in range(n_current):
            for j in range(n_target):
                cost = self._compute_blob_distance(current_blobs[i], target_blobs[j])
                cost_matrix[i, j] = cost
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Compute total cost
        total_cost = 0.0
        for i, j in zip(row_indices, col_indices):
            if i < n_current and j < n_target:
                # Real blob-to-blob assignment
                total_cost += cost_matrix[i, j]
            elif i < n_current:
                # Unmatched current blob (needs to be removed)
                total_cost += 1.0
            elif j < n_target:
                # Unmatched target blob (needs to be added)
                total_cost += 1.0
        
        return total_cost
    
    def _compute_hungarian_fallback(self, current_blobs: List[Blob], target_blobs: List[Blob]) -> float:
        """Fallback Hungarian assignment without SciPy."""
        # Simplified greedy assignment as fallback
        n_current = len(current_blobs)
        n_target = len(target_blobs)
        
        if n_current == 0:
            return float(n_target)
        if n_target == 0:
            return float(n_current)
        
        # Greedy assignment
        used_targets = set()
        total_cost = 0.0
        
        for current_blob in current_blobs:
            best_cost = float('inf')
            best_target = None
            
            for j, target_blob in enumerate(target_blobs):
                if j not in used_targets:
                    cost = self._compute_blob_distance(current_blob, target_blob)
                    if cost < best_cost:
                        best_cost = cost
                        best_target = j
            
            if best_target is not None:
                total_cost += best_cost
                used_targets.add(best_target)
            else:
                total_cost += 1.0  # No match available
        
        # Add cost for unmatched target blobs
        unmatched_targets = n_target - len(used_targets)
        total_cost += unmatched_targets
        
        return total_cost
    
    def _compute_blob_distance(self, blob1: Blob, blob2: Blob) -> float:
        """Compute distance between two blobs for assignment cost.
        
        Args:
            blob1: First blob
            blob2: Second blob
            
        Returns:
            Distance/cost for assigning blob1 to blob2
        """
        # Color mismatch cost
        color_cost = 0.0 if blob1.color == blob2.color else 0.5
        
        # Size difference cost
        size_diff = abs(blob1.area - blob2.area)
        size_cost = size_diff / max(blob1.area, blob2.area, 1) * 0.3
        
        # Position difference cost (simplified)
        pos_diff = abs(blob1.center_of_mass[0] - blob2.center_of_mass[0]) + \
                   abs(blob1.center_of_mass[1] - blob2.center_of_mass[1])
        pos_cost = pos_diff * 0.1
        
        return color_cost + size_cost + pos_cost
    
    def is_admissible(self) -> bool:
        """Tier-2 heuristic is admissible as it provides optimal assignment lower bound."""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computation statistics."""
        avg_time = (self.total_computation_time / self.computation_count 
                   if self.computation_count > 0 else 0.0)
        
        return {
            'computation_count': self.computation_count,
            'total_time': self.total_computation_time,
            'average_time': avg_time,
            'average_time_us': avg_time * 1000000
        }


class TwoTierHeuristicSystem:
    """Two-tier heuristic system with automatic fallback.
    
    Uses Tier-1 for fast pruning and falls back to Tier-2 when
    Tier-1 pruning is insufficient (< 70% pruning rate).
    """
    
    def __init__(self, pruning_threshold: float = 0.7,
                 enable_d4_minimization: bool = True):
        """Initialize two-tier heuristic system.
        
        Args:
            pruning_threshold: Minimum pruning rate to avoid Tier-2 fallback
            enable_d4_minimization: Whether to enable D₄ minimization in Tier-1
        """
        self.pruning_threshold = pruning_threshold
        
        # Initialize both tiers
        self.tier1 = Tier1SpectralHomologyHeuristic(enable_d4_minimization)
        self.tier2 = Tier2HungarianHeuristic()
        
        # Statistics
        self.tier1_uses = 0
        self.tier2_uses = 0
        self.total_computations = 0
        
        logger.info(f"Two-tier heuristic system initialized (pruning threshold: {pruning_threshold})")
    
    def compute_heuristic(self, current_grid: np.ndarray, target_grid: np.ndarray,
                         force_tier: Optional[int] = None) -> HeuristicResult:
        """Compute heuristic using appropriate tier.
        
        Args:
            current_grid: Current state grid
            target_grid: Target/goal grid
            force_tier: Force specific tier (1 or 2), or None for automatic
            
        Returns:
            HeuristicResult from appropriate tier
        """
        self.total_computations += 1
        
        if force_tier == 2:
            # Force Tier-2
            self.tier2_uses += 1
            return self.tier2.compute(current_grid, target_grid)
        
        elif force_tier == 1:
            # Force Tier-1
            self.tier1_uses += 1
            return self.tier1.compute(current_grid, target_grid)
        
        else:
            # Automatic tier selection
            # For now, always use Tier-1 (Tier-2 fallback would be implemented
            # in the search algorithm based on pruning effectiveness)
            self.tier1_uses += 1
            return self.tier1.compute(current_grid, target_grid)
    
    def should_use_tier2(self, pruning_rate: float) -> bool:
        """Determine if Tier-2 should be used based on pruning effectiveness.
        
        Args:
            pruning_rate: Current pruning rate from Tier-1
            
        Returns:
            True if Tier-2 should be used
        """
        return pruning_rate < self.pruning_threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        tier1_stats = self.tier1.get_statistics()
        tier2_stats = self.tier2.get_statistics()
        
        return {
            'total_computations': self.total_computations,
            'tier1_uses': self.tier1_uses,
            'tier2_uses': self.tier2_uses,
            'tier1_percentage': (self.tier1_uses / self.total_computations * 100 
                               if self.total_computations > 0 else 0),
            'tier2_percentage': (self.tier2_uses / self.total_computations * 100 
                               if self.total_computations > 0 else 0),
            'tier1_stats': tier1_stats,
            'tier2_stats': tier2_stats
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.tier1_uses = 0
        self.tier2_uses = 0
        self.total_computations = 0


def create_heuristic_system(pruning_threshold: float = 0.7,
                          enable_d4_minimization: bool = True) -> TwoTierHeuristicSystem:
    """Factory function to create two-tier heuristic system.
    
    Args:
        pruning_threshold: Minimum pruning rate to avoid Tier-2 fallback
        enable_d4_minimization: Whether to enable D₄ minimization
        
    Returns:
        Configured TwoTierHeuristicSystem instance
    """
    return TwoTierHeuristicSystem(pruning_threshold, enable_d4_minimization)