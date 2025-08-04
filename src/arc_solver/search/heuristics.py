"""Two-tier heuristic system for A* search.

Implements the mathematical heuristics described in the technical blueprint:
- Tier 1: L₂ spectral heuristic using 50D feature vectors
- Tier 2: Hungarian assignment for edit distance lower bound
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linear_sum_assignment = None

from arc_solver.core.data_models import Task, FeatureVector
from arc_solver.perception.blob_labeling import create_blob_labeler
from arc_solver.perception.features import (
    create_orbit_signature_computer, create_spectral_feature_computer,
    create_persistence_computer, create_zernike_computer, BlobFeatures
)
from arc_solver.reasoning.dsl_engine import DSLProgram

logger = logging.getLogger(__name__)


@dataclass
class HeuristicResult:
    """Result from heuristic computation."""
    value: float
    computation_time: float
    features_computed: bool = True
    error: Optional[str] = None


class BaseHeuristic(ABC):
    """Abstract base class for heuristics."""
    
    def __init__(self, name: str, max_computation_time: float = 0.001):
        """Initialize heuristic.
        
        Args:
            name: Name of the heuristic
            max_computation_time: Maximum computation time in seconds
        """
        self.name = name
        self.max_computation_time = max_computation_time
        self.computation_count = 0
        self.total_computation_time = 0.0
    
    @abstractmethod
    def compute(self, current_grid: np.ndarray, target_grid: np.ndarray,
                program: Optional[DSLProgram] = None) -> HeuristicResult:
        """Compute heuristic value.
        
        Args:
            current_grid: Current grid state
            target_grid: Target grid state
            program: Current program (optional)
            
        Returns:
            HeuristicResult with heuristic value
        """
        pass
    
    def __call__(self, current_grid: np.ndarray, target_grid: np.ndarray,
                 program: Optional[DSLProgram] = None) -> HeuristicResult:
        """Compute heuristic with timing and statistics."""
        start_time = time.perf_counter()
        
        try:
            result = self.compute(current_grid, target_grid, program)
            
            # Update statistics
            self.computation_count += 1
            self.total_computation_time += result.computation_time
            
            # Check computation time
            if result.computation_time > self.max_computation_time:
                logger.warning(f"{self.name} took {result.computation_time*1000:.2f}ms, "
                             f"exceeds {self.max_computation_time*1000:.2f}ms target")
            
            return result
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            logger.error(f"{self.name} computation failed: {e}")
            
            return HeuristicResult(
                value=float('inf'),  # Worst possible heuristic value
                computation_time=computation_time,
                features_computed=False,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        avg_time = (self.total_computation_time / self.computation_count 
                   if self.computation_count > 0 else 0.0)
        
        return {
            'name': self.name,
            'computation_count': self.computation_count,
            'total_time': self.total_computation_time,
            'average_time': avg_time,
            'average_time_us': avg_time * 1000000
        }


class Tier1Heuristic(BaseHeuristic):
    """Tier 1: L₂ spectral heuristic using 50D feature vectors.
    
    Implements h₁(G) = min_{ρ∈D₄} ||f̄_in - f̄_ρ(G)||₂ where f̄ is the mean
    feature vector across all blobs in the grid.
    """
    
    def __init__(self, max_computation_time: float = 0.0005):
        """Initialize Tier 1 heuristic.
        
        Args:
            max_computation_time: Maximum computation time (500µs target)
        """
        super().__init__("Tier1_L2_Spectral", max_computation_time)
        
        # Initialize feature computers
        self.blob_labeler = create_blob_labeler(use_gpu=False)
        self.orbit_computer = create_orbit_signature_computer()
        self.spectral_computer = create_spectral_feature_computer()
        self.persistence_computer = create_persistence_computer()
        self.zernike_computer = create_zernike_computer()
        
        # Cache for feature vectors
        self.feature_cache = {}
        
        logger.info("Tier 1 L₂ spectral heuristic initialized")
    
    def compute(self, current_grid: np.ndarray, target_grid: np.ndarray,
                program: Optional[DSLProgram] = None) -> HeuristicResult:
        """Compute L₂ distance between mean feature vectors.
        
        This implements the admissible heuristic h₁(G) = min_{ρ∈D₄} ||f̄_in - f̄_ρ(G)||₂
        """
        start_time = time.perf_counter()
        
        try:
            # Extract features from both grids
            target_features = self._extract_mean_features(target_grid)
            current_features = self._extract_mean_features(current_grid)
            
            # Compute L₂ distance
            # For now, we don't consider D₄ symmetries (min over rotations/reflections)
            # This could be added for better heuristic accuracy
            distance = np.linalg.norm(target_features - current_features)
            
            computation_time = time.perf_counter() - start_time
            
            return HeuristicResult(
                value=float(distance),
                computation_time=computation_time,
                features_computed=True
            )
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            raise RuntimeError(f"Tier 1 heuristic computation failed: {e}")
    
    def _extract_mean_features(self, grid: np.ndarray) -> np.ndarray:
        """Extract mean 50D feature vector from grid.
        
        Args:
            grid: Input grid
            
        Returns:
            50-dimensional mean feature vector
        """
        # Check cache first
        grid_hash = hash(grid.tobytes())
        if grid_hash in self.feature_cache:
            return self.feature_cache[grid_hash]
        
        # Extract blobs
        blobs, _ = self.blob_labeler.label_blobs(grid)
        
        if len(blobs) == 0:
            # No blobs - return zero feature vector
            mean_features = np.zeros(50, dtype=np.float32)
        else:
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
            
            # Combine into 50D feature vectors for each blob
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
                feature_vectors.append(feature_vector.to_array())
            
            # Compute mean feature vector
            if feature_vectors:
                mean_features = np.mean(feature_vectors, axis=0).astype(np.float32)
            else:
                mean_features = np.zeros(50, dtype=np.float32)
        
        # Cache result
        self.feature_cache[grid_hash] = mean_features
        
        return mean_features
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_cache.clear()
        logger.info("Tier 1 heuristic cache cleared")


class Tier2Heuristic(BaseHeuristic):
    """Tier 2: Hungarian assignment for edit distance lower bound.
    
    Provides a more accurate but computationally expensive heuristic by solving
    the assignment problem between blobs in current and target grids.
    """
    
    def __init__(self, max_computation_time: float = 0.002):
        """Initialize Tier 2 heuristic.
        
        Args:
            max_computation_time: Maximum computation time (2ms target)
        """
        super().__init__("Tier2_Hungarian", max_computation_time)
        
        self.blob_labeler = create_blob_labeler(use_gpu=False)
        self.scipy_available = SCIPY_AVAILABLE
        
        if not self.scipy_available:
            logger.warning("SciPy not available, using fallback for Tier 2 heuristic")
        
        logger.info(f"Tier 2 Hungarian heuristic initialized (scipy: {self.scipy_available})")
    
    def compute(self, current_grid: np.ndarray, target_grid: np.ndarray,
                program: Optional[DSLProgram] = None) -> HeuristicResult:
        """Compute Hungarian assignment lower bound.
        
        This provides a lower bound on the edit distance between current and target grids
        by solving the assignment problem between their blobs.
        """
        start_time = time.perf_counter()
        
        try:
            if self.scipy_available:
                heuristic_value = self._compute_hungarian_assignment(current_grid, target_grid)
            else:
                heuristic_value = self._compute_fallback_assignment(current_grid, target_grid)
            
            computation_time = time.perf_counter() - start_time
            
            return HeuristicResult(
                value=heuristic_value,
                computation_time=computation_time,
                features_computed=True
            )
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            raise RuntimeError(f"Tier 2 heuristic computation failed: {e}")
    
    def _compute_hungarian_assignment(self, current_grid: np.ndarray, 
                                    target_grid: np.ndarray) -> float:
        """Compute Hungarian assignment using SciPy."""
        # Extract blobs from both grids
        current_blobs, _ = self.blob_labeler.label_blobs(current_grid)
        target_blobs, _ = self.blob_labeler.label_blobs(target_grid)
        
        if len(current_blobs) == 0 and len(target_blobs) == 0:
            return 0.0  # Both grids empty
        
        if len(current_blobs) == 0 or len(target_blobs) == 0:
            return float(max(len(current_blobs), len(target_blobs)))  # One grid empty
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(current_blobs, target_blobs)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Compute total assignment cost
        total_cost = cost_matrix[row_indices, col_indices].sum()
        
        # Add penalty for unassigned blobs
        unassigned_penalty = abs(len(current_blobs) - len(target_blobs))
        
        return float(total_cost + unassigned_penalty)
    
    def _compute_fallback_assignment(self, current_grid: np.ndarray, 
                                   target_grid: np.ndarray) -> float:
        """Compute fallback assignment without SciPy."""
        # Extract blobs
        current_blobs, _ = self.blob_labeler.label_blobs(current_grid)
        target_blobs, _ = self.blob_labeler.label_blobs(target_grid)
        
        if len(current_blobs) == 0 and len(target_blobs) == 0:
            return 0.0
        
        # Simple greedy assignment (not optimal, but fast)
        total_cost = 0.0
        
        # Match blobs greedily by minimum cost
        used_targets = set()
        
        for current_blob in current_blobs:
            min_cost = float('inf')
            best_target = None
            
            for i, target_blob in enumerate(target_blobs):
                if i in used_targets:
                    continue
                
                cost = self._compute_blob_distance(current_blob, target_blob)
                if cost < min_cost:
                    min_cost = cost
                    best_target = i
            
            if best_target is not None:
                total_cost += min_cost
                used_targets.add(best_target)
            else:
                total_cost += 1.0  # Penalty for unmatched blob
        
        # Add penalty for unmatched target blobs
        unmatched_targets = len(target_blobs) - len(used_targets)
        total_cost += unmatched_targets
        
        return total_cost
    
    def _build_cost_matrix(self, current_blobs: List, target_blobs: List) -> np.ndarray:
        """Build cost matrix for Hungarian assignment."""
        n_current = len(current_blobs)
        n_target = len(target_blobs)
        
        # Make square matrix by padding with high costs
        max_size = max(n_current, n_target)
        cost_matrix = np.full((max_size, max_size), 10.0, dtype=np.float32)
        
        # Fill actual costs
        for i in range(n_current):
            for j in range(n_target):
                cost = self._compute_blob_distance(current_blobs[i], target_blobs[j])
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _compute_blob_distance(self, blob1, blob2) -> float:
        """Compute distance between two blobs."""
        # Simple distance based on area, color, and position
        area_diff = abs(blob1.area - blob2.area) / max(blob1.area, blob2.area, 1)
        color_diff = 1.0 if blob1.color != blob2.color else 0.0
        
        # Position difference
        pos_diff = np.sqrt(
            (blob1.center_of_mass[0] - blob2.center_of_mass[0])**2 +
            (blob1.center_of_mass[1] - blob2.center_of_mass[1])**2
        )
        
        return area_diff + color_diff + pos_diff * 0.1


class HeuristicSystem:
    """Combined two-tier heuristic system."""
    
    def __init__(self, use_tier2: bool = True, tier2_threshold: float = 5.0):
        """Initialize heuristic system.
        
        Args:
            use_tier2: Whether to use Tier 2 heuristic
            tier2_threshold: Threshold for switching to Tier 2
        """
        self.tier1 = Tier1Heuristic()
        self.tier2 = Tier2Heuristic() if use_tier2 else None
        self.tier2_threshold = tier2_threshold
        self.use_tier2 = use_tier2
        
        # Statistics
        self.tier1_calls = 0
        self.tier2_calls = 0
        
        logger.info(f"Heuristic system initialized (tier2: {use_tier2})")
    
    def compute_heuristic(self, current_grid: np.ndarray, target_grid: np.ndarray,
                         program: Optional[DSLProgram] = None) -> HeuristicResult:
        """Compute heuristic using two-tier system.
        
        Uses Tier 1 first, then Tier 2 if the Tier 1 value is below threshold.
        """
        # Always compute Tier 1
        tier1_result = self.tier1(current_grid, target_grid, program)
        self.tier1_calls += 1
        
        # Use Tier 2 if enabled and Tier 1 value is low (indicating we're close)
        if (self.use_tier2 and self.tier2 and 
            tier1_result.value < self.tier2_threshold and
            tier1_result.features_computed):
            
            tier2_result = self.tier2(current_grid, target_grid, program)
            self.tier2_calls += 1
            
            # Return the more accurate Tier 2 result
            return HeuristicResult(
                value=tier2_result.value,
                computation_time=tier1_result.computation_time + tier2_result.computation_time,
                features_computed=tier2_result.features_computed,
                error=tier2_result.error
            )
        
        return tier1_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'tier1_calls': self.tier1_calls,
            'tier2_calls': self.tier2_calls,
            'tier2_usage_rate': (self.tier2_calls / max(self.tier1_calls, 1)),
            'tier1_stats': self.tier1.get_stats()
        }
        
        if self.tier2:
            stats['tier2_stats'] = self.tier2.get_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches."""
        self.tier1.clear_cache()
        logger.info("Heuristic system caches cleared")


def create_heuristic_system(use_tier2: bool = True, 
                           tier2_threshold: float = 5.0) -> HeuristicSystem:
    """Factory function to create heuristic system.
    
    Args:
        use_tier2: Whether to enable Tier 2 heuristic
        tier2_threshold: Threshold for Tier 2 activation
        
    Returns:
        Configured HeuristicSystem instance
    """
    return HeuristicSystem(use_tier2=use_tier2, tier2_threshold=tier2_threshold)