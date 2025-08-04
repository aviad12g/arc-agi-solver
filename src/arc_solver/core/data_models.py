"""Core data models for the ARC solver."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import hashlib


@dataclass
class FeatureVector:
    """50-dimensional feature vector for mathematical invariants of a blob."""
    
    orbit_signature: np.ndarray  # D4 symmetry fingerprint (8 dims)
    spectral_features: np.ndarray  # Laplacian eigenvalues λ₂, λ₃, λ₄ (3 dims)
    persistence_landscape: np.ndarray  # Topological features (32 dims)
    zernike_moments: np.ndarray  # Shape descriptors up to order 4 (7 dims)
    
    def __post_init__(self) -> None:
        """Validate feature vector dimensions."""
        assert self.orbit_signature.shape == (8,), f"Expected orbit_signature shape (8,), got {self.orbit_signature.shape}"
        assert self.spectral_features.shape == (3,), f"Expected spectral_features shape (3,), got {self.spectral_features.shape}"
        assert self.persistence_landscape.shape == (32,), f"Expected persistence_landscape shape (32,), got {self.persistence_landscape.shape}"
        assert self.zernike_moments.shape == (7,), f"Expected zernike_moments shape (7,), got {self.zernike_moments.shape}"
    
    def to_array(self) -> np.ndarray:
        """Concatenate all features into a 50-dimensional vector."""
        return np.concatenate([
            self.orbit_signature,
            self.spectral_features,
            self.persistence_landscape,
            self.zernike_moments
        ]).astype(np.float32)
    
    def __hash__(self) -> int:
        """Generate hash for caching based on rounded feature values."""
        rounded = np.round(self.to_array(), decimals=6)
        return int(hashlib.sha1(rounded.tobytes()).hexdigest()[:8], 16)


@dataclass
class Blob:
    """A connected component in the grid with its properties and features."""
    
    id: int
    color: int
    pixels: List[Tuple[int, int]]  # List of (row, col) coordinates
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    center_of_mass: Tuple[float, float]  # (row, col) center of mass
    area: int
    holes: int  # Number of topological holes
    features: Optional[FeatureVector] = None
    
    def __post_init__(self) -> None:
        """Validate blob properties."""
        assert self.area == len(self.pixels), f"Area {self.area} doesn't match pixel count {len(self.pixels)}"
        assert self.area > 0, "Blob must have positive area"
        assert self.holes >= 0, "Number of holes cannot be negative"


@dataclass
class GridState:
    """State representation for search algorithms."""
    
    grid: np.ndarray  # The current grid state
    blobs: List[Blob]  # Extracted blobs from the grid
    program: List[str]  # DSL program sequence that led to this state
    cost: float  # Program length (uniform cost)
    heuristic: float  # Heuristic estimate to goal
    
    def total_cost(self) -> float:
        """Total f-score for A* search."""
        return self.cost + self.heuristic
    
    def canonical_hash(self) -> str:
        """Generate canonical hash for caching."""
        # Normalize grid for consistent hashing
        normalized_grid = self._normalize_grid()
        return hashlib.sha1(normalized_grid.tobytes()).hexdigest()
    
    def _normalize_grid(self) -> np.ndarray:
        """Normalize grid colors for canonical representation."""
        # Map colors to canonical order based on frequency
        unique_colors, counts = np.unique(self.grid, return_counts=True)
        color_order = unique_colors[np.argsort(-counts)]  # Descending by count
        
        # Create mapping to canonical colors [0, 1, 2, ...]
        color_map = {old_color: new_color for new_color, old_color in enumerate(color_order)}
        
        # Apply mapping
        normalized = np.zeros_like(self.grid)
        for old_color, new_color in color_map.items():
            normalized[self.grid == old_color] = new_color
            
        return normalized


@dataclass
class SearchNode:
    """Node in the search tree for A* algorithm."""
    
    state: GridState
    parent: Optional['SearchNode']
    action: str  # DSL operation that led to this node
    depth: int
    
    def __lt__(self, other: 'SearchNode') -> bool:
        """Comparison for priority queue ordering."""
        return self.state.total_cost() < other.state.total_cost()
    
    def get_path(self) -> List[str]:
        """Reconstruct path from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))


@dataclass
class Task:
    """ARC task with training and test examples."""
    
    task_id: str
    train_examples: List[Tuple[np.ndarray, np.ndarray]]  # (input, output) pairs
    test_inputs: List[np.ndarray]
    
    def __post_init__(self) -> None:
        """Validate task structure."""
        assert len(self.train_examples) > 0, "Task must have at least one training example"
        assert len(self.test_inputs) > 0, "Task must have at least one test input"
        
        # Validate that all grids are 2D integer arrays
        for i, (input_grid, output_grid) in enumerate(self.train_examples):
            assert input_grid.ndim == 2, f"Training input {i} must be 2D"
            assert output_grid.ndim == 2, f"Training output {i} must be 2D"
            assert input_grid.dtype == np.int32, f"Training input {i} must be int32"
            assert output_grid.dtype == np.int32, f"Training output {i} must be int32"
        
        for i, test_input in enumerate(self.test_inputs):
            assert test_input.ndim == 2, f"Test input {i} must be 2D"
            assert test_input.dtype == np.int32, f"Test input {i} must be int32"


# Type aliases for clarity
Grid = np.ndarray  # 2D integer array representing the puzzle grid
Color = int  # Integer representing a color in the grid
Position = Tuple[int, int]  # (row, col) position in the grid
BoundingBox = Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)