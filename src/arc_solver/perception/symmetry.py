"""High-performance symmetry detection using bitboard representation."""

import numpy as np
import logging
from typing import Dict, List, Tuple, Set
from enum import Enum
import time

logger = logging.getLogger(__name__)


class SymmetryType(Enum):
    """Types of symmetries in the D₄ dihedral group."""
    IDENTITY = "identity"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    REFLECT_H = "reflect_horizontal"
    REFLECT_V = "reflect_vertical"
    REFLECT_D1 = "reflect_diagonal_1"  # Main diagonal (top-left to bottom-right)
    REFLECT_D2 = "reflect_diagonal_2"  # Anti-diagonal (top-right to bottom-left)


class BitboardSymmetryDetector:
    """Ultra-fast symmetry detection using uint64 bitboard representation."""
    
    def __init__(self, max_grid_size: int = 30):
        """Initialize the symmetry detector.
        
        Args:
            max_grid_size: Maximum grid size for bitboard optimization
        """
        self.max_grid_size = max_grid_size
        self.max_colors = 10  # ARC grids use colors 0-9
        
        # Pre-compute transformation masks for common grid sizes
        self._transform_cache = {}
        self._precompute_transforms()
        
        logger.info(f"Symmetry detector initialized for grids up to {max_grid_size}×{max_grid_size}")
    
    def _precompute_transforms(self):
        """Pre-compute transformation patterns for common grid sizes."""
        common_sizes = [3, 5, 7, 10, 15, 20, 25, 30]
        
        for size in common_sizes:
            if size <= self.max_grid_size:
                self._transform_cache[size] = self._compute_transform_masks(size)
    
    def _compute_transform_masks(self, size: int) -> Dict[SymmetryType, np.ndarray]:
        """Compute transformation index mappings for a given grid size.
        
        Args:
            size: Grid size (assuming square grid)
            
        Returns:
            Dictionary mapping symmetry types to index transformation arrays
        """
        transforms = {}
        
        # Create coordinate arrays
        coords = np.array([(r, c) for r in range(size) for c in range(size)])
        
        # Identity transformation
        transforms[SymmetryType.IDENTITY] = np.arange(size * size)
        
        # Rotation transformations
        transforms[SymmetryType.ROTATE_90] = np.array([
            (size - 1 - c) * size + r for r, c in coords
        ])
        
        transforms[SymmetryType.ROTATE_180] = np.array([
            (size - 1 - r) * size + (size - 1 - c) for r, c in coords
        ])
        
        transforms[SymmetryType.ROTATE_270] = np.array([
            c * size + (size - 1 - r) for r, c in coords
        ])
        
        # Reflection transformations
        transforms[SymmetryType.REFLECT_H] = np.array([
            (size - 1 - r) * size + c for r, c in coords
        ])
        
        transforms[SymmetryType.REFLECT_V] = np.array([
            r * size + (size - 1 - c) for r, c in coords
        ])
        
        transforms[SymmetryType.REFLECT_D1] = np.array([
            c * size + r for r, c in coords
        ])
        
        transforms[SymmetryType.REFLECT_D2] = np.array([
            (size - 1 - c) * size + (size - 1 - r) for r, c in coords
        ])
        
        return transforms
    
    def _grid_to_bitboards(self, grid: np.ndarray) -> Dict[int, int]:
        """Convert grid to bitboard representation for each color.
        
        Args:
            grid: 2D numpy array with integer color values
            
        Returns:
            Dictionary mapping colors to uint64 bitboards
        """
        height, width = grid.shape
        
        # Only use bitboards for small grids that fit in uint64
        if height * width > 64:
            return self._grid_to_large_bitboards(grid)
        
        bitboards = {}
        flat_grid = grid.flatten()
        
        for color in range(self.max_colors):
            if color in flat_grid:
                # Create bitboard for this color
                mask = (flat_grid == color).astype(np.uint64)
                bitboard = 0
                
                for i, bit in enumerate(mask):
                    if bit:
                        bitboard |= (1 << i)
                
                bitboards[color] = bitboard
        
        return bitboards
    
    def _grid_to_large_bitboards(self, grid: np.ndarray) -> Dict[int, np.ndarray]:
        """Handle grids larger than 64 pixels using numpy arrays.
        
        Args:
            grid: 2D numpy array with integer color values
            
        Returns:
            Dictionary mapping colors to boolean arrays
        """
        bitboards = {}
        unique_colors = np.unique(grid)
        
        for color in unique_colors:
            bitboards[color] = (grid == color).flatten()
        
        return bitboards
    
    def _apply_transform_bitboard(self, bitboard: int, transform_indices: np.ndarray, 
                                 grid_size: int) -> int:
        """Apply transformation to a uint64 bitboard.
        
        Args:
            bitboard: Original bitboard
            transform_indices: Index transformation array
            grid_size: Size of the grid
            
        Returns:
            Transformed bitboard
        """
        if grid_size * grid_size > 64:
            # Fall back to array-based transformation
            bits = np.zeros(grid_size * grid_size, dtype=bool)
            for i in range(grid_size * grid_size):
                if bitboard & (1 << i):
                    bits[i] = True
            
            transformed_bits = bits[transform_indices]
            result = 0
            for i, bit in enumerate(transformed_bits):
                if bit:
                    result |= (1 << i)
            return result
        
        # Fast bitboard transformation
        result = 0
        for i in range(grid_size * grid_size):
            if bitboard & (1 << i):
                result |= (1 << transform_indices[i])
        
        return result
    
    def _apply_transform_array(self, bitarray: np.ndarray, 
                              transform_indices: np.ndarray) -> np.ndarray:
        """Apply transformation to a boolean array.
        
        Args:
            bitarray: Original boolean array
            transform_indices: Index transformation array
            
        Returns:
            Transformed boolean array
        """
        return bitarray[transform_indices]
    
    def detect_symmetries(self, grid: np.ndarray) -> Set[SymmetryType]:
        """Detect all symmetries present in the grid.
        
        Args:
            grid: 2D numpy array with integer color values
            
        Returns:
            Set of detected symmetry types
        """
        start_time = time.perf_counter()
        
        height, width = grid.shape
        
        # Only handle square grids for now
        if height != width:
            logger.warning(f"Non-square grid {height}×{width}, symmetry detection limited")
            return {SymmetryType.IDENTITY}
        
        size = height
        symmetries = {SymmetryType.IDENTITY}  # Identity is always present
        
        # Get transformation masks
        if size in self._transform_cache:
            transforms = self._transform_cache[size]
        else:
            transforms = self._compute_transform_masks(size)
        
        # Convert grid to bitboards
        bitboards = self._grid_to_bitboards(grid)
        is_large_grid = size * size > 64
        
        # Test each transformation
        for sym_type, transform_indices in transforms.items():
            if sym_type == SymmetryType.IDENTITY:
                continue
            
            is_symmetric = True
            
            # Check if transformation preserves all color patterns
            for color, bitboard in bitboards.items():
                if is_large_grid:
                    # Use array-based comparison for large grids
                    transformed = self._apply_transform_array(bitboard, transform_indices)
                    if not np.array_equal(bitboard, transformed):
                        is_symmetric = False
                        break
                else:
                    # Use fast bitboard comparison
                    transformed = self._apply_transform_bitboard(
                        bitboard, transform_indices, size
                    )
                    if bitboard != transformed:
                        is_symmetric = False
                        break
            
            if is_symmetric:
                symmetries.add(sym_type)
        
        processing_time = time.perf_counter() - start_time
        
        # Check performance requirement: ≤10µs
        if processing_time > 0.00001:  # 10 microseconds
            logger.warning(f"Symmetry detection took {processing_time*1000000:.2f}µs, exceeds 10µs target")
        
        return symmetries
    
    def get_symmetry_group_order(self, symmetries: Set[SymmetryType]) -> int:
        """Get the order of the symmetry group.
        
        Args:
            symmetries: Set of detected symmetries
            
        Returns:
            Order of the symmetry group (1, 2, 4, or 8)
        """
        return len(symmetries)
    
    def has_rotational_symmetry(self, symmetries: Set[SymmetryType]) -> bool:
        """Check if the grid has any rotational symmetry.
        
        Args:
            symmetries: Set of detected symmetries
            
        Returns:
            True if grid has rotational symmetry
        """
        rotational_symmetries = {
            SymmetryType.ROTATE_90,
            SymmetryType.ROTATE_180,
            SymmetryType.ROTATE_270
        }
        return bool(symmetries & rotational_symmetries)
    
    def has_reflection_symmetry(self, symmetries: Set[SymmetryType]) -> bool:
        """Check if the grid has any reflection symmetry.
        
        Args:
            symmetries: Set of detected symmetries
            
        Returns:
            True if grid has reflection symmetry
        """
        reflection_symmetries = {
            SymmetryType.REFLECT_H,
            SymmetryType.REFLECT_V,
            SymmetryType.REFLECT_D1,
            SymmetryType.REFLECT_D2
        }
        return bool(symmetries & reflection_symmetries)
    
    def apply_symmetry_transform(self, grid: np.ndarray, 
                                symmetry: SymmetryType) -> np.ndarray:
        """Apply a symmetry transformation to a grid.
        
        Args:
            grid: Input grid
            symmetry: Symmetry transformation to apply
            
        Returns:
            Transformed grid
        """
        height, width = grid.shape
        
        if height != width:
            raise ValueError("Only square grids supported for transformations")
        
        size = height
        
        if symmetry == SymmetryType.IDENTITY:
            return grid.copy()
        
        # Get transformation indices
        if size in self._transform_cache:
            transform_indices = self._transform_cache[size][symmetry]
        else:
            transforms = self._compute_transform_masks(size)
            transform_indices = transforms[symmetry]
        
        # Apply transformation
        flat_grid = grid.flatten()
        transformed_flat = flat_grid[transform_indices]
        
        return transformed_flat.reshape(size, size)
    
    def get_canonical_form(self, grid: np.ndarray) -> Tuple[np.ndarray, SymmetryType]:
        """Get the canonical form of a grid under D₄ symmetry group.
        
        Args:
            grid: Input grid
            
        Returns:
            Tuple of (canonical_grid, applied_transformation)
        """
        if grid.shape[0] != grid.shape[1]:
            return grid.copy(), SymmetryType.IDENTITY
        
        # Generate all possible transformations
        all_transforms = [
            SymmetryType.IDENTITY,
            SymmetryType.ROTATE_90,
            SymmetryType.ROTATE_180,
            SymmetryType.ROTATE_270,
            SymmetryType.REFLECT_H,
            SymmetryType.REFLECT_V,
            SymmetryType.REFLECT_D1,
            SymmetryType.REFLECT_D2
        ]
        
        # Find lexicographically smallest representation
        canonical_grid = None
        canonical_transform = SymmetryType.IDENTITY
        
        for transform in all_transforms:
            transformed = self.apply_symmetry_transform(grid, transform)
            
            if canonical_grid is None or self._is_lexicographically_smaller(
                transformed, canonical_grid
            ):
                canonical_grid = transformed
                canonical_transform = transform
        
        return canonical_grid, canonical_transform
    
    def _is_lexicographically_smaller(self, grid1: np.ndarray, grid2: np.ndarray) -> bool:
        """Check if grid1 is lexicographically smaller than grid2.
        
        Args:
            grid1: First grid
            grid2: Second grid
            
        Returns:
            True if grid1 < grid2 lexicographically
        """
        flat1 = grid1.flatten()
        flat2 = grid2.flatten()
        
        for i in range(len(flat1)):
            if flat1[i] < flat2[i]:
                return True
            elif flat1[i] > flat2[i]:
                return False
        
        return False  # They are equal
    
    def benchmark_performance(self, grid_sizes: List[int], 
                            num_trials: int = 100) -> Dict[int, Dict[str, float]]:
        """Benchmark symmetry detection performance.
        
        Args:
            grid_sizes: List of grid sizes to test
            num_trials: Number of trials per size
            
        Returns:
            Performance statistics dictionary
        """
        results = {}
        
        for size in grid_sizes:
            # Generate random test grid
            test_grid = np.random.randint(0, 4, size=(size, size), dtype=np.int32)
            
            times = []
            for _ in range(num_trials):
                start_time = time.perf_counter()
                symmetries = self.detect_symmetries(test_grid)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[size] = {
                "mean_time_us": np.mean(times) * 1000000,
                "std_time_us": np.std(times) * 1000000,
                "min_time_us": np.min(times) * 1000000,
                "max_time_us": np.max(times) * 1000000,
                "meets_target": np.mean(times) <= 0.00001  # 10 microseconds
            }
        
        return results


def create_symmetry_detector(max_grid_size: int = 30) -> BitboardSymmetryDetector:
    """Factory function to create a symmetry detector.
    
    Args:
        max_grid_size: Maximum grid size for optimization
        
    Returns:
        Configured BitboardSymmetryDetector instance
    """
    return BitboardSymmetryDetector(max_grid_size=max_grid_size)


def get_d4_group_elements() -> List[SymmetryType]:
    """Get all elements of the D₄ dihedral group.
    
    Returns:
        List of all 8 symmetry transformations in D₄
    """
    return [
        SymmetryType.IDENTITY,
        SymmetryType.ROTATE_90,
        SymmetryType.ROTATE_180,
        SymmetryType.ROTATE_270,
        SymmetryType.REFLECT_H,
        SymmetryType.REFLECT_V,
        SymmetryType.REFLECT_D1,
        SymmetryType.REFLECT_D2
    ]