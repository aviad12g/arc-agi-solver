"""Core DSL primitives for grid transformations.

This module implements the fundamental operations that can be applied to grids
in the ARC domain. These primitives are designed to be composable and efficient.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
from enum import Enum
import time

logger = logging.getLogger(__name__)


class DSLPrimitive(ABC):
    """Abstract base class for all DSL primitives."""
    
    def __init__(self, name: str, max_execution_time: float = 0.0002):
        """Initialize primitive.
        
        Args:
            name: Name of the primitive
            max_execution_time: Maximum execution time in seconds (200µs target)
        """
        self.name = name
        self.max_execution_time = max_execution_time
    
    @abstractmethod
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Execute the primitive on a grid.
        
        Args:
            grid: Input grid
            **kwargs: Additional parameters
            
        Returns:
            Transformed grid
        """
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """Validate parameters for this primitive.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        pass
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Get all valid parameter combinations for this primitive on the given grid.
        
        Args:
            grid: Input grid to generate parameters for
            
        Returns:
            List of parameter dictionaries
        """
        # Default implementation: return empty parameters for parameterless operations
        return [{}]
    
    def __call__(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Execute primitive with timing and validation."""
        start_time = time.perf_counter()
        
        # Validate parameters
        if not self.validate_params(**kwargs):
            raise ValueError(f"Invalid parameters for {self.name}: {kwargs}")
        
        # Execute primitive
        result = self.execute(grid, **kwargs)
        
        # Check execution time
        execution_time = time.perf_counter() - start_time
        if execution_time > self.max_execution_time:
            logger.warning(f"{self.name} took {execution_time*1000000:.0f}µs, exceeds {self.max_execution_time*1000000:.0f}µs target")
        
        return result


# Geometric Transform Primitives

class Rotate90(DSLPrimitive):
    """Rotate grid 90 degrees clockwise."""
    
    def __init__(self):
        super().__init__("Rotate90")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Rotate grid 90 degrees clockwise."""
        return np.rot90(grid, k=-1)  # k=-1 for clockwise
    
    def validate_params(self, **kwargs) -> bool:
        """No parameters needed for rotation."""
        return True


class Rotate180(DSLPrimitive):
    """Rotate grid 180 degrees."""
    
    def __init__(self):
        super().__init__("Rotate180")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Rotate grid 180 degrees."""
        return np.rot90(grid, k=2)
    
    def validate_params(self, **kwargs) -> bool:
        """No parameters needed for rotation."""
        return True


class ReflectH(DSLPrimitive):
    """Reflect grid horizontally (flip left-right)."""
    
    def __init__(self):
        super().__init__("ReflectH")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Reflect grid horizontally."""
        return np.fliplr(grid)
    
    def validate_params(self, **kwargs) -> bool:
        """No parameters needed for reflection."""
        return True


class ReflectV(DSLPrimitive):
    """Reflect grid vertically (flip up-down)."""
    
    def __init__(self):
        super().__init__("ReflectV")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Reflect grid vertically."""
        return np.flipud(grid)
    
    def validate_params(self, **kwargs) -> bool:
        """No parameters needed for reflection."""
        return True


# Spatial Operation Primitives

class Crop(DSLPrimitive):
    """Crop a rectangular region from the grid."""
    
    def __init__(self):
        super().__init__("Crop")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Crop grid to specified region.
        
        Args:
            r1, r2: Row range (inclusive)
            c1, c2: Column range (inclusive)
        """
        r1 = kwargs['r1']
        r2 = kwargs['r2']
        c1 = kwargs['c1']
        c2 = kwargs['c2']
        
        # Ensure bounds are within grid
        height, width = grid.shape
        r1 = max(0, min(r1, height - 1))
        r2 = max(r1, min(r2, height - 1))
        c1 = max(0, min(c1, width - 1))
        c2 = max(c1, min(c2, width - 1))
        
        return grid[r1:r2+1, c1:c2+1].copy()
    
    def validate_params(self, **kwargs) -> bool:
        """Validate crop parameters."""
        required = ['r1', 'r2', 'c1', 'c2']
        if not all(param in kwargs for param in required):
            return False
        
        r1, r2, c1, c2 = kwargs['r1'], kwargs['r2'], kwargs['c1'], kwargs['c2']
        
        # Check that coordinates are non-negative integers
        if not all(isinstance(x, int) and x >= 0 for x in [r1, r2, c1, c2]):
            return False
        
        # Check that ranges are valid
        return r1 <= r2 and c1 <= c2
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate crop parameter combinations for the grid."""
        height, width = grid.shape
        combinations = []
        
        # Generate a reasonable subset of crop regions (not all possible)
        # Focus on meaningful crops: quarters, halves, etc.
        crop_regions = [
            # Full grid
            (0, height-1, 0, width-1),
            # Halves
            (0, height//2, 0, width-1),  # Top half
            (height//2, height-1, 0, width-1),  # Bottom half
            (0, height-1, 0, width//2),  # Left half
            (0, height-1, width//2, width-1),  # Right half
            # Quarters
            (0, height//2, 0, width//2),  # Top-left
            (0, height//2, width//2, width-1),  # Top-right
            (height//2, height-1, 0, width//2),  # Bottom-left
            (height//2, height-1, width//2, width-1),  # Bottom-right
        ]
        
        for r1, r2, c1, c2 in crop_regions:
            if r1 < height and r2 < height and c1 < width and c2 < width:
                combinations.append({
                    'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2
                })
        
        return combinations


class Paint(DSLPrimitive):
    """Paint a single pixel with a specific color."""
    
    def __init__(self):
        super().__init__("Paint")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Paint pixel at (x, y) with color c.
        
        Args:
            x: Row coordinate
            y: Column coordinate  
            c: Color value
        """
        result = grid.copy()
        x, y, c = kwargs['x'], kwargs['y'], kwargs['c']
        
        # Check bounds
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
            result[x, y] = c
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate paint parameters."""
        required = ['x', 'y', 'c']
        if not all(param in kwargs for param in required):
            return False
        
        x, y, c = kwargs['x'], kwargs['y'], kwargs['c']
        
        # Check that coordinates and color are integers
        if not all(isinstance(val, int) for val in [x, y, c]):
            return False
        
        # Check that coordinates are non-negative and color is valid (0-9)
        return x >= 0 and y >= 0 and 0 <= c <= 9
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate paint parameter combinations for the grid."""
        height, width = grid.shape
        combinations = []
        
        # Get unique colors in the grid
        unique_colors = set(np.unique(grid))
        
        # Add some common colors that might be useful
        useful_colors = unique_colors.union({0, 1, 2, 3, 4, 5})
        
        # Generate paint operations for a subset of positions
        # Focus on corners, edges, and center to keep it manageable
        key_positions = [
            (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),  # Corners
            (height//2, width//2),  # Center
        ]
        
        # Add some edge positions
        if height > 2:
            key_positions.extend([(height//2, 0), (height//2, width-1)])
        if width > 2:
            key_positions.extend([(0, width//2), (height-1, width//2)])
        
        for x, y in key_positions:
            if 0 <= x < height and 0 <= y < width:
                for color in useful_colors:
                    if color != grid[x, y]:  # Only paint if different from current
                        combinations.append({'x': x, 'y': y, 'c': int(color)})
        
        return combinations[:20]  # Limit to prevent explosion


# Color Manipulation Primitives

class MapColors(DSLPrimitive):
    """Remap colors according to a permutation."""
    
    def __init__(self):
        super().__init__("MapColors")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Apply color mapping to grid.
        
        Args:
            perm: List/array of 10 integers representing color mapping
                  perm[i] is the new color for original color i
        """
        perm = kwargs['perm']
        result = grid.copy()
        
        # Apply permutation
        for old_color in range(10):
            if old_color < len(perm):
                new_color = perm[old_color]
                result[grid == old_color] = new_color
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate color mapping parameters."""
        if 'perm' not in kwargs:
            return False
        
        perm = kwargs['perm']
        
        # Check that perm is a sequence of 10 integers
        if not hasattr(perm, '__len__') or len(perm) != 10:
            return False
        
        # Check that all values are valid colors (0-9)
        try:
            return all(isinstance(c, int) and 0 <= c <= 9 for c in perm)
        except (TypeError, ValueError):
            return False
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate color mapping parameter combinations for the grid."""
        combinations = []
        
        # Get unique colors in the grid
        unique_colors = list(set(np.unique(grid)))
        
        # Generate some useful color mappings
        # 1. Identity mapping (no change)
        identity = list(range(10))
        combinations.append({'perm': identity})
        
        # 2. Simple swaps between colors present in the grid
        if len(unique_colors) >= 2:
            for i in range(len(unique_colors)):
                for j in range(i + 1, len(unique_colors)):
                    color1, color2 = unique_colors[i], unique_colors[j]
                    swap_perm = identity.copy()
                    swap_perm[color1], swap_perm[color2] = color2, color1
                    combinations.append({'perm': swap_perm})
        
        # 3. Map all colors to a single color
        for target_color in unique_colors:
            if target_color != 0:  # Don't map everything to background
                single_color_perm = [target_color] * 10
                combinations.append({'perm': single_color_perm})
        
        return combinations[:10]  # Limit to prevent explosion


# Conditional Operation Primitives

class BlobPredicate(ABC):
    """Abstract base class for blob predicates."""
    
    @abstractmethod
    def evaluate(self, blob_pixels: List[Tuple[int, int]], 
                blob_color: int, grid: np.ndarray) -> bool:
        """Evaluate predicate on a blob.
        
        Args:
            blob_pixels: List of (row, col) coordinates of blob pixels
            blob_color: Color of the blob
            grid: Full grid for context
            
        Returns:
            True if predicate is satisfied
        """
        pass


class SizePredicate(BlobPredicate):
    """Predicate based on blob size."""
    
    def __init__(self, min_size: int = 0, max_size: int = float('inf')):
        self.min_size = min_size
        self.max_size = max_size
    
    def evaluate(self, blob_pixels: List[Tuple[int, int]], 
                blob_color: int, grid: np.ndarray) -> bool:
        """Check if blob size is within range."""
        size = len(blob_pixels)
        return self.min_size <= size <= self.max_size


class ColorPredicate(BlobPredicate):
    """Predicate based on blob color."""
    
    def __init__(self, target_colors: List[int]):
        self.target_colors = set(target_colors)
    
    def evaluate(self, blob_pixels: List[Tuple[int, int]], 
                blob_color: int, grid: np.ndarray) -> bool:
        """Check if blob color matches target."""
        return blob_color in self.target_colors


class HorizontalLinePredicate(BlobPredicate):
    """Predicate that checks if a blob forms a horizontal line (height = 1)."""
    def evaluate(self, blob_pixels: List[Tuple[int, int]], blob_color: int, grid: np.ndarray) -> bool:
        rows = [r for r, _ in blob_pixels]
        return max(rows) == min(rows)  # single row


class VerticalLinePredicate(BlobPredicate):
    """Predicate that checks if a blob forms a vertical line (width = 1)."""
    def evaluate(self, blob_pixels: List[Tuple[int, int]], blob_color: int, grid: np.ndarray) -> bool:
        cols = [c for _, c in blob_pixels]
        return max(cols) == min(cols)  # single column


class PaintIf(DSLPrimitive):
    """Conditionally paint blobs based on predicates."""
    
    def __init__(self):
        super().__init__("PaintIf")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Paint blobs that satisfy predicate with new color.
        
        Args:
            predicate: BlobPredicate to evaluate
            new_color: Color to paint matching blobs
        """
        predicate = kwargs['predicate']
        new_color = kwargs['new_color']
        
        result = grid.copy()
        
        # Find all blobs using simple connected component analysis
        visited = np.zeros_like(grid, dtype=bool)
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if visited[r, c] or grid[r, c] == 0:  # Skip visited or background
                    continue
                
                # Find connected component (blob)
                blob_pixels = []
                blob_color = grid[r, c]
                stack = [(r, c)]
                visited[r, c] = True
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    blob_pixels.append((curr_r, curr_c))
                    
                    # Check 4-connected neighbors
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and
                            not visited[nr, nc] and grid[nr, nc] == blob_color):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                # Evaluate predicate
                if predicate.evaluate(blob_pixels, blob_color, grid):
                    # Paint all pixels of this blob
                    for br, bc in blob_pixels:
                        result[br, bc] = new_color
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate PaintIf parameters."""
        required = ['predicate', 'new_color']
        if not all(param in kwargs for param in required):
            return False
        
        predicate = kwargs['predicate']
        new_color = kwargs['new_color']
        
        # Check predicate is a BlobPredicate
        if not isinstance(predicate, BlobPredicate):
            return False
        
        # Check new_color is valid
        return isinstance(new_color, int) and 0 <= new_color <= 9
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate PaintIf parameter combinations for the grid."""
        combinations = []
        
        # Get unique colors in the grid
        unique_colors = list(set(np.unique(grid)))
        
        # Generate simple predicates and color mappings
        for target_color in unique_colors:
            if target_color != 0:  # Don't target background
                # Color-based predicate
                color_predicate = ColorPredicate([target_color])
                
                # Paint with different colors
                for new_color in unique_colors:
                    if new_color != target_color:
                        combinations.append({
                            'predicate': color_predicate,
                            'new_color': new_color
                        })
        
        # Size-based predicates
        if len(unique_colors) > 1:
            # Small blobs (size <= 3)
            small_predicate = SizePredicate(min_size=1, max_size=3)
            # Large blobs (size > 3)  
            large_predicate = SizePredicate(min_size=4, max_size=100)
            
            for predicate in [small_predicate, large_predicate]:
                for new_color in unique_colors:
                    if new_color != 0:  # Don't paint with background
                        combinations.append({
                            'predicate': predicate,
                            'new_color': new_color
                        })
        
        return combinations[:5]  # Limit to prevent explosion


# Factory functions for creating primitives

def create_all_primitives() -> Dict[str, DSLPrimitive]:
    """Create all available DSL primitives.
    
    Returns:
        Dictionary mapping primitive names to instances
    """
    primitives = {
        'Rotate90': Rotate90(),
        'Rotate180': Rotate180(),
        'ReflectH': ReflectH(),
        'ReflectV': ReflectV(),
        'Crop': Crop(),
        'Paint': Paint(),
        'MapColors': MapColors(),
        'PaintIf': PaintIf()
    }
    
    logger.info(f"Created {len(primitives)} DSL primitives")
    return primitives


def create_geometric_primitives() -> Dict[str, DSLPrimitive]:
    """Create only geometric transformation primitives.
    
    Returns:
        Dictionary of geometric primitives
    """
    return {
        'Rotate90': Rotate90(),
        'Rotate180': Rotate180(),
        'ReflectH': ReflectH(),
        'ReflectV': ReflectV()
    }


def create_spatial_primitives() -> Dict[str, DSLPrimitive]:
    """Create spatial operation primitives.
    
    Returns:
        Dictionary of spatial primitives
    """
    return {
        'Crop': Crop(),
        'Paint': Paint()
    }


def create_color_primitives() -> Dict[str, DSLPrimitive]:
    """Create color manipulation primitives.
    
    Returns:
        Dictionary of color primitives
    """
    return {
        'MapColors': MapColors()
    }


def create_conditional_primitives() -> Dict[str, DSLPrimitive]:
    """Create conditional operation primitives.
    
    Returns:
        Dictionary of conditional primitives
    """
    return {
        'PaintIf': PaintIf()
    }