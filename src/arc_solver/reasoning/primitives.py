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

class Translate(DSLPrimitive):
    """Translate (move) the entire grid by a given offset."""
    
    def __init__(self):
        super().__init__("Translate")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Translate grid by (dx, dy) offset with bounds checking.
        
        Args:
            dx: Row offset (positive = down, negative = up)
            dy: Column offset (positive = right, negative = left)
        """
        dx = kwargs['dx']
        dy = kwargs['dy']
        
        height, width = grid.shape
        result = np.zeros_like(grid)
        
        # Copy pixels to new positions, clipping to bounds
        for r in range(height):
            for c in range(width):
                new_r = r + dx
                new_c = c + dy
                
                # Only copy if new position is within bounds
                if 0 <= new_r < height and 0 <= new_c < width:
                    result[new_r, new_c] = grid[r, c]
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate translate parameters."""
        required = ['dx', 'dy']
        if not all(param in kwargs for param in required):
            return False
        
        dx, dy = kwargs['dx'], kwargs['dy']
        
        # Check that offsets are integers
        if not all(isinstance(x, int) for x in [dx, dy]):
            return False
        
        # Allow reasonable translation ranges
        return -10 <= dx <= 10 and -10 <= dy <= 10
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate translate parameter combinations for the grid."""
        height, width = grid.shape
        combinations = []
        
        # Generate useful translation offsets
        # Focus on small translations that are likely to be meaningful
        useful_offsets = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # Single step in each direction
            (1, 1), (1, -1), (-1, 1), (-1, -1),  # Diagonal single steps
            (0, 2), (0, -2), (2, 0), (-2, 0),  # Two steps in cardinal directions
        ]
        
        # Only include offsets that don't move everything out of bounds
        for dx, dy in useful_offsets:
            if (abs(dx) < height and abs(dy) < width):
                combinations.append({'dx': dx, 'dy': dy})
        
        return combinations


class Scale(DSLPrimitive):
    """Scale the grid by a given factor (simple nearest-neighbor)."""
    
    def __init__(self):
        super().__init__("Scale")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Scale grid by the given factor.
        
        Args:
            factor: Scaling factor (integer, 1-4 range for practicality)
        """
        factor = kwargs['factor']
        
        if factor == 1:
            return grid.copy()
        
        height, width = grid.shape
        new_height = height * factor
        new_width = width * factor
        
        # Create scaled grid using nearest-neighbor interpolation
        result = np.zeros((new_height, new_width), dtype=grid.dtype)
        
        for r in range(new_height):
            for c in range(new_width):
                # Map back to original coordinates
                orig_r = r // factor
                orig_c = c // factor
                result[r, c] = grid[orig_r, orig_c]
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate scale parameters."""
        if 'factor' not in kwargs:
            return False
        
        factor = kwargs['factor']
        
        # Check that factor is a positive integer in reasonable range
        return isinstance(factor, int) and 1 <= factor <= 4
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate scale parameter combinations for the grid."""
        combinations = []
        
        # Only generate scale factors that result in reasonable grid sizes
        height, width = grid.shape
        max_dimension = max(height, width)
        
        # Generate useful scale factors
        for factor in [2, 3, 4]:
            # Only include if resulting grid won't be too large
            if max_dimension * factor <= 30:  # Keep within reasonable bounds
                combinations.append({'factor': factor})
        
        return combinations


class Extract(DSLPrimitive):
    """Extract a subregion and return it as a new grid."""
    
    def __init__(self):
        super().__init__("Extract")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Extract a rectangular region from the grid.
        
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
        
        # Extract the region
        extracted = grid[r1:r2+1, c1:c2+1].copy()
        
        # Create a new grid with the extracted region
        # For now, just return the extracted region as-is
        return extracted
    
    def validate_params(self, **kwargs) -> bool:
        """Validate extract parameters."""
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
        """Generate extract parameter combinations for the grid."""
        height, width = grid.shape
        combinations = []
        
        # Generate meaningful extraction regions
        # Focus on regions that might contain interesting patterns
        extract_regions = [
            # Quarters
            (0, height//2, 0, width//2),  # Top-left
            (0, height//2, width//2, width-1),  # Top-right
            (height//2, height-1, 0, width//2),  # Bottom-left
            (height//2, height-1, width//2, width-1),  # Bottom-right
            # Center region
            (height//4, 3*height//4, width//4, 3*width//4),
            # Edge strips
            (0, height-1, 0, width//4),  # Left strip
            (0, height-1, 3*width//4, width-1),  # Right strip
            (0, height//4, 0, width-1),  # Top strip
            (3*height//4, height-1, 0, width-1),  # Bottom strip
        ]
        
        for r1, r2, c1, c2 in extract_regions:
            # Ensure coordinates are valid
            if (r1 < height and r2 < height and c1 < width and c2 < width and
                r1 <= r2 and c1 <= c2 and (r2 - r1 + 1) * (c2 - c1 + 1) > 1):
                combinations.append({
                    'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2
                })
        
        return combinations


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


# Pattern Manipulation Primitives

class FloodFillPrimitive(DSLPrimitive):
    """Fill connected regions using flood fill algorithm."""
    
    def __init__(self):
        super().__init__("FloodFill")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Flood fill starting from given position.
        
        Args:
            start_row: Starting row position
            start_col: Starting column position
            color: Color to fill with
        """
        start_row = kwargs['start_row']
        start_col = kwargs['start_col']
        color = kwargs['color']
        
        rows, cols = grid.shape
        
        # Validate starting position
        if not (0 <= start_row < rows and 0 <= start_col < cols):
            return grid.copy()  # Invalid position, return unchanged
        
        result = grid.copy()
        original_color = result[start_row, start_col]
        
        # No need to fill if already the target color
        if original_color == color:
            return result
        
        # Use iterative flood fill to avoid recursion depth issues
        stack = [(start_row, start_col)]
        visited = set()
        
        while stack:
            row, col = stack.pop()
            
            # Skip if out of bounds or already visited
            if (row, col) in visited or not (0 <= row < rows and 0 <= col < cols):
                continue
            
            # Skip if not the original color
            if result[row, col] != original_color:
                continue
            
            # Fill this cell
            result[row, col] = color
            visited.add((row, col))
            
            # Add neighbors to stack (4-connectivity)
            stack.extend([
                (row - 1, col),  # Up
                (row + 1, col),  # Down
                (row, col - 1),  # Left
                (row, col + 1),  # Right
            ])
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate flood fill parameters."""
        required = ['start_row', 'start_col', 'color']
        if not all(param in kwargs for param in required):
            return False
        
        start_row = kwargs['start_row']
        start_col = kwargs['start_col']
        color = kwargs['color']
        
        # Check that coordinates and color are integers
        if not all(isinstance(val, int) for val in [start_row, start_col, color]):
            return False
        
        # Check that coordinates are non-negative and color is valid (0-9)
        return start_row >= 0 and start_col >= 0 and 0 <= color <= 9
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate flood fill parameter combinations for the grid."""
        rows, cols = grid.shape
        params = []
        
        # Try flood fill from corners and center
        positions = [
            (0, 0),                    # Top-left
            (0, cols - 1),            # Top-right
            (rows - 1, 0),            # Bottom-left
            (rows - 1, cols - 1),     # Bottom-right
            (rows // 2, cols // 2),   # Center
        ]
        
        # Get unique colors in grid to create contrasting fills
        unique_colors = set(np.unique(grid))
        available_colors = [c for c in range(10) if c not in unique_colors]
        if not available_colors:
            available_colors = [1, 2, 3]  # Fallback colors
        
        for row, col in positions:
            if 0 <= row < rows and 0 <= col < cols:
                current_color = grid[row, col]
                for color in available_colors[:3]:  # Limit colors to avoid explosion
                    # Only add if it would actually change something
                    if color != current_color:
                        params.append({
                            'start_row': row,
                            'start_col': col,
                            'color': color
                        })
        
        return params[:15]  # Limit to avoid explosion


class OverlayPrimitive(DSLPrimitive):
    """Overlay one pattern onto another at a specified position."""
    
    def __init__(self):
        super().__init__("Overlay")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Overlay pattern onto grid at specified position.
        
        Args:
            pattern: Pattern to overlay (as numpy array)
            position_row: Row position for top-left of pattern
            position_col: Column position for top-left of pattern
        """
        pattern = kwargs['pattern']
        position_row = kwargs['position_row']
        position_col = kwargs['position_col']
        
        result = grid.copy()
        pattern_rows, pattern_cols = pattern.shape
        grid_rows, grid_cols = grid.shape
        
        # Calculate valid overlay region
        start_row = max(0, position_row)
        end_row = min(grid_rows, position_row + pattern_rows)
        start_col = max(0, position_col)
        end_col = min(grid_cols, position_col + pattern_cols)
        
        # Calculate corresponding pattern region
        pattern_start_row = max(0, -position_row)
        pattern_end_row = pattern_start_row + (end_row - start_row)
        pattern_start_col = max(0, -position_col)
        pattern_end_col = pattern_start_col + (end_col - start_col)
        
        # Overlay pattern (non-zero values only)
        if (end_row > start_row and end_col > start_col and
            pattern_end_row > pattern_start_row and pattern_end_col > pattern_start_col):
            
            pattern_region = pattern[pattern_start_row:pattern_end_row, 
                                  pattern_start_col:pattern_end_col]
            
            # Only overlay non-background pixels (assuming 0 is background)
            mask = pattern_region != 0
            result[start_row:end_row, start_col:end_col][mask] = pattern_region[mask]
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate overlay parameters."""
        required = ['pattern', 'position_row', 'position_col']
        if not all(param in kwargs for param in required):
            return False
        
        pattern = kwargs['pattern']
        position_row = kwargs['position_row']
        position_col = kwargs['position_col']
        
        # Check pattern is a numpy array
        if not isinstance(pattern, np.ndarray):
            return False
        
        # Check positions are integers
        if not all(isinstance(val, int) for val in [position_row, position_col]):
            return False
        
        # Check pattern has valid shape and values
        if pattern.ndim != 2 or pattern.size == 0:
            return False
        
        # Check pattern values are valid colors (0-9)
        if not np.all((pattern >= 0) & (pattern <= 9)):
            return False
        
        return True
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate overlay parameter combinations for the grid."""
        rows, cols = grid.shape
        params = []
        
        # Get unique colors in grid to create contrasting patterns
        unique_colors = set(np.unique(grid))
        available_colors = [c for c in range(1, 10) if c not in unique_colors]
        if not available_colors:
            available_colors = [1, 2, 3]  # Fallback colors
        
        # Create simple patterns to overlay using contrasting colors
        patterns = [
            # Single pixel with contrasting color
            np.array([[available_colors[0]]], dtype=np.int32),
            # 2x2 square
            np.array([[available_colors[0], available_colors[0]], 
                     [available_colors[0], available_colors[0]]], dtype=np.int32),
            # Cross pattern
            np.array([[0, available_colors[0], 0], 
                     [available_colors[0], available_colors[0], available_colors[0]], 
                     [0, available_colors[0], 0]], dtype=np.int32),
            # Line patterns
            np.array([[available_colors[0], available_colors[0], available_colors[0]]], dtype=np.int32),  # Horizontal line
            np.array([[available_colors[0]], [available_colors[0]], [available_colors[0]]], dtype=np.int32),  # Vertical line
        ]
        
        # Try overlaying at different positions
        positions = [
            (0, 0),                    # Top-left
            (0, cols // 2),           # Top-center
            (rows // 2, 0),           # Middle-left
            (rows // 2, cols // 2),   # Center
        ]
        
        for pattern in patterns:
            for pos_row, pos_col in positions:
                if (pos_row < rows and pos_col < cols):
                    params.append({
                        'pattern': pattern,
                        'position_row': pos_row,
                        'position_col': pos_col
                    })
        
        return params[:12]  # Limit to avoid explosion


class RepeatPrimitive(DSLPrimitive):
    """Repeat a pattern multiple times in a given direction."""
    
    def __init__(self):
        super().__init__("Repeat")
    
    def execute(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Repeat pattern in specified direction.
        
        Args:
            pattern: Pattern to repeat (as numpy array)
            count: Number of repetitions
            direction: Direction ('horizontal', 'vertical')
        """
        pattern = kwargs['pattern']
        count = kwargs['count']
        direction = kwargs['direction']
        
        if count <= 0:
            return grid.copy()
        
        pattern_rows, pattern_cols = pattern.shape
        
        if direction == 'horizontal':
            # Repeat horizontally
            result_rows = pattern_rows
            result_cols = pattern_cols * count
            result = np.zeros((result_rows, result_cols), dtype=grid.dtype)
            
            for i in range(count):
                start_col = i * pattern_cols
                end_col = start_col + pattern_cols
                result[:, start_col:end_col] = pattern
                
        elif direction == 'vertical':
            # Repeat vertically
            result_rows = pattern_rows * count
            result_cols = pattern_cols
            result = np.zeros((result_rows, result_cols), dtype=grid.dtype)
            
            for i in range(count):
                start_row = i * pattern_rows
                end_row = start_row + pattern_rows
                result[start_row:end_row, :] = pattern
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'")
        
        # Crop to original grid size if result is larger
        grid_rows, grid_cols = grid.shape
        result = result[:grid_rows, :grid_cols]
        
        return result
    
    def validate_params(self, **kwargs) -> bool:
        """Validate repeat parameters."""
        required = ['pattern', 'count', 'direction']
        if not all(param in kwargs for param in required):
            return False
        
        pattern = kwargs['pattern']
        count = kwargs['count']
        direction = kwargs['direction']
        
        # Check pattern is a numpy array
        if not isinstance(pattern, np.ndarray):
            return False
        
        # Check count is a positive integer
        if not isinstance(count, int) or count <= 0:
            return False
        
        # Check direction is valid
        if direction not in ['horizontal', 'vertical']:
            return False
        
        # Check pattern has valid shape and values
        if pattern.ndim != 2 or pattern.size == 0:
            return False
        
        # Check pattern values are valid colors (0-9)
        if not np.all((pattern >= 0) & (pattern <= 9)):
            return False
        
        return True
    
    def get_parameter_combinations(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate repeat parameter combinations for the grid."""
        rows, cols = grid.shape
        params = []
        
        # Create simple patterns to repeat
        patterns = [
            # Single pixel
            np.array([[1]], dtype=np.int32),
            # 2x1 pattern
            np.array([[1, 2]], dtype=np.int32),
            # 1x2 pattern
            np.array([[1], [2]], dtype=np.int32),
            # 2x2 pattern
            np.array([[1, 2], [3, 4]], dtype=np.int32),
        ]
        
        directions = ['horizontal', 'vertical']
        counts = [2, 3, 4]
        
        for pattern in patterns:
            pattern_rows, pattern_cols = pattern.shape
            for direction in directions:
                for count in counts:
                    # Check if repeated pattern would fit reasonably
                    if direction == 'horizontal' and pattern_cols * count <= cols:
                        params.append({
                            'pattern': pattern,
                            'count': count,
                            'direction': direction
                        })
                    elif direction == 'vertical' and pattern_rows * count <= rows:
                        params.append({
                            'pattern': pattern,
                            'count': count,
                            'direction': direction
                        })
        
        return params[:10]  # Limit to avoid explosion


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
        'Translate': Translate(),
        'Scale': Scale(),
        'Extract': Extract(),
        'Crop': Crop(),
        'Paint': Paint(),
        'MapColors': MapColors(),
        'PaintIf': PaintIf(),
        'FloodFill': FloodFillPrimitive(),
        'Overlay': OverlayPrimitive(),
        'Repeat': RepeatPrimitive()
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