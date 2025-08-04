"""GPU-accelerated blob labeling engine for connected component analysis."""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

from arc_solver.core.data_models import Blob

logger = logging.getLogger(__name__)


class BlobLabeler:
    """High-performance blob labeling with GPU acceleration and CPU fallback."""
    
    def __init__(self, use_gpu: bool = True, max_grid_size: int = 30):
        """Initialize the blob labeler.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
            max_grid_size: Maximum expected grid size for optimization
        """
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.max_grid_size = max_grid_size
        
        if self.use_gpu:
            logger.info("GPU blob labeling enabled")
            self._init_gpu_kernels()
        else:
            if use_gpu and not CUDA_AVAILABLE:
                logger.warning("GPU requested but CUDA not available, using CPU fallback")
            else:
                logger.info("CPU blob labeling enabled")
    
    def _init_gpu_kernels(self):
        """Initialize CUDA kernels for GPU acceleration."""
        if not self.use_gpu:
            return
        
        # Union-Find CUDA kernel for connected components
        self.union_find_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void union_find_kernel(int* labels, int* parent, int height, int width) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = height * width;
            
            if (idx >= total_pixels) return;
            
            int row = idx / width;
            int col = idx % width;
            int current_label = labels[idx];
            
            if (current_label == 0) return;  // Background pixel
            
            // Check 4-connected neighbors
            int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            
            for (int i = 0; i < 4; i++) {
                int nr = row + neighbors[i][0];
                int nc = col + neighbors[i][1];
                
                if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                    int neighbor_idx = nr * width + nc;
                    int neighbor_label = labels[neighbor_idx];
                    
                    if (neighbor_label == current_label) {
                        // Union operation
                        int root1 = find_root(parent, idx);
                        int root2 = find_root(parent, neighbor_idx);
                        
                        if (root1 != root2) {
                            if (root1 < root2) {
                                parent[root2] = root1;
                            } else {
                                parent[root1] = root2;
                            }
                        }
                    }
                }
            }
        }
        
        __device__ int find_root(int* parent, int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];  // Path compression
                x = parent[x];
            }
            return x;
        }
        ''', 'union_find_kernel')
        
        logger.info("GPU kernels initialized successfully")
    
    def label_blobs(self, grid: np.ndarray, connectivity: int = 4) -> Tuple[List[Blob], float]:
        """Label connected components (blobs) in the grid.
        
        Args:
            grid: 2D numpy array with integer color values
            connectivity: 4 or 8 connectivity for blob detection
            
        Returns:
            Tuple of (list of Blob objects, processing time in seconds)
        """
        start_time = time.perf_counter()
        
        try:
            if self.use_gpu and grid.size <= self.max_grid_size * self.max_grid_size:
                blobs = self._label_blobs_gpu(grid, connectivity)
            else:
                blobs = self._label_blobs_cpu(grid, connectivity)
        except Exception as e:
            logger.warning(f"GPU labeling failed, falling back to CPU: {e}")
            blobs = self._label_blobs_cpu(grid, connectivity)
        
        processing_time = time.perf_counter() - start_time
        
        # Verify performance requirement: ≤2ms for 30×30 grid
        if grid.shape[0] <= 30 and grid.shape[1] <= 30 and processing_time > 0.002:
            logger.warning(f"Blob labeling took {processing_time*1000:.2f}ms, exceeds 2ms target")
        
        return blobs, processing_time
    
    def _label_blobs_gpu(self, grid: np.ndarray, connectivity: int) -> List[Blob]:
        """GPU-accelerated blob labeling using CUDA kernels."""
        if not self.use_gpu:
            raise RuntimeError("GPU not available")
        
        # Transfer to GPU
        gpu_grid = cp.asarray(grid, dtype=cp.int32)
        height, width = gpu_grid.shape
        
        # Use CuPy's connected components for initial labeling
        if connectivity == 4:
            structure = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.bool_)
        else:  # connectivity == 8
            structure = cp.ones((3, 3), dtype=cp.bool_)
        
        # Label connected components for each color separately
        unique_colors = cp.unique(gpu_grid)
        all_blobs = []
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            
            # Create binary mask for this color
            color_mask = (gpu_grid == color)
            
            # Label connected components
            labeled_array, num_features = cp_ndimage.label(color_mask, structure=structure)
            
            # Extract blob information
            for blob_id in range(1, num_features + 1):
                blob_mask = (labeled_array == blob_id)
                blob_coords = cp.where(blob_mask)
                
                # Convert to CPU for Blob creation
                coords_cpu = [(int(r), int(c)) for r, c in zip(blob_coords[0].get(), blob_coords[1].get())]
                
                if coords_cpu:  # Ensure blob is not empty
                    # Calculate bounding box
                    rows = [r for r, c in coords_cpu]
                    cols = [c for r, c in coords_cpu]
                    min_row, max_row = min(rows), max(rows)
                    min_col, max_col = min(cols), max(cols)
                    
                    # Calculate center of mass
                    center_row = sum(rows) / len(rows)
                    center_col = sum(cols) / len(cols)
                    
                    blob = Blob(
                        id=len(all_blobs),  # Simple ID assignment
                        color=int(color),
                        pixels=coords_cpu,
                        bounding_box=(min_row, min_col, max_row, max_col),
                        center_of_mass=(center_row, center_col),
                        area=len(coords_cpu),
                        holes=0  # TODO: Implement hole detection
                    )
                    all_blobs.append(blob)
        
        return all_blobs
    
    def _label_blobs_cpu(self, grid: np.ndarray, connectivity: int) -> List[Blob]:
        """CPU fallback blob labeling using flood-fill algorithm."""
        height, width = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        blobs = []
        
        # Define connectivity offsets
        if connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # connectivity == 8
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for row in range(height):
            for col in range(width):
                if visited[row, col] or grid[row, col] == 0:  # Skip visited or background
                    continue
                
                # Start flood-fill for new blob
                color = grid[row, col]
                blob_coords = []
                queue = deque([(row, col)])
                visited[row, col] = True
                
                while queue:
                    r, c = queue.popleft()
                    blob_coords.append((r, c))
                    
                    # Check all neighbors
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        
                        if (0 <= nr < height and 0 <= nc < width and 
                            not visited[nr, nc] and grid[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                # Create blob object
                if blob_coords:
                    # Calculate bounding box
                    rows = [r for r, c in blob_coords]
                    cols = [c for r, c in blob_coords]
                    min_row, max_row = min(rows), max(rows)
                    min_col, max_col = min(cols), max(cols)
                    
                    # Calculate center of mass
                    center_row = sum(rows) / len(rows)
                    center_col = sum(cols) / len(cols)
                    
                    blob = Blob(
                        id=len(blobs),  # Simple ID assignment
                        color=color,
                        pixels=blob_coords,
                        bounding_box=(min_row, min_col, max_row, max_col),
                        center_of_mass=(center_row, center_col),
                        area=len(blob_coords),
                        holes=0  # TODO: Implement hole detection
                    )
                    blobs.append(blob)
        
        return blobs
    
    def get_blob_adjacency_graph(self, blobs: List[Blob], grid_shape: Tuple[int, int]) -> Dict[int, List[int]]:
        """Compute adjacency graph between blobs for spectral analysis.
        
        Args:
            blobs: List of blob objects
            grid_shape: Shape of the original grid
            
        Returns:
            Adjacency list representation of blob graph
        """
        if not blobs:
            return {}
        
        # Create a mapping from coordinates to blob indices
        coord_to_blob = {}
        for blob_idx, blob in enumerate(blobs):
            for coord in blob.pixels:
                coord_to_blob[coord] = blob_idx
        
        # Build adjacency graph
        adjacency = {i: set() for i in range(len(blobs))}
        
        for blob_idx, blob in enumerate(blobs):
            for r, c in blob.pixels:
                # Check 4-connected neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1] and 
                        (nr, nc) in coord_to_blob):
                        neighbor_blob = coord_to_blob[(nr, nc)]
                        if neighbor_blob != blob_idx:
                            adjacency[blob_idx].add(neighbor_blob)
                            adjacency[neighbor_blob].add(blob_idx)
        
        # Convert sets to lists
        return {k: list(v) for k, v in adjacency.items()}
    
    def benchmark_performance(self, grid_sizes: List[Tuple[int, int]], num_trials: int = 10) -> Dict:
        """Benchmark blob labeling performance across different grid sizes.
        
        Args:
            grid_sizes: List of (height, width) tuples to test
            num_trials: Number of trials per grid size
            
        Returns:
            Performance statistics dictionary
        """
        results = {}
        
        for height, width in grid_sizes:
            # Generate random test grid
            test_grid = np.random.randint(0, 5, size=(height, width), dtype=np.int32)
            
            times = []
            for _ in range(num_trials):
                _, processing_time = self.label_blobs(test_grid)
                times.append(processing_time)
            
            results[f"{height}x{width}"] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "meets_target": np.mean(times) <= 0.002 if height <= 30 and width <= 30 else True
            }
        
        return results


def create_blob_labeler(use_gpu: bool = True) -> BlobLabeler:
    """Factory function to create a blob labeler with optimal settings.
    
    Args:
        use_gpu: Whether to enable GPU acceleration
        
    Returns:
        Configured BlobLabeler instance
    """
    return BlobLabeler(use_gpu=use_gpu, max_grid_size=30)