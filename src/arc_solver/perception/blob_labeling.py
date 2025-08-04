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

    def _cuda_connected_components(self, mask, connectivity: int):
        """Apply custom CUDA union-find kernel for connected components.
        
        Args:
            mask: Binary mask (True for foreground, False for background)
            connectivity: 4 or 8 connectivity
            
        Returns:
            Tuple of (labeled_array, num_components)
        """
        height, width = mask.shape
        total_pixels = height * width
        
        # Initialize parent array - each pixel is its own parent initially
        parent = cp.arange(total_pixels, dtype=cp.int32)
        
        # Convert mask to int32 labels (0 for background, 1 for foreground)
        labels = mask.astype(cp.int32)
        
        # Launch union-find kernel
        threads_per_block = 256
        blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block
        
        # Run multiple iterations to ensure convergence
        for iteration in range(10):  # Usually converges in 2-3 iterations
            old_parent = parent.copy()
            
            self.union_find_kernel(
                (blocks_per_grid,), (threads_per_block,),
                (labels, parent, height, width)
            )
            
            # Check for convergence
            if cp.array_equal(parent, old_parent):
                break
        
        # Compress paths and assign dense labels
        return self._compress_and_relabel(labels, parent, height, width)
    
    def _compress_and_relabel(self, labels, parent, height: int, width: int):
        """Compress union-find paths and assign dense component labels.
        
        Args:
            labels: Original foreground/background labels
            parent: Union-find parent array
            height: Grid height
            width: Grid width
            
        Returns:
            Tuple of (dense_labeled_array, num_components)
        """
        # Find root for each pixel
        flat_labels = labels.flatten()
        result = cp.zeros_like(flat_labels)
        
        # Only process foreground pixels
        foreground_mask = flat_labels > 0
        foreground_indices = cp.where(foreground_mask)[0]
        
        if len(foreground_indices) == 0:
            return result.reshape(height, width), 0
        
        # Find roots using path compression
        roots = cp.zeros_like(foreground_indices)
        for i, idx in enumerate(foreground_indices):
            # Path compression: find root and compress path
            root = int(idx)
            path = []
            while parent[root] != root:
                path.append(root)
                root = int(parent[root])
            
            # Compress path
            for p in path:
                parent[p] = root
            
            roots[i] = root
        
        # Get unique roots and assign dense labels
        unique_roots = cp.unique(roots)
        num_components = len(unique_roots)
        
        # Create mapping from root to dense label
        root_to_label = cp.zeros(parent.shape[0], dtype=cp.int32)
        for i, root in enumerate(unique_roots):
            root_to_label[root] = i + 1  # Labels start from 1
        
        # Assign dense labels to result
        for i, idx in enumerate(foreground_indices):
            root = roots[i]
            result[idx] = root_to_label[root]
        
        return result.reshape(height, width), num_components
    
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
        
        # Use custom CUDA union-find kernel for high performance
        unique_colors = cp.unique(gpu_grid)
        all_blobs = []
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            
            # Create binary mask for this color
            color_mask = (gpu_grid == color)
            
            # Apply custom union-find kernel
            try:
                labeled_array, num_features = self._cuda_connected_components(color_mask, connectivity)
            except Exception as e:
                logger.warning(f"CUDA kernel failed, falling back to CuPy: {e}")
                # Fallback to CuPy implementation
                if connectivity == 4:
                    structure = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.bool_)
                else:  # connectivity == 8
                    structure = cp.ones((3, 3), dtype=cp.bool_)
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
                        holes=self._count_holes(grid, coords_cpu, (min_row, min_col, max_row, max_col), int(color))
                    )
                    all_blobs.append(blob)
        
        return all_blobs
    
    def _count_holes(self, grid: np.ndarray, pixel_coords: List[Tuple[int, int]], bounding_box: Tuple[int, int, int, int], blob_color: int) -> int:
        """Count topological holes within a blob using flood-fill.

        Args:
            grid: Original grid array.
            pixel_coords: Coordinates belonging to the blob (row, col).
            bounding_box: (min_row, min_col, max_row, max_col) of the blob.
            blob_color: Color value of the blob.

        Returns:
            Number of background connected components fully enclosed by the blob.
        """
        min_row, min_col, max_row, max_col = bounding_box
        sub_h = max_row - min_row + 1
        sub_w = max_col - min_col + 1
        # Build binary mask of blob inside bounding box
        mask = np.zeros((sub_h, sub_w), dtype=np.uint8)
        for r, c in pixel_coords:
            mask[r - min_row, c - min_col] = 1  # foreground
        # Background mask (0 where foreground)
        bg = 1 - mask
        visited = np.zeros_like(bg, dtype=bool)
        h, w = bg.shape
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        from collections import deque
        # 1. Mark background connected to border (outside) so they're not holes
        queue = deque()
        for r in range(h):
            for c in (0, w - 1):
                if bg[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if bg[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        # 2. Remaining unvisited background components inside mask are holes.
        holes = 0
        for r in range(h):
            for c in range(w):
                if bg[r, c] and not visited[r, c]:
                    holes += 1
                    visited[r, c] = True
                    queue = deque([(r, c)])
                    while queue:
                        rr, cc = queue.popleft()
                        for dr, dc in offsets:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
        return holes

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
                        holes=self._count_holes(grid, blob_coords, (min_row, min_col, max_row, max_col), color)
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