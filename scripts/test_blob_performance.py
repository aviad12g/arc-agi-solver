#!/usr/bin/env python3
"""Performance test for blob labeling engine."""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.perception.blob_labeling import create_blob_labeler

def main():
    """Test blob labeling performance."""
    print("🔬 Testing Blob Labeling Performance")
    print("=" * 50)
    
    # Create labeler
    labeler = create_blob_labeler(use_gpu=False)  # Test CPU performance first
    
    # Test different grid sizes
    test_sizes = [
        (10, 10),
        (20, 20),
        (30, 30),  # Maximum size for 2ms requirement
    ]
    
    for height, width in test_sizes:
        print(f"\n📏 Testing {height}x{width} grid:")
        
        # Generate test grid with multiple colors
        test_grid = np.random.randint(0, 5, size=(height, width), dtype=np.int32)
        
        # Run multiple trials
        times = []
        blob_counts = []
        
        for trial in range(10):
            blobs, processing_time = labeler.label_blobs(test_grid)
            times.append(processing_time)
            blob_counts.append(len(blobs))
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        mean_blobs = np.mean(blob_counts)
        
        print(f"  ⏱️  Mean time: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  📊 Range: {min_time*1000:.2f}ms - {max_time*1000:.2f}ms")
        print(f"  🔵 Average blobs detected: {mean_blobs:.1f}")
        
        # Check performance requirement for 30x30 grids
        if height <= 30 and width <= 30:
            target_time = 0.002  # 2ms
            if mean_time <= target_time:
                print(f"  ✅ Meets performance target (≤{target_time*1000:.0f}ms)")
            else:
                print(f"  ❌ Exceeds performance target ({mean_time*1000:.2f}ms > {target_time*1000:.0f}ms)")
        else:
            print(f"  ℹ️  No performance requirement for this size")
    
    # Test with GPU if available
    print(f"\n🚀 Testing GPU Performance:")
    gpu_labeler = create_blob_labeler(use_gpu=True)
    
    if gpu_labeler.use_gpu:
        print("  GPU acceleration enabled")
        test_grid = np.random.randint(0, 5, size=(30, 30), dtype=np.int32)
        
        gpu_times = []
        for trial in range(10):
            blobs, processing_time = gpu_labeler.label_blobs(test_grid)
            gpu_times.append(processing_time)
        
        gpu_mean = np.mean(gpu_times)
        print(f"  ⏱️  GPU mean time: {gpu_mean*1000:.2f}ms")
        
        if gpu_mean <= 0.002:
            print(f"  ✅ GPU meets performance target")
        else:
            print(f"  ❌ GPU exceeds performance target")
    else:
        print("  GPU not available, using CPU fallback")
    
    print(f"\n🎯 Performance Test Complete!")

if __name__ == "__main__":
    main()