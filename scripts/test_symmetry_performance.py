#!/usr/bin/env python3
"""Performance test for symmetry detection system."""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.perception.symmetry import create_symmetry_detector, SymmetryType

def main():
    """Test symmetry detection performance."""
    print("🔍 Testing Symmetry Detection Performance")
    print("=" * 50)
    
    # Create detector
    detector = create_symmetry_detector(max_grid_size=30)
    print(f"✅ Created symmetry detector (max size: {detector.max_grid_size})")
    
    # Test different grid sizes
    test_sizes = [3, 5, 7, 10, 15, 20, 25, 30]
    
    print(f"\n⏱️  Performance Testing:")
    
    for size in test_sizes:
        print(f"\n📏 Testing {size}×{size} grid:")
        
        # Generate test grid with some symmetry
        test_grid = np.random.randint(0, 3, size=(size, size), dtype=np.int32)
        
        # Run multiple trials
        times = []
        symmetry_counts = []
        
        for trial in range(20):
            start_time = time.perf_counter()
            symmetries = detector.detect_symmetries(test_grid)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            symmetry_counts.append(len(symmetries))
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        mean_symmetries = np.mean(symmetry_counts)
        
        print(f"  ⏱️  Mean time: {mean_time*1000000:.2f}µs ± {std_time*1000000:.2f}µs")
        print(f"  📊 Range: {min_time*1000000:.2f}µs - {max_time*1000000:.2f}µs")
        print(f"  🔄 Average symmetries detected: {mean_symmetries:.1f}")
        
        # Check performance requirement (≤10µs target)
        target_time = 0.00001  # 10 microseconds
        if mean_time <= target_time:
            print(f"  ✅ Meets performance target (≤{target_time*1000000:.0f}µs)")
        else:
            print(f"  ⚠️  Performance: {mean_time*1000000:.2f}µs (target: {target_time*1000000:.0f}µs)")
    
    # Test specific symmetric patterns
    print(f"\n🎨 Testing Symmetric Patterns:")
    
    # Test cross pattern (should have multiple symmetries)
    cross_pattern = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.int32)
    
    start_time = time.perf_counter()
    cross_symmetries = detector.detect_symmetries(cross_pattern)
    cross_time = time.perf_counter() - start_time
    
    print(f"\n  🎯 Cross Pattern:")
    print(f"    Time: {cross_time*1000000:.2f}µs")
    print(f"    Symmetries: {len(cross_symmetries)}")
    print(f"    Types: {[sym.value for sym in cross_symmetries]}")
    print(f"    Has rotational: {detector.has_rotational_symmetry(cross_symmetries)}")
    print(f"    Has reflection: {detector.has_reflection_symmetry(cross_symmetries)}")
    
    # Test fully symmetric pattern
    symmetric_pattern = np.array([
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1]
    ], dtype=np.int32)
    
    start_time = time.perf_counter()
    full_symmetries = detector.detect_symmetries(symmetric_pattern)
    full_time = time.perf_counter() - start_time
    
    print(f"\n  🌟 Fully Symmetric Pattern:")
    print(f"    Time: {full_time*1000000:.2f}µs")
    print(f"    Symmetries: {len(full_symmetries)} (should be 8)")
    print(f"    Group order: {detector.get_symmetry_group_order(full_symmetries)}")
    
    # Test transformation performance
    print(f"\n🔄 Testing Transformations:")
    
    test_grid = np.random.randint(0, 4, size=(10, 10), dtype=np.int32)
    
    transform_times = {}
    for sym_type in [SymmetryType.ROTATE_90, SymmetryType.ROTATE_180, 
                     SymmetryType.REFLECT_H, SymmetryType.REFLECT_V]:
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            transformed = detector.apply_symmetry_transform(test_grid, sym_type)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        transform_times[sym_type] = np.mean(times)
        print(f"  {sym_type.value}: {np.mean(times)*1000000:.2f}µs")
    
    # Test canonical form computation
    print(f"\n📐 Testing Canonical Form:")
    
    canonical_times = []
    for _ in range(10):
        start_time = time.perf_counter()
        canonical, transform = detector.get_canonical_form(test_grid)
        end_time = time.perf_counter()
        canonical_times.append(end_time - start_time)
    
    mean_canonical_time = np.mean(canonical_times)
    print(f"  Canonical form computation: {mean_canonical_time*1000000:.2f}µs")
    
    print(f"\n🎉 Symmetry detection performance test complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())