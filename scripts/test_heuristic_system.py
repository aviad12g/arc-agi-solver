#!/usr/bin/env python3
"""Test script to demonstrate the two-tier heuristic system."""

import numpy as np
import time
from arc_solver.search.heuristics import create_heuristic_system

def main():
    """Demonstrate the heuristic system with sample grids."""
    print("🔍 Testing Two-Tier Heuristic System")
    print("=" * 50)
    
    # Create heuristic system
    heuristic_system = create_heuristic_system(use_tier2=True, tier2_threshold=5.0)
    
    # Test Case 1: Identical grids (should trigger Tier 2)
    print("\n📊 Test Case 1: Identical Grids")
    grid1 = np.array([
        [1, 2, 0],
        [3, 4, 0], 
        [0, 0, 0]
    ], dtype=np.int32)
    
    result = heuristic_system.compute_heuristic(grid1, grid1)
    print(f"Heuristic value: {result.value:.4f}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Features computed: {result.features_computed}")
    
    # Test Case 2: Very different grids (should use only Tier 1)
    print("\n📊 Test Case 2: Very Different Grids")
    grid2 = np.array([
        [5, 6, 7],
        [8, 9, 1],
        [2, 3, 4]
    ], dtype=np.int32)
    
    result = heuristic_system.compute_heuristic(grid1, grid2)
    print(f"Heuristic value: {result.value:.4f}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Features computed: {result.features_computed}")
    
    # Test Case 3: Complex grids with multiple blobs
    print("\n📊 Test Case 3: Complex Multi-Blob Grids")
    complex_grid1 = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ], dtype=np.int32)
    
    complex_grid2 = np.array([
        [2, 2, 0, 1, 1],
        [2, 2, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [4, 4, 0, 3, 3],
        [4, 4, 0, 3, 3]
    ], dtype=np.int32)
    
    result = heuristic_system.compute_heuristic(complex_grid1, complex_grid2)
    print(f"Heuristic value: {result.value:.4f}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Features computed: {result.features_computed}")
    
    # Performance test
    print("\n⚡ Performance Test: 100 Computations")
    start_time = time.perf_counter()
    
    for _ in range(100):
        heuristic_system.compute_heuristic(grid1, grid2)
    
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / 100
    
    print(f"Average computation time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {1/avg_time:.1f} computations/second")
    
    # Statistics
    print("\n📈 System Statistics")
    stats = heuristic_system.get_stats()
    print(f"Tier 1 calls: {stats['tier1_calls']}")
    print(f"Tier 2 calls: {stats['tier2_calls']}")
    print(f"Tier 2 usage rate: {stats['tier2_usage_rate']*100:.1f}%")
    print(f"Tier 1 average time: {stats['tier1_stats']['average_time_us']:.0f}µs")
    
    if 'tier2_stats' in stats:
        print(f"Tier 2 average time: {stats['tier2_stats']['average_time_us']:.0f}µs")
    
    print("\n✅ Heuristic system test completed successfully!")

if __name__ == "__main__":
    main()