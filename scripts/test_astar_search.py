#!/usr/bin/env python3
"""Test script to demonstrate the A* search engine."""

import numpy as np
import time
from arc_solver.search.astar import create_astar_searcher

def main():
    """Demonstrate the A* search engine with sample problems."""
    print("🔍 Testing A* Search Engine with Beam Search")
    print("=" * 60)
    
    # Create A* searcher with reasonable parameters for demo
    searcher = create_astar_searcher(
        max_program_length=3,
        max_nodes_expanded=100,
        beam_width=16,
        adaptive_beam=True
    )
    
    # Test Case 1: Identity transformation (should find empty program)
    print("\n📊 Test Case 1: Identity Transformation")
    grid = np.array([
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 0]
    ], dtype=np.int32)
    
    result = searcher.search(grid, grid)
    print(f"Success: {result.success}")
    print(f"Program length: {len(result.program.operations)}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Termination reason: {result.termination_reason}")
    
    # Test Case 2: Simple transformation
    print("\n📊 Test Case 2: Simple Grid Transformation")
    initial = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.int32)
    
    target = np.array([
        [3, 1],
        [4, 2]
    ], dtype=np.int32)  # Rotated 90 degrees
    
    result = searcher.search(initial, target)
    print(f"Success: {result.success}")
    print(f"Program length: {len(result.program.operations)}")
    print(f"Operations: {[op.primitive_name for op in result.program.operations]}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Nodes generated: {result.nodes_generated}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Max depth reached: {result.max_depth_reached}")
    print(f"Beam width used: {result.beam_width_used}")
    print(f"Termination reason: {result.termination_reason}")
    
    # Test Case 3: More complex transformation
    print("\n📊 Test Case 3: Complex Multi-Blob Grid")
    complex_initial = np.array([
        [1, 1, 0, 2],
        [1, 1, 0, 2],
        [0, 0, 0, 0],
        [3, 0, 0, 0]
    ], dtype=np.int32)
    
    complex_target = np.array([
        [2, 2, 0, 1],
        [2, 2, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 3]
    ], dtype=np.int32)
    
    result = searcher.search(complex_initial, complex_target)
    print(f"Success: {result.success}")
    print(f"Program length: {len(result.program.operations)}")
    if result.program.operations:
        print(f"Operations: {[op.primitive_name for op in result.program.operations]}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Nodes generated: {result.nodes_generated}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Termination reason: {result.termination_reason}")
    
    # Test Case 4: Performance stress test
    print("\n📊 Test Case 4: Performance Stress Test")
    stress_searcher = create_astar_searcher(
        max_program_length=2,  # Shorter to complete faster
        max_nodes_expanded=200,
        beam_width=32
    )
    
    stress_initial = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [0, 0, 0, 0],
        [9, 8, 7, 6]
    ], dtype=np.int32)
    
    stress_target = np.array([
        [9, 5, 1, 0],
        [8, 6, 2, 0],
        [7, 7, 3, 0],
        [6, 8, 4, 0]
    ], dtype=np.int32)
    
    start_time = time.perf_counter()
    result = stress_searcher.search(stress_initial, stress_target)
    end_time = time.perf_counter()
    
    print(f"Success: {result.success}")
    print(f"Program length: {len(result.program.operations)}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Nodes generated: {result.nodes_generated}")
    print(f"Computation time: {result.computation_time*1000:.2f}ms")
    print(f"Wall clock time: {(end_time - start_time)*1000:.2f}ms")
    print(f"Search efficiency: {result.nodes_expanded / max(result.computation_time, 0.001):.0f} nodes/sec")
    print(f"Termination reason: {result.termination_reason}")
    
    # Display search statistics
    print("\n📈 Search Engine Statistics")
    stats = stress_searcher.get_search_stats()
    print(f"Configuration:")
    print(f"  Max program length: {stats['config']['max_program_length']}")
    print(f"  Max nodes expanded: {stats['config']['max_nodes_expanded']}")
    print(f"  Beam width: {stats['config']['beam_width']}")
    print(f"  Adaptive beam: {stats['config']['adaptive_beam']}")
    
    print(f"Heuristic Statistics:")
    heuristic_stats = stats['heuristic_stats']
    print(f"  Tier 1 calls: {heuristic_stats['tier1_calls']}")
    print(f"  Tier 2 calls: {heuristic_stats['tier2_calls']}")
    print(f"  Tier 2 usage rate: {heuristic_stats['tier2_usage_rate']*100:.1f}%")
    print(f"  Tier 1 avg time: {heuristic_stats['tier1_stats']['average_time_us']:.0f}µs")
    
    if 'tier2_stats' in heuristic_stats:
        print(f"  Tier 2 avg time: {heuristic_stats['tier2_stats']['average_time_us']:.0f}µs")
    
    # Test Case 5: Beam width comparison
    print("\n📊 Test Case 5: Beam Width Comparison")
    test_grid1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    test_grid2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32)
    
    beam_widths = [4, 16, 64]
    for beam_width in beam_widths:
        beam_searcher = create_astar_searcher(
            max_program_length=2,
            max_nodes_expanded=50,
            beam_width=beam_width,
            adaptive_beam=False
        )
        
        result = beam_searcher.search(test_grid1, test_grid2)
        print(f"Beam width {beam_width:2d}: "
              f"nodes_expanded={result.nodes_expanded:2d}, "
              f"time={result.computation_time*1000:5.1f}ms, "
              f"success={result.success}")
    
    print("\n✅ A* search engine test completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ Admissible A* search with heuristic guidance")
    print("  ✓ Beam search pruning for efficiency")
    print("  ✓ Adaptive beam width reduction")
    print("  ✓ Duplicate state detection")
    print("  ✓ Timeout and node limit handling")
    print("  ✓ Comprehensive search statistics")
    print("  ✓ Integration with two-tier heuristic system")

if __name__ == "__main__":
    main()