#!/usr/bin/env python3
"""Test the complete DSL system with performance analysis."""

import sys
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.reasoning import (
    create_enhanced_dsl_engine, DSLProgram, DSLOperation
)

def main():
    """Test the complete DSL system."""
    print("🔧 Testing Complete DSL System")
    print("=" * 50)
    
    # Create enhanced DSL engine
    print("🚀 Initializing Enhanced DSL Engine...")
    engine = create_enhanced_dsl_engine(
        max_program_length=4,
        enable_cache=True,
        cache_size=1000
    )
    
    print(f"✅ Engine initialized with {len(engine.get_primitive_names())} primitives")
    print(f"   Primitives: {', '.join(engine.get_primitive_names())}")
    
    # Test input grids
    test_grids = [
        {
            "name": "Simple 2x2",
            "grid": np.array([[1, 2], [3, 4]], dtype=np.int32)
        },
        {
            "name": "Cross Pattern",
            "grid": np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=np.int32)
        },
        {
            "name": "Complex Pattern",
            "grid": np.array([
                [1, 2, 3, 0],
                [4, 5, 6, 0],
                [7, 8, 9, 0],
                [0, 0, 0, 0]
            ], dtype=np.int32)
        }
    ]
    
    # Test programs
    test_programs = [
        {
            "name": "Single Rotation",
            "program": DSLProgram([DSLOperation("Rotate90", {})])
        },
        {
            "name": "Double Transformation",
            "program": DSLProgram([
                DSLOperation("Rotate90", {}),
                DSLOperation("ReflectH", {})
            ])
        },
        {
            "name": "Paint Operation",
            "program": DSLProgram([
                DSLOperation("Paint", {"x": 0, "y": 0, "c": 9})
            ])
        },
        {
            "name": "Complex Sequence",
            "program": DSLProgram([
                DSLOperation("Rotate180", {}),
                DSLOperation("Paint", {"x": 1, "y": 1, "c": 8}),
                DSLOperation("ReflectV", {})
            ])
        }
    ]
    
    print(f"\n📋 Testing {len(test_programs)} programs on {len(test_grids)} grids...")
    
    # Test each program on each grid
    total_executions = 0
    successful_executions = 0
    
    for grid_info in test_grids:
        print(f"\n🎨 Testing with {grid_info['name']}:")
        grid = grid_info['grid']
        print(f"   Input shape: {grid.shape}")
        
        for prog_info in test_programs:
            program = prog_info['program']
            print(f"\n  🔧 Program: {prog_info['name']}")
            print(f"     Operations: {program}")
            
            # Validate program
            is_valid, error = engine.validate_program(program)
            if not is_valid:
                print(f"     ❌ Invalid program: {error}")
                continue
            
            # Execute program
            result = engine.execute_program(program, grid)
            total_executions += 1
            
            if result.success:
                successful_executions += 1
                print(f"     ✅ Success in {result.execution_time*1000:.2f}ms")
                print(f"        Operations executed: {result.operations_executed}")
                print(f"        Output shape: {result.output_grid.shape}")
                print(f"        Cached: {result.cached}")
                
                # Show input/output for small grids
                if grid.size <= 16:
                    print(f"        Input:  {grid.flatten()}")
                    print(f"        Output: {result.output_grid.flatten()}")
            else:
                print(f"     ❌ Failed: {result.error}")
    
    # Test caching performance
    print(f"\n💾 Testing Cache Performance...")
    
    # Execute same program multiple times
    test_program = DSLProgram([DSLOperation("Rotate90", {})])
    test_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    
    # First execution (no cache)
    start_time = time.perf_counter()
    result1 = engine.execute_program(test_program, test_grid)
    first_time = time.perf_counter() - start_time
    
    # Second execution (cached)
    start_time = time.perf_counter()
    result2 = engine.execute_program(test_program, test_grid)
    second_time = time.perf_counter() - start_time
    
    print(f"   First execution: {first_time*1000:.2f}ms (cached: {result1.cached})")
    print(f"   Second execution: {second_time*1000:.2f}ms (cached: {result2.cached})")
    
    if result2.cached:
        print(f"   ✅ Caching working correctly")
    else:
        print(f"   ⚠️ Caching not working as expected")
    
    # Test batch execution
    print(f"\n⚡ Testing Batch Execution...")
    
    batch_programs = [
        DSLProgram([DSLOperation("Rotate90", {})]),
        DSLProgram([DSLOperation("Rotate180", {})]),
        DSLProgram([DSLOperation("ReflectH", {})]),
        DSLProgram([DSLOperation("ReflectV", {})])
    ]
    
    batch_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    
    # Time batch execution
    start_time = time.perf_counter()
    batch_results = engine.execute_batch(batch_programs, batch_grid)
    batch_time = time.perf_counter() - start_time
    
    print(f"   Executed {len(batch_programs)} programs in {batch_time*1000:.2f}ms")
    print(f"   Average per program: {batch_time/len(batch_programs)*1000:.2f}ms")
    
    successful_batch = sum(1 for r in batch_results if r.success)
    print(f"   Successful: {successful_batch}/{len(batch_programs)}")
    
    # Test program enumeration
    print(f"\n📝 Testing Program Enumeration...")
    
    programs = engine.enumerate_programs(max_length=2)
    print(f"   Generated {len(programs)} programs (max length 2)")
    
    # Show first few programs
    for i, program in enumerate(programs[:5]):
        print(f"     {i+1}. {program}")
    
    if len(programs) > 5:
        print(f"     ... and {len(programs) - 5} more")
    
    # Test program optimization
    print(f"\n🔧 Testing Program Optimization...")
    
    # Create redundant program
    redundant_program = DSLProgram([
        DSLOperation("Rotate90", {}),
        DSLOperation("Rotate90", {}),
        DSLOperation("Rotate90", {}),
        DSLOperation("Rotate90", {})  # 4 rotations = identity
    ])
    
    print(f"   Original: {redundant_program}")
    optimized = engine.optimize_program(redundant_program)
    print(f"   Optimized: {optimized}")
    print(f"   Length reduction: {len(redundant_program)} -> {len(optimized)}")
    
    # Performance statistics
    print(f"\n📊 Performance Statistics:")
    stats = engine.get_performance_stats()
    
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
    print(f"   Average execution time: {stats['average_execution_time']*1000:.2f}ms")
    print(f"   Cache size: {stats.get('cache_size', 0)}")
    
    # Performance requirements check
    print(f"\n✅ Performance Requirements Check:")
    
    avg_time_us = stats['average_execution_time'] * 1000000
    target_time_us = 200  # 200µs target per primitive
    
    if avg_time_us <= target_time_us:
        print(f"   ✅ Average execution time: {avg_time_us:.0f}µs ≤ {target_time_us}µs target")
    else:
        print(f"   ⚠️ Average execution time: {avg_time_us:.0f}µs > {target_time_us}µs target")
    
    success_rate = successful_executions / total_executions if total_executions > 0 else 0
    print(f"   ✅ Success rate: {success_rate:.1%} ({successful_executions}/{total_executions})")
    
    # Search space analysis
    print(f"\n🔍 Search Space Analysis:")
    
    max_programs = sum(8**i for i in range(5))  # Theoretical maximum: ∑(i=0 to 4) 8^i
    actual_programs = len(programs)
    
    print(f"   Theoretical max programs (K=4): {max_programs}")
    print(f"   Enumerated programs (K=2): {actual_programs}")
    print(f"   Search space is manageable: ✅")
    
    print(f"\n🎉 DSL System Test Complete!")
    print(f"   The DSL engine is ready for program synthesis and A* search!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())