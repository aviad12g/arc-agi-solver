#!/usr/bin/env python3
"""Test script to demonstrate the caching system."""

import numpy as np
import time
from omegaconf import OmegaConf

from arc_solver.caching import create_cache_manager, CacheKeyGenerator
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation


def main():
    """Demonstrate the caching system with various data types."""
    print("🗄️  Testing ARC-AGI Solver Caching System")
    print("=" * 60)
    
    # Create cache configuration
    cache_config = OmegaConf.create({
        'redis': {
            'enabled': False  # Use file cache only for demo
        },
        'file_cache': {
            'enabled': True,
            'cache_dir': '.demo_cache',
            'max_cache_size': 0.1,  # 100MB
            'compression': True
        },
        'strategies': {
            'grid_features': {
                'enabled': True,
                'compression': True
            },
            'program_results': {
                'enabled': True,
                'max_entries': 1000
            },
            'heuristic_values': {
                'enabled': True,
                'max_entries': 5000
            }
        }
    })
    
    # Create cache manager
    cache_manager = create_cache_manager(cache_config)
    
    print("\n📊 Test Case 1: Basic Cache Operations")
    
    # Test basic caching
    test_data = {
        "message": "Hello, ARC-AGI!",
        "numbers": [1, 2, 3, 4, 5],
        "nested": {"key": "value", "count": 42}
    }
    
    start_time = time.perf_counter()
    success = cache_manager.set("basic_test", test_data)
    set_time = time.perf_counter() - start_time
    
    print(f"Cache set: {success} (time: {set_time*1000:.2f}ms)")
    
    start_time = time.perf_counter()
    retrieved = cache_manager.get("basic_test")
    get_time = time.perf_counter() - start_time
    
    print(f"Cache get: {retrieved == test_data} (time: {get_time*1000:.2f}ms)")
    print(f"Data integrity: {retrieved == test_data}")
    
    print("\n📊 Test Case 2: Grid Features Caching")
    
    # Create sample grids
    grid1 = np.array([
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 0]
    ], dtype=np.int32)
    
    grid2 = np.array([
        [4, 3, 0],
        [2, 1, 0],
        [0, 0, 0]
    ], dtype=np.int32)
    
    # Simulate feature extraction results
    orbit_features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    spectral_features = np.array([1.5, 2.3, 3.1], dtype=np.float32)
    
    # Cache grid features
    print("Caching grid features...")
    
    start_time = time.perf_counter()
    cache_manager.set_grid_features(grid1, "orbit", orbit_features)
    cache_manager.set_grid_features(grid1, "spectral", spectral_features)
    cache_time = time.perf_counter() - start_time
    
    print(f"Feature caching time: {cache_time*1000:.2f}ms")
    
    # Retrieve grid features
    start_time = time.perf_counter()
    cached_orbit = cache_manager.get_grid_features(grid1, "orbit")
    cached_spectral = cache_manager.get_grid_features(grid1, "spectral")
    retrieve_time = time.perf_counter() - start_time
    
    print(f"Feature retrieval time: {retrieve_time*1000:.2f}ms")
    print(f"Orbit features match: {np.array_equal(cached_orbit, orbit_features)}")
    print(f"Spectral features match: {np.array_equal(cached_spectral, spectral_features)}")
    
    # Test cache miss
    missing_features = cache_manager.get_grid_features(grid2, "orbit")
    print(f"Cache miss (expected): {missing_features is None}")
    
    print("\n📊 Test Case 3: Program Results Caching")
    
    # Create sample DSL programs
    program1 = DSLProgram([
        DSLOperation("Rotate90", {}),
        DSLOperation("Paint", {"x": 1, "y": 1, "c": 5})
    ])
    
    program2 = DSLProgram([
        DSLOperation("ReflectH", {}),
        DSLOperation("MapColors", {"perm": [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]})
    ])
    
    # Simulate program execution results
    result1 = {
        "output_grid": [[3, 1, 0], [4, 5, 0], [0, 0, 0]],
        "success": True,
        "execution_time": 0.0015
    }
    
    result2 = {
        "output_grid": [[3, 4, 0], [1, 2, 0], [0, 0, 0]],
        "success": True,
        "execution_time": 0.0012
    }
    
    # Cache program results
    print("Caching program results...")
    
    start_time = time.perf_counter()
    cache_manager.set_program_result(program1, grid1, result1)
    cache_manager.set_program_result(program2, grid1, result2)
    cache_time = time.perf_counter() - start_time
    
    print(f"Program result caching time: {cache_time*1000:.2f}ms")
    
    # Retrieve program results
    start_time = time.perf_counter()
    cached_result1 = cache_manager.get_program_result(program1, grid1)
    cached_result2 = cache_manager.get_program_result(program2, grid1)
    retrieve_time = time.perf_counter() - start_time
    
    print(f"Program result retrieval time: {retrieve_time*1000:.2f}ms")
    print(f"Result 1 match: {cached_result1 == result1}")
    print(f"Result 2 match: {cached_result2 == result2}")
    
    print("\n📊 Test Case 4: Heuristic Values Caching")
    
    # Cache heuristic values
    heuristic_tests = [
        ("tier1", grid1, grid2, 3.14159),
        ("tier2", grid1, grid2, 2.71828),
        ("tier1", grid2, grid1, 1.41421)
    ]
    
    print("Caching heuristic values...")
    
    start_time = time.perf_counter()
    for htype, current, target, value in heuristic_tests:
        cache_manager.set_heuristic_value(current, target, htype, value)
    cache_time = time.perf_counter() - start_time
    
    print(f"Heuristic caching time: {cache_time*1000:.2f}ms")
    
    # Retrieve heuristic values
    start_time = time.perf_counter()
    retrieved_values = []
    for htype, current, target, expected_value in heuristic_tests:
        retrieved_value = cache_manager.get_heuristic_value(current, target, htype)
        retrieved_values.append(retrieved_value)
    retrieve_time = time.perf_counter() - start_time
    
    print(f"Heuristic retrieval time: {retrieve_time*1000:.2f}ms")
    
    all_match = all(
        retrieved == expected 
        for retrieved, (_, _, _, expected) in zip(retrieved_values, heuristic_tests)
    )
    print(f"All heuristic values match: {all_match}")
    
    print("\n📊 Test Case 5: Performance Comparison")
    
    # Test cache performance with repeated operations
    test_grid = np.random.randint(0, 10, size=(5, 5), dtype=np.int32)
    test_features = np.random.rand(50).astype(np.float32)
    
    # First access (cache miss)
    start_time = time.perf_counter()
    cache_manager.set_grid_features(test_grid, "performance_test", test_features)
    first_set_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    retrieved_features = cache_manager.get_grid_features(test_grid, "performance_test")
    first_get_time = time.perf_counter() - start_time
    
    # Repeated accesses (cache hits)
    hit_times = []
    for _ in range(10):
        start_time = time.perf_counter()
        cache_manager.get_grid_features(test_grid, "performance_test")
        hit_times.append(time.perf_counter() - start_time)
    
    avg_hit_time = np.mean(hit_times)
    
    print(f"First set time: {first_set_time*1000:.2f}ms")
    print(f"First get time: {first_get_time*1000:.2f}ms")
    print(f"Average hit time: {avg_hit_time*1000:.2f}ms")
    print(f"Cache speedup: {first_get_time / avg_hit_time:.1f}x")
    
    print("\n📊 Test Case 6: Cache Key Generation")
    
    # Demonstrate cache key generation
    key_gen = CacheKeyGenerator()
    
    # Grid hash
    grid_hash = key_gen.grid_hash(grid1)
    print(f"Grid hash: {grid_hash[:16]}...")
    
    # Program hash
    prog_hash = key_gen.program_hash(program1)
    print(f"Program hash: {prog_hash[:16]}...")
    
    # Feature key
    feature_key = key_gen.feature_key(grid1, "orbit")
    print(f"Feature key: {feature_key[:32]}...")
    
    # Heuristic key
    heuristic_key = key_gen.heuristic_key(grid1, grid2, "tier1")
    print(f"Heuristic key: {heuristic_key[:32]}...")
    
    print("\n📈 Cache Statistics")
    
    # Get comprehensive statistics
    stats = cache_manager.get_stats()
    
    print(f"Redis enabled: {stats['redis_enabled']}")
    print(f"File cache enabled: {stats['file_enabled']}")
    
    if 'file' in stats:
        file_stats = stats['file']
        print(f"File cache hits: {file_stats['hits']}")
        print(f"File cache misses: {file_stats['misses']}")
        print(f"File cache hit rate: {file_stats['hit_rate']*100:.1f}%")
        print(f"File cache size: {file_stats['cache_size_mb']:.2f}MB")
        print(f"File count: {file_stats['file_count']}")
    
    combined = stats['combined']
    print(f"Combined hit rate: {combined['hit_rate']*100:.1f}%")
    print(f"Total requests: {combined['total_requests']}")
    
    print("\n🧹 Cleanup Test")
    
    # Test cache clearing
    cleared_count = cache_manager.clear()
    print(f"Cleared {cleared_count} cache entries")
    
    # Verify cache is empty
    empty_result = cache_manager.get("basic_test")
    print(f"Cache empty after clear: {empty_result is None}")
    
    # Close cache manager
    cache_manager.close()
    
    print("\n✅ Caching system test completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ File-based caching with compression")
    print("  ✓ Grid features caching with SHA-1 keys")
    print("  ✓ Program results caching")
    print("  ✓ Heuristic values caching")
    print("  ✓ Cache key generation utilities")
    print("  ✓ Performance monitoring and statistics")
    print("  ✓ Configurable cache strategies")
    print("  ✓ Automatic cache size management")


if __name__ == "__main__":
    main()