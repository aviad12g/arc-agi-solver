#!/usr/bin/env python3
"""Test integration of symmetry detection with blob labeling."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.perception.blob_labeling import create_blob_labeler
from arc_solver.perception.symmetry import create_symmetry_detector, SymmetryType

def main():
    """Test integration of blob labeling and symmetry detection."""
    print("🔗 Testing Blob Labeling + Symmetry Detection Integration")
    print("=" * 60)
    
    # Create components
    blob_labeler = create_blob_labeler(use_gpu=False)
    symmetry_detector = create_symmetry_detector()
    
    print(f"✅ Created blob labeler and symmetry detector")
    
    # Test patterns
    test_patterns = [
        {
            "name": "Symmetric Cross",
            "grid": np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=np.int32)
        },
        {
            "name": "Two Symmetric Squares",
            "grid": np.array([
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [2, 2, 0, 1, 1],
                [2, 2, 0, 1, 1]
            ], dtype=np.int32)
        },
        {
            "name": "Asymmetric Pattern",
            "grid": np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ], dtype=np.int32)
        }
    ]
    
    for pattern in test_patterns:
        print(f"\n🎨 Testing {pattern['name']}:")
        grid = pattern['grid']
        
        # 1. Detect blobs
        blobs, blob_time = blob_labeler.label_blobs(grid)
        print(f"  🔵 Blob detection: {len(blobs)} blobs in {blob_time*1000:.2f}ms")
        
        # Show blob details
        for i, blob in enumerate(blobs):
            print(f"    Blob {i}: color={blob.color}, area={blob.area}, center={blob.center_of_mass}")
        
        # 2. Detect symmetries
        symmetries = symmetry_detector.detect_symmetries(grid)
        print(f"  🔄 Symmetry detection: {len(symmetries)} symmetries")
        print(f"    Types: {[sym.value for sym in symmetries]}")
        print(f"    Has rotational: {symmetry_detector.has_rotational_symmetry(symmetries)}")
        print(f"    Has reflection: {symmetry_detector.has_reflection_symmetry(symmetries)}")
        
        # 3. Test transformations preserve blob structure
        if len(symmetries) > 1:  # More than just identity
            print(f"  🔄 Testing transformation preservation:")
            
            for sym_type in list(symmetries)[:3]:  # Test first 3 symmetries
                if sym_type == SymmetryType.IDENTITY:
                    continue
                
                # Apply transformation
                transformed_grid = symmetry_detector.apply_symmetry_transform(grid, sym_type)
                
                # Detect blobs in transformed grid
                transformed_blobs, _ = blob_labeler.label_blobs(transformed_grid)
                
                print(f"    {sym_type.value}: {len(blobs)} -> {len(transformed_blobs)} blobs")
                
                # Check if blob count is preserved (should be for true symmetries)
                if len(blobs) == len(transformed_blobs):
                    print(f"      ✅ Blob count preserved")
                else:
                    print(f"      ⚠️  Blob count changed")
        
        # 4. Get canonical form
        canonical_grid, canonical_transform = symmetry_detector.get_canonical_form(grid)
        canonical_blobs, _ = blob_labeler.label_blobs(canonical_grid)
        
        print(f"  📐 Canonical form: applied {canonical_transform.value}")
        print(f"    Original blobs: {len(blobs)}, Canonical blobs: {len(canonical_blobs)}")
        
        # 5. Test adjacency graph with symmetries
        adjacency = blob_labeler.get_blob_adjacency_graph(blobs, grid.shape)
        total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        print(f"  🕸️  Blob adjacency: {len(adjacency)} nodes, {total_edges} edges")
    
    # Test performance of combined operations
    print(f"\n⏱️  Combined Performance Test:")
    
    test_grid = np.random.randint(0, 4, size=(10, 10), dtype=np.int32)
    
    # Time combined operations
    import time
    
    times = []
    for _ in range(10):
        start_time = time.perf_counter()
        
        # Full pipeline
        blobs, _ = blob_labeler.label_blobs(test_grid)
        symmetries = symmetry_detector.detect_symmetries(test_grid)
        adjacency = blob_labeler.get_blob_adjacency_graph(blobs, test_grid.shape)
        canonical, _ = symmetry_detector.get_canonical_form(test_grid)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    mean_time = np.mean(times)
    print(f"  Combined pipeline: {mean_time*1000:.2f}ms average")
    print(f"  Components: blob detection + symmetry detection + adjacency + canonical form")
    
    print(f"\n🎉 Integration test complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())