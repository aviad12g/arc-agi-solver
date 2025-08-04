#!/usr/bin/env python3
"""Simple test of blob labeling with ARC-like patterns."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.perception.blob_labeling import create_blob_labeler

def main():
    """Test blob labeling with ARC-like patterns."""
    print("🧩 Testing Blob Labeling with ARC-like Patterns")
    print("=" * 50)
    
    # Create blob labeler
    labeler = create_blob_labeler(use_gpu=False)
    print(f"✅ Created blob labeler (GPU: {labeler.use_gpu})")
    
    # Test patterns similar to ARC puzzles
    test_patterns = [
        {
            "name": "Cross Pattern",
            "grid": np.array([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=np.int32)
        },
        {
            "name": "Separate Objects",
            "grid": np.array([
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 0, 0, 0, 4],
                [3, 0, 0, 4, 4]
            ], dtype=np.int32)
        },
        {
            "name": "Complex Pattern",
            "grid": np.array([
                [1, 2, 1, 0, 3, 3],
                [2, 2, 2, 0, 3, 0],
                [1, 2, 1, 0, 0, 0],
                [0, 0, 0, 4, 4, 4],
                [5, 5, 0, 4, 0, 4],
                [5, 0, 0, 4, 4, 4]
            ], dtype=np.int32)
        }
    ]
    
    total_time = 0
    total_blobs = 0
    
    for pattern in test_patterns:
        print(f"\n🎨 Testing {pattern['name']}:")
        grid = pattern['grid']
        
        # Test blob detection
        blobs, processing_time = labeler.label_blobs(grid)
        total_time += processing_time
        total_blobs += len(blobs)
        
        print(f"  Grid size: {grid.shape}")
        print(f"  Processing time: {processing_time*1000:.2f}ms")
        print(f"  Blobs detected: {len(blobs)}")
        
        # Show blob details
        blob_colors = {}
        for blob in blobs:
            if blob.color not in blob_colors:
                blob_colors[blob.color] = []
            blob_colors[blob.color].append(blob)
        
        for color, color_blobs in blob_colors.items():
            total_area = sum(blob.area for blob in color_blobs)
            print(f"    Color {color}: {len(color_blobs)} blob(s), total area: {total_area}")
            
            # Show details of first blob of this color
            first_blob = color_blobs[0]
            print(f"      First blob: area={first_blob.area}, bbox={first_blob.bounding_box}")
        
        # Test adjacency graph
        adjacency = labeler.get_blob_adjacency_graph(blobs, grid.shape)
        total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        print(f"    Adjacency graph: {len(adjacency)} nodes, {total_edges} edges")
        
        # Performance check
        if processing_time <= 0.002:
            print(f"    ✅ Meets performance target")
        else:
            print(f"    ⚠️  Performance: {processing_time*1000:.2f}ms")
    
    # Summary
    avg_time = total_time / len(test_patterns)
    avg_blobs = total_blobs / len(test_patterns)
    
    print(f"\n📊 Summary:")
    print(f"  Patterns tested: {len(test_patterns)}")
    print(f"  Total time: {total_time*1000:.2f}ms")
    print(f"  Average time per pattern: {avg_time*1000:.2f}ms")
    print(f"  Total blobs: {total_blobs}")
    print(f"  Average blobs per pattern: {avg_blobs:.1f}")
    
    print(f"\n🎉 Blob labeling test complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())