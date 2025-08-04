#!/usr/bin/env python3
"""Test blob labeling with real ARC data."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.perception.blob_labeling import create_blob_labeler
from arc_solver.integration.io import ARCDataLoader

def main():
    """Test blob labeling with real ARC data."""
    print("🧩 Testing Blob Labeling with Real ARC Data")
    print("=" * 50)
    
    # Load ARC dataset
    dataset_path = "/Users/mazalcohen/Downloads/arc-prize-2025 2"
    
    try:
        loader = ARCDataLoader(dataset_path, use_cache=False)
        print(f"✅ Loaded ARC dataset from {dataset_path}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Create blob labeler
    labeler = create_blob_labeler(use_gpu=False)
    print(f"✅ Created blob labeler (GPU: {labeler.use_gpu})")
    
    # Test on first few tasks
    print(f"\n🔍 Testing blob detection on ARC tasks:")
    
    task_count = 0
    total_time = 0
    total_blobs = 0
    
    for task_id, task in loader.iter_tasks():
        if task_count >= 5:  # Test first 5 tasks
            break
        
        print(f"\n📋 Task {task_id}:")
        
        # Test on training examples
        for i, (input_grid, output_grid) in enumerate(task.train_examples):
            # Test input grid
            input_blobs, input_time = labeler.label_blobs(input_grid)
            output_blobs, output_time = labeler.label_blobs(output_grid)
            
            total_time += input_time + output_time
            total_blobs += len(input_blobs) + len(output_blobs)
            
            print(f"  Example {i}: Input {input_grid.shape} -> {len(input_blobs)} blobs ({input_time*1000:.2f}ms)")
            print(f"             Output {output_grid.shape} -> {len(output_blobs)} blobs ({output_time*1000:.2f}ms)")
            
            # Show blob details for first example
            if i == 0:
                print(f"    Input blob colors: {[blob.color for blob in input_blobs]}")
                print(f"    Output blob colors: {[blob.color for blob in output_blobs]}")
                
                # Show blob properties for first blob
                if input_blobs:
                    first_blob = input_blobs[0]
                    print(f"    First blob: color={first_blob.color}, area={first_blob.area}, "
                          f"bbox={first_blob.bounding_box}, center={first_blob.center_of_mass}")
        
        task_count += 1
    
    # Summary statistics
    avg_time = total_time / (task_count * 2) if task_count > 0 else 0  # 2 grids per example
    avg_blobs = total_blobs / (task_count * 2) if task_count > 0 else 0
    
    print(f"\n📊 Summary Statistics:")
    print(f"  Tasks tested: {task_count}")
    print(f"  Total processing time: {total_time*1000:.2f}ms")
    print(f"  Average time per grid: {avg_time*1000:.2f}ms")
    print(f"  Total blobs detected: {total_blobs}")
    print(f"  Average blobs per grid: {avg_blobs:.1f}")
    
    if avg_time <= 0.002:
        print(f"  ✅ Average performance meets target (≤2ms)")
    else:
        print(f"  ⚠️  Average performance: {avg_time*1000:.2f}ms")
    
    print(f"\n🎉 ARC blob labeling test complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())