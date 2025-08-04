#!/usr/bin/env python3
"""Quick test script to validate the real ARC validation system works."""

import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.cli.commands import ARCSolver
from arc_solver.core.data_models import Task

def test_solver_basic_functionality():
    """Test that the solver can handle basic tasks."""
    print("🧪 Testing basic solver functionality...")
    
    try:
        # Create a simple test task
        train_input = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_output = np.array([[3, 1], [4, 2]], dtype=np.int32)  # 90-degree rotation
        test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
        
        task = Task(
            task_id="test_rotation",
            train_examples=[(train_input, train_output)],
            test_inputs=[test_input]
        )
        
        # Initialize solver
        solver = ARCSolver()
        
        # Solve task
        result = solver.solve_task(task, timeout=5.0)
        
        print(f"✅ Solver initialized and ran successfully")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Runtime: {result.get('computation_time', 0):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Solver test failed: {e}")
        return False

def test_arc_dataset_loading(dataset_path: str):
    """Test loading ARC dataset files."""
    print(f"🧪 Testing ARC dataset loading from: {dataset_path}")
    
    try:
        # Import the validator class directly
        import sys
        import importlib.util
        
        # Load the validator module
        spec = importlib.util.spec_from_file_location("validate_real_arc_dataset", 
                                                     Path(__file__).parent / "validate_real_arc_dataset.py")
        validator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator_module)
        
        RealARCValidator = validator_module.RealARCValidator
        
        validator = RealARCValidator(dataset_path)
        tasks = validator.load_arc_tasks()
        
        print(f"✅ Successfully loaded {len(tasks)} ARC tasks")
        
        if tasks:
            sample_task = tasks[0]
            print(f"   Sample task ID: {sample_task.task_id}")
            print(f"   Training examples: {len(sample_task.train_examples)}")
            print(f"   Test inputs: {len(sample_task.test_inputs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def main():
    """Run validation tests."""
    print("🚀 Testing Real ARC Validation System")
    print("=" * 50)
    
    # Test 1: Basic solver functionality
    solver_works = test_solver_basic_functionality()
    
    # Test 2: Dataset loading (if path provided)
    dataset_works = True
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        dataset_works = test_arc_dataset_loading(dataset_path)
    else:
        print("⚠️  No dataset path provided, skipping dataset loading test")
        print("   Usage: python3 scripts/test_real_arc_validation.py <dataset_path>")
    
    # Summary
    print("\n" + "=" * 50)
    if solver_works and dataset_works:
        print("✅ All tests passed! Ready to run real ARC validation.")
        print("\nNext steps:")
        print("1. Update the dataset path in scripts/run_real_arc_validation.sh")
        print("2. Run: ./scripts/run_real_arc_validation.sh")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before running validation.")
        return 1

if __name__ == "__main__":
    exit(main())