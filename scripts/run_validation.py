#!/usr/bin/env python3
"""Quick validation runner for Task 12.2 completion.

This script runs a simplified validation to demonstrate that Task 12.2 is working.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.cli.commands import ARCSolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainExample:
    """Training example for ARCSolver compatibility."""
    input: np.ndarray
    output: np.ndarray


@dataclass 
class TestExample:
    """Test example for ARCSolver compatibility."""
    input: np.ndarray
    output: np.ndarray = None


@dataclass
class Task:
    """Task class compatible with ARCSolver."""
    task_id: str
    train: List[TrainExample]
    test: List[TestExample]


def create_simple_rotation_task() -> Task:
    """Create a simple rotation task for testing."""
    # Simple 2x2 input
    input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    # 90-degree rotation
    output_grid = np.array([[3, 1], [4, 2]], dtype=np.int32)
    # Test input
    test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
    
    return Task(
        task_id="simple_rotation",
        train=[TrainExample(input_grid, output_grid)],
        test=[TestExample(test_input)]
    )


def run_quick_validation():
    """Run a quick validation to test the system."""
    logger.info("🚀 Starting Task 12.2 Quick Validation")
    
    try:
        # Initialize solver
        logger.info("Initializing ARCSolver...")
        solver = ARCSolver()
        
        # Create test task
        logger.info("Creating test task...")
        task = create_simple_rotation_task()
        
        # Solve task
        logger.info("Solving task...")
        start_time = time.perf_counter()
        result = solver.solve_task(task, timeout=10.0)
        end_time = time.perf_counter()
        
        runtime = end_time - start_time
        
        # Report results
        logger.info("📊 VALIDATION RESULTS:")
        logger.info(f"  ✅ Task solved successfully: {result.get('success', False)}")
        logger.info(f"  ⏱️  Runtime: {runtime:.3f}s (target: ≤0.5s)")
        logger.info(f"  🎯 Prediction: {result.get('predictions', [None])[0]}")
        
        # Check performance targets
        meets_runtime = runtime <= 0.5
        logger.info(f"  📈 Meets runtime target: {'✅' if meets_runtime else '❌'}")
        
        if result.get('success', False):
            logger.info("🎉 VALIDATION PASSED - System is working!")
            return 0
        else:
            logger.info("❌ VALIDATION FAILED - System needs debugging")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    return run_quick_validation()


if __name__ == "__main__":
    exit(main())