#!/usr/bin/env python3
"""Real ARC Dataset Validation Script - Task 12.2 Final Validation.

This script validates the ARC solver against the actual ARC-AGI dataset
to measure real performance on competition tasks.

Performance Targets:
- ≥35% accuracy on ARC-AGI-2 public split (>143 out of 410 tasks)
- ≤0.5s median runtime per puzzle
- ≤5s 95th percentile runtime
- ≤2GB GPU memory usage
- ≤6GB RAM memory usage
"""

import sys
import time
import json
import logging
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.cli.commands import ARCSolver
from arc_solver.core.data_models import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RealARCValidationResult:
    """Container for real ARC validation results."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    accuracy: float
    median_runtime: float
    percentile_95_runtime: float
    max_memory_mb: float
    runtimes: List[float]
    detailed_results: List[Dict[str, Any]]
    meets_accuracy_target: bool
    meets_runtime_targets: bool
    overall_success: bool


class RealARCValidator:
    """Validator for real ARC dataset."""
    
    def __init__(self, arc_dataset_path: str):
        self.arc_dataset_path = Path(arc_dataset_path)
        self.solver = ARCSolver()
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance targets
        self.targets = {
            'accuracy': 0.35,           # ≥35% accuracy
            'median_runtime': 0.5,      # ≤0.5s median runtime
            'percentile_95_runtime': 5.0,  # ≤5s 95th percentile
            'max_ram_memory_gb': 6.0    # ≤6GB RAM memory
        }
        
        logger.info(f"Real ARC validation targets: {self.targets}")
        logger.info(f"Initial memory usage: {self.initial_memory:.1f}MB")
        logger.info(f"ARC dataset path: {self.arc_dataset_path}")
    
    def load_arc_tasks(self, dataset_file: str = "arc-agi_training_challenges.json") -> List[Task]:
        """Load real ARC tasks from the dataset."""
        dataset_path = self.arc_dataset_path / dataset_file
        
        if not dataset_path.exists():
            # Try alternative file names
            alternatives = [
                "arc-agi_evaluation_challenges.json",
                "arc-agi_training_challenges.json", 
                "training_challenges.json",
                "evaluation_challenges.json",
                "challenges.json"
            ]
            
            for alt in alternatives:
                alt_path = self.arc_dataset_path / alt
                if alt_path.exists():
                    dataset_path = alt_path
                    logger.info(f"Using dataset file: {alt}")
                    break
            else:
                raise FileNotFoundError(f"No ARC dataset found in {self.arc_dataset_path}")
        
        logger.info(f"Loading ARC tasks from: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            arc_data = json.load(f)
        
        tasks = []
        for task_id, task_data in arc_data.items():
            try:
                # Convert ARC format to our Task format
                train_examples = []
                for example in task_data['train']:
                    input_grid = np.array(example['input'], dtype=np.int32)
                    output_grid = np.array(example['output'], dtype=np.int32)
                    train_examples.append((input_grid, output_grid))
                
                test_inputs = []
                for test_case in task_data['test']:
                    test_input = np.array(test_case['input'], dtype=np.int32)
                    test_inputs.append(test_input)
                
                task = Task(
                    task_id=task_id,
                    train_examples=train_examples,
                    test_inputs=test_inputs
                )
                tasks.append(task)
                
            except Exception as e:
                logger.warning(f"Failed to load task {task_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(tasks)} real ARC tasks")
        return tasks
    
    def validate_on_real_arc(self, max_tasks: Optional[int] = None, 
                           timeout_per_task: float = 10.0) -> RealARCValidationResult:
        """Validate the solver on real ARC tasks."""
        logger.info("Starting validation on real ARC dataset...")
        
        # Load real ARC tasks
        tasks = self.load_arc_tasks()
        
        if max_tasks and len(tasks) > max_tasks:
            tasks = tasks[:max_tasks]
            logger.info(f"Limited to first {max_tasks} tasks for testing")
        
        logger.info(f"Validating on {len(tasks)} real ARC tasks...")
        logger.info(f"Timeout per task: {timeout_per_task}s")
        
        results = []
        runtimes = []
        successful_tasks = 0
        failed_tasks = 0
        max_memory_usage = self.initial_memory
        
        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}: {task.task_id}")
            
            # Monitor memory before task
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            max_memory_usage = max(max_memory_usage, current_memory)
            
            # Solve task with timing
            start_time = time.perf_counter()
            try:
                result = self.solver.solve_task(task, timeout=timeout_per_task)
                end_time = time.perf_counter()
                runtime = end_time - start_time
                runtimes.append(runtime)
                
                # Determine success
                success = result.get('success', False)
                if success:
                    successful_tasks += 1
                    logger.info(f"✅ Task {task.task_id} solved in {runtime:.3f}s")
                else:
                    failed_tasks += 1
                    logger.info(f"❌ Task {task.task_id} failed in {runtime:.3f}s")
                
                # Store detailed result
                task_result = {
                    'task_id': task.task_id,
                    'success': success,
                    'runtime': runtime,
                    'timeout': runtime >= timeout_per_task * 0.95,
                    'prediction': result.get('predictions', [None])[0],
                    'search_stats': result.get('search_stats', {}),
                    'memory_usage_mb': current_memory
                }
                results.append(task_result)
                
                # Log progress every 10 tasks
                if (i + 1) % 10 == 0:
                    current_accuracy = successful_tasks / (i + 1)
                    current_median = np.median(runtimes) if runtimes else 0
                    logger.info(f"Progress: {i+1}/{len(tasks)} tasks, "
                               f"accuracy: {current_accuracy:.2%}, "
                               f"median runtime: {current_median:.3f}s")
                
            except Exception as e:
                end_time = time.perf_counter()
                runtime = end_time - start_time
                runtimes.append(runtime)
                failed_tasks += 1
                logger.error(f"❌ Task {task.task_id} failed with error: {e}")
                
                task_result = {
                    'task_id': task.task_id,
                    'success': False,
                    'runtime': runtime,
                    'timeout': False,
                    'error': str(e),
                    'memory_usage_mb': current_memory
                }
                results.append(task_result)
        
        # Calculate final metrics
        accuracy = successful_tasks / len(tasks) if tasks else 0.0
        median_runtime = np.median(runtimes) if runtimes else 0.0
        percentile_95_runtime = np.percentile(runtimes, 95) if runtimes else 0.0
        
        # Check targets
        meets_accuracy = accuracy >= self.targets['accuracy']
        meets_median_runtime = median_runtime <= self.targets['median_runtime']
        meets_95th_runtime = percentile_95_runtime <= self.targets['percentile_95_runtime']
        meets_memory = max_memory_usage <= self.targets['max_ram_memory_gb'] * 1024
        
        meets_runtime_targets = meets_median_runtime and meets_95th_runtime
        overall_success = meets_accuracy and meets_runtime_targets and meets_memory
        
        validation_result = RealARCValidationResult(
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            accuracy=accuracy,
            median_runtime=median_runtime,
            percentile_95_runtime=percentile_95_runtime,
            max_memory_mb=max_memory_usage,
            runtimes=runtimes,
            detailed_results=results,
            meets_accuracy_target=meets_accuracy,
            meets_runtime_targets=meets_runtime_targets,
            overall_success=overall_success
        )
        
        return validation_result
    
    def print_validation_report(self, result: RealARCValidationResult) -> None:
        """Print detailed validation report."""
        print("\n" + "="*80)
        print("REAL ARC DATASET VALIDATION REPORT - TASK 12.2")
        print("="*80)
        
        # Overall status
        status = "✅ PASSED" if result.overall_success else "❌ FAILED"
        print(f"Overall Validation Status: {status}")
        
        # Metrics summary
        print(f"\nPerformance Metrics:")
        print(f"  Total Tasks: {result.total_tasks}")
        print(f"  Successful: {result.successful_tasks}")
        print(f"  Failed: {result.failed_tasks}")
        
        # Accuracy validation
        accuracy_status = "✅" if result.meets_accuracy_target else "❌"
        print(f"\nAccuracy Validation:")
        print(f"  {accuracy_status} Accuracy: {result.accuracy:.2%} (target: ≥{self.targets['accuracy']:.0%})")
        print(f"  Tasks solved: {result.successful_tasks}/{result.total_tasks}")
        
        if result.meets_accuracy_target:
            print(f"  🎯 EXCEEDS ARC-AGI COMPETITION REQUIREMENT!")
        else:
            shortfall = self.targets['accuracy'] - result.accuracy
            needed_tasks = int(shortfall * result.total_tasks)
            print(f"  ⚠️  Need {needed_tasks} more successful tasks to reach target")
        
        # Runtime validation
        median_status = "✅" if result.median_runtime <= self.targets['median_runtime'] else "❌"
        p95_status = "✅" if result.percentile_95_runtime <= self.targets['percentile_95_runtime'] else "❌"
        print(f"\nRuntime Validation:")
        print(f"  {median_status} Median Runtime: {result.median_runtime:.3f}s (target: ≤{self.targets['median_runtime']}s)")
        print(f"  {p95_status} 95th Percentile: {result.percentile_95_runtime:.3f}s (target: ≤{self.targets['percentile_95_runtime']}s)")
        
        # Memory validation
        memory_status = "✅" if result.max_memory_mb <= self.targets['max_ram_memory_gb'] * 1024 else "❌"
        print(f"\nMemory Validation:")
        print(f"  {memory_status} Peak RAM: {result.max_memory_mb:.1f}MB (target: ≤{self.targets['max_ram_memory_gb']*1024:.0f}MB)")
        
        # Runtime distribution
        if result.runtimes:
            print(f"\nRuntime Distribution:")
            print(f"  Min: {np.min(result.runtimes):.3f}s")
            print(f"  25th percentile: {np.percentile(result.runtimes, 25):.3f}s")
            print(f"  Median: {result.median_runtime:.3f}s")
            print(f"  75th percentile: {np.percentile(result.runtimes, 75):.3f}s")
            print(f"  95th percentile: {result.percentile_95_runtime:.3f}s")
            print(f"  Max: {np.max(result.runtimes):.3f}s")
        
        # Success/failure analysis
        if result.detailed_results:
            successful_tasks = [r for r in result.detailed_results if r['success']]
            failed_tasks = [r for r in result.detailed_results if not r['success']]
            
            if successful_tasks:
                avg_success_time = np.mean([r['runtime'] for r in successful_tasks])
                print(f"\nSuccessful Tasks Analysis:")
                print(f"  Average solve time: {avg_success_time:.3f}s")
                print(f"  Fastest solve: {min(r['runtime'] for r in successful_tasks):.3f}s")
                print(f"  Slowest solve: {max(r['runtime'] for r in successful_tasks):.3f}s")
            
            if failed_tasks:
                avg_fail_time = np.mean([r['runtime'] for r in failed_tasks])
                timeout_count = sum(1 for r in failed_tasks if r.get('timeout', False))
                print(f"\nFailed Tasks Analysis:")
                print(f"  Average attempt time: {avg_fail_time:.3f}s")
                print(f"  Timeouts: {timeout_count}/{len(failed_tasks)}")
        
        # Final assessment
        print(f"\n" + "="*80)
        if result.overall_success:
            print("🎉 SYSTEM READY FOR ARC-AGI COMPETITION SUBMISSION!")
            print("All performance targets met on real ARC dataset.")
        else:
            print("⚠️  SYSTEM NEEDS OPTIMIZATION BEFORE SUBMISSION")
            if not result.meets_accuracy_target:
                print(f"   • Accuracy too low: {result.accuracy:.1%} < {self.targets['accuracy']:.0%}")
            if not result.meets_runtime_targets:
                print(f"   • Runtime too slow: median {result.median_runtime:.3f}s > {self.targets['median_runtime']}s")
        print("="*80)
    
    def save_validation_report(self, result: RealARCValidationResult, 
                             filename: str = "real_arc_validation_report.json") -> None:
        """Save validation report to JSON file."""
        try:
            report_data = {
                'validation_summary': {
                    'dataset': 'Real ARC-AGI Dataset',
                    'total_tasks': result.total_tasks,
                    'successful_tasks': result.successful_tasks,
                    'failed_tasks': result.failed_tasks,
                    'accuracy': result.accuracy,
                    'meets_accuracy_target': result.meets_accuracy_target,
                    'meets_runtime_targets': result.meets_runtime_targets,
                    'overall_success': result.overall_success
                },
                'performance_metrics': {
                    'accuracy': result.accuracy,
                    'median_runtime': result.median_runtime,
                    'percentile_95_runtime': result.percentile_95_runtime,
                    'max_memory_mb': result.max_memory_mb,
                    'runtime_distribution': {
                        'min': float(np.min(result.runtimes)) if result.runtimes else 0,
                        'q25': float(np.percentile(result.runtimes, 25)) if result.runtimes else 0,
                        'median': result.median_runtime,
                        'q75': float(np.percentile(result.runtimes, 75)) if result.runtimes else 0,
                        'q95': result.percentile_95_runtime,
                        'max': float(np.max(result.runtimes)) if result.runtimes else 0
                    }
                },
                'targets': self.targets,
                'detailed_results': result.detailed_results
            }
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Real ARC validation report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")


def main():
    """Run real ARC dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ARC solver on real ARC dataset")
    parser.add_argument("dataset_path", help="Path to ARC dataset directory")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to test")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout per task in seconds")
    parser.add_argument("--output", help="Output file for validation report")
    
    args = parser.parse_args()
    
    logger.info("Starting Task 12.2 - Real ARC Dataset Validation")
    
    try:
        # Initialize validator
        validator = RealARCValidator(args.dataset_path)
        
        # Run validation on real ARC tasks
        logger.info("Running validation on real ARC dataset...")
        result = validator.validate_on_real_arc(
            max_tasks=args.max_tasks, 
            timeout_per_task=args.timeout
        )
        
        # Print report
        validator.print_validation_report(result)
        
        # Save report
        output_file = args.output or "real_arc_validation_report.json"
        validator.save_validation_report(result, output_file)
        
        # Return appropriate exit code
        return 0 if result.overall_success else 1
        
    except KeyboardInterrupt:
        logger.info("Real ARC validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Real ARC validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())