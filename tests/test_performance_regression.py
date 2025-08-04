"""Performance regression tests for continuous integration.

This module contains automated performance regression tests that ensure
the system maintains performance targets across updates and changes.
"""

import pytest
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from arc_solver.cli.commands import ARCSolver
from arc_solver.core.data_models import Task, TrainExample, TestExample, Grid
from arc_solver.perception.blob_labeling import create_blob_labeler
from arc_solver.reasoning.dsl_engine import create_dsl_engine, DSLOperation
from arc_solver.search.astar import create_astar_searcher

logger = logging.getLogger(__name__)


class PerformanceRegressionTester:
    """Tests for performance regression detection."""
    
    def __init__(self):
        self.baseline_file = Path("performance_baseline.json")
        self.tolerance = 0.1  # 10% tolerance for performance regression
    
    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load performance baseline from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")
        return None
    
    def save_baseline(self, metrics: Dict[str, Any]) -> None:
        """Save performance baseline to file."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Performance baseline saved to {self.baseline_file}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def check_regression(self, current_metrics: Dict[str, Any], 
                        baseline_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check for performance regression against baseline."""
        regression_results = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                
                # Calculate relative change
                if baseline_value > 0:
                    relative_change = (current_value - baseline_value) / baseline_value
                    
                    # Check if regression (performance got worse)
                    # For time metrics, increase is bad; for accuracy, decrease is bad
                    if 'time' in metric_name.lower() or 'runtime' in metric_name.lower():
                        is_regression = relative_change > self.tolerance
                    elif 'accuracy' in metric_name.lower():
                        is_regression = relative_change < -self.tolerance
                    else:
                        is_regression = abs(relative_change) > self.tolerance
                    
                    regression_results[metric_name] = {
                        'regression': is_regression,
                        'current': current_value,
                        'baseline': baseline_value,
                        'relative_change': relative_change
                    }
        
        return regression_results


class TestPerformanceRegression:
    """Performance regression test suite."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tester = PerformanceRegressionTester()
    
    def test_blob_labeling_performance_regression(self):
        """Test blob labeling performance hasn't regressed."""
        labeler = create_blob_labeler(use_gpu=False)
        
        # Standard test grid
        np.random.seed(42)
        test_grid = np.random.randint(0, 4, size=(20, 20), dtype=np.int32)
        
        # Measure performance
        times = []
        for _ in range(20):
            start_time = time.perf_counter()
            blobs, processing_time = labeler.label_blobs(test_grid)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        current_metrics = {
            'blob_labeling_avg_time': np.mean(times),
            'blob_labeling_max_time': np.max(times),
            'blob_count': len(blobs)
        }
        
        # Check against baseline
        baseline = self.tester.load_baseline()
        if baseline and 'blob_labeling' in baseline:
            regression_results = self.tester.check_regression(
                current_metrics, baseline['blob_labeling']
            )
            
            for metric, result in regression_results.items():
                if result['regression']:
                    pytest.fail(f"Performance regression in {metric}: "
                               f"{result['current']:.6f} vs baseline {result['baseline']:.6f} "
                               f"({result['relative_change']:+.1%})")
        
        # Performance targets
        assert current_metrics['blob_labeling_avg_time'] < 0.05  # 50ms for 20x20
        assert current_metrics['blob_labeling_max_time'] < 0.1   # 100ms max
        
        logger.info(f"Blob labeling performance: {current_metrics['blob_labeling_avg_time']*1000:.2f}ms avg")
    
    def test_dsl_operation_performance_regression(self):
        """Test DSL operation performance hasn't regressed."""
        engine = create_dsl_engine()
        
        # Test grid
        test_grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ], dtype=np.int32)
        
        # Test operations
        operations = [
            DSLOperation('Rotate90', {}),
            DSLOperation('ReflectH', {}),
            DSLOperation('Paint', {'x': 1, 'y': 1, 'c': 0})
        ]
        
        current_metrics = {}
        
        for operation in operations:
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                engine.apply_operation(test_grid, operation)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            
            current_metrics[f'{operation.primitive_name}_avg_time'] = avg_time
            current_metrics[f'{operation.primitive_name}_max_time'] = max_time
            
            # Performance target: ≤200µs
            assert avg_time <= 0.0002, f"{operation.primitive_name} avg time {avg_time*1000000:.1f}µs > 200µs"
            assert max_time <= 0.001,  f"{operation.primitive_name} max time {max_time*1000000:.1f}µs > 1000µs"
        
        # Check against baseline
        baseline = self.tester.load_baseline()
        if baseline and 'dsl_operations' in baseline:
            regression_results = self.tester.check_regression(
                current_metrics, baseline['dsl_operations']
            )
            
            for metric, result in regression_results.items():
                if result['regression']:
                    pytest.fail(f"Performance regression in {metric}: "
                               f"{result['current']*1000000:.1f}µs vs baseline {result['baseline']*1000000:.1f}µs "
                               f"({result['relative_change']:+.1%})")
        
        logger.info(f"DSL operations performance: {[f'{op.primitive_name}={current_metrics[f\"{op.primitive_name}_avg_time\"]*1000000:.1f}µs' for op in operations]}")
    
    def test_search_performance_regression(self):
        """Test search performance hasn't regressed."""
        searcher = create_astar_searcher(
            max_program_length=3,
            max_nodes_expanded=100,
            beam_width=16
        )
        
        # Test problems
        test_cases = [
            {
                'name': 'simple_rotation',
                'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
                'target': np.array([[3, 1], [4, 2]], dtype=np.int32)
            },
            {
                'name': 'reflection',
                'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                'target': np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32)
            }
        ]
        
        current_metrics = {}
        
        for case in test_cases:
            times = []
            nodes_expanded = []
            success_count = 0
            
            for _ in range(5):
                start_time = time.perf_counter()
                result = searcher.search(case['input'], case['target'])
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                nodes_expanded.append(result.nodes_expanded)
                if result.success:
                    success_count += 1
            
            current_metrics[f'{case["name"]}_avg_time'] = np.mean(times)
            current_metrics[f'{case["name"]}_avg_nodes'] = np.mean(nodes_expanded)
            current_metrics[f'{case["name"]}_success_rate'] = success_count / 5
            
            # Performance targets
            assert np.mean(times) < 1.0  # Should complete within 1 second
            assert np.mean(nodes_expanded) <= 100  # Should not expand too many nodes
        
        # Check against baseline
        baseline = self.tester.load_baseline()
        if baseline and 'search' in baseline:
            regression_results = self.tester.check_regression(
                current_metrics, baseline['search']
            )
            
            for metric, result in regression_results.items():
                if result['regression']:
                    pytest.fail(f"Performance regression in {metric}: "
                               f"{result['current']:.4f} vs baseline {result['baseline']:.4f} "
                               f"({result['relative_change']:+.1%})")
        
        logger.info(f"Search performance: avg_time={np.mean([current_metrics[k] for k in current_metrics if 'avg_time' in k]):.3f}s")
    
    def test_end_to_end_performance_regression(self):
        """Test end-to-end performance hasn't regressed."""
        solver = ARCSolver()
        
        # Create test tasks
        test_tasks = []
        for i in range(5):
            train_input = np.array([[i, i+1], [i+2, i+3]], dtype=np.int32)
            train_output = np.rot90(train_input)
            test_input = np.array([[i+10, i+11], [i+12, i+13]], dtype=np.int32)
            
            task = Task(
                task_id=f"regression_test_{i}",
                train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
                test_inputs=[Grid(test_input)]
            )
            test_tasks.append(task)
        
        # Measure performance
        times = []
        success_count = 0
        
        for task in test_tasks:
            start_time = time.perf_counter()
            result = solver.solve_task(task, timeout=5.0)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            if result.get('success', False):
                success_count += 1
        
        current_metrics = {
            'end_to_end_avg_time': np.mean(times),
            'end_to_end_max_time': np.max(times),
            'end_to_end_success_rate': success_count / len(test_tasks)
        }
        
        # Performance targets
        assert current_metrics['end_to_end_avg_time'] <= 2.0  # Average ≤2s
        assert current_metrics['end_to_end_max_time'] <= 5.0  # Max ≤5s
        assert current_metrics['end_to_end_success_rate'] >= 0.6  # ≥60% success on simple tasks
        
        # Check against baseline
        baseline = self.tester.load_baseline()
        if baseline and 'end_to_end' in baseline:
            regression_results = self.tester.check_regression(
                current_metrics, baseline['end_to_end']
            )
            
            for metric, result in regression_results.items():
                if result['regression']:
                    pytest.fail(f"Performance regression in {metric}: "
                               f"{result['current']:.4f} vs baseline {result['baseline']:.4f} "
                               f"({result['relative_change']:+.1%})")
        
        logger.info(f"End-to-end performance: {current_metrics['end_to_end_avg_time']:.3f}s avg, "
                   f"{current_metrics['end_to_end_success_rate']:.1%} success")
    
    def test_memory_usage_regression(self):
        """Test memory usage hasn't regressed."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        solver = ARCSolver()
        
        # Process multiple tasks to test memory accumulation
        for i in range(10):
            train_input = np.random.randint(0, 3, size=(5, 5), dtype=np.int32)
            train_output = np.rot90(train_input)
            test_input = np.random.randint(0, 3, size=(5, 5), dtype=np.int32)
            
            task = Task(
                task_id=f"memory_test_{i}",
                train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
                test_inputs=[Grid(test_input)]
            )
            
            solver.solve_task(task, timeout=2.0)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        current_metrics = {
            'memory_increase_mb': memory_increase,
            'final_memory_mb': final_memory
        }
        
        # Memory targets
        assert memory_increase < 500  # ≤500MB increase
        assert final_memory < 2000    # ≤2GB total (conservative)
        
        # Check against baseline
        baseline = self.tester.load_baseline()
        if baseline and 'memory' in baseline:
            regression_results = self.tester.check_regression(
                current_metrics, baseline['memory']
            )
            
            for metric, result in regression_results.items():
                if result['regression']:
                    pytest.fail(f"Memory regression in {metric}: "
                               f"{result['current']:.1f}MB vs baseline {result['baseline']:.1f}MB "
                               f"({result['relative_change']:+.1%})")
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")


def create_performance_baseline():
    """Create performance baseline for regression testing."""
    logger.info("Creating performance baseline...")
    
    tester = PerformanceRegressionTester()
    
    # Run all performance tests and collect metrics
    test_instance = TestPerformanceRegression()
    test_instance.setup_method()
    
    baseline_metrics = {}
    
    try:
        # Blob labeling baseline
        test_instance.test_blob_labeling_performance_regression()
        logger.info("✓ Blob labeling baseline created")
        
        # DSL operations baseline  
        test_instance.test_dsl_operation_performance_regression()
        logger.info("✓ DSL operations baseline created")
        
        # Search baseline
        test_instance.test_search_performance_regression()
        logger.info("✓ Search baseline created")
        
        # End-to-end baseline
        test_instance.test_end_to_end_performance_regression()
        logger.info("✓ End-to-end baseline created")
        
        # Memory baseline
        test_instance.test_memory_usage_regression()
        logger.info("✓ Memory baseline created")
        
        logger.info("Performance baseline creation completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to create performance baseline: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create baseline if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--create-baseline":
        create_performance_baseline()
    else:
        # Run regression tests
        pytest.main([__file__, "-v", "--tb=short"])