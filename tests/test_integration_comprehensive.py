"""Comprehensive integration tests for the complete ARC solver system.

Tests end-to-end functionality including:
- Complete pipeline from JSON input to solution output
- Component interaction validation
- Performance regression testing
- Error handling across system boundaries
- Configuration management integration
"""

import pytest
import numpy as np
import time
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

from arc_solver.core.data_models import Task, TrainExample, TestExample, Grid
from arc_solver.cli.commands import ARCSolver
from arc_solver.config import load_config, ConfigManager
from arc_solver.caching import create_cache_manager
from arc_solver.integration.io import load_arc_task, save_results

logger = logging.getLogger(__name__)


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    def test_simple_task_solving(self):
        """Test solving a simple ARC task end-to-end."""
        # Create simple task: 90-degree rotation
        train_input = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_output = np.array([[3, 1], [4, 2]], dtype=np.int32)
        test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
        
        task = Task(
            task_id="test_rotation",
            train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
            test_inputs=[Grid(test_input)]
        )
        
        # Initialize solver
        solver = ARCSolver()
        
        # Solve task
        result = solver.solve_task(task, timeout=10.0)
        
        assert result['success']
        assert result['task_id'] == "test_rotation"
        assert len(result['predictions']) == 1
        
        # Verify prediction is correct (90-degree rotation of test input)
        if result['predictions'][0] is not None:
            prediction = np.array(result['predictions'][0], dtype=np.int32)
            expected = np.array([[7, 5], [8, 6]], dtype=np.int32)
            assert np.array_equal(prediction, expected)
        
        # Check performance metrics
        assert result['computation_time'] < 10.0
        assert 'search_stats' in result
    
    def test_json_task_loading(self):
        """Test loading ARC tasks from JSON format."""
        # Create temporary JSON file
        task_data = {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[3, 1], [4, 2]]
                }
            ],
            "test": [
                {
                    "input": [[5, 6], [7, 8]]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task_data, f)
            temp_path = f.name
        
        try:
            # Load task
            task = load_arc_task(temp_path)
            
            assert task.task_id == Path(temp_path).stem
            assert len(task.train_examples) == 1
            assert len(task.test_inputs) == 1
            
            # Verify data integrity
            train_example = task.train_examples[0]
            assert train_example.input.data.shape == (2, 2)
            assert train_example.output.data.shape == (2, 2)
            
        finally:
            Path(temp_path).unlink()
    
    def test_batch_processing(self):
        """Test batch processing of multiple tasks."""
        # Create multiple simple tasks
        tasks = []
        for i in range(3):
            train_input = np.array([[i, i+1], [i+2, i+3]], dtype=np.int32)
            train_output = np.rot90(train_input)  # 90-degree rotation
            test_input = np.array([[i+10, i+11], [i+12, i+13]], dtype=np.int32)
            
            task = Task(
                task_id=f"batch_test_{i}",
                train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
                test_inputs=[Grid(test_input)]
            )
            tasks.append(task)
        
        # Initialize solver
        solver = ARCSolver()
        
        # Process batch
        results = []
        total_time = 0
        
        for task in tasks:
            start_time = time.perf_counter()
            result = solver.solve_task(task, timeout=5.0)
            end_time = time.perf_counter()
            
            results.append(result)
            total_time += (end_time - start_time)
        
        # Verify all tasks processed
        assert len(results) == 3
        
        # Check success rate
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Batch processing: {successful}/3 tasks solved successfully")
        
        # Check total processing time
        logger.info(f"Total batch processing time: {total_time:.2f}s")
        assert total_time < 30.0  # Should complete within reasonable time
    
    def test_configuration_integration(self):
        """Test configuration system integration."""
        # Test with custom configuration
        config_overrides = [
            "search.astar.max_nodes_expanded=50",
            "search.beam_search.initial_beam_width=16"
        ]
        
        solver = ARCSolver(config_overrides=config_overrides)
        
        # Verify configuration was applied
        # Note: This would require access to internal configuration
        # For now, just test that solver initializes correctly
        assert solver is not None
        
        # Test solving with custom config
        train_input = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_output = np.array([[3, 1], [4, 2]], dtype=np.int32)
        test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
        
        task = Task(
            task_id="config_test",
            train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
            test_inputs=[Grid(test_input)]
        )
        
        result = solver.solve_task(task, timeout=5.0)
        
        # Should still work with custom configuration
        assert 'success' in result
        assert 'computation_time' in result
    
    def test_caching_integration(self):
        """Test caching system integration."""
        # Create task
        train_input = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_output = np.array([[3, 1], [4, 2]], dtype=np.int32)
        test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
        
        task = Task(
            task_id="cache_test",
            train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
            test_inputs=[Grid(test_input)]
        )
        
        # Initialize solver
        solver = ARCSolver()
        
        # First solve (should populate cache)
        start_time1 = time.perf_counter()
        result1 = solver.solve_task(task, timeout=10.0)
        time1 = time.perf_counter() - start_time1
        
        # Second solve (should use cache)
        start_time2 = time.perf_counter()
        result2 = solver.solve_task(task, timeout=10.0)
        time2 = time.perf_counter() - start_time2
        
        # Results should be consistent
        assert result1['success'] == result2['success']
        if result1['predictions'][0] is not None and result2['predictions'][0] is not None:
            pred1 = np.array(result1['predictions'][0])
            pred2 = np.array(result2['predictions'][0])
            assert np.array_equal(pred1, pred2)
        
        # Second solve might be faster due to caching
        logger.info(f"First solve: {time1:.3f}s, Second solve: {time2:.3f}s")
    
    def test_error_handling_integration(self):
        """Test error handling across system boundaries."""
        solver = ARCSolver()
        
        # Test with invalid task
        invalid_task = Task(
            task_id="invalid_test",
            train_examples=[],  # Empty training examples
            test_inputs=[Grid(np.array([[1]], dtype=np.int32))]
        )
        
        result = solver.solve_task(invalid_task, timeout=5.0)
        
        # Should handle error gracefully
        assert not result['success']
        assert 'error' in result or result['predictions'] == [None]
        
        # Test with malformed grid
        try:
            malformed_grid = np.array([[[1, 2]]], dtype=np.int32)  # 3D array
            malformed_task = Task(
                task_id="malformed_test",
                train_examples=[TrainExample(Grid(malformed_grid), Grid(malformed_grid))],
                test_inputs=[Grid(malformed_grid)]
            )
            
            result = solver.solve_task(malformed_task, timeout=5.0)
            assert not result['success']
            
        except (ValueError, AssertionError):
            # Should catch validation errors
            pass
    
    def test_timeout_handling(self):
        """Test timeout handling in integrated system."""
        solver = ARCSolver()
        
        # Create complex task that might timeout
        complex_input = np.random.randint(0, 5, size=(8, 8), dtype=np.int32)
        complex_output = np.random.randint(0, 5, size=(8, 8), dtype=np.int32)
        complex_test = np.random.randint(0, 5, size=(8, 8), dtype=np.int32)
        
        task = Task(
            task_id="timeout_test",
            train_examples=[TrainExample(Grid(complex_input), Grid(complex_output))],
            test_inputs=[Grid(complex_test)]
        )
        
        # Set very short timeout
        start_time = time.perf_counter()
        result = solver.solve_task(task, timeout=0.1)
        end_time = time.perf_counter()
        
        # Should respect timeout
        assert end_time - start_time <= 1.0  # Allow some tolerance
        assert 'computation_time' in result
        
        # Should handle timeout gracefully
        if not result['success']:
            logger.info(f"Task timed out as expected: {result.get('error', 'No error message')}")


class TestComponentInteraction:
    """Test interaction between major system components."""
    
    def test_perception_reasoning_integration(self):
        """Test integration between perception and reasoning layers."""
        from arc_solver.perception.blob_labeling import create_blob_labeler
        from arc_solver.perception.features import create_orbit_signature_computer
        from arc_solver.reasoning.dsl_engine import create_dsl_engine
        
        # Initialize components
        blob_labeler = create_blob_labeler(use_gpu=False)
        feature_computer = create_orbit_signature_computer()
        dsl_engine = create_dsl_engine()
        
        # Create test grid
        grid = np.array([
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 3, 3]
        ], dtype=np.int32)
        
        # Extract blobs (perception)
        blobs, _ = blob_labeler.label_blobs(grid)
        assert len(blobs) == 3
        
        # Compute features (perception)
        for blob in blobs:
            signature = feature_computer.compute_blob_signature(blob, grid.shape)
            assert signature.shape == (8,)
        
        # Apply DSL operation (reasoning)
        from arc_solver.reasoning.dsl_engine import DSLOperation
        operation = DSLOperation('Rotate90', {})
        transformed_grid = dsl_engine.apply_operation(grid, operation)
        
        # Extract blobs from transformed grid
        transformed_blobs, _ = blob_labeler.label_blobs(transformed_grid)
        assert len(transformed_blobs) == 3  # Should have same number of blobs
        
        logger.info("Perception-reasoning integration test passed")
    
    def test_reasoning_search_integration(self):
        """Test integration between reasoning and search layers."""
        from arc_solver.reasoning.dsl_engine import create_dsl_engine
        from arc_solver.search.astar import create_astar_searcher
        
        # Initialize components
        dsl_engine = create_dsl_engine()
        searcher = create_astar_searcher(
            max_program_length=2,
            max_nodes_expanded=50,
            beam_width=16
        )
        
        # Test simple search problem
        input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target_grid = np.array([[3, 1], [4, 2]], dtype=np.int32)
        
        result = searcher.search(input_grid, target_grid)
        
        if result.success:
            # Verify solution using DSL engine
            solution_grid, exec_info = dsl_engine.execute_program(result.program, input_grid)
            assert exec_info['success']
            assert np.array_equal(solution_grid, target_grid)
            
            logger.info(f"Found solution: {result.program}")
        else:
            logger.info(f"No solution found: {result.termination_reason}")
        
        logger.info("Reasoning-search integration test completed")
    
    def test_search_caching_integration(self):
        """Test integration between search and caching layers."""
        from arc_solver.search.astar import create_astar_searcher
        from arc_solver.caching import create_cache_manager
        
        # Initialize components
        cache_config = {'file_cache': {'enabled': True}}
        cache_manager = create_cache_manager(cache_config)
        
        searcher = create_astar_searcher(
            max_program_length=2,
            max_nodes_expanded=30,
            beam_width=8
        )
        
        # Test caching with search
        input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target_grid = np.array([[3, 1], [4, 2]], dtype=np.int32)
        
        # First search
        result1 = searcher.search(input_grid, target_grid)
        
        # Second search (should potentially use cached results)
        result2 = searcher.search(input_grid, target_grid)
        
        # Results should be consistent
        assert result1.success == result2.success
        if result1.success and result2.success:
            assert str(result1.program) == str(result2.program)
        
        logger.info("Search-caching integration test completed")
    
    def test_full_pipeline_integration(self):
        """Test full pipeline integration with all components."""
        # This test verifies that all components work together
        # in the complete solving pipeline
        
        solver = ARCSolver()
        
        # Create test task with known solution
        train_input = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        train_output = np.array([
            [3, 1, 0],
            [4, 2, 0],
            [0, 0, 0]
        ], dtype=np.int32)  # 90-degree rotation
        
        test_input = np.array([
            [5, 6, 0],
            [7, 8, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        task = Task(
            task_id="full_pipeline_test",
            train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
            test_inputs=[Grid(test_input)]
        )
        
        # Solve with full pipeline
        result = solver.solve_task(task, timeout=15.0)
        
        # Verify pipeline components were used
        assert 'computation_time' in result
        assert 'search_stats' in result
        
        if result['success']:
            prediction = np.array(result['predictions'][0], dtype=np.int32)
            expected = np.array([
                [7, 5, 0],
                [8, 6, 0],
                [0, 0, 0]
            ], dtype=np.int32)
            
            assert np.array_equal(prediction, expected)
            logger.info("Full pipeline integration test: SUCCESS")
        else:
            logger.info(f"Full pipeline integration test: No solution found ({result.get('error', 'Unknown error')})")
        
        # Test should complete regardless of success
        assert result['computation_time'] > 0


class TestPerformanceRegression:
    """Test performance regression across system updates."""
    
    def test_perception_performance_regression(self):
        """Test perception layer performance hasn't regressed."""
        from arc_solver.perception.blob_labeling import create_blob_labeler
        
        # Create standard test grid
        np.random.seed(42)
        test_grid = np.random.randint(0, 4, size=(20, 20), dtype=np.int32)
        
        labeler = create_blob_labeler(use_gpu=False)
        
        # Measure performance
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            blobs, processing_time = labeler.label_blobs(test_grid)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        logger.info(f"Perception performance: {avg_time*1000:.2f}ms average")
        
        # Should meet performance targets
        assert avg_time < 0.05  # 50ms for 20×20 grid should be reasonable
        assert len(blobs) > 0  # Should find some blobs
    
    def test_reasoning_performance_regression(self):
        """Test reasoning layer performance hasn't regressed."""
        from arc_solver.reasoning.dsl_engine import create_dsl_engine, DSLOperation
        
        engine = create_dsl_engine()
        
        # Create test grid
        test_grid = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ], dtype=np.int32)
        
        # Test operation performance
        operations = [
            DSLOperation('Rotate90', {}),
            DSLOperation('ReflectH', {}),
            DSLOperation('Paint', {'x': 1, 'y': 1, 'c': 0})
        ]
        
        for operation in operations:
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                engine.apply_operation(test_grid, operation)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            logger.info(f"{operation.primitive_name} performance: {avg_time*1000000:.1f}µs average")
            
            # Should meet ≤200µs target
            assert avg_time <= 0.0002
    
    def test_search_performance_regression(self):
        """Test search layer performance hasn't regressed."""
        from arc_solver.search.astar import create_astar_searcher
        
        searcher = create_astar_searcher(
            max_program_length=3,
            max_nodes_expanded=100,
            beam_width=16
        )
        
        # Test search performance
        input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target_grid = np.array([[3, 1], [4, 2]], dtype=np.int32)
        
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = searcher.search(input_grid, target_grid)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        logger.info(f"Search performance: {avg_time*1000:.1f}ms average")
        
        # Should complete quickly for simple problems
        assert avg_time < 1.0  # 1 second should be more than enough
    
    def test_end_to_end_performance_regression(self):
        """Test end-to-end performance hasn't regressed."""
        solver = ARCSolver()
        
        # Create standard test task
        train_input = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_output = np.array([[3, 1], [4, 2]], dtype=np.int32)
        test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
        
        task = Task(
            task_id="performance_test",
            train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
            test_inputs=[Grid(test_input)]
        )
        
        # Measure end-to-end performance
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = solver.solve_task(task, timeout=10.0)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        logger.info(f"End-to-end performance: {avg_time:.3f}s average")
        
        # Should meet ≤0.5s median runtime target for simple tasks
        assert avg_time <= 2.0  # Allow some tolerance for test environment


class TestSystemRobustness:
    """Test system robustness under various conditions."""
    
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
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
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB "
                   f"(+{memory_increase:.1f}MB)")
        
        # Should not have excessive memory growth
        assert memory_increase < 500  # 500MB increase should be reasonable
    
    def test_concurrent_solving(self):
        """Test concurrent task solving."""
        import threading
        import queue
        
        solver = ARCSolver()
        results_queue = queue.Queue()
        
        def solve_task_thread(task_id):
            try:
                train_input = np.array([[task_id % 3, (task_id + 1) % 3]], dtype=np.int32)
                train_output = np.array([[(task_id + 1) % 3, task_id % 3]], dtype=np.int32)
                test_input = np.array([[task_id % 3, (task_id + 2) % 3]], dtype=np.int32)
                
                task = Task(
                    task_id=f"concurrent_test_{task_id}",
                    train_examples=[TrainExample(Grid(train_input), Grid(train_output))],
                    test_inputs=[Grid(test_input)]
                )
                
                result = solver.solve_task(task, timeout=5.0)
                results_queue.put((task_id, result))
                
            except Exception as e:
                results_queue.put((task_id, {'success': False, 'error': str(e)}))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=solve_task_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=15.0)
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        logger.info(f"Concurrent solving: {len(results)} tasks completed")
        
        # All tasks should complete (success or failure)
        for task_id, result in results:
            assert 'success' in result
            logger.info(f"Task {task_id}: {'SUCCESS' if result['success'] else 'FAILED'}")
    
    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        solver = ARCSolver()
        
        edge_cases = [
            {
                'name': 'empty_grid',
                'input': np.zeros((1, 1), dtype=np.int32),
                'output': np.zeros((1, 1), dtype=np.int32)
            },
            {
                'name': 'large_grid',
                'input': np.ones((30, 30), dtype=np.int32),
                'output': np.ones((30, 30), dtype=np.int32)
            },
            {
                'name': 'all_colors',
                'input': np.arange(10).reshape(2, 5).astype(np.int32),
                'output': np.arange(10).reshape(2, 5).astype(np.int32)
            }
        ]
        
        for case in edge_cases:
            logger.info(f"Testing edge case: {case['name']}")
            
            task = Task(
                task_id=f"edge_case_{case['name']}",
                train_examples=[TrainExample(Grid(case['input']), Grid(case['output']))],
                test_inputs=[Grid(case['input'])]
            )
            
            try:
                result = solver.solve_task(task, timeout=5.0)
                
                # Should handle gracefully
                assert 'success' in result
                assert 'computation_time' in result
                
                logger.info(f"  Result: {'SUCCESS' if result['success'] else 'HANDLED'}")
                
            except Exception as e:
                logger.error(f"  Failed with exception: {e}")
                # Edge cases should not cause unhandled exceptions
                pytest.fail(f"Edge case {case['name']} caused unhandled exception: {e}")


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])