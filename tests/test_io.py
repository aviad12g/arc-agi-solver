"""Tests for data loading and I/O operations."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from arc_solver.integration.io import ARCDataLoader, PaletteCanonicalizer, verify_dataset_integrity


class TestARCDataLoader:
    """Test the ARC data loader."""
    
    def create_test_task_data(self):
        """Create sample task data for testing."""
        return {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[4, 3], [2, 1]]
                },
                {
                    "input": [[5, 6], [7, 8]],
                    "output": [[8, 7], [6, 5]]
                }
            ],
            "test": [
                {
                    "input": [[9, 10], [11, 12]]
                }
            ]
        }
    
    def test_load_single_task_file(self):
        """Test loading a single task from individual JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test task file
            task_data = self.create_test_task_data()
            task_file = temp_path / "test_task.json"
            with open(task_file, 'w') as f:
                json.dump(task_data, f)
            
            # Load task
            loader = ARCDataLoader(temp_path)
            task = loader.load_task("test_task", use_cache=False)
            
            assert task.task_id == "test_task"
            assert len(task.train_examples) == 2
            assert len(task.test_inputs) == 1
            
            # Check first training example
            input_grid, output_grid = task.train_examples[0]
            assert input_grid.shape == (2, 2)
            assert output_grid.shape == (2, 2)
            assert input_grid.dtype == np.int32
            assert output_grid.dtype == np.int32
            np.testing.assert_array_equal(input_grid, [[1, 2], [3, 4]])
            np.testing.assert_array_equal(output_grid, [[4, 3], [2, 1]])
    
    def test_load_from_combined_file(self):
        """Test loading task from combined JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create combined file with multiple tasks
            combined_data = {
                "task1": self.create_test_task_data(),
                "task2": {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]]}]
                }
            }
            
            combined_file = temp_path / "arc-agi_training_challenges.json"
            with open(combined_file, 'w') as f:
                json.dump(combined_data, f)
            
            # Load specific task
            loader = ARCDataLoader(temp_path)
            task = loader.load_task("task1", use_cache=False)
            
            assert task.task_id == "task1"
            assert len(task.train_examples) == 2
            assert len(task.test_inputs) == 1
    
    def test_caching(self):
        """Test task caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test task
            task_data = self.create_test_task_data()
            task_file = temp_path / "cache_test.json"
            with open(task_file, 'w') as f:
                json.dump(task_data, f)
            
            loader = ARCDataLoader(temp_path)
            
            # Load and cache task
            task1 = loader.load_task("cache_test", use_cache=True)
            
            # Verify cache file was created
            cache_file = loader.cache_dir / "cache_test.npz"
            assert cache_file.exists()
            
            # Load from cache
            task2 = loader.load_task("cache_test", use_cache=True)
            
            # Verify tasks are equivalent
            assert task1.task_id == task2.task_id
            assert len(task1.train_examples) == len(task2.train_examples)
            assert len(task1.test_inputs) == len(task2.test_inputs)
            
            # Check array equality
            for (in1, out1), (in2, out2) in zip(task1.train_examples, task2.train_examples):
                np.testing.assert_array_equal(in1, in2)
                np.testing.assert_array_equal(out1, out2)
    
    def test_iter_tasks(self):
        """Test iterating over all tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple task files
            for i in range(3):
                task_data = self.create_test_task_data()
                task_file = temp_path / f"task_{i}.json"
                with open(task_file, 'w') as f:
                    json.dump(task_data, f)
            
            loader = ARCDataLoader(temp_path)
            
            # Collect all tasks
            tasks = list(loader.iter_tasks(use_cache=False))
            
            assert len(tasks) == 3
            task_ids = [task_id for task_id, _ in tasks]
            assert "task_0" in task_ids
            assert "task_1" in task_ids
            assert "task_2" in task_ids
    
    def test_get_task_ids(self):
        """Test getting list of task IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "task_a.json").touch()
            (temp_path / "task_b.json").touch()
            (temp_path / "not_json.txt").touch()  # Should be ignored
            
            loader = ARCDataLoader(temp_path)
            task_ids = loader.get_task_ids()
            
            assert len(task_ids) == 2
            assert "task_a" in task_ids
            assert "task_b" in task_ids
            assert "not_json" not in task_ids
    
    def test_nonexistent_task(self):
        """Test loading nonexistent task raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ARCDataLoader(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                loader.load_task("nonexistent_task")
    
    def test_invalid_data_directory(self):
        """Test that invalid data directory raises error."""
        with pytest.raises(FileNotFoundError):
            ARCDataLoader("/nonexistent/directory")


class TestPaletteCanonicalizer:
    """Test the palette canonicalizer."""
    
    def test_canonicalize_simple_grid(self):
        """Test canonicalizing a simple grid."""
        # Grid with colors: 0 (background), 3 (2 pixels), 1 (1 pixel), 2 (1 pixel)
        grid = np.array([
            [0, 3, 1],
            [3, 0, 2]
        ], dtype=np.int32)
        
        canonical, mapping = PaletteCanonicalizer.canonicalize_grid(grid)
        
        # Expected mapping: 0->0 (background), 3->1 (most frequent), 1->2, 2->3
        # But 1 < 2, so 1->2, 2->3 (ties broken by original value)
        expected_mapping = {0: 0, 3: 1, 1: 2, 2: 3}
        assert mapping == expected_mapping
        
        expected_canonical = np.array([
            [0, 1, 2],
            [1, 0, 3]
        ], dtype=np.int32)
        np.testing.assert_array_equal(canonical, expected_canonical)
    
    def test_canonicalize_background_only(self):
        """Test canonicalizing grid with only background."""
        grid = np.array([[0, 0], [0, 0]], dtype=np.int32)
        
        canonical, mapping = PaletteCanonicalizer.canonicalize_grid(grid)
        
        assert mapping == {0: 0}
        np.testing.assert_array_equal(canonical, grid)
    
    def test_canonicalize_no_background(self):
        """Test canonicalizing grid without background (no zeros)."""
        grid = np.array([[1, 2], [2, 1]], dtype=np.int32)
        
        canonical, mapping = PaletteCanonicalizer.canonicalize_grid(grid)
        
        # Both colors have same frequency, so 1 < 2 means 1->1, 2->2
        expected_mapping = {1: 1, 2: 2}
        assert mapping == expected_mapping
        
        expected_canonical = np.array([[1, 2], [2, 1]], dtype=np.int32)
        np.testing.assert_array_equal(canonical, expected_canonical)
    
    def test_apply_mapping(self):
        """Test applying a color mapping."""
        grid = np.array([[1, 2], [3, 1]], dtype=np.int32)
        mapping = {1: 5, 2: 6, 3: 7}
        
        result = PaletteCanonicalizer.apply_mapping(grid, mapping)
        
        expected = np.array([[5, 6], [7, 5]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_frequency_ordering(self):
        """Test that colors are ordered by frequency correctly."""
        # Create grid where color frequencies are: 5 (4 times), 3 (2 times), 1 (1 time)
        grid = np.array([
            [0, 5, 5, 3],
            [5, 5, 3, 1]
        ], dtype=np.int32)
        
        canonical, mapping = PaletteCanonicalizer.canonicalize_grid(grid)
        
        # Expected: 0->0, 5->1 (most frequent), 3->2, 1->3
        expected_mapping = {0: 0, 5: 1, 3: 2, 1: 3}
        assert mapping == expected_mapping


class TestDatasetVerification:
    """Test dataset verification functionality."""
    
    def test_verify_valid_dataset(self):
        """Test verifying a valid dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid tasks
            for i in range(2):
                task_data = {
                    "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                    "test": [{"input": [[3, 4]]}]
                }
                task_file = temp_path / f"task_{i}.json"
                with open(task_file, 'w') as f:
                    json.dump(task_data, f)
            
            stats = verify_dataset_integrity(temp_path)
            
            assert stats["total_tasks"] == 2
            assert stats["valid_tasks"] == 2
            assert stats["invalid_tasks"] == 0
            assert stats["total_train_examples"] == 2
            assert stats["total_test_examples"] == 2
            assert len(stats["errors"]) == 0
    
    def test_verify_invalid_dataset(self):
        """Test verifying dataset with invalid tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid task (no training examples)
            invalid_task = {
                "train": [],  # Invalid: no training examples
                "test": [{"input": [[1, 2]]}]
            }
            task_file = temp_path / "invalid_task.json"
            with open(task_file, 'w') as f:
                json.dump(invalid_task, f)
            
            stats = verify_dataset_integrity(temp_path)
            
            assert stats["total_tasks"] == 1
            assert stats["valid_tasks"] == 0
            assert stats["invalid_tasks"] == 1
            assert len(stats["errors"]) == 1
            assert "training example" in stats["errors"][0]