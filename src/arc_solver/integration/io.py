"""Data loading and I/O operations for ARC tasks."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union, Any
import numpy as np
from arc_solver.core.data_models import Task, TrainExample, TestExample, Grid
from dataclasses import dataclass


class ARCDataLoader:
    """Loader for ARC dataset with caching capabilities."""
    
    def __init__(self, data_dir: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing ARC JSON files
            cache_dir: Directory for cached .npz files (optional)
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_task(self, task_id: str, use_cache: bool = True) -> Task:
        """Load a single ARC task by ID.
        
        Args:
            task_id: Task identifier
            use_cache: Whether to use cached .npz files
            
        Returns:
            Task object with loaded data
        """
        # Try cache first if enabled
        if use_cache:
            cached_task = self._load_from_cache(task_id)
            if cached_task is not None:
                return cached_task
        
        # Load from JSON
        task_data = self._load_task_json(task_id)
        task = self._parse_task_data(task_id, task_data)
        
        # Cache the result
        if use_cache:
            self._save_to_cache(task)
        
        return task
    
    def cache_task(self, task_id: str) -> None:
        """Cache a single task as .npz file.
        
        Args:
            task_id: Task identifier to cache
        """
        task = self.load_task(task_id, use_cache=False)
        self._save_to_cache(task)
    
    def iter_tasks(self, use_cache: bool = True, skip_invalid: bool = True) -> Iterator[Tuple[str, Task]]:
        """Iterate over all available tasks.
        
        Args:
            use_cache: Whether to use cached files
            skip_invalid: Whether to skip invalid tasks or raise errors
            
        Yields:
            Tuples of (task_id, Task)
        """
        task_files = list(self.data_dir.glob("*.json"))
        
        for task_file in task_files:
            task_id = task_file.stem
            try:
                task = self.load_task(task_id, use_cache=use_cache)
                yield task_id, task
            except Exception as e:
                if skip_invalid:
                    print(f"Warning: Failed to load task {task_id}: {e}")
                    continue
                else:
                    raise
    
    def get_task_ids(self) -> List[str]:
        """Get list of all available task IDs.
        
        Returns:
            List of task identifiers
        """
        task_ids = set()
        
        # Add individual task files
        for f in self.data_dir.glob("*.json"):
            # Skip combined files
            if f.name not in ["arc-agi_training_challenges.json", 
                             "arc-agi_evaluation_challenges.json", 
                             "arc-agi_test_challenges.json",
                             "arc-agi_training_solutions.json",
                             "arc-agi_evaluation_solutions.json",
                             "sample_submission.json"]:
                task_ids.add(f.stem)
        
        # Add task IDs from combined files
        combined_files = [
            self.data_dir / "arc-agi_training_challenges.json",
            self.data_dir / "arc-agi_evaluation_challenges.json", 
            self.data_dir / "arc-agi_test_challenges.json"
        ]
        
        for combined_file in combined_files:
            if combined_file.exists():
                try:
                    with open(combined_file, 'r') as f:
                        all_tasks = json.load(f)
                        task_ids.update(all_tasks.keys())
                except Exception as e:
                    print(f"Warning: Failed to read {combined_file}: {e}")
        
        return sorted(list(task_ids))
    
    def _load_task_json(self, task_id: str) -> Dict:
        """Load task data from JSON file."""
        # First try direct task file
        task_file = self.data_dir / f"{task_id}.json"
        if task_file.exists():
            with open(task_file, 'r') as f:
                return json.load(f)
        
        # Try loading from combined files
        combined_files = [
            self.data_dir / "arc-agi_training_challenges.json",
            self.data_dir / "arc-agi_evaluation_challenges.json",
            self.data_dir / "arc-agi_test_challenges.json"
        ]
        
        for combined_file in combined_files:
            if combined_file.exists():
                with open(combined_file, 'r') as f:
                    all_tasks = json.load(f)
                    if task_id in all_tasks:
                        return all_tasks[task_id]
        
        raise FileNotFoundError(f"Task {task_id} not found in {self.data_dir}")
    
    def _parse_task_data(self, task_id: str, task_data: Dict) -> Task:
        """Parse JSON task data into Task object."""
        # Parse training examples
        train_examples = []
        for example in task_data.get("train", []):
            input_grid = np.array(example["input"], dtype=np.int32)
            output_grid = np.array(example["output"], dtype=np.int32)
            train_examples.append((input_grid, output_grid))
        
        # Parse test inputs
        test_inputs = []
        for example in task_data.get("test", []):
            input_grid = np.array(example["input"], dtype=np.int32)
            test_inputs.append(input_grid)
        
        return Task(
            task_id=task_id,
            train_examples=train_examples,
            test_inputs=test_inputs
        )
    
    def _load_from_cache(self, task_id: str) -> Optional[Task]:
        """Load task from cached .npz file."""
        cache_file = self.cache_dir / f"{task_id}.npz"
        
        if not cache_file.exists():
            return None
        
        try:
            data = np.load(cache_file, allow_pickle=True)
            
            # Reconstruct training examples
            train_examples = []
            train_inputs = data["train_inputs"]
            train_outputs = data["train_outputs"]
            
            for i in range(len(train_inputs)):
                train_examples.append((train_inputs[i], train_outputs[i]))
            
            # Reconstruct test inputs
            test_inputs = list(data["test_inputs"])
            
            return Task(
                task_id=task_id,
                train_examples=train_examples,
                test_inputs=test_inputs
            )
        
        except Exception as e:
            print(f"Warning: Failed to load cached task {task_id}: {e}")
            return None
    
    def _save_to_cache(self, task: Task) -> None:
        """Save task to cached .npz file."""
        cache_file = self.cache_dir / f"{task.task_id}.npz"
        
        # Separate training inputs and outputs
        train_inputs = [example[0] for example in task.train_examples]
        train_outputs = [example[1] for example in task.train_examples]
        
        np.savez_compressed(
            cache_file,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_inputs=task.test_inputs
        )


class PaletteCanonicalizer:
    """Canonicalizes color palettes for consistent representation."""
    
    @staticmethod
    def canonicalize_grid(grid: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """Canonicalize grid colors by frequency.
        
        Remaps colors by descending pixel count; ties broken by ascending original color index.
        Background (0) is never remapped.
        
        Args:
            grid: Input grid with arbitrary color values
            
        Returns:
            Tuple of (canonicalized_grid, color_mapping)
        """
        # Count color frequencies (excluding background)
        unique_colors, counts = np.unique(grid, return_counts=True)
        
        # Separate background from other colors
        bg_mask = unique_colors == 0
        bg_colors = unique_colors[bg_mask]
        bg_counts = counts[bg_mask]
        
        other_colors = unique_colors[~bg_mask]
        other_counts = counts[~bg_mask]
        
        # Sort other colors by count (descending), then by original value (ascending)
        if len(other_colors) > 0:
            # Create sorting key: (-count, original_color)
            sort_keys = list(zip(-other_counts, other_colors))
            sorted_indices = np.argsort(sort_keys, axis=0)[:, 0]
            sorted_colors = other_colors[sorted_indices]
        else:
            sorted_colors = np.array([], dtype=np.int32)
        
        # Create color mapping
        color_mapping = {}
        
        # Background stays as 0
        if len(bg_colors) > 0:
            color_mapping[0] = 0
        
        # Map other colors to 1, 2, 3, ...
        for i, original_color in enumerate(sorted_colors):
            color_mapping[original_color] = i + 1
        
        # Apply mapping
        canonicalized = np.zeros_like(grid)
        for original, canonical in color_mapping.items():
            canonicalized[grid == original] = canonical
        
        return canonicalized, color_mapping
    
    @staticmethod
    def apply_mapping(grid: np.ndarray, color_mapping: Dict[int, int]) -> np.ndarray:
        """Apply a color mapping to a grid.
        
        Args:
            grid: Input grid
            color_mapping: Dictionary mapping original colors to new colors
            
        Returns:
            Grid with colors remapped
        """
        result = np.zeros_like(grid)
        for original, new in color_mapping.items():
            result[grid == original] = new
        return result


def load_arc_dataset(data_dir: Union[str, Path]) -> ARCDataLoader:
    """Convenience function to create an ARC data loader.
    
    Args:
        data_dir: Directory containing ARC dataset files
        
    Returns:
        Configured ARCDataLoader instance
    """
    return ARCDataLoader(data_dir)


def verify_dataset_integrity(data_dir: Union[str, Path]) -> Dict[str, int]:
    """Verify the integrity of an ARC dataset.
    
    Args:
        data_dir: Directory containing ARC dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    loader = ARCDataLoader(data_dir)
    
    stats = {
        "total_tasks": 0,
        "valid_tasks": 0,
        "invalid_tasks": 0,
        "total_train_examples": 0,
        "total_test_examples": 0,
        "errors": []
    }
    
    # Get all task IDs first
    task_ids = loader.get_task_ids()
    
    for task_id in task_ids:
        stats["total_tasks"] += 1
        
        try:
            task = loader.load_task(task_id, use_cache=False)
            
            # Validate task structure
            assert len(task.train_examples) > 0, "No training examples"
            assert len(task.test_inputs) > 0, "No test inputs"
            
            # Validate grid properties
            for i, (input_grid, output_grid) in enumerate(task.train_examples):
                assert input_grid.ndim == 2, f"Train input {i} not 2D"
                assert output_grid.ndim == 2, f"Train output {i} not 2D"
                assert input_grid.dtype == np.int32, f"Train input {i} not int32"
                assert output_grid.dtype == np.int32, f"Train output {i} not int32"
            
            for i, test_input in enumerate(task.test_inputs):
                assert test_input.ndim == 2, f"Test input {i} not 2D"
                assert test_input.dtype == np.int32, f"Test input {i} not int32"
            
            stats["valid_tasks"] += 1
            stats["total_train_examples"] += len(task.train_examples)
            stats["total_test_examples"] += len(task.test_inputs)
            
        except Exception as e:
            stats["invalid_tasks"] += 1
            stats["errors"].append(f"Task {task_id}: {e}")
    
    return stats


# Convenience functions expected by tests
@dataclass
class LoadedTaskShim:
    """Lightweight task shim exposing wrapper lists as expected by some tests."""
    task_id: str
    train_examples: List[TrainExample]
    test_inputs: List[TestExample]


def load_arc_task(task_file: Union[str, Path]) -> LoadedTaskShim:
    """Load a single ARC task from a JSON file path.
    
    Parses JSON with keys 'train' and 'test'. The task_id is derived from the filename stem.
    """
    path = Path(task_file)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path, 'r') as f:
        task_data = json.load(f)
    
    # Return legacy-style Task with wrapper classes to satisfy tests
    train_examples_wrapped: List[TrainExample] = []
    for ex in task_data.get("train", []):
        input_grid = Grid(np.array(ex["input"], dtype=np.int32))
        output_grid = Grid(np.array(ex["output"], dtype=np.int32))
        train_examples_wrapped.append(TrainExample(input_grid, output_grid))
    
    test_inputs_wrapped: List[TestExample] = []
    for ex in task_data.get("test", []):
        input_grid = Grid(np.array(ex["input"], dtype=np.int32))
        test_inputs_wrapped.append(TestExample(input_grid))

    # Return shim that preserves wrapped examples for tests that inspect wrappers directly
    return LoadedTaskShim(task_id=path.stem,
                          train_examples=train_examples_wrapped,
                          test_inputs=test_inputs_wrapped)


def save_results(results: Dict[str, Any], output_file: Union[str, Path]) -> None:
    """Save results dictionary to a JSON file."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)