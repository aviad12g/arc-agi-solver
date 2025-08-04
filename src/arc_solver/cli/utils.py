"""CLI utility functions."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import signal
import threading

from arc_solver.core.data_models import Task
from arc_solver.integration.io import ARCDataLoader


class TimeoutHandler:
    """Handle timeouts for individual task processing."""
    
    def __init__(self, timeout_seconds: float):
        """Initialize timeout handler.
        
        Args:
            timeout_seconds: Timeout in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        self.timer: Optional[threading.Timer] = None
    
    def _timeout_callback(self):
        """Called when timeout occurs."""
        self.timed_out = True
    
    def __enter__(self):
        """Start timeout timer."""
        if self.timeout_seconds > 0:
            self.timer = threading.Timer(self.timeout_seconds, self._timeout_callback)
            self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timeout timer."""
        if self.timer:
            self.timer.cancel()


def setup_logging(level: int = logging.INFO, 
                 format_string: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        if level <= logging.DEBUG:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(levelname)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Reduce noise from some libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def load_task_from_file(file_path: Union[str, Path]) -> Task:
    """Load ARC task from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded Task object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Task file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            task_data = json.load(f)
        
        # Use ARCDataLoader to convert JSON to Task object
        # For now, we'll create a simple Task object directly
        from arc_solver.core.data_models import Task, TrainExample, TestExample
        
        train_examples = []
        for example in task_data.get('train', []):
            train_examples.append(TrainExample(
                input=np.array(example['input'], dtype=np.int32),
                output=np.array(example['output'], dtype=np.int32)
            ))
        
        test_examples = []
        for example in task_data.get('test', []):
            test_examples.append(TestExample(
                input=np.array(example['input'], dtype=np.int32),
                output=np.array(example.get('output', []), dtype=np.int32) if 'output' in example else None
            ))
        
        task = Task(
            task_id=file_path.stem,
            train=train_examples,
            test=test_examples
        )
        
        return task
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load task from {file_path}: {e}")


def save_results(results: Dict[str, Any], 
                output_path: Union[str, Path],
                pretty: bool = True) -> None:
    """Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        pretty: Whether to pretty-print JSON
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {k: convert_numpy(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(serializable_results, f, indent=2, sort_keys=True)
        else:
            json.dump(serializable_results, f)


def find_task_files(input_path: Union[str, Path], 
                   max_files: Optional[int] = None) -> List[Path]:
    """Find ARC task files in directory or from file list.
    
    Args:
        input_path: Directory path or file containing list of paths
        max_files: Maximum number of files to return
        
    Returns:
        List of task file paths
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Input is a file - check if it's a task file or file list
        if input_path.suffix.lower() == '.json':
            # Single task file
            return [input_path]
        else:
            # File containing list of paths
            with open(input_path, 'r') as f:
                file_paths = [Path(line.strip()) for line in f if line.strip()]
            return file_paths[:max_files] if max_files else file_paths
    
    elif input_path.is_dir():
        # Directory - find all JSON files
        json_files = list(input_path.rglob('*.json'))
        json_files.sort()  # Consistent ordering
        return json_files[:max_files] if max_files else json_files
    
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_memory(bytes_used: int) -> str:
    """Format memory usage in human-readable format.
    
    Args:
        bytes_used: Memory usage in bytes
        
    Returns:
        Formatted memory string
    """
    if bytes_used < 1024:
        return f"{bytes_used}B"
    elif bytes_used < 1024**2:
        return f"{bytes_used/1024:.1f}KB"
    elif bytes_used < 1024**3:
        return f"{bytes_used/1024**2:.1f}MB"
    else:
        return f"{bytes_used/1024**3:.1f}GB"


class ProgressReporter:
    """Progress reporting for batch processing."""
    
    def __init__(self, total_tasks: int, report_interval: int = 10):
        """Initialize progress reporter.
        
        Args:
            total_tasks: Total number of tasks
            report_interval: Report progress every N tasks
        """
        self.total_tasks = total_tasks
        self.report_interval = report_interval
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
    
    def update(self, success: bool = False) -> None:
        """Update progress.
        
        Args:
            success: Whether the task was successful
        """
        self.completed_tasks += 1
        if success:
            self.successful_tasks += 1
        
        # Report progress at intervals
        if (self.completed_tasks % self.report_interval == 0 or 
            self.completed_tasks == self.total_tasks):
            self._report_progress()
    
    def _report_progress(self) -> None:
        """Report current progress."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate rates
        tasks_per_second = self.completed_tasks / elapsed if elapsed > 0 else 0
        success_rate = self.successful_tasks / self.completed_tasks if self.completed_tasks > 0 else 0
        
        # Estimate remaining time
        remaining_tasks = self.total_tasks - self.completed_tasks
        eta = remaining_tasks / tasks_per_second if tasks_per_second > 0 else 0
        
        print(f"Progress: {self.completed_tasks}/{self.total_tasks} "
              f"({self.completed_tasks/self.total_tasks*100:.1f}%) | "
              f"Success: {self.successful_tasks} ({success_rate*100:.1f}%) | "
              f"Rate: {tasks_per_second:.1f} tasks/s | "
              f"ETA: {format_duration(eta)}")


def create_result_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics from batch results.
    
    Args:
        results: List of individual task results
        
    Returns:
        Summary statistics dictionary
    """
    if not results:
        return {
            'total_tasks': 0,
            'successful_tasks': 0,
            'success_rate': 0.0,
            'average_time': 0.0,
            'median_time': 0.0,
            'total_time': 0.0
        }
    
    successful_results = [r for r in results if r.get('success', False)]
    times = [r.get('computation_time', 0.0) for r in results]
    
    # Calculate statistics
    total_tasks = len(results)
    successful_tasks = len(successful_results)
    success_rate = successful_tasks / total_tasks
    
    total_time = sum(times)
    average_time = total_time / total_tasks if total_tasks > 0 else 0.0
    
    # Calculate median time
    sorted_times = sorted(times)
    n = len(sorted_times)
    if n == 0:
        median_time = 0.0
    elif n % 2 == 0:
        median_time = (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2
    else:
        median_time = sorted_times[n//2]
    
    # Calculate percentiles
    p95_time = sorted_times[int(0.95 * n)] if n > 0 else 0.0
    
    return {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'failed_tasks': total_tasks - successful_tasks,
        'success_rate': success_rate,
        'average_time': average_time,
        'median_time': median_time,
        'p95_time': p95_time,
        'total_time': total_time,
        'min_time': min(times) if times else 0.0,
        'max_time': max(times) if times else 0.0
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Print batch processing summary.
    
    Args:
        summary: Summary statistics dictionary
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Total tasks:      {summary['total_tasks']}")
    print(f"Successful:       {summary['successful_tasks']} ({summary['success_rate']*100:.1f}%)")
    print(f"Failed:           {summary['failed_tasks']}")
    
    print(f"\nTiming Statistics:")
    print(f"Total time:       {format_duration(summary['total_time'])}")
    print(f"Average time:     {format_duration(summary['average_time'])}")
    print(f"Median time:      {format_duration(summary['median_time'])}")
    print(f"95th percentile:  {format_duration(summary['p95_time'])}")
    print(f"Min time:         {format_duration(summary['min_time'])}")
    print(f"Max time:         {format_duration(summary['max_time'])}")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if summary['median_time'] <= 0.5:
        print("✅ Median runtime meets requirement (≤0.5s)")
    else:
        print(f"❌ Median runtime ({format_duration(summary['median_time'])}) exceeds requirement (≤0.5s)")
    
    if summary['p95_time'] <= 5.0:
        print("✅ 95th percentile runtime meets requirement (≤5s)")
    else:
        print(f"❌ 95th percentile runtime ({format_duration(summary['p95_time'])}) exceeds requirement (≤5s)")
    
    if summary['success_rate'] >= 0.35:
        print(f"✅ Success rate meets requirement (≥35%)")
    else:
        print(f"❌ Success rate ({summary['success_rate']*100:.1f}%) below requirement (≥35%)")