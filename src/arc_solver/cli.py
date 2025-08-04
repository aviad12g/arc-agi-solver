"""Command-line interface for the ARC solver."""

import argparse
import sys
from pathlib import Path
from typing import Optional

def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ARC-AGI Solver: Mathematical perception and DSL reasoning"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a single ARC task")
    solve_parser.add_argument("task_file", help="Path to task JSON file")
    solve_parser.add_argument("--config", help="Path to config file")
    solve_parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process batch of tasks")
    batch_parser.add_argument("folder", help="Path to folder containing task files")
    batch_parser.add_argument("--timeout", type=int, default=30, help="Timeout per task")
    batch_parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    batch_parser.add_argument("--output", help="Output file for results")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "solve":
        return solve_task(args.task_file, args.config, args.timeout)
    elif args.command == "batch":
        return batch_process(args.folder, args.timeout, args.threads, args.output)
    elif args.command == "test":
        return run_tests(args.coverage)
    
    return 0


def solve_task(task_file: str, config: Optional[str], timeout: int) -> int:
    """Solve a single ARC task."""
    print(f"Solving task: {task_file}")
    print(f"Config: {config}")
    print(f"Timeout: {timeout}s")
    
    # TODO: Implement actual solving logic
    print("Task solving not yet implemented")
    return 0


def batch_process(folder: str, timeout: int, threads: int, output: Optional[str]) -> int:
    """Process batch of tasks."""
    print(f"Processing batch in folder: {folder}")
    print(f"Timeout: {timeout}s per task")
    print(f"Threads: {threads}")
    print(f"Output: {output}")
    
    # TODO: Implement batch processing
    print("Batch processing not yet implemented")
    return 0


def run_tests(coverage: bool) -> int:
    """Run the test suite."""
    import subprocess
    
    cmd = ["python3", "-m", "pytest"]
    if coverage:
        cmd.extend(["--cov=arc_solver", "--cov-report=html"])
    
    env = {"PYTHONPATH": "src"}
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())