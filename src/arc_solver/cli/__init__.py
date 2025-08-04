"""Command-line interface for ARC-AGI solver.

This module provides CLI commands for solving individual puzzles and batch processing.
"""

from .main import main_cli
from .commands import solve_command, batch_command, config_command
from .utils import setup_logging, load_task_from_file, save_results

__all__ = [
    'main_cli',
    'solve_command',
    'batch_command', 
    'config_command',
    'setup_logging',
    'load_task_from_file',
    'save_results'
]