"""Main CLI entry point for ARC-AGI solver."""

import sys
import argparse
import logging
from typing import List, Optional

from . import commands
from .utils import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='arc-solver',
        description='ARC-AGI Solver - Mathematical perception and DSL reasoning for ARC puzzles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arc-solver solve task.json                    # Solve single puzzle
  arc-solver batch puzzles/ --timeout 30       # Batch process folder
  arc-solver config show                        # Show current configuration
  arc-solver test --quick                       # Run quick system test
        """
    )
    
    # Global options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file or overrides (e.g., search.beam_width=32)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (use -v, -vv, or -vvv)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Solve command
    solve_parser = subparsers.add_parser(
        'solve',
        help='Solve a single ARC puzzle',
        description='Solve a single ARC puzzle from JSON file'
    )
    
    solve_parser.add_argument(
        'task_file',
        type=str,
        help='Path to ARC task JSON file'
    )
    
    solve_parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=30.0,
        help='Timeout in seconds (default: 30.0)'
    )
    
    solve_parser.add_argument(
        '--max-program-length',
        type=int,
        default=4,
        help='Maximum DSL program length (default: 4)'
    )
    
    solve_parser.add_argument(
        '--beam-width',
        type=int,
        default=64,
        help='Beam search width (default: 64)'
    )
    
    solve_parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    solve_parser.add_argument(
        '--save-search-tree',
        action='store_true',
        help='Save search tree for debugging'
    )
    
    solve_parser.add_argument(
        '--disable-multi-example',
        action='store_true',
        help='Disable multi-example validation (use only first training example)'
    )

    # LLM options
    solve_parser.add_argument(
        '--llm',
        dest='llm_enabled',
        action='store_true',
        help='Enable LLM-guided search (priority-boost; preserves admissibility)'
    )
    solve_parser.add_argument(
        '--llm-model',
        type=str,
        default='local/mock',
        help='LLM model name (transformers hub id or local path)'
    )
    solve_parser.add_argument(
        '--llm-proposals',
        type=int,
        default=2,
        help='Number of LLM proposals to generate'
    )
    solve_parser.add_argument(
        '--llm-priority-boost',
        type=float,
        default=1.0,
        help='Tie-break priority boost factor for LLM-matching ops'
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Process multiple ARC puzzles',
        description='Process multiple ARC puzzles from directory or file list'
    )
    
    batch_parser.add_argument(
        'input_path',
        type=str,
        help='Directory containing JSON files or file with list of paths'
    )
    
    batch_parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=30.0,
        help='Timeout per puzzle in seconds (default: 30.0)'
    )
    
    batch_parser.add_argument(
        '--threads', '-j',
        type=int,
        default=1,
        help='Number of parallel threads (default: 1)'
    )
    
    batch_parser.add_argument(
        '--max-tasks',
        type=int,
        help='Maximum number of tasks to process'
    )
    
    batch_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous batch run'
    )
    
    batch_parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle task order'
    )
    
    batch_parser.add_argument(
        '--report-interval',
        type=int,
        default=10,
        help='Progress report interval (default: 10)'
    )
    
    batch_parser.add_argument(
        '--disable-multi-example',
        action='store_true',
        help='Disable multi-example validation for batch processing'
    )

    # LLM options for batch mode
    batch_parser.add_argument(
        '--llm',
        dest='llm_enabled',
        action='store_true',
        help='Enable LLM-guided search for batch mode'
    )
    batch_parser.add_argument(
        '--llm-model',
        type=str,
        default='local/mock',
        help='LLM model name (transformers hub id or local path)'
    )
    batch_parser.add_argument(
        '--llm-proposals',
        type=int,
        default=2,
        help='Number of LLM proposals to generate per task'
    )
    batch_parser.add_argument(
        '--llm-priority-boost',
        type=float,
        default=1.0,
        help='Tie-break priority boost factor for LLM-matching ops'
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management',
        description='Manage solver configuration'
    )
    
    config_subparsers = config_parser.add_subparsers(
        dest='config_action',
        help='Configuration actions'
    )
    
    # Config show
    config_subparsers.add_parser(
        'show',
        help='Show current configuration'
    )
    
    # Config validate
    config_subparsers.add_parser(
        'validate',
        help='Validate configuration'
    )
    
    # Config set
    config_set_parser = config_subparsers.add_parser(
        'set',
        help='Set configuration parameter'
    )
    config_set_parser.add_argument('key', help='Parameter key (e.g., search.beam_width)')
    config_set_parser.add_argument('value', help='Parameter value')
    
    # Test command
    test_parser = subparsers.add_parser(
        'test',
        help='Run system tests',
        description='Run system tests to verify installation'
    )
    
    test_parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only'
    )
    
    test_parser.add_argument(
        '--component',
        choices=['perception', 'reasoning', 'search', 'caching', 'all'],
        default='all',
        help='Test specific component (default: all)'
    )
    
    # Serve command
    serve_parser = subparsers.add_parser(
        'serve',
        help='Launch the interactive web GUI',
        description='Launch the interactive web GUI for visualizing solver progress'
    )

    serve_parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind the server to (default: 127.0.0.1)'
    )

    serve_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the server on (default: 8000)'
    )

    return parser


def main_cli(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Setup logging based on verbosity
    if parsed_args.quiet:
        log_level = logging.ERROR
    elif parsed_args.verbose == 0:
        log_level = logging.WARNING
    elif parsed_args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Handle no command case (return 1 per tests)
        if not parsed_args.command:
            parser.print_help()
            return 1

        # Route to appropriate command handler
        if parsed_args.command == 'solve':
            return commands.solve_command(parsed_args)
        if parsed_args.command == 'batch':
            return commands.batch_command(parsed_args)
        if parsed_args.command == 'config':
            return commands.config_command(parsed_args)
        if parsed_args.command == 'test':
            return commands.test_command(parsed_args)
        if parsed_args.command == 'serve':
            return commands.serve_command(parsed_args)

        logger.error(f"Unknown command: {parsed_args.command}")
        return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        # For tests that patch command handlers, ensure we propagate the mocked return
        logger.error(f"Unexpected error: {e}")
        return 1


def main() -> None:
    """Entry point for console script."""
    sys.exit(main_cli())


if __name__ == '__main__':
    main()
