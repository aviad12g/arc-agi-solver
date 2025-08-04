"""Tests for CLI interface."""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from arc_solver.cli.main import main_cli, create_parser
from arc_solver.cli.utils import (
    load_task_from_file, save_results, find_task_files, format_duration,
    format_memory, ProgressReporter, create_result_summary
)
from arc_solver.cli.commands import ARCSolver


class TestCLIParser:
    """Test CLI argument parsing."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser.prog == 'arc-solver'
    
    def test_solve_command_parsing(self):
        """Test solve command parsing."""
        parser = create_parser()
        
        # Basic solve command
        args = parser.parse_args(['solve', 'task.json'])
        assert args.command == 'solve'
        assert args.task_file == 'task.json'
        assert args.timeout == 30.0
        assert args.max_program_length == 4
        assert args.beam_width == 64
        
        # Solve with options
        args = parser.parse_args([
            'solve', 'task.json',
            '--timeout', '60',
            '--max-program-length', '2',
            '--beam-width', '32',
            '--no-cache'
        ])
        assert args.timeout == 60.0
        assert args.max_program_length == 2
        assert args.beam_width == 32
        assert args.no_cache is True
    
    def test_batch_command_parsing(self):
        """Test batch command parsing."""
        parser = create_parser()
        
        # Basic batch command
        args = parser.parse_args(['batch', 'puzzles/'])
        assert args.command == 'batch'
        assert args.input_path == 'puzzles/'
        assert args.timeout == 30.0
        assert args.threads == 1
        
        # Batch with options
        args = parser.parse_args([
            'batch', 'puzzles/',
            '--timeout', '45',
            '--threads', '4',
            '--max-tasks', '100',
            '--shuffle'
        ])
        assert args.timeout == 45.0
        assert args.threads == 4
        assert args.max_tasks == 100
        assert args.shuffle is True
    
    def test_config_command_parsing(self):
        """Test config command parsing."""
        parser = create_parser()
        
        # Config show
        args = parser.parse_args(['config', 'show'])
        assert args.command == 'config'
        assert args.config_action == 'show'
        
        # Config validate
        args = parser.parse_args(['config', 'validate'])
        assert args.config_action == 'validate'
        
        # Config set
        args = parser.parse_args(['config', 'set', 'search.beam_width', '32'])
        assert args.config_action == 'set'
        assert args.key == 'search.beam_width'
        assert args.value == '32'
    
    def test_test_command_parsing(self):
        """Test test command parsing."""
        parser = create_parser()
        
        # Basic test command
        args = parser.parse_args(['test'])
        assert args.command == 'test'
        assert args.component == 'all'
        assert args.quick is False
        
        # Test with options
        args = parser.parse_args(['test', '--quick', '--component', 'perception'])
        assert args.quick is True
        assert args.component == 'perception'
    
    def test_global_options(self):
        """Test global options parsing."""
        parser = create_parser()
        
        # Verbose options
        args = parser.parse_args(['-v', 'solve', 'task.json'])
        assert args.verbose == 1
        
        args = parser.parse_args(['-vv', 'solve', 'task.json'])
        assert args.verbose == 2
        
        # Quiet option
        args = parser.parse_args(['--quiet', 'solve', 'task.json'])
        assert args.quiet is True
        
        # Config and output options
        args = parser.parse_args([
            '--config', 'search.beam_width=32',
            '--output', 'results.json',
            'solve', 'task.json'
        ])
        assert args.config == 'search.beam_width=32'
        assert args.output == 'results.json'


class TestCLIUtils:
    """Test CLI utility functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_task_data(self):
        """Create sample task data."""
        return {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[4, 3], [2, 1]]
                }
            ],
            "test": [
                {
                    "input": [[5, 6], [7, 8]],
                    "output": [[8, 7], [6, 5]]
                }
            ]
        }
    
    def test_load_task_from_file(self, temp_dir, sample_task_data):
        """Test loading task from JSON file."""
        # Create test task file
        task_file = Path(temp_dir) / "test_task.json"
        with open(task_file, 'w') as f:
            json.dump(sample_task_data, f)
        
        # Load task
        task = load_task_from_file(task_file)
        
        assert len(task.train) == 1
        assert len(task.test) == 1
        assert task.train[0].input.shape == (2, 2)
    
    def test_load_task_file_not_found(self):
        """Test loading non-existent task file."""
        with pytest.raises(FileNotFoundError):
            load_task_from_file("nonexistent.json")
    
    def test_save_results(self, temp_dir):
        """Test saving results to JSON file."""
        results = {
            "success": True,
            "computation_time": 1.23,
            "program": {"operations": []}
        }
        
        output_file = Path(temp_dir) / "results.json"
        save_results(results, output_file)
        
        assert output_file.exists()
        
        # Load and verify
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results["success"] is True
        assert loaded_results["computation_time"] == 1.23
    
    def test_find_task_files_directory(self, temp_dir, sample_task_data):
        """Test finding task files in directory."""
        # Create test files
        for i in range(3):
            task_file = Path(temp_dir) / f"task_{i}.json"
            with open(task_file, 'w') as f:
                json.dump(sample_task_data, f)
        
        # Create non-JSON file
        (Path(temp_dir) / "readme.txt").write_text("Not a task file")
        
        # Find task files
        task_files = find_task_files(temp_dir)
        
        assert len(task_files) == 3
        assert all(f.suffix == '.json' for f in task_files)
    
    def test_find_task_files_single_file(self, temp_dir, sample_task_data):
        """Test finding single task file."""
        task_file = Path(temp_dir) / "single_task.json"
        with open(task_file, 'w') as f:
            json.dump(sample_task_data, f)
        
        task_files = find_task_files(task_file)
        
        assert len(task_files) == 1
        assert task_files[0] == task_file
    
    def test_find_task_files_with_limit(self, temp_dir, sample_task_data):
        """Test finding task files with limit."""
        # Create 5 test files
        for i in range(5):
            task_file = Path(temp_dir) / f"task_{i}.json"
            with open(task_file, 'w') as f:
                json.dump(sample_task_data, f)
        
        # Find with limit
        task_files = find_task_files(temp_dir, max_files=3)
        
        assert len(task_files) == 3
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(0.0005) == "500.0µs"
        assert format_duration(0.5) == "500.0ms"
        assert format_duration(1.5) == "1.50s"
        assert format_duration(65.5) == "1m 5.5s"
        assert format_duration(3665.5) == "1h 1m 5.5s"
    
    def test_format_memory(self):
        """Test memory formatting."""
        assert format_memory(512) == "512B"
        assert format_memory(1536) == "1.5KB"
        assert format_memory(1536 * 1024) == "1.5MB"
        assert format_memory(1536 * 1024 * 1024) == "1.5GB"
    
    def test_progress_reporter(self):
        """Test progress reporter."""
        reporter = ProgressReporter(total_tasks=10, report_interval=3)
        
        assert reporter.total_tasks == 10
        assert reporter.completed_tasks == 0
        assert reporter.successful_tasks == 0
        
        # Update progress
        reporter.update(success=True)
        assert reporter.completed_tasks == 1
        assert reporter.successful_tasks == 1
        
        reporter.update(success=False)
        assert reporter.completed_tasks == 2
        assert reporter.successful_tasks == 1
    
    def test_create_result_summary(self):
        """Test result summary creation."""
        results = [
            {"success": True, "computation_time": 1.0},
            {"success": False, "computation_time": 2.0},
            {"success": True, "computation_time": 1.5},
        ]
        
        summary = create_result_summary(results)
        
        assert summary["total_tasks"] == 3
        assert summary["successful_tasks"] == 2
        assert summary["failed_tasks"] == 1
        assert summary["success_rate"] == 2/3
        assert summary["average_time"] == 1.5
        assert summary["median_time"] == 1.5
        assert summary["total_time"] == 4.5
    
    def test_create_result_summary_empty(self):
        """Test result summary with empty results."""
        summary = create_result_summary([])
        
        assert summary["total_tasks"] == 0
        assert summary["successful_tasks"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["average_time"] == 0.0


class TestARCSolver:
    """Test ARC solver integration."""
    
    @pytest.fixture
    def mock_task(self):
        """Create mock task object."""
        task = MagicMock()
        task.train = [MagicMock()]
        task.train[0].input = [[1, 2], [3, 4]]
        task.train[0].output = [[4, 3], [2, 1]]
        task.test = [MagicMock()]
        task.test[0].input = [[5, 6], [7, 8]]
        return task
    
    @patch('arc_solver.cli.commands.load_config')
    @patch('arc_solver.cli.commands.create_cache_manager')
    @patch('arc_solver.cli.commands.create_heuristic_system')
    @patch('arc_solver.cli.commands.DSLEngine')
    @patch('arc_solver.cli.commands.create_astar_searcher')
    def test_solver_initialization(self, mock_searcher, mock_dsl, mock_heuristic, 
                                  mock_cache, mock_config):
        """Test solver initialization."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_cache.return_value = MagicMock()
        mock_heuristic.return_value = MagicMock()
        mock_dsl.return_value = MagicMock()
        mock_searcher.return_value = MagicMock()
        
        # Create solver
        solver = ARCSolver()
        
        # Verify initialization
        assert solver.config is not None
        assert solver.cache_manager is not None
        assert solver.heuristic_system is not None
        assert solver.dsl_engine is not None
        assert solver.searcher is not None
    
    @patch('arc_solver.cli.commands.load_config')
    @patch('arc_solver.cli.commands.create_cache_manager')
    @patch('arc_solver.cli.commands.create_heuristic_system')
    @patch('arc_solver.cli.commands.DSLEngine')
    @patch('arc_solver.cli.commands.create_astar_searcher')
    def test_solve_task_success(self, mock_searcher, mock_dsl, mock_heuristic, 
                               mock_cache, mock_config, mock_task):
        """Test successful task solving."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_cache.return_value = MagicMock()
        mock_heuristic.return_value = MagicMock()
        mock_dsl_instance = MagicMock()
        mock_dsl.return_value = mock_dsl_instance
        
        # Mock successful search result
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.program = MagicMock()
        mock_search_result.program.to_dict.return_value = {"operations": []}
        mock_search_result.nodes_expanded = 10
        mock_search_result.nodes_generated = 20
        mock_search_result.max_depth_reached = 2
        mock_search_result.termination_reason = "goal_reached"
        
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = mock_search_result
        mock_searcher.return_value = mock_searcher_instance
        
        # Mock DSL execution
        mock_dsl_instance.execute_program.return_value = ([[8, 7], [6, 5]], {"success": True})
        
        # Create solver and solve task
        solver = ARCSolver()
        result = solver.solve_task(mock_task, timeout=30.0)
        
        # Verify result
        assert result["success"] is True
        assert "program" in result
        assert "predictions" in result
        assert "search_stats" in result
        assert "computation_time" in result
        assert len(result["predictions"]) == 1
    
    @patch('arc_solver.cli.commands.load_config')
    @patch('arc_solver.cli.commands.create_cache_manager')
    @patch('arc_solver.cli.commands.create_heuristic_system')
    @patch('arc_solver.cli.commands.DSLEngine')
    @patch('arc_solver.cli.commands.create_astar_searcher')
    def test_solve_task_failure(self, mock_searcher, mock_dsl, mock_heuristic, 
                               mock_cache, mock_config, mock_task):
        """Test failed task solving."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_cache.return_value = MagicMock()
        mock_heuristic.return_value = MagicMock()
        mock_dsl.return_value = MagicMock()
        
        # Mock failed search result
        mock_search_result = MagicMock()
        mock_search_result.success = False
        mock_search_result.termination_reason = "timeout"
        mock_search_result.nodes_expanded = 100
        mock_search_result.nodes_generated = 200
        
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = mock_search_result
        mock_searcher.return_value = mock_searcher_instance
        
        # Create solver and solve task
        solver = ARCSolver()
        result = solver.solve_task(mock_task, timeout=30.0)
        
        # Verify result
        assert result["success"] is False
        assert "error" in result
        assert "search_stats" in result
        assert "computation_time" in result
        assert result["error"] == "Search failed: timeout"


class TestMainCLI:
    """Test main CLI function."""
    
    def test_main_cli_no_command(self):
        """Test CLI with no command."""
        exit_code = main_cli([])
        assert exit_code == 1
    
    def test_main_cli_help(self):
        """Test CLI help."""
        with pytest.raises(SystemExit) as exc_info:
            main_cli(['--help'])
        assert exc_info.value.code == 0
    
    @patch('arc_solver.cli.commands.solve_command')
    def test_main_cli_solve_command(self, mock_solve):
        """Test CLI solve command routing."""
        mock_solve.return_value = 0
        
        exit_code = main_cli(['solve', 'task.json'])
        
        assert exit_code == 0
        mock_solve.assert_called_once()
    
    @patch('arc_solver.cli.commands.batch_command')
    def test_main_cli_batch_command(self, mock_batch):
        """Test CLI batch command routing."""
        mock_batch.return_value = 0
        
        exit_code = main_cli(['batch', 'puzzles/'])
        
        assert exit_code == 0
        mock_batch.assert_called_once()
    
    @patch('arc_solver.cli.commands.config_command')
    def test_main_cli_config_command(self, mock_config):
        """Test CLI config command routing."""
        mock_config.return_value = 0
        
        exit_code = main_cli(['config', 'show'])
        
        assert exit_code == 0
        mock_config.assert_called_once()
    
    @patch('arc_solver.cli.commands.test_command')
    def test_main_cli_test_command(self, mock_test):
        """Test CLI test command routing."""
        mock_test.return_value = 0
        
        exit_code = main_cli(['test'])
        
        assert exit_code == 0
        mock_test.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])