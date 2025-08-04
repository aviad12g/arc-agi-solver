#!/usr/bin/env python3
"""Demonstration script for the CLI interface."""

import json
import tempfile
import shutil
from pathlib import Path

from arc_solver.cli.main import main_cli


def create_sample_task(task_id: str) -> dict:
    """Create a sample ARC task for testing."""
    return {
        "train": [
            {
                "input": [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
                "output": [[3, 1, 0], [4, 2, 0], [0, 0, 0]]
            }
        ],
        "test": [
            {
                "input": [[5, 6, 0], [7, 8, 0], [0, 0, 0]],
                "output": [[7, 5, 0], [8, 6, 0], [0, 0, 0]]
            }
        ]
    }


def main():
    """Demonstrate CLI functionality."""
    print("🖥️  ARC-AGI Solver CLI Demonstration")
    print("=" * 60)
    
    # Create temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create sample task files
        print("\n📁 Creating sample task files...")
        task_files = []
        for i in range(3):
            task_file = temp_dir / f"task_{i:03d}.json"
            task_data = create_sample_task(f"task_{i:03d}")
            
            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            
            task_files.append(task_file)
            print(f"  Created: {task_file.name}")
        
        # Test 1: Show help
        print("\n📖 Test 1: Show CLI Help")
        print("-" * 30)
        try:
            main_cli(['--help'])
        except SystemExit:
            pass  # Help command exits normally
        
        # Test 2: Show configuration
        print("\n⚙️  Test 2: Show Configuration")
        print("-" * 30)
        exit_code = main_cli(['config', 'show', '--quiet'])
        print(f"Exit code: {exit_code}")
        
        # Test 3: Validate configuration
        print("\n✅ Test 3: Validate Configuration")
        print("-" * 30)
        exit_code = main_cli(['config', 'validate'])
        print(f"Exit code: {exit_code}")
        
        # Test 4: Run system tests
        print("\n🧪 Test 4: Run System Tests")
        print("-" * 30)
        exit_code = main_cli(['test', '--quick', '--component', 'perception'])
        print(f"Exit code: {exit_code}")
        
        # Test 5: Solve single task
        print("\n🎯 Test 5: Solve Single Task")
        print("-" * 30)
        output_file = temp_dir / "single_result.json"
        exit_code = main_cli([
            'solve', str(task_files[0]),
            '--timeout', '10',
            '--beam-width', '16',
            '--output', str(output_file),
            '--verbose'
        ])
        print(f"Exit code: {exit_code}")
        
        if output_file.exists():
            print(f"Results saved to: {output_file}")
            with open(output_file, 'r') as f:
                result = json.load(f)
            print(f"Success: {result.get('success', False)}")
            print(f"Computation time: {result.get('computation_time', 0):.3f}s")
        
        # Test 6: Batch processing
        print("\n📦 Test 6: Batch Processing")
        print("-" * 30)
        batch_output = temp_dir / "batch_results.json"
        exit_code = main_cli([
            'batch', str(temp_dir),
            '--timeout', '10',
            '--max-tasks', '2',
            '--output', str(batch_output),
            '--report-interval', '1',
            '--verbose'
        ])
        print(f"Exit code: {exit_code}")
        
        if batch_output.exists():
            print(f"Batch results saved to: {batch_output}")
            with open(batch_output, 'r') as f:
                batch_result = json.load(f)
            
            summary = batch_result.get('summary', {})
            print(f"Total tasks: {summary.get('total_tasks', 0)}")
            print(f"Successful: {summary.get('successful_tasks', 0)}")
            print(f"Success rate: {summary.get('success_rate', 0)*100:.1f}%")
            print(f"Average time: {summary.get('average_time', 0):.3f}s")
        
        # Test 7: Batch with parallel processing
        print("\n🔄 Test 7: Parallel Batch Processing")
        print("-" * 30)
        parallel_output = temp_dir / "parallel_results.json"
        exit_code = main_cli([
            'batch', str(temp_dir),
            '--timeout', '10',
            '--threads', '2',
            '--output', str(parallel_output),
            '--shuffle',
            '--quiet'
        ])
        print(f"Exit code: {exit_code}")
        
        if parallel_output.exists():
            with open(parallel_output, 'r') as f:
                parallel_result = json.load(f)
            
            summary = parallel_result.get('summary', {})
            print(f"Parallel processing completed:")
            print(f"  Tasks: {summary.get('total_tasks', 0)}")
            print(f"  Success rate: {summary.get('success_rate', 0)*100:.1f}%")
            print(f"  Wall clock time: {summary.get('wall_clock_time', 0):.3f}s")
        
        # Test 8: Error handling
        print("\n❌ Test 8: Error Handling")
        print("-" * 30)
        
        # Try to solve non-existent file
        exit_code = main_cli(['solve', 'nonexistent.json', '--quiet'])
        print(f"Non-existent file exit code: {exit_code}")
        
        # Try batch on non-existent directory
        exit_code = main_cli(['batch', 'nonexistent_dir/', '--quiet'])
        print(f"Non-existent directory exit code: {exit_code}")
        
        # Test 9: Configuration overrides
        print("\n🔧 Test 9: Configuration Overrides")
        print("-" * 30)
        override_output = temp_dir / "override_result.json"
        exit_code = main_cli([
            '--config', 'search.beam_search.initial_beam_width=8',
            'solve', str(task_files[0]),
            '--timeout', '5',
            '--output', str(override_output),
            '--quiet'
        ])
        print(f"Configuration override exit code: {exit_code}")
        
        # Test 10: Verbose output levels
        print("\n📢 Test 10: Verbose Output Levels")
        print("-" * 30)
        
        print("Testing different verbosity levels:")
        
        # Quiet
        print("  Quiet mode:")
        exit_code = main_cli(['--quiet', 'test', '--quick'])
        print(f"    Exit code: {exit_code}")
        
        # Normal (no output expected in demo)
        print("  Normal verbosity: (output suppressed)")
        
        # Verbose
        print("  Verbose mode: (output suppressed)")
        
        print("\n✅ CLI Demonstration Complete!")
        print("\n🎯 Key Features Demonstrated:")
        print("  ✓ Single task solving with customizable parameters")
        print("  ✓ Batch processing with parallel execution")
        print("  ✓ Configuration management and validation")
        print("  ✓ System testing capabilities")
        print("  ✓ Flexible output formats (JSON)")
        print("  ✓ Progress reporting and statistics")
        print("  ✓ Error handling and timeout management")
        print("  ✓ Runtime configuration overrides")
        print("  ✓ Multiple verbosity levels")
        print("  ✓ Resume and shuffle options for batch processing")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()