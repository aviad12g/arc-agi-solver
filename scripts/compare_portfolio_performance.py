#!/usr/bin/env python3
"""Compare portfolio scheduler performance vs direct A* search.

This script runs the same benchmark tasks with portfolio enabled vs disabled
to quantify the performance gains from multi-lane orchestration.

Usage:
    python scripts/compare_portfolio_performance.py [--quick] [--output results.json]
    python scripts/compare_portfolio_performance.py --comprehensive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, Any, List
import statistics

# Add src to path for imports
sys.path.insert(0, str(sys.path[0]).replace('/scripts', '/src'))

from arc_solver.cli.commands import ARCSolver


def run_comparison(quick: bool = True) -> Dict[str, Any]:
    """Run performance comparison between portfolio enabled vs disabled."""

    print("=== PORTFOLIO PERFORMANCE COMPARISON ===")
    print("Running benchmark with portfolio enabled vs disabled...")

    # Configure for portfolio enabled
    portfolio_overrides = [
        "search.portfolio.enabled=true",
        "search.advanced.canonicalization.enabled=true",
        "search.advanced.unsat_cache.enabled=true",
        "search.advanced.unsat_cache.signatures_file=unsat_signatures.jsonl",
        "search.advanced.deadend_predictor.enabled=true",
        "search.advanced.deadend_predictor.model_file=deadend_model.json",
        "search.advanced.pdb.enabled=true",
    ]

    # Configure for portfolio disabled (direct A* only)
    direct_overrides = [
        "search.portfolio.enabled=false",
        "search.advanced.canonicalization.enabled=true",
        "search.advanced.unsat_cache.enabled=true",
        "search.advanced.unsat_cache.signatures_file=unsat_signatures.jsonl",
        "search.advanced.deadend_predictor.enabled=true",
        "search.advanced.deadend_predictor.model_file=deadend_model.json",
        "search.advanced.pdb.enabled=true",
    ]

    if quick:
        # Quick settings for fast comparison
        portfolio_overrides.extend([
            "search.astar.max_program_length=4",
            "search.astar.max_nodes_expanded=600",
            "search.beam_search.initial_beam_width=32",
        ])
        direct_overrides.extend([
            "search.astar.max_program_length=4",
            "search.astar.max_nodes_expanded=600",
            "search.beam_search.initial_beam_width=32",
        ])
        timeout = 5.0
        description = "quick benchmark"
    else:
        # Comprehensive settings
        portfolio_overrides.extend([
            "search.astar.max_program_length=6",
            "search.astar.max_nodes_expanded=2000",
            "search.beam_search.initial_beam_width=64",
        ])
        direct_overrides.extend([
            "search.astar.max_program_length=6",
            "search.astar.max_nodes_expanded=2000",
            "search.beam_search.initial_beam_width=64",
        ])
        timeout = 15.0
        description = "comprehensive benchmark"

    print(f"Configuration: {description}")
    print(f"Timeout per task: {timeout}s")
    print()

    # Create synthetic test tasks
    tasks = [
        # Simple rotation task
        {
            "task_id": "rotation_test",
            "input": [[1, 2], [3, 4]],
            "output": [[3, 1], [4, 2]]
        },
        # Simple reflection task
        {
            "task_id": "reflection_test",
            "input": [[1, 2], [3, 4]],
            "output": [[2, 1], [4, 3]]
        },
        # Color permutation task
        {
            "task_id": "color_perm_test",
            "input": [[1, 2], [2, 1]],
            "output": [[2, 1], [1, 2]]
        },
        # Translation task
        {
            "task_id": "translate_test",
            "input": [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            "output": [[0, 0, 0], [1, 0, 1], [1, 0, 1]]
        }
    ]

    # Convert to Task objects
    test_tasks = []
    for task_data in tasks:
        class MockTask:
            def __init__(self, data):
                self.task_id = data["task_id"]
                self.train_examples = [(data["input"], data["output"])]
                self.test_inputs = [data["input"]]  # Use same input as test

        test_tasks.append(MockTask(task_data))

    results = {
        "configuration": description,
        "portfolio_enabled": {},
        "portfolio_disabled": {},
        "comparison": {}
    }

    # Test with portfolio enabled
    print("🔄 Testing with PORTFOLIO ENABLED...")
    solver_portfolio = ARCSolver(portfolio_overrides)
    portfolio_results = []

    for task in test_tasks:
        start_time = time.time()
        result = solver_portfolio.solve_task(task, timeout=timeout, use_multi_example=False)
        elapsed = time.time() - start_time

        task_result = {
            "task_id": task.task_id,
            "success": result.get("success", False),
            "solve_time": result.get("computation_time", elapsed),
            "nodes_expanded": result.get("search_stats", {}).get("nodes_expanded", 0),
            "nodes_generated": result.get("search_stats", {}).get("nodes_generated", 0),
        }
        portfolio_results.append(task_result)
        print(f"  {task.task_id}: {'✅' if task_result['success'] else '❌'} "
              f"{task_result['solve_time']:.3f}s, {task_result['nodes_expanded']} nodes")

    results["portfolio_enabled"] = {
        "tasks": portfolio_results,
        "summary": {
            "total_tasks": len(portfolio_results),
            "successful_tasks": len([r for r in portfolio_results if r["success"]]),
            "success_rate": len([r for r in portfolio_results if r["success"]]) / len(portfolio_results),
            "median_solve_time": statistics.median([r["solve_time"] for r in portfolio_results]),
            "mean_solve_time": statistics.mean([r["solve_time"] for r in portfolio_results]),
            "median_nodes_expanded": statistics.median([r["nodes_expanded"] for r in portfolio_results]),
            "total_time": sum([r["solve_time"] for r in portfolio_results])
        }
    }

    # Test with portfolio disabled
    print("\n🎯 Testing with PORTFOLIO DISABLED...")
    solver_direct = ARCSolver(direct_overrides)
    direct_results = []

    for task in test_tasks:
        start_time = time.time()
        result = solver_direct.solve_task(task, timeout=timeout, use_multi_example=False)
        elapsed = time.time() - start_time

        task_result = {
            "task_id": task.task_id,
            "success": result.get("success", False),
            "solve_time": result.get("computation_time", elapsed),
            "nodes_expanded": result.get("search_stats", {}).get("nodes_expanded", 0),
            "nodes_generated": result.get("search_stats", {}).get("nodes_generated", 0),
        }
        direct_results.append(task_result)
        print(f"  {task.task_id}: {'✅' if task_result['success'] else '❌'} "
              f"{task_result['solve_time']:.3f}s, {task_result['nodes_expanded']} nodes")

    results["portfolio_disabled"] = {
        "tasks": direct_results,
        "summary": {
            "total_tasks": len(direct_results),
            "successful_tasks": len([r for r in direct_results if r["success"]]),
            "success_rate": len([r for r in direct_results if r["success"]]) / len(direct_results),
            "median_solve_time": statistics.median([r["solve_time"] for r in direct_results]),
            "mean_solve_time": statistics.mean([r["solve_time"] for r in direct_results]),
            "median_nodes_expanded": statistics.median([r["nodes_expanded"] for r in direct_results]),
            "total_time": sum([r["solve_time"] for r in direct_results])
        }
    }

    # Calculate comparison metrics
    p_enabled = results["portfolio_enabled"]["summary"]
    p_disabled = results["portfolio_disabled"]["summary"]

    results["comparison"] = {
        "median_solve_time_improvement": ((p_disabled["median_solve_time"] - p_enabled["median_solve_time"]) /
                                        p_disabled["median_solve_time"]) * 100,
        "median_nodes_reduction": ((p_disabled["median_nodes_expanded"] - p_enabled["median_nodes_expanded"]) /
                                 p_disabled["median_nodes_expanded"]) * 100 if p_disabled["median_nodes_expanded"] > 0 else 0,
        "total_time_improvement": ((p_disabled["total_time"] - p_enabled["total_time"]) /
                                 p_disabled["total_time"]) * 100,
        "success_rate_difference": p_enabled["success_rate"] - p_disabled["success_rate"]
    }

    return results


def print_comparison(results: Dict[str, Any]) -> None:
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📊 PORTFOLIO PERFORMANCE COMPARISON RESULTS")
    print("="*80)

    enabled = results["portfolio_enabled"]["summary"]
    disabled = results["portfolio_disabled"]["summary"]
    comparison = results["comparison"]

    print("\n🎯 PORTFOLIO ENABLED:")
    print(f"  Success Rate: {enabled['success_rate']:.1%} ({enabled['successful_tasks']}/{enabled['total_tasks']})")
    print(f"  Median Time: {enabled['median_solve_time']:.3f}s")
    print(f"  Mean Time:   {enabled['mean_solve_time']:.3f}s")
    print(f"  Median Nodes: {enabled['median_nodes_expanded']}")

    print("\n🔄 PORTFOLIO DISABLED:")
    print(f"  Success Rate: {disabled['success_rate']:.1%} ({disabled['successful_tasks']}/{disabled['total_tasks']})")
    print(f"  Median Time: {disabled['median_solve_time']:.3f}s")
    print(f"  Mean Time:   {disabled['mean_solve_time']:.3f}s")
    print(f"  Median Nodes: {disabled['median_nodes_expanded']}")

    print("\n📈 IMPROVEMENTS:")
    print(f"  Median Time Improvement: {comparison['median_solve_time_improvement']:.1f}%")
    print(f"  Median Nodes Reduction:  {comparison['median_nodes_reduction']:.1f}%")
    print(f"  Total Time Improvement:  {comparison['total_time_improvement']:.1f}%")
    print(f"  Success Rate Difference: {comparison['success_rate_difference']:.1%}")

    if comparison["median_solve_time_improvement"] > 0:
        print("\n✅ PORTFOLIO SCHEDULER SHOWS SIGNIFICANT PERFORMANCE GAINS!")
    else:
        print("\n⚠️  PORTFOLIO SCHEDULER OVERHEAD MAY OUTWEIGH BENEFITS FOR SIMPLE TASKS")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare portfolio vs direct A* performance")
    parser.add_argument("--quick", action="store_true", default=True,
                       help="Run quick comparison with tight limits (default: True)")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive comparison with relaxed limits")
    parser.add_argument("--output", "-o", type=str,
                       help="Output JSON file for detailed results")

    args = parser.parse_args()

    if args.comprehensive:
        args.quick = False

    try:
        # Run comparison
        results = run_comparison(quick=args.quick)

        # Print formatted results
        print_comparison(results)

        # Save detailed results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Detailed results saved to {args.output}")

        return 0

    except Exception as e:
        print(f"❌ Comparison failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
