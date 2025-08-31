"""Deterministic end-to-end micro-benchmarks for ARC solver.

Runs a few synthetic tasks (no dataset required) with deterministic settings
and reports timing and success. Useful for quick regressions and CI smoke tests.

Usage:
    python scripts/benchmark_e2e_deterministic.py [--quick] [--output FILE.json]
"""

from __future__ import annotations

import time
import argparse
import json
import sys
from typing import List, Tuple, Dict, Any
import numpy as np
import statistics

from arc_solver.cli.commands import ARCSolver


def make_task(train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_inputs: List[np.ndarray], task_id: str):
    class Mock:
        pass

    class TrainEx:
        def __init__(self, i, o):
            self.input = i
            self.output = o

    class TestEx:
        def __init__(self, i):
            self.input = i

    t = Mock()
    t.task_id = task_id
    t.train = [TrainEx(tp[0].tolist(), tp[1].tolist()) for tp in train_pairs]
    t.test = [TestEx(ti.tolist()) for ti in test_inputs]
    return t


def rotate90_task() -> Tuple[str, object]:
    src = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dst = np.rot90(src, -1)
    test = [np.array([[5, 6], [7, 8]], dtype=np.int32)]
    return "rotate90", make_task([(src, dst)], test, "rotate90")


def reflectv_task() -> Tuple[str, object]:
    src = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dst = np.flipud(src)
    test = [np.array([[8, 7], [6, 5]], dtype=np.int32)]
    return "reflectv", make_task([(src, dst)], test, "reflectv")


def color_perm_task() -> Tuple[str, object]:
    src = np.array([[1, 2], [2, 1]], dtype=np.int32)
    # Map 1->3, 2->4
    dst = np.where(src == 1, 3, np.where(src == 2, 4, src)).astype(np.int32)
    test = [np.array([[2, 1], [1, 2]], dtype=np.int32)]
    return "colorperm", make_task([(src, dst)], test, "colorperm")


def translate_task() -> Tuple[str, object]:
    src = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int32)
    dst = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 1]], dtype=np.int32)  # shift down+right (1,1)
    test = [np.array([[0, 2, 0], [0, 2, 0], [0, 2, 0]], dtype=np.int32)]
    return "translate", make_task([(src, dst)], test, "translate")


def run(quick: bool = True) -> Dict[str, Any]:
    """Run the benchmark and return structured results.

    Args:
        quick: Whether to use quick settings (tight timeouts, fewer tasks)

    Returns:
        Dictionary with benchmark results
    """
    tasks = [rotate90_task(), reflectv_task(), color_perm_task(), translate_task()]

    if quick:
        # Quick settings for CI/regression testing
        overrides = [
            # Deterministic settings
            "development.testing.deterministic_mode=true",
            # Tight A* limits for quick runs
            "search.astar.max_program_length=4",
            "search.astar.max_nodes_expanded=600",
            "search.beam_search.initial_beam_width=32",
        ]
        timeout = 5.0
    else:
        # More thorough settings for detailed analysis
        overrides = [
            # Deterministic settings
            "development.testing.deterministic_mode=true",
            # Allow more exploration
            "search.astar.max_program_length=6",
            "search.astar.max_nodes_expanded=2000",
            "search.beam_search.initial_beam_width=64",
        ]
        timeout = 15.0

    solver = ARCSolver(overrides)

    results = []
    t0 = time.perf_counter()
    for name, task in tasks:
        start = time.perf_counter()
        res = solver.solve_task(task, timeout=timeout, use_multi_example=False)
        dt = time.perf_counter() - start
        ok = bool(res.get("success", False))
        stats = res.get("search_stats", {})
        results.append({
            "task_name": name,
            "success": ok,
            "solve_time": dt,
            "nodes_expanded": stats.get("nodes_expanded", 0),
            "nodes_generated": stats.get("nodes_generated", 0),
            "max_depth": stats.get("max_depth_reached", 0)
        })

    total_time = time.perf_counter() - t0

    # Calculate summary statistics
    successful_results = [r for r in results if r["success"]]
    solve_times = [r["solve_time"] for r in results]
    nodes_expanded = [r["nodes_expanded"] for r in results]

    summary = {
        "total_tasks": len(results),
        "successful_tasks": len(successful_results),
        "success_rate": len(successful_results) / len(results) if results else 0.0,
        "median_solve_time": statistics.median(solve_times) if solve_times else 0.0,
        "mean_solve_time": statistics.mean(solve_times) if solve_times else 0.0,
        "median_nodes_expanded": statistics.median(nodes_expanded) if nodes_expanded else 0,
        "total_time": total_time,
        "timestamp": time.time(),
        "config": "quick" if quick else "thorough",
        "results": results
    }

    return summary


def print_summary(results: Dict[str, Any]) -> None:
    """Print human-readable summary to stdout."""
    print("Deterministic E2E Benchmarks")
    print("=" * 40)

    for result in results["results"]:
        status = "OK" if result["success"] else "FAIL"
        print(f"{result['task_name']:12s} {status:4s} time={result['solve_time']*1000:.1f}ms nodes={result['nodes_expanded']}")

    print("-" * 40)
    print(f"Success rate: {results['success_rate']:.3f} ({results['successful_tasks']}/{results['total_tasks']})")
    print(f"Median solve time: {results['median_solve_time']:.3f}s")
    print(f"Median nodes expanded: {results['median_nodes_expanded']}")
    print(f"Total time: {results['total_time']:.3f}s")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Run deterministic E2E benchmarks for ARC solver")
    parser.add_argument("--quick", action="store_true", default=True,
                       help="Run quick benchmark with tight limits (default: True)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output JSON file path (if not specified, prints to stdout only)")

    args = parser.parse_args()

    try:
        # Run the benchmark
        results = run(quick=args.quick)

        # Print human-readable summary
        print_summary(results)

        # Write JSON output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults written to {args.output}")

        # Exit with success/failure based on results
        success_rate = results["success_rate"]
        if success_rate >= 0.5:  # At least 50% success rate
            sys.exit(0)
        else:
            print(f"Warning: Low success rate ({success_rate:.1%})")
            sys.exit(1)

    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

