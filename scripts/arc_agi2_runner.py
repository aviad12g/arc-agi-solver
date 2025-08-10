#!/usr/bin/env python3
"""ARC-AGI-2 smoke runner.

Runs a fixed-seed, deterministic evaluation on a directory of ARC-AGI-2
JSON tasks and writes results and metadata to an output JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from arc_solver.cli.commands import ARCSolver
from arc_solver.integration.io import load_arc_task


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 smoke evaluation runner")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with ARC tasks (.json)")
    parser.add_argument("--output", type=str, required=True, help="Output results JSON path")
    parser.add_argument("--max-tasks", type=int, default=10, help="Max tasks to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    args = parser.parse_args()

    # Deterministic seeds
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)

    # Initialize solver with deterministic overrides
    overrides = [
        "development.testing.deterministic_mode=true",
        f"development.testing.random_seed={args.seed}",
    ]
    solver = ARCSolver(config_overrides=overrides)

    data_dir = Path(args.data_dir)
    task_files = sorted([p for p in data_dir.glob("*.json")])[: args.max_tasks]

    results: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for task_path in task_files:
        task = load_arc_task(task_path)
        res = solver.solve_task(task, timeout=30.0)
        res["task_file"] = str(task_path)
        results.append(res)

    total_time = time.perf_counter() - start
    success = sum(1 for r in results if r.get("success"))
    summary = {
        "total_tasks": len(results),
        "successful_tasks": success,
        "success_rate": (success / max(1, len(results))),
        "total_time_sec": total_time,
        "seed": args.seed,
        "env": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


