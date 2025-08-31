#!/usr/bin/env python3
"""Seed the retrieval store with solved ARC tasks for cross-run reuse.

This script populates the retrieval store with solved examples that can be
reused across different runs, enabling the system to quickly solve similar tasks.

Usage:
    python scripts/seed_retrieval_store.py [--input arc_tasks.json] [--output retrieval_store.jsonl]
    python scripts/seed_retrieval_store.py --synthetic  # Use synthetic examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.cli.commands import ARCSolver
from arc_solver.core.data_models import Task


def create_synthetic_solved_examples() -> List[Dict[str, Any]]:
    """Create synthetic solved examples for seeding the retrieval store."""
    examples = []

    # Example 1: Simple rotation task
    task_data = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[3, 1], [4, 2]]
            }
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]]
            }
        ]
    }

    examples.append({
        "task_id": "synthetic_rotation",
        "task_data": task_data,
        "solved_program": [
            {"primitive": "Rotate90", "parameters": {}}
        ],
        "solve_time": 0.001,
        "confidence": 0.95
    })

    # Example 2: Simple reflection task
    task_data = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[2, 1], [4, 3]]
            }
        ],
        "test": [
            {
                "input": [[8, 7], [6, 5]]
            }
        ]
    }

    examples.append({
        "task_id": "synthetic_reflection",
        "task_data": task_data,
        "solved_program": [
            {"primitive": "ReflectH", "parameters": {}}
        ],
        "solve_time": 0.001,
        "confidence": 0.95
    })

    return examples


def solve_and_store_examples(solver: ARCSolver, examples: List[Dict[str, Any]], output_file: str) -> int:
    """Solve examples and store successful solutions in the retrieval store."""
    stored_count = 0

    for example in examples:
        try:
            print(f"Solving {example['task_id']}...")

            # Convert task data to Task object
            task_data = example['task_data']
            task = Task(
                task_id=example['task_id'],
                train_examples=[
                    (np.array(train['input'], dtype=np.int32),
                     np.array(train['output'], dtype=np.int32))
                    for train in task_data['train']
                ],
                test_inputs=[
                    np.array(test['input'], dtype=np.int32)
                    for test in task_data['test']
                ]
            )

            # Solve the task
            result = solver.solve_task(task, timeout=10.0, use_multi_example=False)

            if result.get('success', False) and 'program' in result:
                # Store successful solution
                solved_entry = {
                    "task_id": example['task_id'],
                    "signature": example.get('signature', example['task_id']),  # Use task_id as simple signature
                    "program": result['program'],
                    "solve_time": result.get('computation_time', 0.0),
                    "confidence": 0.9,  # Default confidence for stored solutions
                    "timestamp": result.get('timestamp', 0)
                }

                # Append to retrieval store
                with open(output_file, 'a') as f:
                    json.dump(solved_entry, f)
                    f.write('\n')

                stored_count += 1
                print(f"✅ Stored solution for {example['task_id']}")

            else:
                print(f"❌ Failed to solve {example['task_id']}")

        except Exception as e:
            print(f"❌ Error processing {example['task_id']}: {e}")
            continue

    return stored_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed retrieval store with solved ARC tasks")
    parser.add_argument("--input", "-i", type=str,
                       help="Input JSON file with ARC tasks to solve and store")
    parser.add_argument("--output", "-o", type=str, default="retrieval_store.jsonl",
                       help="Output retrieval store file (default: retrieval_store.jsonl)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic examples instead of input file")
    parser.add_argument("--max-tasks", type=int, default=10,
                       help="Maximum number of tasks to process (default: 10)")

    args = parser.parse_args()

    # Initialize solver with retrieval enabled
    overrides = [
        "search.portfolio.enabled=true",
        f"search.advanced.retrieval.store_file={args.output}",
        "search.advanced.canonicalization.enabled=true",
        "search.advanced.unsat_cache.enabled=false",  # Disable for seeding
    ]

    solver = ARCSolver(overrides)

    if args.synthetic:
        print("Using synthetic examples for seeding...")
        examples = create_synthetic_solved_examples()
    elif args.input:
        print(f"Loading examples from {args.input}...")
        try:
            with open(args.input, 'r') as f:
                examples = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load input file: {e}")
            return 1
    else:
        print("❌ Must specify either --synthetic or --input")
        return 1

    # Limit number of tasks if specified
    examples = examples[:args.max_tasks]

    print(f"Processing {len(examples)} examples...")

    # Solve and store examples
    stored_count = solve_and_store_examples(solver, examples, args.output)

    print(f"✅ Successfully stored {stored_count}/{len(examples)} solutions in {args.output}")

    if stored_count > 0:
        print("\n💡 Retrieval store is now populated!")
        print("Future runs will automatically reuse these solutions for similar tasks.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
