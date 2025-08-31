#!/usr/bin/env python3
"""Train a dead-end predictor model from ARC task execution logs.

This script analyzes solved and unsolved ARC tasks to train a simple
dead-end predictor that can identify likely-unsolvable states early.

Usage:
    python scripts/train_deadend_predictor.py --input solved_tasks.jsonl --unsolved unsolved_tasks.jsonl --output deadend_model.json
    python scripts/train_deadend_predictor.py --synthetic  # Use synthetic examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from arc_solver.cli.commands import ARCSolver
from arc_solver.core.data_models import Task


def extract_features(grid: np.ndarray) -> Dict[str, Any]:
    """Extract comprehensive features from a grid for dead-end prediction.

    Enhanced features for better dead-end detection:
    - c_shape: input grid shape
    - t_shape: target grid shape (if available)
    - c_nz: number of non-zero pixels in current grid
    - t_nz: number of non-zero pixels in target grid (if available)
    - Grid complexity metrics
    - Color distribution analysis
    - Spatial patterns
    """
    features = {
        "c_shape": list(grid.shape),
        "c_nz": int(np.count_nonzero(grid)),
    }

    height, width = grid.shape
    total_pixels = height * width

    # Enhanced color distribution features
    unique_colors, counts = np.unique(grid, return_counts=True)
    features["num_colors"] = len(unique_colors)
    features["color_entropy"] = sum((count / total_pixels) * np.log(count / total_pixels)
                                   for count in counts if count > 0)

    # Color concentration metrics
    if len(unique_colors) > 1:
        features["most_common_color_ratio"] = max(counts) / total_pixels
        features["color_uniformity"] = 1.0 - (max(counts) / total_pixels)  # Higher = more uniform

    # Grid structure features
    features["aspect_ratio"] = height / width if width > 0 else 1.0
    features["density"] = features["c_nz"] / total_pixels

    # Edge and corner analysis
    if height > 1 and width > 1:
        features["corner_pixels"] = int(grid[0, 0] + grid[0, -1] + grid[-1, 0] + grid[-1, -1])
        features["edge_pixels"] = int(np.sum(grid[0, :]) + np.sum(grid[-1, :]) +
                                     np.sum(grid[1:-1, 0]) + np.sum(grid[1:-1, -1]))

    # Pattern complexity (simplified)
    if features["c_nz"] > 0:
        # Estimate spatial complexity
        features["spatial_variance"] = float(np.var(grid))
        # Symmetry indicators
        features["horizontal_symmetry"] = float(np.mean(grid == np.fliplr(grid)))
        features["vertical_symmetry"] = float(np.mean(grid == np.flipud(grid)))

    return features


def analyze_task_execution(task: Task, solver: ARCSolver, max_time: float = 30.0) -> Dict[str, Any]:
    """Analyze a single task execution to extract dead-end features."""
    try:
        # Run the solver and capture execution details
        result = solver.solve_task(task, timeout=max_time, use_multi_example=False)

        # Extract features from input grid
        input_grid = task.train_examples[0][0]  # First training input
        features = extract_features(input_grid)

        # Add target features and differences if available
        if len(task.test_inputs) > 0:
            target_grid = task.test_inputs[0]
            target_features = extract_features(target_grid)

            # Add target features
            features.update({
                "t_shape": target_features["c_shape"],
                "t_nz": target_features["c_nz"],
                "t_colors": target_features.get("num_colors", 0),
            })

            # Add difference features (key for dead-end detection)
            features.update({
                "shape_diff": abs(features["c_shape"][0] - target_features["c_shape"][0]) +
                             abs(features["c_shape"][1] - target_features["c_shape"][1]),
                "nz_diff": abs(features["c_nz"] - target_features["c_nz"]),
                "color_diff": abs(features["num_colors"] - target_features.get("num_colors", 0)),

                # Complexity differences
                "density_diff": abs(features.get("density", 0) - target_features.get("density", 0)),
                "entropy_diff": abs(features.get("color_entropy", 0) - target_features.get("color_entropy", 0)),

                # Spatial pattern differences
                "h_symmetry_diff": abs(features.get("horizontal_symmetry", 0) - target_features.get("horizontal_symmetry", 0)),
                "v_symmetry_diff": abs(features.get("vertical_symmetry", 0) - target_features.get("vertical_symmetry", 0)),
            })

        # Add execution outcome
        features.update({
            "solved": result.get("success", False),
            "solve_time": result.get("computation_time", max_time),
            "task_id": getattr(task, 'task_id', 'unknown'),
        })

        # Add search statistics if available
        search_stats = result.get("search_stats", {})
        if search_stats:
            features.update({
                "nodes_expanded": search_stats.get("nodes_expanded", 0),
                "nodes_generated": search_stats.get("nodes_generated", 0),
                "max_depth": search_stats.get("max_depth_reached", 0),
            })

        return features

    except Exception as e:
        print(f"Error analyzing task: {e}")
        return {}


def generate_synthetic_training_data() -> Tuple[List[Dict], List[Dict]]:
    """Generate synthetic solved and unsolved examples for training."""

    solved_examples = []
    unsolved_examples = []

    # Create some solvable tasks
    solvable_tasks = [
        # Simple rotation
        {
            "train": [(np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[3, 1], [4, 2]], dtype=np.int32))],
            "test": [np.array([[5, 6], [7, 8]], dtype=np.int32)]
        },
        # Simple reflection
        {
            "train": [(np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[2, 1], [4, 3]], dtype=np.int32))],
            "test": [np.array([[8, 7], [6, 5]], dtype=np.int32)]
        }
    ]

    # Create some unsolvable tasks (based on our UNSAT cases)
    unsolvable_tasks = [
        # Shape change violation
        {
            "train": [(np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[1, 2, 5], [3, 4, 6], [7, 8, 9]], dtype=np.int32))],
            "test": [np.array([[5, 6], [7, 8]], dtype=np.int32)]
        },
        # Color mismatch
        {
            "train": [(np.array([[1, 1], [1, 1]], dtype=np.int32), np.array([[1, 2], [1, 2]], dtype=np.int32))],
            "test": [np.array([[1, 1], [1, 1]], dtype=np.int32)]
        }
    ]

    # Convert to Task objects and analyze
    solver = ARCSolver()

    for i, task_data in enumerate(solvable_tasks):
        task = Task(
            task_id=f"solvable_{i}",
            train_examples=task_data["train"],
            test_inputs=task_data["test"]
        )

        features = analyze_task_execution(task, solver, max_time=10.0)
        if features and features.get("solved", False):
            solved_examples.append(features)

    for i, task_data in enumerate(unsolvable_tasks):
        task = Task(
            task_id=f"unsolvable_{i}",
            train_examples=task_data["train"],
            test_inputs=task_data["test"]
        )

        features = analyze_task_execution(task, solver, max_time=5.0)
        if features and not features.get("solved", True):  # Only include unsolved ones
            unsolved_examples.append(features)

    return solved_examples, unsolved_examples


def train_enhanced_model(solved_data: List[Dict], unsolved_data: List[Dict]) -> Dict[str, Any]:
    """Train an enhanced dead-end predictor model with sophisticated features.

    Uses statistical thresholds based on solved vs unsolved task features,
    including difference metrics and pattern analysis.
    """

    model = {
        "version": "2.0",
        "description": "Enhanced dead-end predictor with difference features and pattern analysis",
        "training_data": {
            "solved_count": len(solved_data),
            "unsolved_count": len(unsolved_data),
            "total_count": len(solved_data) + len(unsolved_data)
        },
        "thresholds": {},
        "feature_importance": {}
    }

    if not solved_data or not unsolved_data:
        print("Warning: Insufficient training data")
        return model

    # Enhanced feature set including difference metrics
    features_to_analyze = [
        # Basic features
        "c_nz", "num_colors", "solve_time", "nodes_expanded",
        # Difference features (most important for dead-end detection)
        "shape_diff", "nz_diff", "color_diff",
        "density_diff", "entropy_diff",
        "h_symmetry_diff", "v_symmetry_diff",
        # Pattern features
        "aspect_ratio", "density", "spatial_variance",
        "horizontal_symmetry", "vertical_symmetry"
    ]

    for feature in features_to_analyze:
        solved_values = [d.get(feature, 0) for d in solved_data if feature in d and d.get(feature, 0) != 0]
        unsolved_values = [d.get(feature, 0) for d in unsolved_data if feature in d and d.get(feature, 0) != 0]

        if solved_values and unsolved_values:
            # Calculate statistical measures
            solved_median = statistics.median(solved_values)
            unsolved_median = statistics.median(unsolved_values)

            # Use more sophisticated threshold calculation
            if feature in ["solve_time", "nodes_expanded", "shape_diff", "nz_diff", "color_diff"]:
                # Higher difference values suggest dead-end
                threshold = solved_median + (unsolved_median - solved_median) * 0.7  # Conservative
                direction = "higher_bad"
            elif feature in ["density_diff", "entropy_diff", "h_symmetry_diff", "v_symmetry_diff"]:
                # Difference-based features - use absolute threshold
                threshold = unsolved_median * 0.5  # Half the unsolved median
                direction = "difference_based"
            else:
                # Distribution-based for other features
                threshold = (solved_median + unsolved_median) / 2
                direction = "distribution_based"

            model["thresholds"][feature] = {
                "threshold": threshold,
                "solved_median": solved_median,
                "unsolved_median": unsolved_median,
                "direction": direction,
                "solved_std": statistics.stdev(solved_values) if len(solved_values) > 1 else 0,
                "unsolved_std": statistics.stdev(unsolved_values) if len(unsolved_values) > 1 else 0
            }

            # Calculate feature importance based on separation
            separation = abs(solved_median - unsolved_median)
            if solved_median + unsolved_median > 0:
                separation /= (solved_median + unsolved_median) / 2  # Normalize
            model["feature_importance"][feature] = separation

    # Sort features by importance
    sorted_features = sorted(model["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    model["top_features"] = [f[0] for f in sorted_features[:10]]  # Top 10 most important

    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train dead-end predictor model")
    parser.add_argument("--input", "-i", type=str,
                       help="Input JSONL file with solved task features")
    parser.add_argument("--unsolved", "-u", type=str,
                       help="Input JSONL file with unsolved task features")
    parser.add_argument("--output", "-o", type=str, default="deadend_model.json",
                       help="Output model file (default: deadend_model.json)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic training data")

    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic training data...")
        solved_data, unsolved_data = generate_synthetic_training_data()
    elif args.input and args.unsolved:
        print("Loading training data from files...")
        solved_data = []
        unsolved_data = []

        try:
            with open(args.input, 'r') as f:
                for line in f:
                    solved_data.append(json.loads(line.strip()))

            with open(args.unsolved, 'r') as f:
                for line in f:
                    unsolved_data.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Error loading training data: {e}")
            return 1
    else:
        print("Must specify either --synthetic or both --input and --unsolved")
        return 1

    print(f"Training on {len(solved_data)} solved and {len(unsolved_data)} unsolved examples")

    # Train the enhanced model
    model = train_enhanced_model(solved_data, unsolved_data)

    # Save the model
    with open(args.output, 'w') as f:
        json.dump(model, f, indent=2)

    print(f"✅ Model trained and saved to {args.output}")
    print(f"   - Solved examples: {model['training_data']['solved_count']}")
    print(f"   - Unsolved examples: {model['training_data']['unsolved_count']}")
    print(f"   - Thresholds: {len(model['thresholds'])}")

    if model['thresholds']:
        print("\nLearned thresholds:")
        for feature, data in model['thresholds'].items():
            thr = data.get('threshold', 0)
            direction = data.get('direction', 'n/a')
            print(f"  {feature}: thr={thr:.2f} dir={direction}")
    if 'nz_diff_threshold' in model or 'colors_diff_threshold' in model:
        print("\nLearned diff thresholds:")
        if 'nz_diff_threshold' in model:
            print(f"  nz_diff_threshold={model['nz_diff_threshold']}")
        if 'colors_diff_threshold' in model:
            print(f"  colors_diff_threshold={model['colors_diff_threshold']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
