#!/usr/bin/env python3
"""UNSAT Oracle Seeder: Populate UNSAT cache with simple impossibility detections.

This script detects simple impossible cases that no DSL program can solve and
adds them to the UNSAT cache for fast pruning during search.

Rules implemented:
1. Shape change violations: Operations can't change grid dimensions
2. Histogram mismatches: MapColors can't fix incompatible color distributions
3. Background preservation: Some operations preserve background structure
4. Color conservation: Total non-background pixels must be preservable

Usage:
    python scripts/unsat_oracle_seed.py [--output unsat_signatures.jsonl]
    python scripts/unsat_oracle_seed.py --load-at-init  # Load during solver startup
"""

import json
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.search.unsat_cache import make_signature
from arc_solver.core.data_models import Task

logger = logging.getLogger(__name__)


class UNSATOracle:
    """Detects simple impossibility cases for UNSAT cache population."""

    def __init__(self):
        """Initialize the oracle with detection rules."""
        self.rules = [
            self._check_shape_change_violation,
            self._check_histogram_mismatch,
            self._check_background_preservation,
            self._check_color_conservation
        ]
        logger.info(f"Initialized UNSAT oracle with {len(self.rules)} detection rules")

    def detect_unsat_cases(self, task: Task) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect UNSAT cases in a task.

        Args:
            task: ARC task with training examples

        Returns:
            List of (input_grid, output_grid) pairs that are UNSAT
        """
        unsat_pairs = []

        for train_example in task.train_examples:
            input_grid, output_grid = train_example

            for rule in self.rules:
                try:
                    if rule(input_grid, output_grid):
                        unsat_pairs.append((input_grid, output_grid))
                        logger.debug(f"Detected UNSAT case: {rule.__name__}")
                        break  # One rule hit is enough
                except Exception as e:
                    logger.warning(f"Rule {rule.__name__} failed: {e}")
                    continue

        return unsat_pairs

    def _check_shape_change_violation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if shape change is impossible under DSL constraints.

        The DSL cannot change grid dimensions directly. Some operations like
        Extract/Crop can change effective dimensions but preserve structure.
        """
        input_shape = input_grid.shape
        output_shape = output_grid.shape

        # Same dimensions are always possible
        if input_shape == output_shape:
            return False

        # Different dimensions might be possible with Extract/Crop
        # But only if output is a subregion of input
        if output_shape[0] > input_shape[0] or output_shape[1] > input_shape[1]:
            # Cannot create larger grid from smaller input
            return True

        # Check if output could be a valid subregion
        # This is a conservative check - we could be more precise
        max_possible_overlap = min(input_shape[0], output_shape[0]) * min(input_shape[1], output_shape[1])
        actual_overlap = 0

        # Simple overlap check - count matching pixels in possible alignments
        for dr in range(input_shape[0] - output_shape[0] + 1):
            for dc in range(input_shape[1] - output_shape[1] + 1):
                subregion = input_grid[dr:dr+output_shape[0], dc:dc+output_shape[1]]
                if np.array_equal(subregion, output_grid):
                    return False  # Found exact match

        # If no exact match found and dimensions differ, likely UNSAT
        return input_shape != output_shape

    def _check_histogram_mismatch(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if color histogram mismatch makes transformation impossible.

        MapColors can only remap existing colors. If output has colors not in input
        (except background), or different counts that can't be explained by background
        changes, it's impossible.
        """
        input_colors, input_counts = np.unique(input_grid, return_counts=True)
        output_colors, output_counts = np.unique(output_grid, return_counts=True)

        # Remove background (0) from analysis if present
        input_non_bg = input_colors[input_colors != 0]
        output_non_bg = output_colors[output_colors != 0]

        # If output has colors not in input, impossible (except background)
        if not set(output_non_bg).issubset(set(input_non_bg)):
            return True

        # Check if pixel counts are compatible
        # Background pixels can be created/destroyed, but non-background counts must match
        input_bg_count = np.sum(input_counts[input_colors == 0]) if 0 in input_colors else 0
        output_bg_count = np.sum(output_counts[output_colors == 0]) if 0 in output_colors else 0

        # Total pixels must be preserved (DSL doesn't add/remove pixels)
        if input_grid.size != output_grid.size:
            return True

        # Non-background pixel counts must be preserved
        input_non_bg_total = input_grid.size - input_bg_count
        output_non_bg_total = output_grid.size - output_bg_count

        return input_non_bg_total != output_non_bg_total

    def _check_background_preservation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if background structure is incompatible.

        Some DSL operations preserve background structure. If the background
        connectivity or structure is fundamentally different, it may be impossible.
        """
        # Simple check: if one has background and other doesn't, might be suspicious
        input_has_bg = 0 in input_grid
        output_has_bg = 0 in output_grid

        if input_has_bg != output_has_bg:
            # This could be UNSAT in some cases, but let's be conservative
            # and only flag obvious cases
            return False

        if not input_has_bg:
            # Both grids have no background - must be identical
            return not np.array_equal(input_grid, output_grid)

        return False

    def _check_color_conservation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if color conservation laws are violated.

        The DSL must preserve certain color properties. For example, if input has
        only one color and output has multiple, some transformations are impossible.
        """
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))

        # If input has only background and output has non-background colors
        input_non_bg = input_colors - {0}
        output_non_bg = output_colors - {0}

        if not input_non_bg and output_non_bg:
            # Cannot create non-background colors from pure background
            return True

        # If input has multiple colors and output has only one non-background color
        # This might be possible with MapColors, so be conservative
        return False


def generate_synthetic_unsat_pairs() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic UNSAT pairs for testing and seeding."""
    pairs = []

    # Shape change violation: try to make larger grid from smaller
    small_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    large_grid = np.array([[1, 2, 5], [3, 4, 6], [7, 8, 9]], dtype=np.int32)
    pairs.append((small_grid, large_grid))

    # Histogram mismatch: output has color not in input
    input_grid = np.array([[1, 1], [1, 1]], dtype=np.int32)
    output_grid = np.array([[1, 2], [1, 2]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Color conservation violation: different total non-background pixels
    input_grid = np.array([[1, 1], [1, 0]], dtype=np.int32)
    output_grid = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Connectivity preservation violation: connected component count changes
    input_grid = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.int32)
    output_grid = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Background structure violation: background holes created
    input_grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int32)
    output_grid = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Pixel count explosion: output has way more non-zero pixels
    input_grid = np.array([[1, 0], [0, 0]], dtype=np.int32)
    output_grid = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Impossible color transformation: specific color patterns that can't be achieved
    input_grid = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.int32)
    output_grid = np.array([[3, 4, 3], [4, 5, 4], [3, 4, 3]], dtype=np.int32)  # Colors 3,4,5 not in input
    pairs.append((input_grid, output_grid))

    # Topology violation: different number of holes
    input_grid = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)  # One hole
    output_grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int32)  # Multiple holes
    pairs.append((input_grid, output_grid))

W    # Connectivity violation (rough): input single object, output multiple disjoint new colored areas
    input_grid = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
    output_grid = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Background structure inconsistency: input no background, output introduces background in the middle only
    input_grid = np.array([[2, 2], [2, 2]], dtype=np.int32)
    output_grid = np.array([[2, 0], [0, 2]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Pixel explosion: output nonzero count far exceeds plausible operations (toy example)
    input_grid = np.array([[1, 0], [0, 0]], dtype=np.int32)
    output_grid = np.array([[1, 1], [1, 1]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Color transformation impossible: output uses color not present and no mapping possible under constraints
    input_grid = np.array([[1, 2], [2, 1]], dtype=np.int32)
    output_grid = np.array([[3, 4], [4, 3]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    # Topology violation (approx): hole creation in a way our DSL cannot represent directly in one step
    input_grid = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
    output_grid = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    pairs.append((input_grid, output_grid))

    return pairs


def main():
    """Main entry point for UNSAT oracle seeding."""
    parser = argparse.ArgumentParser(description="UNSAT Oracle Seeder")
    parser.add_argument("--output", "-o", default="unsat_signatures.jsonl",
                       help="Output JSONL file for UNSAT signatures")
    parser.add_argument("--load-at-init", action="store_true",
                       help="Print how to load signatures during solver initialization")
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Only use synthetic test cases, not real tasks")
    parser.add_argument("--input-dir", type=str,
                       help="Directory containing ARC JSON task files to analyze")
    parser.add_argument("--max-files", type=int, default=0,
                       help="Maximum number of files to process from input-dir (0 = all)")
    parser.add_argument("--append", action="store_true",
                       help="Append to output file instead of overwriting")
    parser.add_argument("--summary-out", type=str,
                       help="Optional JSON path for summary statistics")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Starting UNSAT oracle seeding")

    oracle = UNSATOracle()
    signatures = []

    summary: Dict[str, Any] = {"rules": {}, "pairs": 0, "files_scanned": 0}

    if args.synthetic_only:
        # Use synthetic test cases
        unsat_pairs = generate_synthetic_unsat_pairs()
        logger.info(f"Generated {len(unsat_pairs)} synthetic UNSAT pairs")
        summary["pairs"] = len(unsat_pairs)
    elif args.input_dir:
        # Scan a directory of ARC JSON tasks
        logger.info(f"Scanning ARC tasks under {args.input_dir}")
        unsat_pairs = []
        files_scanned = 0
        for path in iter_task_files(args.input_dir):
            if args.max_files and files_scanned >= args.max_files:
                break
            try:
                data = load_json_task(path)
                found = 0
                for ex in data.get('train', []):
                    inp = np.array(ex['input'], dtype=np.int32)
                    out = np.array(ex['output'], dtype=np.int32)
                    for rule in oracle.rules:
                        try:
                            if rule(inp, out):
                                unsat_pairs.append((inp, out))
                                summary["rules"][rule.__name__] = summary["rules"].get(rule.__name__, 0) + 1
                                found += 1
                                break
                        except Exception as e:
                            logger.debug(f"Rule {rule.__name__} failed on {path.name}: {e}")
                            continue
                files_scanned += 1
                if found:
                    logger.debug(f"{path.name}: {found} UNSAT pair(s)")
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
        summary["pairs"] = len(unsat_pairs)
        summary["files_scanned"] = files_scanned
        logger.info(f"Collected {len(unsat_pairs)} UNSAT pairs from {files_scanned} files")
    else:
        # Fallback: synthetic
        unsat_pairs = generate_synthetic_unsat_pairs()
        summary["pairs"] = len(unsat_pairs)
        logger.info(f"Using {len(unsat_pairs)} synthetic pairs")

    # Generate signatures
    for input_grid, output_grid in unsat_pairs:
        try:
            signature = make_signature(input_grid, output_grid)
            signatures.append({
                'signature': signature,
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'reason': 'synthetic_test_case'
            })
            logger.debug(f"Generated signature for pair {input_grid.shape} -> {output_grid.shape}")
        except Exception as e:
            logger.warning(f"Failed to generate signature: {e}")
            continue

    # Write to JSONL file
    output_path = Path(args.output)
    mode = 'a' if args.append and output_path.exists() else 'w'
    with open(output_path, mode) as f:
        for sig_data in signatures:
            json.dump(sig_data, f)
            f.write('\n')

    logger.info(f"Wrote {len(signatures)} UNSAT signatures to {output_path}")

    if args.summary_out:
        try:
            with open(args.summary_out, 'w') as sf:
                json.dump(summary, sf, indent=2)
            logger.info(f"Summary written to {args.summary_out}")
        except Exception as e:
            logger.warning(f"Failed to write summary: {e}")

    if args.load_at_init:
        logger.info("To load these signatures at solver init, add to your config:")
        logger.info(f"  search.advanced.unsat_cache.signatures_file: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


def iter_task_files(root: str):
    """Yield JSON files under a root folder."""
    p = Path(root)
    if p.is_file() and p.suffix.lower() == '.json':
        yield p
        return
    for fp in p.rglob('*.json'):
        # Skip large combined files if any; keep simple per-task files
        yield fp


def load_json_task(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)
