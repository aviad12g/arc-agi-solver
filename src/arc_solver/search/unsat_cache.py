"""UNSAT cache and signatures for fast, safe pruning.

This module provides a conservative signature function over grid pairs and a
simple in-memory cache of signatures known to be UNSAT (i.e., provably no
solution under current DSL/constraints). It is intentionally minimal and
opt-in; by default, nothing is marked UNSAT.

Integrations should only query this cache and skip expansions when a signature
is marked UNSAT. Populating the cache should come from certified sources
(e.g., SAT/ILP oracles or invariant proofs), not heuristics.

Supports loading pre-computed signatures from JSONL files for bootstrapping.
"""

from __future__ import annotations

from typing import Tuple, Optional
import threading
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _histogram_signature(grid: np.ndarray) -> Tuple[int, Tuple[int, ...]]:
    """Return (size, sorted color multiset) signature.

    This ignores absolute color labels and just keeps the multiset of counts,
    which is stable under permutations. Background (0) is included; higher-level
    logic can decide whether to drop/keep it for a given family.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.asarray(grid)
    size = int(grid.size)
    unique, counts = np.unique(grid, return_counts=True)
    # Sort counts in descending order to get a multiset signature
    multiset = tuple(sorted((int(c) for c in counts), reverse=True))
    return size, multiset


def make_signature(current: np.ndarray, target: np.ndarray) -> Tuple:
    """Build a conservative, reusable signature for UNSAT caching.

    Components:
      - current.shape, target.shape
      - color-count multisets (descending)
      - number of non-zero pixels (approx topology proxy)
    """
    ch, cw = current.shape
    th, tw = target.shape
    c_size, c_hist = _histogram_signature(current)
    t_size, t_hist = _histogram_signature(target)
    c_nz = int(np.count_nonzero(current))
    t_nz = int(np.count_nonzero(target))
    return (
        (int(ch), int(cw)),
        (int(th), int(tw)),
        (c_size, c_hist, c_nz),
        (t_size, t_hist, t_nz),
    )


class UNSATCache:
    """Thread-safe in-memory UNSAT cache keyed by signatures."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._set = set()

    def add_unsat(self, signature: Tuple) -> None:
        """Add a signature to the UNSAT cache."""
        with self._lock:
            self._set.add(signature)

    def is_unsat(self, signature: Tuple) -> bool:
        """Check if a signature is marked as UNSAT."""
        with self._lock:
            return signature in self._set

    def clear(self) -> None:
        """Clear all signatures from the cache."""
        with self._lock:
            self._set.clear()

    def load_from_jsonl(self, filepath: str) -> int:
        """Load UNSAT signatures from a JSONL file.

        Args:
            filepath: Path to JSONL file containing signature data

        Returns:
            Number of signatures loaded
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"UNSAT signatures file not found: {filepath}")
            return 0

        loaded_count = 0
        with self._lock:
            try:
                with open(path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            if 'signature' in data:
                                # Convert nested lists to tuples for hashing
                                signature = self._list_to_tuple(data['signature'])
                                self._set.add(signature)
                                loaded_count += 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at line {line_num} in {filepath}: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num} in {filepath}: {e}")
                            continue

            except Exception as e:
                logger.error(f"Failed to load UNSAT signatures from {filepath}: {e}")
                return 0

        logger.info(f"Loaded {loaded_count} UNSAT signatures from {filepath}")
        return loaded_count

    def size(self) -> int:
        """Get the number of signatures in the cache."""
        with self._lock:
            return len(self._set)

    def _list_to_tuple(self, obj):
        """Convert nested lists to tuples for hashing."""
        if isinstance(obj, list):
            return tuple(self._list_to_tuple(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple((k, self._list_to_tuple(v)) for k, v in obj.items())
        else:
            return obj

