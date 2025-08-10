"""Golden tests for primitives, heuristics, and end-to-end pipeline.

These ensure correctness remains stable across refactors.
"""
import numpy as np

from arc_solver.reasoning.dsl_engine import create_dsl_engine, DSLOperation
from arc_solver.search.heuristics import create_heuristic_system


def test_primitive_rotate90_golden():
  engine = create_dsl_engine()
  grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
  op = DSLOperation('Rotate90', {})
  out = engine.apply_operation(grid, op)
  expected = np.array([[3, 1], [4, 2]], dtype=np.int32)
  assert np.array_equal(out, expected)


def test_heuristic_symmetry_invariant_mean_l2_not_negative():
  system = create_heuristic_system(use_tier2=False)
  g1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
  g2 = np.array([[3, 1], [4, 2]], dtype=np.int32)
  res = system.compute_heuristic(g1, g2)
  assert res.value >= 0.0


