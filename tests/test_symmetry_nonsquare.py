import numpy as np
from arc_solver.perception.symmetry import BitboardSymmetryDetector, SymmetryType


def test_vertical_reflection_rectangle():
    grid = np.array([
        [1, 0, 0, 1],
        [2, 0, 0, 2]
    ], dtype=np.int32)
    detector = BitboardSymmetryDetector()
    sym = detector.detect_symmetries(grid)
    assert SymmetryType.REFLECT_V in sym


def test_no_horizontal_symmetry_rectangle():
    grid = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)
    detector = BitboardSymmetryDetector()
    sym = detector.detect_symmetries(grid)
    assert SymmetryType.REFLECT_H not in sym
