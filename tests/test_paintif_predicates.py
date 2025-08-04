import numpy as np
from arc_solver.reasoning.primitives import PaintIf, HorizontalLinePredicate, VerticalLinePredicate


def test_horizontal_line_predicate():
    grid = np.array([
        [0, 1, 1, 1],
        [0, 0, 0, 0]
    ], dtype=np.int32)

    primitive = PaintIf()
    pred = HorizontalLinePredicate()
    new_grid = primitive.execute(grid, predicate=pred, new_color=2)

    # entire horizontal line should become color 2
    assert (new_grid[0] == np.array([0, 2, 2, 2])).all()


def test_vertical_line_predicate():
    grid = np.array([
        [0, 1],
        [0, 1],
        [0, 1]
    ], dtype=np.int32)
    primitive = PaintIf()
    pred = VerticalLinePredicate()
    new_grid = primitive.execute(grid, predicate=pred, new_color=3)

    # vertical line column (col 1) becomes 3
    assert (new_grid[:, 1] == 3).all()
