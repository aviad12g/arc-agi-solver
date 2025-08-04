import numpy as np
from arc_solver.perception.blob_labeling import BlobLabeler


def test_single_hole_detection():
    """A 3x3 ring of color 1 has one hole in the centre."""
    grid = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.int32)

    labeler = BlobLabeler(use_gpu=False)
    blobs, _ = labeler.label_blobs(grid)

    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.holes == 1


def test_multiple_holes_detection():
    """A 6x6 ring with two separate interior holes."""
    grid = np.array([
        [2, 2, 2, 2, 2, 2],
        [2, 0, 2, 2, 0, 2],
        [2, 0, 2, 2, 0, 2],
        [2, 0, 2, 2, 0, 2],
        [2, 0, 2, 2, 0, 2],
        [2, 2, 2, 2, 2, 2]
    ], dtype=np.int32)

    labeler = BlobLabeler(use_gpu=False)
    blobs, _ = labeler.label_blobs(grid)

    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.holes == 2
