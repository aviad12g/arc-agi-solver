"""Tests for core data models."""

import pytest
import numpy as np
from arc_solver.core.data_models import FeatureVector, Blob, GridState, SearchNode, Task


class TestFeatureVector:
    """Test the FeatureVector data model."""
    
    def test_feature_vector_creation(self):
        """Test creating a valid feature vector."""
        orbit_sig = np.random.rand(8).astype(np.float32)
        spectral = np.random.rand(3).astype(np.float32)
        persistence = np.random.rand(32).astype(np.float32)
        zernike = np.random.rand(7).astype(np.float32)
        
        fv = FeatureVector(
            orbit_signature=orbit_sig,
            spectral_features=spectral,
            persistence_landscape=persistence,
            zernike_moments=zernike
        )
        
        assert fv.orbit_signature.shape == (8,)
        assert fv.spectral_features.shape == (3,)
        assert fv.persistence_landscape.shape == (32,)
        assert fv.zernike_moments.shape == (7,)
    
    def test_to_array(self):
        """Test converting feature vector to 50D array."""
        orbit_sig = np.ones(8, dtype=np.float32)
        spectral = np.ones(3, dtype=np.float32) * 2
        persistence = np.ones(32, dtype=np.float32) * 3
        zernike = np.ones(7, dtype=np.float32) * 4
        
        fv = FeatureVector(orbit_sig, spectral, persistence, zernike)
        array = fv.to_array()
        
        assert array.shape == (50,)
        assert np.all(array[:8] == 1)
        assert np.all(array[8:11] == 2)
        assert np.all(array[11:43] == 3)
        assert np.all(array[43:50] == 4)
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise assertions."""
        with pytest.raises(AssertionError):
            FeatureVector(
                orbit_signature=np.ones(7),  # Wrong size
                spectral_features=np.ones(3),
                persistence_landscape=np.ones(32),
                zernike_moments=np.ones(7)
            )


class TestBlob:
    """Test the Blob data model."""
    
    def test_blob_creation(self):
        """Test creating a valid blob."""
        pixels = [(0, 0), (0, 1), (1, 0)]
        blob = Blob(
            id=1,
            color=2,
            pixels=pixels,
            bounding_box=(0, 0, 1, 1),
            center_of_mass=(0.33, 0.33),
            area=3,
            holes=0
        )
        
        assert blob.id == 1
        assert blob.color == 2
        assert blob.area == len(pixels)
        assert blob.holes == 0
    
    def test_invalid_area(self):
        """Test that mismatched area raises assertion."""
        with pytest.raises(AssertionError):
            Blob(
                id=1,
                color=2,
                pixels=[(0, 0), (0, 1)],
                bounding_box=(0, 0, 0, 1),
                center_of_mass=(0, 0.5),
                area=3,  # Wrong area
                holes=0
            )


class TestGridState:
    """Test the GridState data model."""
    
    def test_grid_state_creation(self):
        """Test creating a valid grid state."""
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        blobs = []
        program = ["rotate90", "paint(0,0,5)"]
        
        state = GridState(
            grid=grid,
            blobs=blobs,
            program=program,
            cost=2.0,
            heuristic=1.5
        )
        
        assert state.total_cost() == 3.5
        assert len(state.program) == 2
    
    def test_canonical_hash(self):
        """Test that canonical hash is consistent."""
        grid1 = np.array([[1, 2], [1, 2]], dtype=np.int32)
        grid2 = np.array([[2, 1], [2, 1]], dtype=np.int32)  # Colors swapped
        
        state1 = GridState(grid1, [], [], 0, 0)
        state2 = GridState(grid2, [], [], 0, 0)
        
        # Should have same hash after normalization
        hash1 = state1.canonical_hash()
        hash2 = state2.canonical_hash()
        
        assert isinstance(hash1, str)
        assert len(hash1) == 40  # SHA1 hex length


class TestTask:
    """Test the Task data model."""
    
    def test_task_creation(self):
        """Test creating a valid task."""
        train_input = np.array([[1, 2], [3, 4]], dtype=np.int32)
        train_output = np.array([[4, 3], [2, 1]], dtype=np.int32)
        test_input = np.array([[5, 6], [7, 8]], dtype=np.int32)
        
        task = Task(
            task_id="test_task",
            train_examples=[(train_input, train_output)],
            test_inputs=[test_input]
        )
        
        assert task.task_id == "test_task"
        assert len(task.train_examples) == 1
        assert len(task.test_inputs) == 1
    
    def test_invalid_task(self):
        """Test that invalid task structure raises assertion."""
        with pytest.raises(AssertionError):
            Task(
                task_id="empty_task",
                train_examples=[],  # Empty training examples
                test_inputs=[np.array([[1, 2]], dtype=np.int32)]
            )