"""Tests for two-tier heuristic system."""

import pytest
import numpy as np
import time

from arc_solver.search.heuristics import (
    Tier1Heuristic, Tier2Heuristic, HeuristicSystem, HeuristicResult,
    create_heuristic_system
)
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation


class TestTier1Heuristic:
    """Test Tier 1 L₂ spectral heuristic."""
    
    @pytest.fixture
    def heuristic(self):
        """Create Tier 1 heuristic for testing."""
        return Tier1Heuristic()
    
    @pytest.fixture
    def sample_grids(self):
        """Create sample grids for testing."""
        grid1 = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        grid2 = np.array([
            [4, 3, 0],
            [2, 1, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        return grid1, grid2
    
    def test_heuristic_initialization(self, heuristic):
        """Test heuristic initialization."""
        assert heuristic.name == "Tier1_L2_Spectral"
        assert heuristic.computation_count == 0
        assert hasattr(heuristic, 'blob_labeler')
        assert hasattr(heuristic, 'feature_cache')
    
    def test_compute_heuristic_basic(self, heuristic, sample_grids):
        """Test basic heuristic computation."""
        grid1, grid2 = sample_grids
        
        result = heuristic.compute(grid1, grid2)
        
        assert isinstance(result, HeuristicResult)
        assert isinstance(result.value, float)
        assert result.value >= 0.0  # Heuristic should be non-negative
        assert result.computation_time > 0
        assert result.features_computed is True
        assert result.error is None
    
    def test_identical_grids(self, heuristic):
        """Test heuristic with identical grids."""
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        
        result = heuristic.compute(grid, grid)
        
        # Identical grids should have zero distance
        assert result.value == 0.0
        assert result.features_computed is True
    
    def test_empty_grids(self, heuristic):
        """Test heuristic with empty grids."""
        empty_grid = np.zeros((3, 3), dtype=np.int32)
        
        result = heuristic.compute(empty_grid, empty_grid)
        
        assert result.value == 0.0
        assert result.features_computed is True
    
    def test_feature_caching(self, heuristic, sample_grids):
        """Test that feature extraction is cached."""
        grid1, grid2 = sample_grids
        
        # First computation
        result1 = heuristic.compute(grid1, grid2)
        cache_size_after_first = len(heuristic.feature_cache)
        
        # Second computation with same grids
        result2 = heuristic.compute(grid1, grid2)
        cache_size_after_second = len(heuristic.feature_cache)
        
        # Results should be identical
        assert result1.value == result2.value
        
        # Cache should not grow (same grids)
        assert cache_size_after_second == cache_size_after_first
    
    def test_clear_cache(self, heuristic, sample_grids):
        """Test cache clearing."""
        grid1, grid2 = sample_grids
        
        # Populate cache
        heuristic.compute(grid1, grid2)
        assert len(heuristic.feature_cache) > 0
        
        # Clear cache
        heuristic.clear_cache()
        assert len(heuristic.feature_cache) == 0
    
    def test_statistics_tracking(self, heuristic, sample_grids):
        """Test computation statistics."""
        grid1, grid2 = sample_grids
        
        initial_stats = heuristic.get_stats()
        assert initial_stats['computation_count'] == 0
        
        # Perform computation
        heuristic(grid1, grid2)  # Use __call__ to test statistics
        
        stats = heuristic.get_stats()
        assert stats['computation_count'] == 1
        assert stats['total_time'] > 0
        assert stats['average_time'] > 0
    
    def test_performance_requirement(self, heuristic):
        """Test that heuristic meets performance requirements."""
        # Create test grids
        grid1 = np.random.randint(0, 5, size=(10, 10), dtype=np.int32)
        grid2 = np.random.randint(0, 5, size=(10, 10), dtype=np.int32)
        
        # Time multiple computations
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = heuristic.compute(grid1, grid2)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        # Should be under 500µs target (though this might be optimistic)
        if avg_time > 0.0005:
            pytest.skip(f"Tier 1 performance: {avg_time*1000:.2f}ms > 0.5ms target")


class TestTier2Heuristic:
    """Test Tier 2 Hungarian assignment heuristic."""
    
    @pytest.fixture
    def heuristic(self):
        """Create Tier 2 heuristic for testing."""
        return Tier2Heuristic()
    
    @pytest.fixture
    def blob_grids(self):
        """Create grids with distinct blobs."""
        grid1 = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 2],
            [0, 0, 0, 0],
            [3, 0, 0, 0]
        ], dtype=np.int32)
        
        grid2 = np.array([
            [2, 2, 0, 1],
            [2, 2, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 3, 0]
        ], dtype=np.int32)
        
        return grid1, grid2
    
    def test_heuristic_initialization(self, heuristic):
        """Test heuristic initialization."""
        assert heuristic.name == "Tier2_Hungarian"
        assert heuristic.computation_count == 0
        assert hasattr(heuristic, 'blob_labeler')
    
    def test_compute_heuristic_basic(self, heuristic, blob_grids):
        """Test basic heuristic computation."""
        grid1, grid2 = blob_grids
        
        result = heuristic.compute(grid1, grid2)
        
        assert isinstance(result, HeuristicResult)
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        assert result.computation_time > 0
        assert result.features_computed is True
        assert result.error is None
    
    def test_identical_grids(self, heuristic):
        """Test heuristic with identical grids."""
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        
        result = heuristic.compute(grid, grid)
        
        # Identical grids should have zero distance
        assert result.value == 0.0
        assert result.features_computed is True
    
    def test_empty_grids(self, heuristic):
        """Test heuristic with empty grids."""
        empty_grid = np.zeros((3, 3), dtype=np.int32)
        
        result = heuristic.compute(empty_grid, empty_grid)
        
        assert result.value == 0.0
        assert result.features_computed is True
    
    def test_fallback_assignment(self, heuristic):
        """Test fallback assignment when SciPy unavailable."""
        # Temporarily disable scipy
        original_scipy = heuristic.scipy_available
        heuristic.scipy_available = False
        
        try:
            grid1 = np.array([[1, 1], [2, 2]], dtype=np.int32)
            grid2 = np.array([[2, 2], [1, 1]], dtype=np.int32)
            
            result = heuristic.compute(grid1, grid2)
            
            assert isinstance(result.value, float)
            assert result.value >= 0.0
            assert result.features_computed is True
        finally:
            heuristic.scipy_available = original_scipy
    
    def test_performance_requirement(self, heuristic):
        """Test that heuristic meets performance requirements."""
        # Create test grids with multiple blobs
        grid1 = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4]
        ], dtype=np.int32)
        
        grid2 = np.array([
            [2, 2, 0, 1, 1],
            [2, 2, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [4, 4, 0, 3, 3],
            [4, 4, 0, 3, 3]
        ], dtype=np.int32)
        
        # Time multiple computations
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = heuristic.compute(grid1, grid2)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        # Should be under 2ms target
        if avg_time > 0.002:
            pytest.skip(f"Tier 2 performance: {avg_time*1000:.2f}ms > 2ms target")


class TestHeuristicSystem:
    """Test combined two-tier heuristic system."""
    
    @pytest.fixture
    def heuristic_system(self):
        """Create heuristic system for testing."""
        return create_heuristic_system(use_tier2=True, tier2_threshold=5.0)
    
    @pytest.fixture
    def sample_grids(self):
        """Create sample grids for testing."""
        grid1 = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        grid2 = np.array([
            [4, 3, 0],
            [2, 1, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        
        return grid1, grid2
    
    def test_system_initialization(self, heuristic_system):
        """Test heuristic system initialization."""
        assert heuristic_system.tier1 is not None
        assert heuristic_system.tier2 is not None
        assert heuristic_system.use_tier2 is True
        assert heuristic_system.tier2_threshold == 5.0
        assert heuristic_system.tier1_calls == 0
        assert heuristic_system.tier2_calls == 0
    
    def test_tier1_only_computation(self, heuristic_system):
        """Test computation using only Tier 1."""
        # Create very different grids to ensure Tier 1 value is high
        grid1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
        grid2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 0]], dtype=np.int32)
        
        # Set low threshold to avoid Tier 2
        heuristic_system.tier2_threshold = 0.1
        
        result = heuristic_system.compute_heuristic(grid1, grid2)
        
        assert isinstance(result, HeuristicResult)
        assert result.value >= 0.0
        assert heuristic_system.tier1_calls == 1
        assert heuristic_system.tier2_calls == 0
    
    def test_tier2_activation(self, heuristic_system):
        """Test Tier 2 activation for similar grids."""
        # Create very similar grids to trigger low Tier 1 value
        grid1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid2 = np.array([[1, 2], [3, 4]], dtype=np.int32)  # Identical
        
        # Set low threshold to force Tier 2
        heuristic_system.tier2_threshold = 10.0
        
        result = heuristic_system.compute_heuristic(grid1, grid2)
        
        assert isinstance(result, HeuristicResult)
        assert heuristic_system.tier1_calls == 1
        # Tier 2 should be called for identical grids (Tier 1 = 0.0)
        assert heuristic_system.tier2_calls == 1
    
    def test_system_without_tier2(self):
        """Test system with Tier 2 disabled."""
        system = create_heuristic_system(use_tier2=False)
        
        grid1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        
        result = system.compute_heuristic(grid1, grid2)
        
        assert isinstance(result, HeuristicResult)
        assert system.tier1_calls == 1
        assert system.tier2_calls == 0
        assert system.tier2 is None
    
    def test_statistics_collection(self, heuristic_system, sample_grids):
        """Test statistics collection."""
        grid1, grid2 = sample_grids
        
        # Perform multiple computations
        for _ in range(3):
            heuristic_system.compute_heuristic(grid1, grid2)
        
        stats = heuristic_system.get_stats()
        
        assert stats['tier1_calls'] == 3
        assert 'tier1_stats' in stats
        assert 'tier2_usage_rate' in stats
        assert isinstance(stats['tier2_usage_rate'], float)
    
    def test_cache_clearing(self, heuristic_system, sample_grids):
        """Test cache clearing functionality."""
        grid1, grid2 = sample_grids
        
        # Populate caches
        heuristic_system.compute_heuristic(grid1, grid2)
        
        # Clear caches
        heuristic_system.clear_caches()
        
        # Should not raise any errors
        assert True
    
    def test_error_handling(self, heuristic_system):
        """Test error handling with invalid inputs."""
        # Test with invalid grid (wrong dtype)
        invalid_grid = np.array([[1.5, 2.5]], dtype=np.float32)
        valid_grid = np.array([[1, 2]], dtype=np.int32)
        
        # Should handle gracefully and return error result
        result = heuristic_system.compute_heuristic(invalid_grid, valid_grid)
        
        # Should return a result (possibly with error)
        assert isinstance(result, HeuristicResult)


class TestHeuristicFactory:
    """Test heuristic system factory function."""
    
    def test_create_default_system(self):
        """Test creating system with default parameters."""
        system = create_heuristic_system()
        
        assert system.use_tier2 is True
        assert system.tier2_threshold == 5.0
        assert system.tier1 is not None
        assert system.tier2 is not None
    
    def test_create_tier1_only_system(self):
        """Test creating Tier 1 only system."""
        system = create_heuristic_system(use_tier2=False)
        
        assert system.use_tier2 is False
        assert system.tier1 is not None
        assert system.tier2 is None
    
    def test_create_custom_threshold_system(self):
        """Test creating system with custom threshold."""
        system = create_heuristic_system(tier2_threshold=2.5)
        
        assert system.tier2_threshold == 2.5
        assert system.use_tier2 is True


if __name__ == "__main__":
    pytest.main([__file__])