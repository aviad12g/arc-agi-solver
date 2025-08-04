"""Tests for caching system."""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from arc_solver.caching import (
    CacheManager, create_cache_manager, CacheKeyGenerator,
    RedisCache, MockRedisCache, FileCache
)
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation


class TestCacheKeyGenerator:
    """Test cache key generation."""
    
    def test_grid_hash(self):
        """Test grid hash generation."""
        grid1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid3 = np.array([[4, 3], [2, 1]], dtype=np.int32)
        
        hash1 = CacheKeyGenerator.grid_hash(grid1)
        hash2 = CacheKeyGenerator.grid_hash(grid2)
        hash3 = CacheKeyGenerator.grid_hash(grid3)
        
        # Same grids should have same hash
        assert hash1 == hash2
        
        # Different grids should have different hash
        assert hash1 != hash3
        
        # Hash should be consistent
        assert CacheKeyGenerator.grid_hash(grid1) == hash1
    
    def test_program_hash(self):
        """Test program hash generation."""
        op1 = DSLOperation("Rotate90", {})
        op2 = DSLOperation("Paint", {"x": 1, "y": 2, "c": 3})
        
        prog1 = DSLProgram([op1, op2])
        prog2 = DSLProgram([op1, op2])
        prog3 = DSLProgram([op2, op1])  # Different order
        
        hash1 = CacheKeyGenerator.program_hash(prog1)
        hash2 = CacheKeyGenerator.program_hash(prog2)
        hash3 = CacheKeyGenerator.program_hash(prog3)
        
        # Same programs should have same hash
        assert hash1 == hash2
        
        # Different programs should have different hash
        assert hash1 != hash3
    
    def test_grid_pair_hash(self):
        """Test grid pair hash generation."""
        grid1 = np.array([[1, 2]], dtype=np.int32)
        grid2 = np.array([[3, 4]], dtype=np.int32)
        
        hash1 = CacheKeyGenerator.grid_pair_hash(grid1, grid2)
        hash2 = CacheKeyGenerator.grid_pair_hash(grid1, grid2)
        hash3 = CacheKeyGenerator.grid_pair_hash(grid2, grid1)  # Swapped order
        
        # Same pairs should have same hash
        assert hash1 == hash2
        
        # Different order should have different hash
        assert hash1 != hash3
    
    def test_feature_key(self):
        """Test feature key generation."""
        grid = np.array([[1, 2]], dtype=np.int32)
        
        key1 = CacheKeyGenerator.feature_key(grid, "orbit")
        key2 = CacheKeyGenerator.feature_key(grid, "orbit")
        key3 = CacheKeyGenerator.feature_key(grid, "spectral")
        
        # Same grid and feature type should have same key
        assert key1 == key2
        
        # Different feature type should have different key
        assert key1 != key3
        
        # Key should have correct format
        assert key1.startswith("features:orbit:")


class TestFileCache:
    """Test file-based caching."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_file_cache_initialization(self, temp_cache_dir):
        """Test file cache initialization."""
        cache = FileCache(
            cache_dir=temp_cache_dir,
            max_cache_size=0.1,  # 100MB
            compression=True
        )
        
        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.compression is True
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_file_cache_basic_operations(self, temp_cache_dir):
        """Test basic file cache operations."""
        cache = FileCache(cache_dir=temp_cache_dir)
        
        # Test set and get
        test_data = {"key": "value", "number": 42}
        assert cache.set("test_key", test_data) is True
        
        retrieved = cache.get("test_key")
        assert retrieved == test_data
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        assert cache.misses == 1
        
        # Test exists
        assert cache.exists("test_key") is True
        assert cache.exists("nonexistent") is False
        
        # Test delete
        assert cache.delete("test_key") is True
        assert cache.get("test_key") is None
        assert cache.exists("test_key") is False
    
    def test_file_cache_compression(self, temp_cache_dir):
        """Test file cache compression."""
        # Test with compression
        cache_compressed = FileCache(cache_dir=temp_cache_dir, compression=True)
        
        test_data = "x" * 1000  # Large string
        cache_compressed.set("compressed", test_data)
        retrieved = cache_compressed.get("compressed")
        assert retrieved == test_data
        
        # Test without compression
        cache_uncompressed = FileCache(
            cache_dir=Path(temp_cache_dir) / "uncompressed", 
            compression=False
        )
        
        cache_uncompressed.set("uncompressed", test_data)
        retrieved = cache_uncompressed.get("uncompressed")
        assert retrieved == test_data
    
    def test_file_cache_stats(self, temp_cache_dir):
        """Test file cache statistics."""
        cache = FileCache(cache_dir=temp_cache_dir)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        
        # Add some data
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Get some data
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['file_count'] == 2
        
        # Reset stats
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0


class TestMockRedisCache:
    """Test mock Redis cache."""
    
    def test_mock_redis_basic_operations(self):
        """Test basic mock Redis operations."""
        cache = MockRedisCache()
        
        # Test connection
        assert cache.connect() is True
        assert cache.is_connected() is True
        
        # Test set and get
        assert cache.set("test_key", "test_value") is True
        assert cache.get("test_key") == "test_value"
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        assert cache.misses == 1
        
        # Test exists
        assert cache.exists("test_key") is True
        assert cache.exists("nonexistent") is False
        
        # Test delete
        assert cache.delete("test_key") is True
        assert cache.get("test_key") is None
        
        # Test clear
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cleared = cache.clear()
        assert cleared == 2
        assert cache.get("key1") is None
    
    def test_mock_redis_stats(self):
        """Test mock Redis statistics."""
        cache = MockRedisCache()
        
        stats = cache.get_stats()
        assert stats['mock'] is True
        assert stats['connected'] is True
        
        # Test operations
        cache.set("key", "value")
        cache.get("key")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestCacheManager:
    """Test cache manager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_config(self, temp_cache_dir):
        """Create cache configuration."""
        return OmegaConf.create({
            'redis': {
                'enabled': False  # Use mock Redis
            },
            'file_cache': {
                'enabled': True,
                'cache_dir': temp_cache_dir,
                'max_cache_size': 0.1,
                'compression': True
            },
            'strategies': {
                'grid_features': {
                    'enabled': True,
                    'compression': True
                },
                'program_results': {
                    'enabled': True,
                    'max_entries': 100
                },
                'heuristic_values': {
                    'enabled': True,
                    'max_entries': 200
                }
            }
        })
    
    def test_cache_manager_initialization(self, cache_config):
        """Test cache manager initialization."""
        manager = CacheManager(cache_config)
        
        assert manager.file_cache is not None
        assert manager.redis_cache is None  # Disabled
        assert manager.strategies is not None
    
    def test_cache_manager_basic_operations(self, cache_config):
        """Test basic cache manager operations."""
        manager = CacheManager(cache_config)
        
        # Test set and get
        test_data = {"test": "data"}
        assert manager.set("test_key", test_data) is True
        
        retrieved = manager.get("test_key")
        assert retrieved == test_data
        
        # Test exists and delete
        assert manager.exists("test_key") is True
        assert manager.delete("test_key") is True
        assert manager.exists("test_key") is False
    
    def test_cache_manager_grid_features(self, cache_config):
        """Test grid features caching."""
        manager = CacheManager(cache_config)
        
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        features = np.array([0.1, 0.2, 0.3])
        
        # Test caching features
        assert manager.set_grid_features(grid, "orbit", features) is True
        
        # Test retrieving features
        retrieved = manager.get_grid_features(grid, "orbit")
        assert np.array_equal(retrieved, features)
        
        # Test different feature type
        assert manager.get_grid_features(grid, "spectral") is None
    
    def test_cache_manager_program_results(self, cache_config):
        """Test program results caching."""
        manager = CacheManager(cache_config)
        
        program = DSLProgram([DSLOperation("Rotate90", {})])
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = {"output_grid": [[3, 1], [4, 2]], "success": True}
        
        # Test caching result
        assert manager.set_program_result(program, grid, result) is True
        
        # Test retrieving result
        retrieved = manager.get_program_result(program, grid)
        assert retrieved == result
        
        # Test different program
        different_program = DSLProgram([DSLOperation("Rotate180", {})])
        assert manager.get_program_result(different_program, grid) is None
    
    def test_cache_manager_heuristic_values(self, cache_config):
        """Test heuristic values caching."""
        manager = CacheManager(cache_config)
        
        current_grid = np.array([[1, 2]], dtype=np.int32)
        target_grid = np.array([[2, 1]], dtype=np.int32)
        heuristic_value = 3.14
        
        # Test caching heuristic
        assert manager.set_heuristic_value(
            current_grid, target_grid, "tier1", heuristic_value
        ) is True
        
        # Test retrieving heuristic
        retrieved = manager.get_heuristic_value(
            current_grid, target_grid, "tier1"
        )
        assert retrieved == heuristic_value
        
        # Test different heuristic type
        assert manager.get_heuristic_value(
            current_grid, target_grid, "tier2"
        ) is None
    
    def test_cache_manager_stats(self, cache_config):
        """Test cache manager statistics."""
        manager = CacheManager(cache_config)
        
        stats = manager.get_stats()
        
        assert 'redis_enabled' in stats
        assert 'file_enabled' in stats
        assert 'strategies' in stats
        assert 'combined' in stats
        
        assert stats['redis_enabled'] is False
        assert stats['file_enabled'] is True
        
        # Test with some operations
        manager.set("test_key", "test_value")
        manager.get("test_key")  # Hit
        manager.get("nonexistent")  # Miss
        
        stats = manager.get_stats()
        combined = stats['combined']
        assert combined['total_hits'] >= 1
        assert combined['total_requests'] >= 2
        assert combined['hit_rate'] > 0
    
    def test_cache_manager_with_redis_enabled(self, cache_config, temp_cache_dir):
        """Test cache manager with Redis enabled (mock)."""
        # Enable Redis in config
        cache_config.redis.enabled = True
        
        manager = CacheManager(cache_config)
        
        # Should have both caches
        assert manager.file_cache is not None
        assert manager.redis_cache is not None
        
        # Test auto cache type (should use both)
        test_data = {"test": "data"}
        assert manager.set("test_key", test_data, cache_type='auto') is True
        
        # Should be able to retrieve from either cache
        assert manager.get("test_key", cache_type='redis') == test_data
        assert manager.get("test_key", cache_type='file') == test_data


class TestCacheFactory:
    """Test cache factory function."""
    
    def test_create_cache_manager_default(self):
        """Test creating cache manager with default config."""
        manager = create_cache_manager()
        
        assert manager.file_cache is not None
        assert manager.redis_cache is None  # Disabled by default
        
        stats = manager.get_stats()
        assert stats['file_enabled'] is True
        assert stats['redis_enabled'] is False
    
    def test_create_cache_manager_custom_config(self):
        """Test creating cache manager with custom config."""
        config = OmegaConf.create({
            'redis': {'enabled': False},
            'file_cache': {
                'enabled': True,
                'cache_dir': '.test_cache',
                'max_cache_size': 0.5
            },
            'strategies': {
                'grid_features': {'enabled': False}
            }
        })
        
        manager = create_cache_manager(config)
        
        assert manager.file_cache is not None
        assert manager.strategies['grid_features']['enabled'] is False


if __name__ == "__main__":
    pytest.main([__file__])