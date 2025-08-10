"""Main cache manager that coordinates different cache backends."""

import logging
from typing import Any, Optional, Dict, Union, List
from omegaconf import DictConfig

from .redis_cache import RedisCache, MockRedisCache, REDIS_AVAILABLE
from .file_cache import FileCache
from .cache_keys import CacheKeyGenerator

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages multiple cache backends with fallback strategies."""
    
    def __init__(self, config: DictConfig):
        """Initialize cache manager with configuration.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.key_generator = CacheKeyGenerator()
        
        # Initialize cache backends
        self.redis_cache: Optional[Union[RedisCache, MockRedisCache]] = None
        self.file_cache: Optional[FileCache] = None
        
        self._init_redis_cache()
        self._init_file_cache()
        
        # Cache strategies
        self.strategies = config.get('strategies', {})
        
        logger.info("Cache manager initialized")
    
    def _init_redis_cache(self) -> None:
        """Initialize Redis cache if enabled."""
        redis_config = self.config.get('redis', {})
        
        if not redis_config.get('enabled', False):
            logger.info("Redis cache disabled")
            return
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using mock cache")
            self.redis_cache = MockRedisCache()
            return
        
        try:
            self.redis_cache = RedisCache(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                connection_timeout=redis_config.get('connection_timeout', 5.0),
                default_ttl=redis_config.get('ttl', 86400)
            )
            
            # Test connection
            if self.redis_cache.connect():
                logger.info("Redis cache initialized successfully")
            else:
                logger.warning("Redis connection failed, using mock cache")
                self.redis_cache = MockRedisCache()
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.redis_cache = MockRedisCache()
    
    def _init_file_cache(self) -> None:
        """Initialize file cache if enabled."""
        file_config = self.config.get('file_cache', {})
        
        if not file_config.get('enabled', True):
            logger.info("File cache disabled")
            return
        
        try:
            self.file_cache = FileCache(
                cache_dir=file_config.get('cache_dir', '.cache/arc_solver'),
                max_cache_size=file_config.get('max_cache_size', 1.0),
                default_ttl=file_config.get('cache_ttl', 86400),
                compression=file_config.get('compression', True)
            )
            
            logger.info("File cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize file cache: {e}")
            self.file_cache = None
    
    def get(self, key: str, cache_type: str = 'auto') -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            cache_type: Cache type ('redis', 'file', 'auto')
            
        Returns:
            Cached value or None if not found
        """
        if cache_type == 'auto':
            # Try Redis first, then file cache
            if self.redis_cache:
                value = self.redis_cache.get(key)
                if value is not None:
                    return value
            
            if self.file_cache:
                return self.file_cache.get(key)
            
            return None
            
        elif cache_type == 'redis' and self.redis_cache:
            return self.redis_cache.get(key)
            
        elif cache_type == 'file' and self.file_cache:
            return self.file_cache.get(key)
        
        return None
    
    def set(self, key: str, value: Any, cache_type: str = 'auto', ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Cache type ('redis', 'file', 'auto')
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        success = False
        
        if cache_type == 'auto':
            # Set in both caches if available
            if self.redis_cache:
                success = self.redis_cache.set(key, value, ttl) or success
            
            if self.file_cache:
                success = self.file_cache.set(key, value, ttl) or success
            
            return success
            
        elif cache_type == 'redis' and self.redis_cache:
            return self.redis_cache.set(key, value, ttl)
            
        elif cache_type == 'file' and self.file_cache:
            return self.file_cache.set(key, value, ttl)
        
        return False
    
    def delete(self, key: str, cache_type: str = 'auto') -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            cache_type: Cache type ('redis', 'file', 'auto')
            
        Returns:
            True if successful
        """
        success = False
        
        if cache_type == 'auto':
            # Delete from both caches
            if self.redis_cache:
                success = self.redis_cache.delete(key) or success
            
            if self.file_cache:
                success = self.file_cache.delete(key) or success
            
            return success
            
        elif cache_type == 'redis' and self.redis_cache:
            return self.redis_cache.delete(key)
            
        elif cache_type == 'file' and self.file_cache:
            return self.file_cache.delete(key)
        
        return False
    
    def exists(self, key: str, cache_type: str = 'auto') -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            cache_type: Cache type ('redis', 'file', 'auto')
            
        Returns:
            True if key exists
        """
        if cache_type == 'auto':
            # Check Redis first, then file cache
            if self.redis_cache and self.redis_cache.exists(key):
                return True
            
            if self.file_cache and self.file_cache.exists(key):
                return True
            
            return False
            
        elif cache_type == 'redis' and self.redis_cache:
            return self.redis_cache.exists(key)
            
        elif cache_type == 'file' and self.file_cache:
            return self.file_cache.exists(key)
        
        return False
    
    def clear(self, cache_type: str = 'auto', pattern: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            cache_type: Cache type ('redis', 'file', 'auto')
            pattern: Key pattern to match
            
        Returns:
            Number of entries cleared
        """
        total_cleared = 0
        
        if cache_type == 'auto':
            # Clear both caches
            if self.redis_cache:
                total_cleared += self.redis_cache.clear(pattern)
            
            if self.file_cache:
                total_cleared += self.file_cache.clear(pattern)
            
            return total_cleared
            
        elif cache_type == 'redis' and self.redis_cache:
            return self.redis_cache.clear(pattern)
            
        elif cache_type == 'file' and self.file_cache:
            return self.file_cache.clear(pattern)
        
        return 0
    
    # High-level caching methods for specific data types
    
    def get_grid_features(self, grid, feature_type: str) -> Optional[Any]:
        """Get cached grid features.
        
        Args:
            grid: Input grid
            feature_type: Type of features
            
        Returns:
            Cached features or None
        """
        if not self.strategies.get('grid_features', {}).get('enabled', True):
            return None
        
        key = self.key_generator.feature_key(grid, feature_type)
        return self.get(key)
    
    def set_grid_features(self, grid, feature_type: str, features: Any) -> bool:
        """Cache grid features.
        
        Args:
            grid: Input grid
            feature_type: Type of features
            features: Features to cache
            
        Returns:
            True if successful
        """
        if not self.strategies.get('grid_features', {}).get('enabled', True):
            return False
        
        key = self.key_generator.feature_key(grid, feature_type)
        
        # Use compression for features
        cache_type = 'file' if self.strategies.get('grid_features', {}).get('compression', True) else 'auto'
        
        return self.set(key, features, cache_type)
    
    def get_program_result(self, program, grid) -> Optional[Any]:
        """Get cached program execution result.
        
        Args:
            program: DSL program
            grid: Input grid
            
        Returns:
            Cached result or None
        """
        if not self.strategies.get('program_results', {}).get('enabled', True):
            return None
        
        key = self.key_generator.program_result_key(program, grid)
        return self.get(key)
    
    def set_program_result(self, program, grid, result: Any) -> bool:
        """Cache program execution result.
        
        Args:
            program: DSL program
            grid: Input grid
            result: Execution result
            
        Returns:
            True if successful
        """
        if not self.strategies.get('program_results', {}).get('enabled', True):
            return False
        
        key = self.key_generator.program_result_key(program, grid)
        
        # Check max entries limit
        max_entries = self.strategies.get('program_results', {}).get('max_entries', 10000)
        # TODO: Implement LRU eviction for max_entries
        
        return self.set(key, result)
    
    def get_heuristic_value(self, current_grid, target_grid, heuristic_type: str) -> Optional[float]:
        """Get cached heuristic value.
        
        Args:
            current_grid: Current grid state
            target_grid: Target grid state
            heuristic_type: Type of heuristic
            
        Returns:
            Cached heuristic value or None
        """
        if not self.strategies.get('heuristic_values', {}).get('enabled', True):
            return None
        
        key = self.key_generator.heuristic_key(current_grid, target_grid, heuristic_type)
        return self.get(key)
    
    def set_heuristic_value(self, current_grid, target_grid, heuristic_type: str, value: float) -> bool:
        """Cache heuristic value.
        
        Args:
            current_grid: Current grid state
            target_grid: Target grid state
            heuristic_type: Type of heuristic
            value: Heuristic value
            
        Returns:
            True if successful
        """
        if not self.strategies.get('heuristic_values', {}).get('enabled', True):
            return False
        
        key = self.key_generator.heuristic_key(current_grid, target_grid, heuristic_type)
        
        # Check max entries limit
        max_entries = self.strategies.get('heuristic_values', {}).get('max_entries', 50000)
        # TODO: Implement LRU eviction for max_entries
        
        return self.set(key, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = {
            'redis_enabled': self.redis_cache is not None,
            'file_enabled': self.file_cache is not None,
            'strategies': self.strategies
        }
        
        if self.redis_cache:
            stats['redis'] = self.redis_cache.get_stats()
        
        if self.file_cache:
            stats['file'] = self.file_cache.get_stats()
        
        # Calculate combined hit rate
        total_hits = 0
        total_requests = 0
        
        if self.redis_cache:
            redis_stats = self.redis_cache.get_stats()
            total_hits += redis_stats['hits']
            total_requests += redis_stats['hits'] + redis_stats['misses']
        
        if self.file_cache:
            file_stats = self.file_cache.get_stats()
            total_hits += file_stats['hits']
            total_requests += file_stats['hits'] + file_stats['misses']
        
        stats['combined'] = {
            'total_hits': total_hits,
            'total_requests': total_requests,
            'hit_rate': total_hits / max(total_requests, 1)
        }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset all cache statistics."""
        if self.redis_cache:
            self.redis_cache.reset_stats()
        
        if self.file_cache:
            self.file_cache.reset_stats()
        
        logger.info("All cache statistics reset")
    
    def close(self) -> None:
        """Close all cache connections."""
        if self.redis_cache:
            self.redis_cache.disconnect()
        
        logger.info("Cache manager closed")


def create_cache_manager(config: Optional[DictConfig] = None) -> CacheManager:
    """Factory function to create cache manager.
    
    Args:
        config: Cache configuration (uses default if None)
        
    Returns:
        Configured CacheManager instance
    """
    if config is None:
        # Create default configuration
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'redis': {
                'enabled': False
            },
            'file_cache': {
                'enabled': True,
                'cache_dir': '.cache/arc_solver',
                'max_cache_size': 1.0,
                'compression': True
            },
            'strategies': {
                'grid_features': {
                    'enabled': True,
                    'compression': True
                },
                'program_results': {
                    'enabled': True,
                    'max_entries': 10000
                },
                'heuristic_values': {
                    'enabled': True,
                    'max_entries': 50000
                }
            }
        })
    
    return CacheManager(config)