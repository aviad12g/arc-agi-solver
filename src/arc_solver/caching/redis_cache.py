"""Redis-based caching implementation."""

import json
import pickle
import logging
from typing import Any, Optional, Dict, Union
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 connection_timeout: float = 5.0,
                 default_ttl: int = 86400):  # 24 hours
        """Initialize Redis cache.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            connection_timeout: Connection timeout in seconds
            default_ttl: Default time-to-live in seconds
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.connection_timeout = connection_timeout
        self.default_ttl = default_ttl
        
        self.client: Optional[redis.Redis] = None
        self.connected = False
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        
        logger.info(f"Redis cache initialized: {host}:{port}/{db}")
    
    def connect(self) -> bool:
        """Connect to Redis server.
        
        Returns:
            True if connection successful
        """
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.connection_timeout,
                decode_responses=False  # Keep binary data as bytes
            )
            
            # Test connection
            self.client.ping()
            self.connected = True
            
            logger.info(f"Connected to Redis server: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Redis server."""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error disconnecting from Redis: {e}")
        
        self.connected = False
        self.client = None
        logger.info("Disconnected from Redis server")
    
    def is_connected(self) -> bool:
        """Check if connected to Redis server.
        
        Returns:
            True if connected
        """
        if not self.connected or not self.client:
            return False
        
        try:
            self.client.ping()
            return True
        except Exception:
            self.connected = False
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_connected():
            if not self.connect():
                return None
        
        try:
            data = self.client.get(key)
            if data is None:
                self.misses += 1
                return None
            
            # Deserialize data
            value = pickle.loads(data)
            self.hits += 1
            
            logger.debug(f"Cache hit: {key}")
            return value
            
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {e}")
            self.errors += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            if not self.connect():
                return False
        
        try:
            # Serialize data
            data = pickle.dumps(value)
            
            # Set with TTL
            ttl = ttl or self.default_ttl
            success = self.client.setex(key, ttl, data)
            
            if success:
                logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            
            return bool(success)
            
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {e}")
            self.errors += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            if not self.connect():
                return False
        
        try:
            deleted = self.client.delete(key)
            logger.debug(f"Cache delete: {key} (deleted: {deleted})")
            return deleted > 0
            
        except Exception as e:
            logger.warning(f"Cache delete error for key '{key}': {e}")
            self.errors += 1
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self.is_connected():
            if not self.connect():
                return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.warning(f"Cache exists error for key '{key}': {e}")
            self.errors += 1
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            pattern: Key pattern to match (clears all if None)
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected():
            if not self.connect():
                return 0
        
        try:
            if pattern:
                # Delete keys matching pattern
                keys = self.client.keys(pattern)
                if keys:
                    deleted = self.client.delete(*keys)
                    logger.info(f"Cleared {deleted} cache entries matching '{pattern}'")
                    return deleted
                return 0
            else:
                # Clear entire database
                self.client.flushdb()
                logger.info("Cleared entire cache database")
                return -1  # Unknown count
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.errors += 1
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = {
            'hits': self.hits,
            'misses': self.misses,
            'errors': self.errors,
            'hit_rate': self.hits / max(self.hits + self.misses, 1),
            'connected': self.connected,
            'host': self.host,
            'port': self.port,
            'db': self.db
        }
        
        # Add Redis server info if connected
        if self.is_connected():
            try:
                info = self.client.info()
                stats.update({
                    'redis_version': info.get('redis_version', 'unknown'),
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                })
            except Exception as e:
                logger.warning(f"Failed to get Redis info: {e}")
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        logger.info("Cache statistics reset")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class MockRedisCache:
    """Mock Redis cache for testing without Redis server."""
    
    def __init__(self, *args, **kwargs):
        """Initialize mock cache."""
        self.data = {}
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.connected = True
        
        logger.info("Mock Redis cache initialized")
    
    def connect(self) -> bool:
        """Mock connect."""
        return True
    
    def disconnect(self) -> None:
        """Mock disconnect."""
        pass
    
    def is_connected(self) -> bool:
        """Mock connection check."""
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Mock get."""
        if key in self.data:
            self.hits += 1
            return self.data[key]
        else:
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Mock set."""
        self.data[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """Mock delete."""
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Mock exists."""
        return key in self.data
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """Mock clear."""
        count = len(self.data)
        self.data.clear()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Mock stats."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'errors': self.errors,
            'hit_rate': self.hits / max(self.hits + self.misses, 1),
            'connected': True,
            'mock': True
        }
    
    def reset_stats(self) -> None:
        """Mock reset stats."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass