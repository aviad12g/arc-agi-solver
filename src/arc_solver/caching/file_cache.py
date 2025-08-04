"""File-based caching implementation."""

import os
import pickle
import json
import gzip
import logging
import time
import shutil
from typing import Any, Optional, Dict, Union
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class FileCache:
    """File-based cache implementation."""
    
    def __init__(self,
                 cache_dir: Union[str, Path] = ".cache/arc_solver",
                 max_cache_size: float = 1.0,  # GB
                 default_ttl: int = 86400,  # 24 hours
                 compression: bool = True):
        """Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_cache_size: Maximum cache size in GB
            default_ttl: Default time-to-live in seconds
            compression: Whether to compress cache files
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size * 1024 * 1024 * 1024  # Convert to bytes
        self.default_ttl = default_ttl
        self.compression = compression
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        
        logger.info(f"File cache initialized: {self.cache_dir} "
                   f"(max_size: {max_cache_size:.1f}GB, compression: {compression})")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Hash key to create safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Create subdirectory based on first 2 characters of hash
        subdir = self.cache_dir / key_hash[:2]
        subdir.mkdir(exist_ok=True)
        
        # Add extension based on compression
        ext = ".pkl.gz" if self.compression else ".pkl"
        return subdir / f"{key_hash}{ext}"
    
    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path for cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Path to metadata file
        """
        return cache_path.with_suffix(cache_path.suffix + ".meta")
    
    def _is_expired(self, cache_path: Path, ttl: Optional[int] = None) -> bool:
        """Check if cache file is expired.
        
        Args:
            cache_path: Path to cache file
            ttl: Time-to-live in seconds
            
        Returns:
            True if expired
        """
        if not cache_path.exists():
            return True
        
        # Check TTL
        ttl = ttl or self.default_ttl
        if ttl <= 0:
            return False  # No expiration
        
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age > ttl
    
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            cache_path = self._get_cache_path(key)
            
            # Check if file exists and is not expired
            if not cache_path.exists() or self._is_expired(cache_path, ttl):
                self.misses += 1
                return None
            
            # Load data
            if self.compression:
                with gzip.open(cache_path, 'rb') as f:
                    value = pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
            
            self.hits += 1
            logger.debug(f"File cache hit: {key}")
            return value
            
        except Exception as e:
            logger.warning(f"File cache get error for key '{key}': {e}")
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
        try:
            cache_path = self._get_cache_path(key)
            metadata_path = self._get_metadata_path(cache_path)
            
            # Ensure parent directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            if self.compression:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            
            # Save metadata
            metadata = {
                'key': key,
                'created_at': time.time(),
                'ttl': ttl or self.default_ttl,
                'size': cache_path.stat().st_size
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"File cache set: {key}")
            
            # Check cache size and cleanup if needed
            self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.warning(f"File cache set error for key '{key}': {e}")
            self.errors += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            cache_path = self._get_cache_path(key)
            metadata_path = self._get_metadata_path(cache_path)
            
            deleted = False
            
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            if deleted:
                logger.debug(f"File cache delete: {key}")
            
            return deleted
            
        except Exception as e:
            logger.warning(f"File cache delete error for key '{key}': {e}")
            self.errors += 1
            return False
    
    def exists(self, key: str, ttl: Optional[int] = None) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
            
        Returns:
            True if key exists and not expired
        """
        try:
            cache_path = self._get_cache_path(key)
            return cache_path.exists() and not self._is_expired(cache_path, ttl)
        except Exception as e:
            logger.warning(f"File cache exists error for key '{key}': {e}")
            self.errors += 1
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            pattern: Key pattern to match (clears all if None)
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            
            if pattern:
                # TODO: Implement pattern matching for file cache
                logger.warning("Pattern matching not implemented for file cache")
                return 0
            else:
                # Clear entire cache directory
                if self.cache_dir.exists():
                    for item in self.cache_dir.rglob("*"):
                        if item.is_file():
                            item.unlink()
                            deleted_count += 1
                    
                    # Remove empty directories
                    for item in self.cache_dir.rglob("*"):
                        if item.is_dir() and not any(item.iterdir()):
                            item.rmdir()
                
                logger.info(f"Cleared {deleted_count} cache files")
                return deleted_count
                
        except Exception as e:
            logger.error(f"File cache clear error: {e}")
            self.errors += 1
            return 0
    
    def get_cache_size(self) -> int:
        """Get current cache size in bytes.
        
        Returns:
            Cache size in bytes
        """
        try:
            total_size = 0
            for item in self.cache_dir.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
            return total_size
        except Exception as e:
            logger.warning(f"Error calculating cache size: {e}")
            return 0
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup cache if it exceeds maximum size."""
        try:
            current_size = self.get_cache_size()
            
            if current_size <= self.max_cache_size:
                return
            
            logger.info(f"Cache size ({current_size / 1024 / 1024:.1f}MB) exceeds limit "
                       f"({self.max_cache_size / 1024 / 1024:.1f}MB), cleaning up...")
            
            # Get all cache files with their access times
            cache_files = []
            for item in self.cache_dir.rglob("*.pkl*"):
                if item.is_file() and not item.name.endswith('.meta'):
                    try:
                        stat = item.stat()
                        cache_files.append((item, stat.st_atime, stat.st_size))
                    except Exception:
                        continue
            
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove files until we're under the limit
            bytes_to_remove = current_size - int(self.max_cache_size * 0.8)  # Remove to 80% of limit
            bytes_removed = 0
            files_removed = 0
            
            for cache_path, _, file_size in cache_files:
                if bytes_removed >= bytes_to_remove:
                    break
                
                try:
                    # Remove cache file and metadata
                    metadata_path = self._get_metadata_path(cache_path)
                    
                    cache_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    bytes_removed += file_size
                    files_removed += 1
                    
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_path}: {e}")
            
            logger.info(f"Cache cleanup completed: removed {files_removed} files "
                       f"({bytes_removed / 1024 / 1024:.1f}MB)")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        try:
            cache_size = self.get_cache_size()
            
            # Count files
            file_count = 0
            for item in self.cache_dir.rglob("*.pkl*"):
                if item.is_file() and not item.name.endswith('.meta'):
                    file_count += 1
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'errors': self.errors,
                'hit_rate': self.hits / max(self.hits + self.misses, 1),
                'cache_dir': str(self.cache_dir),
                'cache_size_bytes': cache_size,
                'cache_size_mb': cache_size / 1024 / 1024,
                'max_cache_size_mb': self.max_cache_size / 1024 / 1024,
                'file_count': file_count,
                'compression': self.compression
            }
            
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {
                'hits': self.hits,
                'misses': self.misses,
                'errors': self.errors,
                'hit_rate': self.hits / max(self.hits + self.misses, 1),
                'error': str(e)
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        logger.info("File cache statistics reset")