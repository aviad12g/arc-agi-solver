"""Enhanced DSL wrapper with performance optimizations and program management.

This module provides an optimized interface for DSL operations with features like
program caching, batch execution, and performance monitoring.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

from .dsl_engine import DSLEngine, DSLProgram, DSLOperation, create_dsl_engine
from .primitives import create_all_primitives

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Enhanced result from DSL program execution."""
    output_grid: np.ndarray
    success: bool
    execution_time: float
    operations_executed: int
    program_hash: str
    error: Optional[str] = None
    cached: bool = False


class ProgramCache:
    """Cache for DSL program execution results."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize program cache.
        
        Args:
            max_size: Maximum number of cached results
        """
        self.max_size = max_size
        self.cache: Dict[str, ExecutionResult] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, program: DSLProgram, input_grid: np.ndarray) -> str:
        """Create cache key from program and input grid."""
        program_hash = str(hash(program))
        grid_hash = hashlib.md5(input_grid.tobytes()).hexdigest()[:8]
        return f"{program_hash}_{grid_hash}"
    
    def get(self, program: DSLProgram, input_grid: np.ndarray) -> Optional[ExecutionResult]:
        """Get cached result if available."""
        key = self._make_key(program, input_grid)
        
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                
                result = self.cache[key]
                # Create new result with cached flag
                cached_result = ExecutionResult(
                    output_grid=result.output_grid.copy(),
                    success=result.success,
                    execution_time=result.execution_time,
                    operations_executed=result.operations_executed,
                    program_hash=result.program_hash,
                    error=result.error,
                    cached=True
                )
                
                self.hits += 1
                return cached_result
            
            self.misses += 1
            return None
    
    def put(self, program: DSLProgram, input_grid: np.ndarray, result: ExecutionResult):
        """Cache execution result."""
        key = self._make_key(program, input_grid)
        
        with self.lock:
            # Remove oldest entries if cache is full
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            # Add new result
            self.cache[key] = result
            self.access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class EnhancedDSLEngine:
    """Enhanced DSL engine with caching, batching, and performance optimizations."""
    
    def __init__(self, max_program_length: int = 4, 
                 enable_cache: bool = True,
                 cache_size: int = 10000,
                 max_execution_time: float = 0.001):
        """Initialize enhanced DSL engine.
        
        Args:
            max_program_length: Maximum program length (K=4 in spec)
            enable_cache: Whether to enable result caching
            cache_size: Maximum cache size
            max_execution_time: Maximum execution time per program
        """
        self.base_engine = create_dsl_engine(max_program_length)
        self.max_program_length = max_program_length
        self.max_execution_time = max_execution_time
        
        # Caching
        self.enable_cache = enable_cache
        self.cache = ProgramCache(cache_size) if enable_cache else None
        
        # Performance tracking
        self.total_executions = 0
        self.total_cache_hits = 0
        self.total_execution_time = 0.0
        
        # Thread pool for batch operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Enhanced DSL engine initialized (cache: {enable_cache}, max_length: {max_program_length})")
    
    def execute_program(self, program: DSLProgram, 
                       input_grid: np.ndarray,
                       use_cache: bool = True) -> ExecutionResult:
        """Execute DSL program with caching and enhanced error handling.
        
        Args:
            program: DSL program to execute
            input_grid: Input grid
            use_cache: Whether to use cache for this execution
            
        Returns:
            ExecutionResult with detailed information
        """
        start_time = time.perf_counter()
        
        # Check cache first
        if self.enable_cache and use_cache and self.cache:
            cached_result = self.cache.get(program, input_grid)
            if cached_result is not None:
                self.total_cache_hits += 1
                self.total_executions += 1  # Count cached executions too
                return cached_result
        
        # Execute program
        try:
            output_grid, exec_info = self.base_engine.execute_program(program, input_grid)
            
            program_hash = str(hash(program))
            result = ExecutionResult(
                output_grid=output_grid,
                success=exec_info['success'],
                execution_time=exec_info['execution_time'],
                operations_executed=exec_info['operations_executed'],
                program_hash=program_hash,
                error=exec_info.get('error'),
                cached=False
            )
            
            # Cache successful results
            if self.enable_cache and result.success and self.cache:
                self.cache.put(program, input_grid, result)
            
            # Update statistics
            self.total_executions += 1
            self.total_execution_time += result.execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Program execution failed: {e}")
            
            return ExecutionResult(
                output_grid=input_grid.copy(),
                success=False,
                execution_time=execution_time,
                operations_executed=0,
                program_hash=str(hash(program)),
                error=str(e),
                cached=False
            )
    
    def execute_batch(self, programs: List[DSLProgram], 
                     input_grid: np.ndarray,
                     max_workers: Optional[int] = None) -> List[ExecutionResult]:
        """Execute multiple programs in parallel.
        
        Args:
            programs: List of programs to execute
            input_grid: Input grid (same for all programs)
            max_workers: Maximum number of worker threads
            
        Returns:
            List of execution results
        """
        if max_workers is None:
            max_workers = min(len(programs), 4)
        
        def execute_single(program):
            return self.execute_program(program, input_grid)
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(execute_single, programs))
        
        return results
    
    def validate_program(self, program: DSLProgram) -> Tuple[bool, Optional[str]]:
        """Validate program using base engine."""
        return self.base_engine.validate_program(program)
    
    def create_operation(self, primitive_name: str, **parameters) -> DSLOperation:
        """Create operation using base engine."""
        return self.base_engine.create_operation(primitive_name, **parameters)
    
    def create_empty_program(self) -> DSLProgram:
        """Create empty program."""
        return self.base_engine.create_empty_program()
    
    def get_primitive_names(self) -> List[str]:
        """Get available primitive names."""
        return self.base_engine.get_primitive_names()
    
    def enumerate_programs(self, max_length: Optional[int] = None,
                          filter_valid: bool = True) -> List[DSLProgram]:
        """Enumerate programs with optional filtering.
        
        Args:
            max_length: Maximum program length
            filter_valid: Whether to filter out invalid programs
            
        Returns:
            List of programs
        """
        programs = self.base_engine.enumerate_programs(max_length)
        
        if filter_valid:
            valid_programs = []
            for program in programs:
                is_valid, _ = self.validate_program(program)
                if is_valid:
                    valid_programs.append(program)
            return valid_programs
        
        return programs
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_stats = self.base_engine.get_execution_stats()
        
        stats = {
            'total_executions': self.total_executions,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': (self.total_execution_time / self.total_executions 
                                     if self.total_executions > 0 else 0.0),
            'cache_enabled': self.enable_cache,
            'cache_hits': self.total_cache_hits
        }
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats.update({
                'cache_size': cache_stats['size'],
                'cache_hit_rate': cache_stats['hit_rate'],
                'cache_hits_total': cache_stats['hits'],
                'cache_misses_total': cache_stats['misses']
            })
        
        # Add base engine stats
        stats.update({
            'base_engine_executions': base_stats['total_executions'],
            'base_engine_time': base_stats['total_time']
        })
        
        return stats
    
    def clear_cache(self):
        """Clear the execution cache."""
        if self.cache:
            self.cache.clear()
            logger.info("DSL execution cache cleared")
    
    def optimize_program(self, program: DSLProgram) -> DSLProgram:
        """Apply basic optimizations to a program.
        
        This is a placeholder for more sophisticated optimizations.
        """
        # Remove redundant operations (simplified)
        optimized_ops = []
        
        for op in program.operations:
            # Skip redundant rotations (4 × 90° = identity)
            if (len(optimized_ops) >= 3 and 
                all(prev_op.primitive_name == "Rotate90" for prev_op in optimized_ops[-3:]) and
                op.primitive_name == "Rotate90"):
                # Remove the last 3 Rotate90 operations instead of adding the 4th
                optimized_ops = optimized_ops[:-3]
                continue
            
            # Skip double reflections (2 × same reflection = identity)
            if (len(optimized_ops) >= 1 and
                optimized_ops[-1].primitive_name == op.primitive_name and
                op.primitive_name in ["ReflectH", "ReflectV"]):
                # Remove the previous reflection instead of adding the duplicate
                optimized_ops.pop()
                continue
            
            optimized_ops.append(op)
        
        return DSLProgram(optimized_ops, program.max_length)
    
    def find_equivalent_programs(self, target_program: DSLProgram,
                                test_grids: List[np.ndarray]) -> List[DSLProgram]:
        """Find programs that produce equivalent results on test grids.
        
        Args:
            target_program: Program to find equivalents for
            test_grids: Grids to test equivalence on
            
        Returns:
            List of equivalent programs
        """
        # Execute target program on all test grids
        target_results = []
        for grid in test_grids:
            result = self.execute_program(target_program, grid)
            if result.success:
                target_results.append(result.output_grid)
            else:
                return []  # Can't find equivalents if target fails
        
        # Test other programs
        equivalent_programs = []
        candidate_programs = self.enumerate_programs(max_length=len(target_program))
        
        for candidate in candidate_programs:
            if candidate == target_program:
                continue
            
            # Test on all grids
            is_equivalent = True
            for i, grid in enumerate(test_grids):
                result = self.execute_program(candidate, grid)
                if not result.success or not np.array_equal(result.output_grid, target_results[i]):
                    is_equivalent = False
                    break
            
            if is_equivalent:
                equivalent_programs.append(candidate)
        
        return equivalent_programs
    
    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


def create_enhanced_dsl_engine(max_program_length: int = 4,
                              enable_cache: bool = True,
                              cache_size: int = 10000) -> EnhancedDSLEngine:
    """Factory function to create enhanced DSL engine.
    
    Args:
        max_program_length: Maximum program length
        enable_cache: Whether to enable caching
        cache_size: Cache size
        
    Returns:
        Configured EnhancedDSLEngine instance
    """
    return EnhancedDSLEngine(
        max_program_length=max_program_length,
        enable_cache=enable_cache,
        cache_size=cache_size
    )