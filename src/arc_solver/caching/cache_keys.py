"""Cache key generation utilities."""

import hashlib
import json
import numpy as np
from typing import Any, Dict, List, Union
from arc_solver.reasoning.dsl_engine import DSLProgram


class CacheKeyGenerator:
    """Generates consistent cache keys for different data types."""
    
    @staticmethod
    def grid_hash(grid: np.ndarray) -> str:
        """Generate SHA-1 hash for a grid.
        
        Args:
            grid: Input grid
            
        Returns:
            SHA-1 hash string
        """
        # Ensure consistent dtype and byte order
        grid_bytes = grid.astype(np.int32).tobytes()
        return hashlib.sha1(grid_bytes).hexdigest()
    
    @staticmethod
    def program_hash(program: DSLProgram) -> str:
        """Generate hash for a DSL program.
        
        Args:
            program: DSL program
            
        Returns:
            SHA-1 hash string
        """
        # Convert program to consistent string representation
        program_dict = program.to_dict()
        program_str = json.dumps(program_dict, sort_keys=True)
        return hashlib.sha1(program_str.encode()).hexdigest()
    
    @staticmethod
    def grid_pair_hash(grid1: np.ndarray, grid2: np.ndarray) -> str:
        """Generate hash for a pair of grids.
        
        Args:
            grid1: First grid
            grid2: Second grid
            
        Returns:
            SHA-1 hash string
        """
        hash1 = CacheKeyGenerator.grid_hash(grid1)
        hash2 = CacheKeyGenerator.grid_hash(grid2)
        combined = f"{hash1}:{hash2}"
        return hashlib.sha1(combined.encode()).hexdigest()
    
    @staticmethod
    def program_grid_hash(program: DSLProgram, grid: np.ndarray) -> str:
        """Generate hash for program + grid combination.
        
        Args:
            program: DSL program
            grid: Input grid
            
        Returns:
            SHA-1 hash string
        """
        prog_hash = CacheKeyGenerator.program_hash(program)
        grid_hash = CacheKeyGenerator.grid_hash(grid)
        combined = f"{prog_hash}:{grid_hash}"
        return hashlib.sha1(combined.encode()).hexdigest()
    
    @staticmethod
    def feature_key(grid: np.ndarray, feature_type: str) -> str:
        """Generate cache key for grid features.
        
        Args:
            grid: Input grid
            feature_type: Type of features (e.g., 'orbit', 'spectral')
            
        Returns:
            Cache key string
        """
        grid_hash = CacheKeyGenerator.grid_hash(grid)
        return f"features:{feature_type}:{grid_hash}"
    
    @staticmethod
    def heuristic_key(current_grid: np.ndarray, target_grid: np.ndarray, 
                     heuristic_type: str) -> str:
        """Generate cache key for heuristic values.
        
        Args:
            current_grid: Current grid state
            target_grid: Target grid state
            heuristic_type: Type of heuristic (e.g., 'tier1', 'tier2')
            
        Returns:
            Cache key string
        """
        pair_hash = CacheKeyGenerator.grid_pair_hash(current_grid, target_grid)
        return f"heuristic:{heuristic_type}:{pair_hash}"
    
    @staticmethod
    def program_result_key(program: DSLProgram, grid: np.ndarray) -> str:
        """Generate cache key for program execution results.
        
        Args:
            program: DSL program
            grid: Input grid
            
        Returns:
            Cache key string
        """
        prog_grid_hash = CacheKeyGenerator.program_grid_hash(program, grid)
        return f"program_result:{prog_grid_hash}"
    
    @staticmethod
    def custom_key(prefix: str, *args: Any) -> str:
        """Generate custom cache key from arbitrary arguments.
        
        Args:
            prefix: Key prefix
            *args: Arguments to include in key
            
        Returns:
            Cache key string
        """
        # Convert arguments to strings and hash
        arg_strs = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg_strs.append(CacheKeyGenerator.grid_hash(arg))
            elif isinstance(arg, DSLProgram):
                arg_strs.append(CacheKeyGenerator.program_hash(arg))
            else:
                arg_strs.append(str(arg))
        
        combined = ":".join([prefix] + arg_strs)
        return hashlib.sha1(combined.encode()).hexdigest()[:16]  # Shorter for custom keys