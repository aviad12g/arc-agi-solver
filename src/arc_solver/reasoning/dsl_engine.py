"""DSL Engine for program synthesis and execution.

This module provides the core engine for creating, validating, and executing
DSL programs composed of primitive operations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import hashlib

from .primitives import DSLPrimitive, create_all_primitives

logger = logging.getLogger(__name__)


@dataclass
class DSLOperation:
    """Represents a single operation in a DSL program."""
    primitive_name: str
    parameters: Dict[str, Any]
    
    def __hash__(self) -> int:
        """Generate hash for operation."""
        param_str = str(sorted(self.parameters.items()))
        return hash((self.primitive_name, param_str))
    
    def __str__(self) -> str:
        """String representation of operation."""
        if not self.parameters:
            return self.primitive_name
        
        param_strs = [f"{k}={v}" for k, v in self.parameters.items()]
        return f"{self.primitive_name}({', '.join(param_strs)})"


class DSLProgram:
    """Represents a complete DSL program as a sequence of operations."""
    
    def __init__(self, operations: List[DSLOperation], max_length: int = 4):
        """Initialize DSL program.
        
        Args:
            operations: List of operations in execution order
            max_length: Maximum allowed program length
        """
        if len(operations) > max_length:
            raise ValueError(f"Program length {len(operations)} exceeds maximum {max_length}")
        
        self.operations = operations
        self.max_length = max_length
        self._hash = None
    
    def __len__(self) -> int:
        """Get program length."""
        return len(self.operations)
    
    def __hash__(self) -> int:
        """Generate hash for program."""
        if self._hash is None:
            self._hash = hash(tuple(self.operations))
        return self._hash
    
    def __eq__(self, other) -> bool:
        """Check program equality."""
        if not isinstance(other, DSLProgram):
            return False
        return self.operations == other.operations
    
    def __str__(self) -> str:
        """String representation of program."""
        if not self.operations:
            return "EmptyProgram"
        
        op_strs = [str(op) for op in self.operations]
        return " -> ".join(op_strs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert program to dictionary representation."""
        return {
            'operations': [
                {
                    'primitive': op.primitive_name,
                    'parameters': op.parameters
                }
                for op in self.operations
            ],
            'length': len(self.operations)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSLProgram':
        """Create program from dictionary representation."""
        operations = []
        for op_data in data['operations']:
            operation = DSLOperation(
                primitive_name=op_data['primitive'],
                parameters=op_data['parameters']
            )
            operations.append(operation)
        
        return cls(operations)
    
    def copy(self) -> 'DSLProgram':
        """Create a copy of the program."""
        return DSLProgram([
            DSLOperation(op.primitive_name, op.parameters.copy())
            for op in self.operations
        ], self.max_length)
    
    def append(self, operation: DSLOperation) -> 'DSLProgram':
        """Create new program with appended operation."""
        if len(self.operations) >= self.max_length:
            raise ValueError(f"Cannot append to program at maximum length {self.max_length}")
        
        new_operations = self.operations + [operation]
        return DSLProgram(new_operations, self.max_length)


class DSLEngine:
    """Engine for executing DSL programs and managing primitives."""
    
    def __init__(self, max_program_length: int = 4, 
                 max_execution_time: float = 0.001):
        """Initialize DSL engine.
        
        Args:
            max_program_length: Maximum length of programs (K=4 in spec)
            max_execution_time: Maximum execution time per program
        """
        self.max_program_length = max_program_length
        self.max_execution_time = max_execution_time
        
        # Load all primitives
        self.primitives = create_all_primitives()
        
        # Execution statistics
        self.execution_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"DSL Engine initialized with {len(self.primitives)} primitives")
    
    def execute_program(self, program: DSLProgram, 
                       input_grid: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute a DSL program on an input grid.
        
        Args:
            program: DSL program to execute
            input_grid: Input grid
            
        Returns:
            Tuple of (output_grid, execution_info)
        """
        start_time = time.perf_counter()
        
        try:
            current_grid = input_grid.copy()
            execution_info = {
                'success': True,
                'operations_executed': 0,
                'execution_time': 0.0,
                'error': None
            }
            
            # Execute each operation in sequence
            for i, operation in enumerate(program.operations):
                if operation.primitive_name not in self.primitives:
                    raise ValueError(f"Unknown primitive: {operation.primitive_name}")
                
                primitive = self.primitives[operation.primitive_name]
                
                # Execute primitive
                current_grid = primitive(current_grid, **operation.parameters)
                execution_info['operations_executed'] += 1
                
                # Check for timeout
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time > self.max_execution_time:
                    raise TimeoutError(f"Program execution exceeded {self.max_execution_time*1000:.1f}ms")
            
            execution_info['execution_time'] = time.perf_counter() - start_time
            
            # Update statistics
            self.execution_count += 1
            self.total_execution_time += execution_info['execution_time']
            
            return current_grid, execution_info
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            execution_info = {
                'success': False,
                'operations_executed': i if 'i' in locals() else 0,
                'execution_time': execution_time,
                'error': str(e)
            }
            
            logger.warning(f"Program execution failed: {e}")
            return input_grid.copy(), execution_info
    
    def validate_program(self, program: DSLProgram) -> Tuple[bool, Optional[str]]:
        """Validate a DSL program.
        
        Args:
            program: Program to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check program length
        if len(program) > self.max_program_length:
            return False, f"Program length {len(program)} exceeds maximum {self.max_program_length}"
        
        # Check each operation
        for i, operation in enumerate(program.operations):
            # Check primitive exists
            if operation.primitive_name not in self.primitives:
                return False, f"Unknown primitive at position {i}: {operation.primitive_name}"
            
            # Validate parameters
            primitive = self.primitives[operation.primitive_name]
            try:
                if not primitive.validate_params(**operation.parameters):
                    return False, f"Invalid parameters for {operation.primitive_name} at position {i}"
            except Exception as e:
                return False, f"Parameter validation error at position {i}: {e}"
        
        return True, None
    
    def create_operation(self, primitive_name: str, **parameters) -> DSLOperation:
        """Create a DSL operation with validation.
        
        Args:
            primitive_name: Name of the primitive
            **parameters: Parameters for the primitive
            
        Returns:
            DSLOperation instance
        """
        if primitive_name not in self.primitives:
            raise ValueError(f"Unknown primitive: {primitive_name}")
        
        primitive = self.primitives[primitive_name]
        if not primitive.validate_params(**parameters):
            raise ValueError(f"Invalid parameters for {primitive_name}: {parameters}")
        
        return DSLOperation(primitive_name, parameters)
    
    def get_available_operations(self, grid: np.ndarray) -> List[DSLOperation]:
        """Get all available operations that can be applied to the given grid.
        
        Args:
            grid: Input grid to generate operations for
            
        Returns:
            List of available DSL operations
        """
        operations = []
        
        # Generate operations for each primitive
        for primitive_name, primitive in self.primitives.items():
            try:
                # Get parameter combinations for this primitive and grid
                param_combinations = primitive.get_parameter_combinations(grid)
                
                for params in param_combinations:
                    try:
                        # Validate parameters before creating operation
                        if primitive.validate_params(**params):
                            operation = DSLOperation(primitive_name, params)
                            operations.append(operation)
                        else:
                            logger.debug(f"Invalid parameters for {primitive_name}: {params}")
                    except Exception as e:
                        logger.debug(f"Failed to validate parameters for {primitive_name}: {e}")
                        continue
                    
            except Exception as e:
                logger.debug(f"Failed to generate operations for {primitive_name}: {e}")
                continue
        
        return operations
    
    def apply_operation(self, grid: np.ndarray, operation: DSLOperation) -> np.ndarray:
        """Apply a single DSL operation to a grid.
        
        Args:
            grid: Input grid
            operation: Operation to apply
            
        Returns:
            Transformed grid
        """
        if operation.primitive_name not in self.primitives:
            raise ValueError(f"Unknown primitive: {operation.primitive_name}")
        
        primitive = self.primitives[operation.primitive_name]
        
        try:
            return primitive(grid, **operation.parameters)
        except Exception as e:
            logger.warning(f"Failed to apply operation {operation}: {e}")
            return grid.copy()  # Return unchanged grid on failure
    
    def create_empty_program(self) -> DSLProgram:
        """Create an empty DSL program."""
        return DSLProgram([], self.max_program_length)
    
    def get_primitive_names(self) -> List[str]:
        """Get list of available primitive names."""
        return list(self.primitives.keys())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        avg_time = (self.total_execution_time / self.execution_count 
                   if self.execution_count > 0 else 0.0)
        
        return {
            'total_executions': self.execution_count,
            'total_time': self.total_execution_time,
            'average_time': avg_time,
            'average_time_us': avg_time * 1000000
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    def enumerate_programs(self, max_length: Optional[int] = None) -> List[DSLProgram]:
        """Enumerate all possible programs up to a given length.
        
        Args:
            max_length: Maximum program length (uses engine default if None)
            
        Returns:
            List of all possible programs
        """
        if max_length is None:
            max_length = self.max_program_length
        
        programs = []
        
        # Add empty program
        programs.append(self.create_empty_program())
        
        # Generate programs of each length
        for length in range(1, max_length + 1):
            programs.extend(self._generate_programs_of_length(length))
        
        return programs
    
    def _generate_programs_of_length(self, length: int) -> List[DSLProgram]:
        """Generate all programs of a specific length.
        
        This is a simplified version - in practice, we'd need more sophisticated
        parameter generation for primitives that require parameters.
        """
        if length == 0:
            return [self.create_empty_program()]
        
        programs = []
        
        # For now, generate programs with parameter-free primitives
        parameter_free_primitives = ['Rotate90', 'Rotate180', 'ReflectH', 'ReflectV']
        
        if length == 1:
            # Single operation programs
            for prim_name in parameter_free_primitives:
                operation = DSLOperation(prim_name, {})
                program = DSLProgram([operation], self.max_program_length)
                programs.append(program)
        else:
            # Multi-operation programs (simplified generation)
            # In practice, this would be more sophisticated
            shorter_programs = self._generate_programs_of_length(length - 1)
            
            for base_program in shorter_programs:
                for prim_name in parameter_free_primitives:
                    try:
                        operation = DSLOperation(prim_name, {})
                        new_program = base_program.append(operation)
                        programs.append(new_program)
                    except ValueError:
                        # Program too long
                        continue
        
        return programs


def create_dsl_engine(max_program_length: int = 4) -> DSLEngine:
    """Factory function to create a DSL engine.
    
    Args:
        max_program_length: Maximum program length
        
    Returns:
        Configured DSLEngine instance
    """
    return DSLEngine(max_program_length=max_program_length)