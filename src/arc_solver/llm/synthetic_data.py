"""Synthetic data generation for LLM training.

This module generates synthetic ARC-like tasks and their corresponding
DSL programs for training the LLM proposer.
"""

import numpy as np
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation, DSLEngine
from arc_solver.perception.blob_labeling import create_blob_labeler

logger = logging.getLogger(__name__)


@dataclass
class SyntheticTask:
    """A synthetic ARC-like task with known solution."""
    
    input_grid: np.ndarray
    target_grid: np.ndarray
    program: DSLProgram
    task_type: str
    difficulty: str


class SyntheticDataGenerator:
    """Generator for synthetic ARC-like training data."""
    
    def __init__(self, 
                 grid_sizes: List[Tuple[int, int]] = None,
                 colors: List[int] = None,
                 max_program_length: int = 4):
        """Initialize synthetic data generator.
        
        Args:
            grid_sizes: List of (height, width) tuples for grid sizes
            colors: List of available colors (0-9)
            max_program_length: Maximum program length
        """
        self.grid_sizes = grid_sizes or [(3, 3), (4, 4), (5, 5), (6, 6)]
        self.colors = colors or list(range(10))  # ARC uses colors 0-9
        self.max_program_length = max_program_length
        
        self.dsl_engine = DSLEngine(max_program_length=max_program_length, max_execution_time=0.01, adaptive_length_limits=True)
        self.blob_labeler = create_blob_labeler(use_gpu=False)
        
        # Task type generators
        self.task_generators = {
            'rotation': self._generate_rotation_task,
            'reflection': self._generate_reflection_task,
            'color_mapping': self._generate_color_mapping_task,
            'cropping': self._generate_cropping_task,
            'painting': self._generate_painting_task,
            'composite': self._generate_composite_task
        }
        
        logger.info(f"Synthetic data generator initialized with {len(self.task_generators)} task types")
    
    def generate_training_set(self, 
                            num_tasks: int = 300,
                            task_types: Optional[List[str]] = None,
                            difficulty_distribution: Optional[Dict[str, float]] = None) -> List[SyntheticTask]:
        """Generate a set of synthetic training tasks.
        
        Args:
            num_tasks: Number of tasks to generate
            task_types: List of task types to include (None for all)
            difficulty_distribution: Distribution of difficulty levels
            
        Returns:
            List of synthetic tasks
        """
        if task_types is None:
            task_types = list(self.task_generators.keys())
        
        if difficulty_distribution is None:
            difficulty_distribution = {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
        
        tasks = []
        
        for i in range(num_tasks):
            try:
                # Select task type and difficulty
                task_type = random.choice(task_types)
                difficulty = np.random.choice(
                    list(difficulty_distribution.keys()),
                    p=list(difficulty_distribution.values())
                )
                
                # Generate task
                task = self.task_generators[task_type](difficulty)
                if task is not None:
                    tasks.append(task)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated {i + 1}/{num_tasks} synthetic tasks")
                    
            except Exception as e:
                logger.warning(f"Failed to generate task {i}: {e}")
                continue
        
        logger.info(f"Generated {len(tasks)} synthetic tasks successfully")
        return tasks
    
    def _generate_rotation_task(self, difficulty: str) -> Optional[SyntheticTask]:
        """Generate a rotation-based task.
        
        Args:
            difficulty: Task difficulty level
            
        Returns:
            SyntheticTask or None if generation failed
        """
        try:
            # Create input grid
            input_grid = self._create_random_grid(difficulty)
            
            # Choose rotation
            rotations = {
                'easy': [1],  # 90 degrees only
                'medium': [1, 2, 3],  # 90, 180, 270 degrees
                'hard': [1, 2, 3]  # Same as medium for rotations
            }
            
            rotation_steps = random.choice(rotations[difficulty])
            
            # Apply rotation
            target_grid = np.rot90(input_grid, rotation_steps)
            
            # Create program
            if rotation_steps == 1:
                program = DSLProgram([DSLOperation('Rotate90', {})])
            elif rotation_steps == 2:
                program = DSLProgram([DSLOperation('Rotate180', {})])
            else:  # rotation_steps == 3
                # 270 degrees = 3 * 90 degrees
                program = DSLProgram([
                    DSLOperation('Rotate90', {}),
                    DSLOperation('Rotate90', {}),
                    DSLOperation('Rotate90', {})
                ])
            
            return SyntheticTask(
                input_grid=input_grid,
                target_grid=target_grid,
                program=program,
                task_type='rotation',
                difficulty=difficulty
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate rotation task: {e}")
            return None
    
    def _generate_reflection_task(self, difficulty: str) -> Optional[SyntheticTask]:
        """Generate a reflection-based task.
        
        Args:
            difficulty: Task difficulty level
            
        Returns:
            SyntheticTask or None if generation failed
        """
        try:
            # Create input grid
            input_grid = self._create_random_grid(difficulty)
            
            # Choose reflection type
            reflection_types = ['horizontal', 'vertical']
            reflection_type = random.choice(reflection_types)
            
            # Apply reflection
            if reflection_type == 'horizontal':
                target_grid = np.fliplr(input_grid)
                program = DSLProgram([DSLOperation('ReflectH', {})])
            else:  # vertical
                target_grid = np.flipud(input_grid)
                program = DSLProgram([DSLOperation('ReflectV', {})])
            
            return SyntheticTask(
                input_grid=input_grid,
                target_grid=target_grid,
                program=program,
                task_type='reflection',
                difficulty=difficulty
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate reflection task: {e}")
            return None
    
    def _generate_color_mapping_task(self, difficulty: str) -> Optional[SyntheticTask]:
        """Generate a color mapping task.
        
        Args:
            difficulty: Task difficulty level
            
        Returns:
            SyntheticTask or None if generation failed
        """
        try:
            # Create input grid
            input_grid = self._create_random_grid(difficulty)
            
            # Get unique colors in input
            unique_colors = np.unique(input_grid).tolist()
            
            if len(unique_colors) < 2:
                return None  # Need at least 2 colors for mapping
            
            # Create color mapping based on difficulty
            if difficulty == 'easy':
                # Simple swap of 2 colors
                if len(unique_colors) >= 2:
                    color1, color2 = random.sample(unique_colors, 2)
                    color_mapping = {color1: color2, color2: color1}
                else:
                    return None
            else:
                # More complex mappings
                num_colors_to_map = min(len(unique_colors), 3 if difficulty == 'medium' else 4)
                colors_to_map = random.sample(unique_colors, num_colors_to_map)
                
                # Create random permutation
                new_colors = colors_to_map.copy()
                random.shuffle(new_colors)
                color_mapping = dict(zip(colors_to_map, new_colors))
            
            # Apply color mapping
            target_grid = input_grid.copy()
            for old_color, new_color in color_mapping.items():
                target_grid[input_grid == old_color] = new_color
            
            # Create program
            program = DSLProgram([DSLOperation('MapColors', {'mapping': color_mapping})])
            
            return SyntheticTask(
                input_grid=input_grid,
                target_grid=target_grid,
                program=program,
                task_type='color_mapping',
                difficulty=difficulty
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate color mapping task: {e}")
            return None
    
    def _generate_cropping_task(self, difficulty: str) -> Optional[SyntheticTask]:
        """Generate a cropping task.
        
        Args:
            difficulty: Task difficulty level
            
        Returns:
            SyntheticTask or None if generation failed
        """
        try:
            # Create larger input grid for cropping
            size_multiplier = {'easy': 1.5, 'medium': 2.0, 'hard': 2.5}
            base_size = random.choice(self.grid_sizes)
            input_size = (
                int(base_size[0] * size_multiplier[difficulty]),
                int(base_size[1] * size_multiplier[difficulty])
            )
            
            input_grid = self._create_specific_grid(input_size, difficulty)
            
            # Choose crop region
            h, w = input_grid.shape
            
            if difficulty == 'easy':
                # Crop to center region
                crop_h, crop_w = h // 2, w // 2
                r1, c1 = h // 4, w // 4
                r2, c2 = r1 + crop_h, c1 + crop_w
            else:
                # Random crop region
                crop_h = random.randint(2, h - 1)
                crop_w = random.randint(2, w - 1)
                r1 = random.randint(0, h - crop_h)
                c1 = random.randint(0, w - crop_w)
                r2, c2 = r1 + crop_h, c1 + crop_w
            
            # Apply crop
            target_grid = input_grid[r1:r2, c1:c2].copy()
            
            # Create program
            program = DSLProgram([DSLOperation('Crop', {'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2})])
            
            return SyntheticTask(
                input_grid=input_grid,
                target_grid=target_grid,
                program=program,
                task_type='cropping',
                difficulty=difficulty
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate cropping task: {e}")
            return None
    
    def _generate_painting_task(self, difficulty: str) -> Optional[SyntheticTask]:
        """Generate a painting task.
        
        Args:
            difficulty: Task difficulty level
            
        Returns:
            SyntheticTask or None if generation failed
        """
        try:
            # Create input grid
            input_grid = self._create_random_grid(difficulty)
            target_grid = input_grid.copy()
            
            # Choose number of paint operations based on difficulty
            num_paints = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
            
            operations = []
            h, w = input_grid.shape
            
            for _ in range(num_paints):
                # Choose random position and color
                x, y = random.randint(0, h - 1), random.randint(0, w - 1)
                new_color = random.choice(self.colors)
                
                # Apply paint
                target_grid[x, y] = new_color
                
                # Add operation
                operations.append(DSLOperation('Paint', {'x': x, 'y': y, 'c': new_color}))
            
            program = DSLProgram(operations)
            
            return SyntheticTask(
                input_grid=input_grid,
                target_grid=target_grid,
                program=program,
                task_type='painting',
                difficulty=difficulty
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate painting task: {e}")
            return None
    
    def _generate_composite_task(self, difficulty: str) -> Optional[SyntheticTask]:
        """Generate a composite task with multiple operations.
        
        Args:
            difficulty: Task difficulty level
            
        Returns:
            SyntheticTask or None if generation failed
        """
        try:
            # Create input grid
            input_grid = self._create_random_grid(difficulty)
            current_grid = input_grid.copy()
            
            # Choose number of operations based on difficulty
            num_operations = {'easy': 2, 'medium': 3, 'hard': 4}[difficulty]
            
            operations = []
            
            for i in range(num_operations):
                # Choose operation type (avoid cropping in composite tasks)
                operation_types = ['rotation', 'reflection', 'color_mapping', 'painting']
                op_type = random.choice(operation_types)
                
                if op_type == 'rotation':
                    rotation_steps = random.choice([1, 2, 3])
                    current_grid = np.rot90(current_grid, rotation_steps)
                    
                    if rotation_steps == 1:
                        operations.append(DSLOperation('Rotate90', {}))
                    elif rotation_steps == 2:
                        operations.append(DSLOperation('Rotate180', {}))
                    else:
                        # For composite tasks, use single Rotate90 operations
                        operations.append(DSLOperation('Rotate90', {}))
                        if len(operations) < num_operations:
                            operations.append(DSLOperation('Rotate90', {}))
                            if len(operations) < num_operations:
                                operations.append(DSLOperation('Rotate90', {}))
                        break  # Avoid exceeding operation limit
                
                elif op_type == 'reflection':
                    if random.choice([True, False]):
                        current_grid = np.fliplr(current_grid)
                        operations.append(DSLOperation('ReflectH', {}))
                    else:
                        current_grid = np.flipud(current_grid)
                        operations.append(DSLOperation('ReflectV', {}))
                
                elif op_type == 'color_mapping' and i == 0:  # Only at beginning
                    unique_colors = np.unique(current_grid).tolist()
                    if len(unique_colors) >= 2:
                        color1, color2 = random.sample(unique_colors, 2)
                        color_mapping = {color1: color2, color2: color1}
                        
                        for old_color, new_color in color_mapping.items():
                            current_grid[current_grid == old_color] = new_color
                        
                        operations.append(DSLOperation('MapColors', {'mapping': color_mapping}))
                
                elif op_type == 'painting':
                    h, w = current_grid.shape
                    x, y = random.randint(0, h - 1), random.randint(0, w - 1)
                    new_color = random.choice(self.colors)
                    
                    current_grid[x, y] = new_color
                    operations.append(DSLOperation('Paint', {'x': x, 'y': y, 'c': new_color}))
                
                # Check if we've reached the operation limit
                if len(operations) >= num_operations:
                    break
            
            if not operations:
                return None
            
            program = DSLProgram(operations[:num_operations])  # Ensure we don't exceed limit
            
            return SyntheticTask(
                input_grid=input_grid,
                target_grid=current_grid,
                program=program,
                task_type='composite',
                difficulty=difficulty
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate composite task: {e}")
            return None
    
    def _create_random_grid(self, difficulty: str) -> np.ndarray:
        """Create a random grid based on difficulty.
        
        Args:
            difficulty: Difficulty level
            
        Returns:
            Random grid
        """
        # Choose grid size based on difficulty
        if difficulty == 'easy':
            candidates = self.grid_sizes[:2]
        elif difficulty == 'medium':
            candidates = self.grid_sizes[1:3]
        else:  # hard
            candidates = self.grid_sizes[2:] or self.grid_sizes  # Fallback if empty

        size = random.choice(candidates)
        
        return self._create_specific_grid(size, difficulty)
    
    def _create_specific_grid(self, size: Tuple[int, int], difficulty: str) -> np.ndarray:
        """Create a grid with specific size and complexity.
        
        Args:
            size: (height, width) of the grid
            difficulty: Difficulty level
            
        Returns:
            Generated grid
        """
        h, w = size
        
        # Choose number of colors based on difficulty
        num_colors = {'easy': 2, 'medium': 3, 'hard': 4}[difficulty]
        available_colors = random.sample(self.colors, num_colors)
        
        # Create grid with patterns
        grid = np.zeros((h, w), dtype=np.int32)
        
        if difficulty == 'easy':
            # Simple patterns
            for i in range(h):
                for j in range(w):
                    grid[i, j] = available_colors[(i + j) % len(available_colors)]
        
        elif difficulty == 'medium':
            # More complex patterns
            # Create some structure with random elements
            for i in range(h):
                for j in range(w):
                    if (i + j) % 2 == 0:
                        grid[i, j] = available_colors[0]
                    else:
                        grid[i, j] = random.choice(available_colors[1:])
        
        else:  # hard
            # Complex patterns with noise
            # Create structured regions with random noise
            for i in range(h):
                for j in range(w):
                    if i < h // 2 and j < w // 2:
                        grid[i, j] = available_colors[0]
                    elif i < h // 2:
                        grid[i, j] = available_colors[1]
                    elif j < w // 2:
                        grid[i, j] = available_colors[2] if len(available_colors) > 2 else available_colors[1]
                    else:
                        grid[i, j] = random.choice(available_colors)
            
            # Add some random noise
            noise_positions = random.sample(
                [(i, j) for i in range(h) for j in range(w)],
                min(3, h * w // 4)
            )
            for i, j in noise_positions:
                grid[i, j] = random.choice(available_colors)
        
        return grid
    
    def export_training_data(self, 
                           tasks: List[SyntheticTask],
                           output_file: str,
                           format: str = 'json') -> None:
        """Export synthetic tasks to file for training.
        
        Args:
            tasks: List of synthetic tasks
            output_file: Output file path
            format: Export format ('json' or 'jsonl')
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Export as single JSON file
            data = []
            for task in tasks:
                task_data = {
                    'input_grid': task.input_grid.tolist(),
                    'target_grid': task.target_grid.tolist(),
                    'program': task.program.to_dict(),
                    'task_type': task.task_type,
                    'difficulty': task.difficulty
                }
                data.append(task_data)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'jsonl':
            # Export as JSONL file (one JSON object per line)
            with open(output_path, 'w') as f:
                for task in tasks:
                    task_data = {
                        'input_grid': task.input_grid.tolist(),
                        'target_grid': task.target_grid.tolist(),
                        'program': task.program.to_dict(),
                        'task_type': task.task_type,
                        'difficulty': task.difficulty
                    }
                    f.write(json.dumps(task_data) + '\n')
        
        logger.info(f"Exported {len(tasks)} synthetic tasks to {output_path}")


def create_synthetic_data_generator(grid_sizes: List[Tuple[int, int]] = None,
                                  max_program_length: int = 4) -> SyntheticDataGenerator:
    """Factory function to create synthetic data generator.
    
    Args:
        grid_sizes: List of grid sizes to use
        max_program_length: Maximum program length
        
    Returns:
        Configured SyntheticDataGenerator
    """
    return SyntheticDataGenerator(
        grid_sizes=grid_sizes,
        max_program_length=max_program_length
    )