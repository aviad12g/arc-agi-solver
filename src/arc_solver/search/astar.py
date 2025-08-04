"""A* search algorithm for ARC-AGI solver.

This module implements the A* search algorithm with beam search pruning
for finding optimal DSL programs that transform input grids to target grids.
"""

import heapq
import time
import logging
from typing import Optional, List, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from arc_solver.search.heuristics import HeuristicSystem, create_heuristic_system
from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation
from arc_solver.core.data_models import GridState

logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    """Node in the A* search tree."""
    grid: np.ndarray
    program: DSLProgram
    cost: float  # g(n) - actual cost from start
    heuristic: float  # h(n) - heuristic estimate to goal
    parent: Optional['SearchNode'] = None
    action: Optional[DSLOperation] = None
    depth: int = 0
    
    def __post_init__(self):
        """Compute grid hash for efficient comparison."""
        self.grid_hash = hash(self.grid.tobytes())
    
    @property
    def f_score(self) -> float:
        """Total estimated cost f(n) = g(n) + h(n)."""
        return self.cost + self.heuristic
    
    def __lt__(self, other: 'SearchNode') -> bool:
        """Comparison for priority queue (lower f_score has higher priority)."""
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        # Tie-breaking: prefer lower cost (shorter programs)
        if self.cost != other.cost:
            return self.cost < other.cost
        # Final tie-breaking: prefer lower depth
        return self.depth < other.depth
    
    def __eq__(self, other: 'SearchNode') -> bool:
        """Equality based on grid state."""
        return np.array_equal(self.grid, other.grid)
    
    def __hash__(self) -> int:
        """Hash based on grid state."""
        return self.grid_hash
    
    def get_program_sequence(self) -> List[DSLOperation]:
        """Get the sequence of operations from root to this node."""
        operations = []
        node = self
        while node.parent is not None:
            if node.action is not None:
                operations.append(node.action)
            node = node.parent
        return list(reversed(operations))


@dataclass 
class SearchResult:
    """Result from A* search."""
    success: bool
    program: Optional[DSLProgram] = None
    final_grid: Optional[np.ndarray] = None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    computation_time: float = 0.0
    max_depth_reached: int = 0
    beam_width_used: int = 0
    termination_reason: str = "unknown"
    heuristic_stats: Optional[Dict[str, Any]] = None


@dataclass
class SearchConfig:
    """Configuration for A* search."""
    max_program_length: int = 4  # K=4 from requirements
    max_nodes_expanded: int = 600  # Target from requirements
    max_computation_time: float = 30.0  # 30 second timeout
    beam_width: int = 64  # Initial beam width
    adaptive_beam: bool = True  # Enable adaptive beam width
    min_beam_width: int = 8  # Minimum beam width
    beam_reduction_factor: float = 0.7  # Factor to reduce beam when needed
    duplicate_detection: bool = True  # Enable duplicate state detection
    early_termination: bool = True  # Enable early termination on exact match


class AStarSearcher:
    """A* search algorithm with beam search pruning."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize A* searcher.
        
        Args:
            config: Search configuration parameters
        """
        self.config = config or SearchConfig()
        self.heuristic_system = create_heuristic_system(use_tier2=True, tier2_threshold=5.0)
        self.dsl_engine = DSLEngine()
        
        # Search statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_depth_reached = 0
        self.beam_width_used = self.config.beam_width
        
        logger.info(f"A* searcher initialized with max_length={self.config.max_program_length}, "
                   f"beam_width={self.config.beam_width}")
    
    def search(self, initial_grid: np.ndarray, target_grid: np.ndarray) -> SearchResult:
        """Search for optimal DSL program to transform initial_grid to target_grid.
        
        Args:
            initial_grid: Starting grid state
            target_grid: Goal grid state
            
        Returns:
            SearchResult with program and statistics
        """
        start_time = time.perf_counter()
        
        # Reset statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_depth_reached = 0
        self.beam_width_used = self.config.beam_width
        
        logger.info(f"Starting A* search: {initial_grid.shape} -> {target_grid.shape}")
        
        try:
            # Check if initial grid already matches target
            if np.array_equal(initial_grid, target_grid):
                computation_time = time.perf_counter() - start_time
                return SearchResult(
                    success=True,
                    program=DSLProgram([]),  # Empty program
                    final_grid=initial_grid.copy(),
                    nodes_expanded=0,
                    nodes_generated=1,
                    computation_time=computation_time,
                    termination_reason="initial_match"
                )
            
            # Initialize search
            initial_heuristic = self.heuristic_system.compute_heuristic(initial_grid, target_grid)
            
            root_node = SearchNode(
                grid=initial_grid.copy(),
                program=DSLProgram([]),
                cost=0.0,
                heuristic=initial_heuristic.value,
                depth=0
            )
            
            # Priority queue for open nodes (min-heap)
            open_queue = [root_node]
            heapq.heapify(open_queue)
            
            # Closed set for duplicate detection
            closed_set: Set[int] = set() if self.config.duplicate_detection else set()
            
            # Best node found so far
            best_node = root_node
            
            while open_queue and self.nodes_expanded < self.config.max_nodes_expanded:
                # Check timeout
                if time.perf_counter() - start_time > self.config.max_computation_time:
                    break
                
                # Beam search pruning: keep only top beam_width nodes
                if len(open_queue) > self.beam_width_used:
                    open_queue = heapq.nsmallest(self.beam_width_used, open_queue)
                    heapq.heapify(open_queue)
                
                # Get best node from open queue
                current_node = heapq.heappop(open_queue)
                self.nodes_expanded += 1
                
                # Update statistics
                self.max_depth_reached = max(self.max_depth_reached, current_node.depth)
                
                # Check for goal state
                if np.array_equal(current_node.grid, target_grid):
                    computation_time = time.perf_counter() - start_time
                    return self._create_success_result(
                        current_node, computation_time, "goal_reached"
                    )
                
                # Skip if already explored (duplicate detection)
                if self.config.duplicate_detection:
                    if current_node.grid_hash in closed_set:
                        continue
                    closed_set.add(current_node.grid_hash)
                
                # Update best node if this has lower heuristic
                if current_node.heuristic < best_node.heuristic:
                    best_node = current_node
                
                # Expand node if within depth limit
                if current_node.depth < self.config.max_program_length:
                    successors = self._expand_node(current_node, target_grid)
                    
                    for successor in successors:
                        # Skip if already in closed set
                        if (self.config.duplicate_detection and 
                            successor.grid_hash in closed_set):
                            continue
                        
                        heapq.heappush(open_queue, successor)
                        self.nodes_generated += 1
                
                # Adaptive beam width reduction if search is taking too long
                if (self.config.adaptive_beam and 
                    self.nodes_expanded > self.config.max_nodes_expanded // 2 and
                    self.beam_width_used > self.config.min_beam_width):
                    
                    new_beam_width = max(
                        self.config.min_beam_width,
                        int(self.beam_width_used * self.config.beam_reduction_factor)
                    )
                    
                    if new_beam_width < self.beam_width_used:
                        self.beam_width_used = new_beam_width
                        logger.info(f"Reduced beam width to {self.beam_width_used}")
            
            # Search completed without finding exact solution
            computation_time = time.perf_counter() - start_time
            
            # Determine termination reason
            if self.nodes_expanded >= self.config.max_nodes_expanded:
                termination_reason = "max_nodes_reached"
            elif time.perf_counter() - start_time > self.config.max_computation_time:
                termination_reason = "timeout"
            elif not open_queue:
                termination_reason = "search_exhausted"
            else:
                termination_reason = "unknown"
            
            # Return best partial result
            return self._create_partial_result(
                best_node, computation_time, termination_reason
            )
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            logger.error(f"A* search failed with error: {e}")
            
            return SearchResult(
                success=False,
                nodes_expanded=self.nodes_expanded,
                nodes_generated=self.nodes_generated,
                computation_time=computation_time,
                termination_reason=f"error: {str(e)}"
            )
    
    def _expand_node(self, node: SearchNode, target_grid: np.ndarray) -> List[SearchNode]:
        """Expand a search node by applying all possible DSL operations.
        
        Args:
            node: Node to expand
            target_grid: Target grid for heuristic computation
            
        Returns:
            List of successor nodes
        """
        successors = []
        
        # Get all available DSL operations
        available_operations = self.dsl_engine.get_available_operations(node.grid)
        
        for operation in available_operations:
            try:
                # Apply operation to get new grid
                new_grid = self.dsl_engine.apply_operation(node.grid, operation)
                
                # Skip if operation had no effect
                if np.array_equal(new_grid, node.grid):
                    continue
                
                # Create new program with added operation
                new_program = DSLProgram(node.program.operations + [operation])
                
                # Compute heuristic for new state
                heuristic_result = self.heuristic_system.compute_heuristic(new_grid, target_grid)
                
                # Create successor node
                successor = SearchNode(
                    grid=new_grid,
                    program=new_program,
                    cost=node.cost + 1.0,  # Uniform cost (each operation costs 1)
                    heuristic=heuristic_result.value,
                    parent=node,
                    action=operation,
                    depth=node.depth + 1
                )
                
                successors.append(successor)
                
            except Exception as e:
                # Log operation failure but continue with other operations
                logger.debug(f"Operation {operation} failed on grid: {e}")
                continue
        
        return successors
    
    def _create_success_result(self, node: SearchNode, computation_time: float, 
                             termination_reason: str) -> SearchResult:
        """Create successful search result."""
        heuristic_stats = self.heuristic_system.get_stats()
        
        return SearchResult(
            success=True,
            program=node.program,
            final_grid=node.grid.copy(),
            nodes_expanded=self.nodes_expanded,
            nodes_generated=self.nodes_generated,
            computation_time=computation_time,
            max_depth_reached=self.max_depth_reached,
            beam_width_used=self.beam_width_used,
            termination_reason=termination_reason,
            heuristic_stats=heuristic_stats
        )
    
    def _create_partial_result(self, best_node: SearchNode, computation_time: float,
                             termination_reason: str) -> SearchResult:
        """Create partial search result when exact solution not found."""
        heuristic_stats = self.heuristic_system.get_stats()
        
        return SearchResult(
            success=False,
            program=best_node.program,
            final_grid=best_node.grid.copy(),
            nodes_expanded=self.nodes_expanded,
            nodes_generated=self.nodes_generated,
            computation_time=computation_time,
            max_depth_reached=self.max_depth_reached,
            beam_width_used=self.beam_width_used,
            termination_reason=termination_reason,
            heuristic_stats=heuristic_stats
        )
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get detailed search statistics."""
        heuristic_stats = self.heuristic_system.get_stats()
        
        return {
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'max_depth_reached': self.max_depth_reached,
            'beam_width_used': self.beam_width_used,
            'heuristic_stats': heuristic_stats,
            'config': {
                'max_program_length': self.config.max_program_length,
                'max_nodes_expanded': self.config.max_nodes_expanded,
                'beam_width': self.config.beam_width,
                'adaptive_beam': self.config.adaptive_beam
            }
        }


def create_astar_searcher(max_program_length: int = 4,
                         max_nodes_expanded: int = 600,
                         beam_width: int = 64,
                         adaptive_beam: bool = True) -> AStarSearcher:
    """Factory function to create A* searcher with custom configuration.
    
    Args:
        max_program_length: Maximum DSL program length (K=4 from requirements)
        max_nodes_expanded: Maximum nodes to expand (≤600 from requirements)
        beam_width: Initial beam width for pruning
        adaptive_beam: Enable adaptive beam width reduction
        
    Returns:
        Configured AStarSearcher instance
    """
    config = SearchConfig(
        max_program_length=max_program_length,
        max_nodes_expanded=max_nodes_expanded,
        beam_width=beam_width,
        adaptive_beam=adaptive_beam
    )
    
    return AStarSearcher(config)