"""A* search algorithm for ARC-AGI solver.

This module implements the A* search algorithm with beam search pruning
for finding optimal DSL programs that transform input grids to target grids.
"""

import heapq
import os
import time
import logging
from typing import Optional, List, Dict, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from arc_solver.search.heuristics import HeuristicSystem, create_heuristic_system
from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation
from arc_solver.core.data_models import GridState
from arc_solver.caching import create_cache_manager
from arc_solver.reasoning.version_space import compute_constraints

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
    # Multi-example tracking
    valid_examples: Optional[Set[int]] = None  # Set of example indices this node is valid for
    example_scores: Optional[Dict[int, float]] = None  # Performance score per example
    
    def __post_init__(self):
        """Compute grid hash for efficient comparison."""
        # Use D4-canonicalized hash for square grids to reduce symmetric duplicates
        self.grid_hash = self._compute_canonical_hash(self.grid)
        if self.valid_examples is None:
            self.valid_examples = set()
        if self.example_scores is None:
            self.example_scores = {}

    def _compute_canonical_hash(self, grid: np.ndarray) -> int:
        """Compute a symmetry-canonical hash for the grid (D4 for square grids).
        For rectangular grids, fall back to direct bytes hash.
        """
        h, w = grid.shape
        if h == w:
            candidates = []
            # Rotations
            candidates.append(grid)
            candidates.append(np.rot90(grid, 1))
            candidates.append(np.rot90(grid, 2))
            candidates.append(np.rot90(grid, 3))
            # Reflections
            candidates.append(np.fliplr(grid))
            candidates.append(np.flipud(grid))
            # Diagonal reflections (transpose and anti-diagonal)
            candidates.append(grid.T)
            candidates.append(np.rot90(grid.T, 2))
            # Pick lexicographically smallest bytes as canonical
            canonical_bytes = min(c.tobytes() for c in candidates)
            return hash(canonical_bytes)
        # Non-square: use direct bytes
        return hash(grid.tobytes())
    
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
    
    def is_valid_for_example(self, example_idx: int) -> bool:
        """Check if this node is valid for a specific training example."""
        return example_idx in self.valid_examples
    
    def add_example_validation(self, example_idx: int, score: float) -> None:
        """Mark this node as valid for a specific example with a score."""
        self.valid_examples.add(example_idx)
        self.example_scores[example_idx] = score
    
    def remove_example_validation(self, example_idx: int) -> None:
        """Mark this node as invalid for a specific example."""
        self.valid_examples.discard(example_idx)
        self.example_scores.pop(example_idx, None)
    
    def get_example_score(self, example_idx: int) -> float:
        """Get the performance score for a specific example."""
        return self.example_scores.get(example_idx, 0.0)
    
    def get_average_example_score(self) -> float:
        """Get the average performance score across all valid examples."""
        if not self.example_scores:
            return 0.0
        return sum(self.example_scores.values()) / len(self.example_scores)
    
    def is_valid_for_all_examples(self, total_examples: int) -> bool:
        """Check if this node is valid for all training examples."""
        return len(self.valid_examples) == total_examples


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
    # Multi-example validation info
    candidates_generated: int = 0
    examples_validated: int = 0
    validation_success_rate: float = 0.0


@dataclass
class SearchStatistics:
    """Detailed search statistics for optimization."""
    nodes_expanded: int = 0
    nodes_generated: int = 0
    nodes_pruned: int = 0
    duplicate_states: int = 0
    max_depth_reached: int = 0
    beam_reductions: int = 0
    heuristic_computations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_branching_factor: float = 0.0
    search_efficiency: float = 0.0  # nodes_expanded / nodes_generated
    
    def update_branching_factor(self, total_successors: int) -> None:
        """Update average branching factor."""
        if self.nodes_expanded > 0:
            self.average_branching_factor = (
                (self.average_branching_factor * (self.nodes_expanded - 1) + total_successors) 
                / self.nodes_expanded
            )
        else:
            self.average_branching_factor = total_successors
    
    def compute_efficiency(self) -> None:
        """Compute search efficiency metrics."""
        if self.nodes_generated > 0:
            self.search_efficiency = self.nodes_expanded / self.nodes_generated
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'nodes_pruned': self.nodes_pruned,
            'duplicate_states': self.duplicate_states,
            'max_depth_reached': self.max_depth_reached,
            'beam_reductions': self.beam_reductions,
            'heuristic_computations': self.heuristic_computations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'average_branching_factor': self.average_branching_factor,
            'search_efficiency': self.search_efficiency
        }


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
    # New optimization parameters
    parallel_expansion: bool = False  # Enable parallel node expansion
    max_threads: int = 4  # Maximum threads for parallel expansion
    incremental_search: bool = True  # Enable incremental search optimizations
    adaptive_beam_quality_threshold: float = 0.8  # Threshold for beam quality adaptation
    statistics_tracking: bool = True  # Enable detailed statistics tracking


class AStarSearcher:
    """A* search algorithm with beam search pruning and optimizations."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize A* searcher.
        
        Args:
            config: Search configuration parameters
        """
        self.config = config or SearchConfig()
        # Initialize heuristic system with optional gating from config
        try:

            # Defaults
            use_tier2 = True
            tier2_threshold = 5.0
            max_blob = 10**9
            min_depth = 0
            max_calls = 10**9
            dedupe = False
            greedy_large = False

            # Prefer Hydra configuration if available
            from arc_solver.config import get_config as _get_cfg
            cfg = _get_cfg()
            if cfg is not None and 'search' in cfg and 'heuristics' in cfg.search:
                hcfg = cfg.search.heuristics
                try:
                    if 'use_tier2' in hcfg:
                        use_tier2 = bool(hcfg.use_tier2)
                except Exception:
                    pass
                try:
                    if 'tier2_threshold' in hcfg:
                        tier2_threshold = float(hcfg.tier2_threshold)
                except Exception:
                    pass
                # Gating group
                try:
                    gating = hcfg.get('gating', {})
                    if gating and getattr(gating, 'enable', False):
                        if 'max_blob_for_tier2' in gating:
                            max_blob = int(gating.max_blob_for_tier2)
                        if 'min_depth_for_tier2' in gating:
                            min_depth = int(gating.min_depth_for_tier2)
                        if 'max_tier2_calls' in gating:
                            max_calls = int(gating.max_tier2_calls)
                        if 'dedupe_pairs' in gating:
                            dedupe = bool(gating.dedupe_pairs)
                        if 'use_greedy_fallback_when_large' in gating:
                            greedy_large = bool(gating.use_greedy_fallback_when_large)
                except Exception:
                    pass

            # Environment variables can override for quick experiments
            try:
                if 'ARC_HEUR_USE_T2' in os.environ:
                    use_tier2 = os.environ['ARC_HEUR_USE_T2'].lower() == 'true'
                if 'ARC_HEUR_T2_THRESHOLD' in os.environ:
                    tier2_threshold = float(os.environ['ARC_HEUR_T2_THRESHOLD'])
                if 'ARC_HEUR_T2_MAX_BLOB' in os.environ:
                    max_blob = int(os.environ['ARC_HEUR_T2_MAX_BLOB'])
                if 'ARC_HEUR_T2_MIN_DEPTH' in os.environ:
                    min_depth = int(os.environ['ARC_HEUR_T2_MIN_DEPTH'])
                if 'ARC_HEUR_T2_MAX_CALLS' in os.environ:
                    max_calls = int(os.environ['ARC_HEUR_T2_MAX_CALLS'])
                if 'ARC_HEUR_T2_DEDUPE' in os.environ:
                    dedupe = os.environ['ARC_HEUR_T2_DEDUPE'].lower() == 'true'
                if 'ARC_HEUR_T2_GREEDY_LARGE' in os.environ:
                    greedy_large = os.environ['ARC_HEUR_T2_GREEDY_LARGE'].lower() == 'true'
            except Exception:
                pass
        except Exception:
            # Fall back to safe defaults
            use_tier2 = True
            tier2_threshold = 5.0
            max_blob = 10**9
            min_depth = 0
            max_calls = 10**9
            dedupe = False
            greedy_large = False

        self.heuristic_system = create_heuristic_system(
            use_tier2=use_tier2,
            tier2_threshold=tier2_threshold,
            max_blob_for_tier2=max_blob,
            min_depth_for_tier2=min_depth,
            max_tier2_calls=max_calls,
            dedupe_pairs=dedupe,
            use_greedy_fallback_when_large=greedy_large,
        )
        # Ensure DSLEngine uses the configured execution cap (default to 10ms if not provided)
        try:
            from arc_solver.config import get_config as _get_cfg
            cfg = _get_cfg()
            if cfg is not None and 'reasoning' in cfg and 'dsl_engine' in cfg.reasoning:
                dsl_cfg = cfg.reasoning.dsl_engine
                max_exec_time = float(dsl_cfg.get('max_execution_time', 0.01))
                max_prog_len = int(dsl_cfg.get('max_program_length', 4))
            else:
                max_exec_time = 0.01
                max_prog_len = 4
        except Exception:
            max_exec_time = 0.01
            max_prog_len = 4
        self.dsl_engine = DSLEngine(max_program_length=max_prog_len, max_execution_time=max_exec_time, adaptive_length_limits=True)
        
        # Enhanced search statistics
        self.statistics = SearchStatistics()
        self.beam_width_used = self.config.beam_width
        # Public counters for backward-compatibility with tests
        self.nodes_expanded: int = 0
        self.nodes_generated: int = 0
        self.max_depth_reached: int = 0
        
        # Enhanced caching system
        self.state_cache: Dict[int, SearchNode] = {}
        self.partial_results: Dict[str, Any] = {}
        self.program_validation_cache: Dict[str, Dict[str, Any]] = {}
        self.pattern_cache: Dict[str, List[DSLOperation]] = {}
        # External cache manager for sharing across components
        try:
            self.cache_manager = create_cache_manager()
        except Exception:
            self.cache_manager = None
        
        # Cache performance metrics
        self.cache_stats = {
            'state_cache_hits': 0,
            'state_cache_misses': 0,
            'program_cache_hits': 0,
            'program_cache_misses': 0,
            'pattern_cache_hits': 0,
            'pattern_cache_misses': 0,
            'cache_size': 0,
            'cache_cleanup_count': 0
        }
        
        # Cache limits for automatic cleanup
        self.max_cache_size = 10000
        self.cache_cleanup_threshold = 0.8
        
        # Parallel processing setup
        if self.config.parallel_expansion:
            import concurrent.futures
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_threads
            )
        else:
            self.thread_pool = None
        
        logger.info(f"A* searcher initialized with max_length={self.config.max_program_length}, "
                   f"beam_width={self.config.beam_width}, parallel={self.config.parallel_expansion}")
    
    def __del__(self):
        """Cleanup thread pool if used."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
    
    def _expand_node_parallel(self, node: SearchNode, target_grid: np.ndarray, deadline: float) -> List[SearchNode]:
        """Expand node using parallel processing.
        
        Args:
            node: Node to expand
            target_grid: Target grid for heuristic computation
            
        Returns:
            List of successor nodes
        """
        if not self.config.parallel_expansion or not self.thread_pool:
            return self._expand_node(node, target_grid, deadline)
        
        # Get available operations
        operations = self.dsl_engine.get_available_operations(node.grid)
        
        if not operations:
            return []
        
        # Split operations into chunks for parallel processing
        chunk_size = max(1, len(operations) // self.config.max_threads)
        operation_chunks = [
            operations[i:i + chunk_size] 
            for i in range(0, len(operations), chunk_size)
        ]
        
        # Process chunks in parallel
        futures = []
        for chunk in operation_chunks:
            future = self.thread_pool.submit(
                self._process_operation_chunk, node, chunk, target_grid, deadline
            )
            futures.append(future)
        
        # Collect results
        successors = []
        for future in futures:
            try:
                chunk_successors = future.result(timeout=1.0)  # 1 second timeout per chunk
                successors.extend(chunk_successors)
            except Exception as e:
                logger.warning(f"Parallel expansion chunk failed: {e}")
        
        return successors
    
    def _process_operation_chunk(self, node: SearchNode, operations: List[DSLOperation], 
                               target_grid: np.ndarray, deadline: float) -> List[SearchNode]:
        """Process a chunk of operations for parallel expansion.
        
        Args:
            node: Parent node
            operations: Operations to process
            target_grid: Target grid for heuristic computation
            
        Returns:
            List of successor nodes from this chunk
        """
        successors = []
        
        for operation in operations:
            if time.perf_counter() > deadline:
                break
            try:
                # Apply operation
                new_grid = self.dsl_engine.apply_operation(node.grid, operation)

                # Create new program
                new_program = node.program.append(operation)

                # Optional: conservative canonicalization of programs for caches
                try:
                    # Read canonicalization flag from Hydra config
                    canonicalization_enabled = True  # Default to enabled
                    saturation_enabled = False
                    validate_smt = False
                    try:
                        from arc_solver.config import get_config as _get_cfg
                        cfg = _get_cfg()
                        if cfg is not None and 'search' in cfg and 'advanced' in cfg.search:
                            advanced_cfg = cfg.search.advanced
                            if 'canonicalization' in advanced_cfg:
                                ccfg = advanced_cfg.canonicalization
                                canonicalization_enabled = bool(ccfg.get('enabled', True))
                                saturation_enabled = bool(ccfg.get('saturation_enabled', False))
                                validate_smt = bool(ccfg.get('validate_smt', False))
                    except Exception:
                        pass

                    if canonicalization_enabled:
                        if saturation_enabled:
                            from arc_solver.reasoning.egraph import saturate_program
                            new_program = saturate_program(new_program, max_iters=5, validate_smt=validate_smt)
                        else:
                            from arc_solver.reasoning.rewrite_rules import canonicalize_program
                            new_program = canonicalize_program(new_program)
                except Exception:
                    pass

                # Compute heuristic
                heuristic_result = self.heuristic_system.compute_heuristic(new_grid, target_grid)
                self.statistics.heuristic_computations += 1

                # Create successor node
                successor = SearchNode(
                    grid=new_grid,
                    program=new_program,
                    cost=node.cost + 1.0,  # Unit cost per operation
                    heuristic=heuristic_result.value,
                    parent=node,
                    action=operation,
                    depth=node.depth + 1
                )

                # Optional UNSAT cache check (configured via Hydra)
                try:
                    # Read UNSAT cache flag from Hydra config
                    unsat_cache_enabled = False  # Default to disabled for safety
                    try:
                        from arc_solver.config import get_config as _get_cfg
                        cfg = _get_cfg()
                        if cfg is not None and 'search' in cfg and 'advanced' in cfg.search:
                            advanced_cfg = cfg.search.advanced
                            if 'unsat_cache' in advanced_cfg and 'enabled' in advanced_cfg.unsat_cache:
                                unsat_cache_enabled = bool(advanced_cfg.unsat_cache.enabled)
                    except Exception:
                        pass

                    if unsat_cache_enabled:
                        from arc_solver.search.unsat_cache import UNSATCache, make_signature
                        if not hasattr(self, '_unsat_cache'):
                            self._unsat_cache = UNSATCache()

                            # Load pre-computed signatures if configured
                            try:
                                from arc_solver.config import get_config as _get_cfg
                                cfg = _get_cfg()
                                if cfg is not None and 'search' in cfg and 'advanced' in cfg.search:
                                    advanced_cfg = cfg.search.advanced
                                    if ('unsat_cache' in advanced_cfg and
                                        'signatures_file' in advanced_cfg.unsat_cache and
                                        advanced_cfg.unsat_cache.signatures_file):
                                        signatures_file = str(advanced_cfg.unsat_cache.signatures_file)
                                        loaded_count = self._unsat_cache.load_from_jsonl(signatures_file)
                                        if loaded_count > 0:
                                            logger.info(f"Loaded {loaded_count} pre-computed UNSAT signatures")
                            except Exception as e:
                                logger.warning(f"Failed to load UNSAT signatures: {e}")

                        sig = make_signature(new_grid, target_grid)
                        if self._unsat_cache.is_unsat(sig):
                            # Skip known-impossible region
                            continue
                except Exception:
                    pass

                # Optional dead-end predictor gating (configured via Hydra)
                try:
                    deadend_enabled = False
                    try:
                        from arc_solver.config import get_config as _get_cfg
                        cfg = _get_cfg()
                        if cfg is not None and 'search' in cfg and 'advanced' in cfg.search:
                            advanced_cfg = cfg.search.advanced
                            if 'deadend_predictor' in advanced_cfg and 'enabled' in advanced_cfg.deadend_predictor:
                                deadend_enabled = bool(advanced_cfg.deadend_predictor.enabled)
                    except Exception:
                        pass

                    if deadend_enabled:
                        from arc_solver.search.dead_end_predictor import DeadEndPredictor
                        if not hasattr(self, '_deadend_pred'):
                            self._deadend_pred = DeadEndPredictor()
                        if self._deadend_pred.should_gate(new_grid, target_grid):
                            continue
                except Exception:
                    pass

                successors.append(successor)

            except Exception as e:
                logger.debug(f"Failed to apply operation {operation}: {e}")
                continue
        
        return successors
    
    def _compute_beam_quality(self, open_queue: List[SearchNode]) -> float:
        """Compute quality metric for current beam.
        
        Args:
            open_queue: Current open queue
            
        Returns:
            Quality score (0.0 to 1.0, higher is better)
        """
        if not open_queue:
            return 0.0
        
        # Compute diversity of heuristic values
        heuristic_values = [node.heuristic for node in open_queue]
        
        if len(set(heuristic_values)) <= 1:
            return 0.5  # Low diversity
        
        # Compute coefficient of variation (std/mean) as diversity measure
        mean_h = np.mean(heuristic_values)
        std_h = np.std(heuristic_values)
        
        if mean_h == 0:
            return 0.5
        
        diversity = min(1.0, std_h / mean_h)
        
        # Compute progress metric (lower heuristic values are better)
        min_h = min(heuristic_values)
        progress = max(0.0, 1.0 - min_h / 10.0)  # Normalize assuming max heuristic ~10
        
        # Combine diversity and progress
        quality = 0.6 * progress + 0.4 * diversity
        
        return min(1.0, max(0.0, quality))
    
    def _adaptive_beam_scheduling(self, open_queue: List[SearchNode]) -> int:
        """Compute adaptive beam width based on search quality.
        
        Args:
            open_queue: Current open queue
            
        Returns:
            Recommended beam width
        """
        if not self.config.adaptive_beam:
            return self.beam_width_used
        
        # Compute beam quality
        quality = self._compute_beam_quality(open_queue)
        
        # Adjust beam width based on quality
        if quality > self.config.adaptive_beam_quality_threshold:
            # High quality - can afford larger beam
            new_width = min(
                self.config.beam_width,
                int(self.beam_width_used * 1.2)
            )
        else:
            # Low quality - reduce beam to focus search
            new_width = max(
                self.config.min_beam_width,
                int(self.beam_width_used * 0.9)
            )
        
        return new_width
    
    def _update_statistics(self, node: SearchNode, successors: List[SearchNode]) -> None:
        """Update search statistics.
        
        Args:
            node: Expanded node
            successors: Generated successor nodes
        """
        if not self.config.statistics_tracking:
            return
        
        self.statistics.nodes_expanded += 1
        self.statistics.nodes_generated += len(successors)
        self.statistics.max_depth_reached = max(
            self.statistics.max_depth_reached, node.depth
        )
        self.statistics.update_branching_factor(len(successors))
    
    def _check_incremental_cache(self, node: SearchNode) -> Optional[SearchNode]:
        """Check incremental search cache for previously computed results.
        
        Args:
            node: Node to check
            
        Returns:
            Cached node if found, None otherwise
        """
        if not self.config.incremental_search:
            return None
        
        cache_key = node.grid_hash
        if cache_key in self.state_cache:
            cached_node = self.state_cache[cache_key]
            self.statistics.cache_hits += 1
            return cached_node
        
        self.statistics.cache_misses += 1
        return None
    
    def _update_incremental_cache(self, node: SearchNode) -> None:
        """Update incremental search cache.
        
        Args:
            node: Node to cache
        """
        if not self.config.incremental_search:
            return
        
        cache_key = node.grid_hash
        
        # Only cache if this is a better path to this state
        if (cache_key not in self.state_cache or 
            node.cost < self.state_cache[cache_key].cost):
            self.state_cache[cache_key] = node
            
            # Update cache size and trigger cleanup if needed
            self._update_cache_size()
    
    def _generate_program_cache_key(self, program: DSLProgram, grid_hash: int) -> str:
        """Generate intelligent cache key for program validation results.
        
        Args:
            program: DSL program
            grid_hash: Hash of the grid the program was applied to
            
        Returns:
            Cache key string
        """
        # Create key based on program structure and grid
        program_str = str(program)
        return f"{program_str}_{grid_hash}"
    
    def _cache_program_validation(self, program: DSLProgram, grid: np.ndarray, 
                                 result_grid: np.ndarray, success: bool) -> None:
        """Cache program validation results.
        
        Args:
            program: DSL program that was executed
            grid: Input grid
            result_grid: Output grid from program execution
            success: Whether the program executed successfully
        """
        cache_key = self._generate_program_cache_key(program, hash(grid.tobytes()))

        # In-memory cache
        self.program_validation_cache[cache_key] = {
            'result_grid': result_grid.copy(),
            'success': success,
            'program_length': len(program),
            'access_count': 1,
            'last_access': time.time()
        }

        # External cache manager (shared)
        if getattr(self, 'cache_manager', None) is not None:
            try:
                payload = {
                    'result_grid': result_grid.copy(),
                    'success': success,
                    'program_length': len(program),
                }
                self.cache_manager.set_program_result(program, grid, payload)
            except Exception:
                pass

        self._update_cache_size()
    
    def _get_cached_program_validation(self, program: DSLProgram, grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get cached program validation result.
        
        Args:
            program: DSL program
            grid: Input grid
            
        Returns:
            Cached validation result or None
        """
        cache_key = self._generate_program_cache_key(program, hash(grid.tobytes()))

        # Check in-memory cache first
        if cache_key in self.program_validation_cache:
            cached_result = self.program_validation_cache[cache_key]
            cached_result['access_count'] += 1
            cached_result['last_access'] = time.time()
            self.cache_stats['program_cache_hits'] += 1
            return cached_result

        # Check external cache manager
        if getattr(self, 'cache_manager', None) is not None:
            try:
                cached = self.cache_manager.get_program_result(program, grid)
                if cached is not None:
                    # Normalize cached format to in-memory format
                    normalized = {
                        'result_grid': cached.get('result_grid'),
                        'success': cached.get('success', False),
                        'program_length': cached.get('program_length', len(program)),
                        'access_count': 1,
                        'last_access': time.time()
                    }
                    self.program_validation_cache[cache_key] = normalized
                    self.cache_stats['program_cache_hits'] += 1
                    return normalized
            except Exception:
                pass

        self.cache_stats['program_cache_misses'] += 1
        return None
    
    def _warm_pattern_cache(self, grid: np.ndarray) -> None:
        """Warm cache with common patterns for the given grid.
        
        Args:
            grid: Grid to analyze for pattern caching
        """
        # Generate pattern key based on grid characteristics
        height, width = grid.shape
        unique_colors = len(np.unique(grid))
        pattern_key = f"h{height}_w{width}_c{unique_colors}"
        
        if pattern_key in self.pattern_cache:
            self.cache_stats['pattern_cache_hits'] += 1
            return
        
        # Generate common operations for this pattern
        common_operations = []
        
        # Add geometric transformations (always useful)
        common_operations.extend([
            DSLOperation('Rotate90', {}),
            DSLOperation('Rotate180', {}),
            DSLOperation('ReflectH', {}),
            DSLOperation('ReflectV', {})
        ])
        
        # Add size-appropriate operations
        if height == width and height <= 10:  # Small square grids
            common_operations.extend([
                DSLOperation('Scale', {'factor': 2}),
                DSLOperation('Extract', {'r1': 0, 'r2': height//2, 'c1': 0, 'c2': width//2})
            ])
        
        # Add color-based operations if multiple colors present
        if unique_colors > 2:
            # Simple color swaps
            identity = list(range(10))
            swap_perm = identity.copy()
            swap_perm[1], swap_perm[2] = swap_perm[2], swap_perm[1]
            common_operations.append(DSLOperation('MapColors', {'perm': swap_perm}))
        
        self.pattern_cache[pattern_key] = common_operations
        self.cache_stats['pattern_cache_misses'] += 1
        self._update_cache_size()
    
    def _get_pattern_operations(self, grid: np.ndarray) -> List[DSLOperation]:
        """Get cached common operations for grid pattern.
        
        Args:
            grid: Grid to get operations for
            
        Returns:
            List of common operations for this pattern
        """
        height, width = grid.shape
        unique_colors = len(np.unique(grid))
        pattern_key = f"h{height}_w{width}_c{unique_colors}"
        
        if pattern_key in self.pattern_cache:
            self.cache_stats['pattern_cache_hits'] += 1
            return self.pattern_cache[pattern_key]
        
        self.cache_stats['pattern_cache_misses'] += 1
        return []
    
    def _update_cache_size(self) -> None:
        """Update cache size metrics and trigger cleanup if needed."""
        total_size = (len(self.state_cache) + 
                     len(self.program_validation_cache) + 
                     len(self.pattern_cache))
        
        self.cache_stats['cache_size'] = total_size
        
        # Trigger cleanup if cache is getting too large
        if total_size > self.max_cache_size * self.cache_cleanup_threshold:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Perform automatic cache cleanup based on access patterns."""
        current_time = time.time()
        cleanup_count = 0
        
        # Clean up program validation cache based on access patterns
        items_to_remove = []
        for key, cached_item in self.program_validation_cache.items():
            # Remove items that haven't been accessed recently and have low access count
            time_since_access = current_time - cached_item['last_access']
            if (time_since_access > 300 and  # 5 minutes
                cached_item['access_count'] < 3):
                items_to_remove.append(key)
        
        for key in items_to_remove:
            del self.program_validation_cache[key]
            cleanup_count += 1
        
        # Clean up state cache - keep only the most promising nodes
        if len(self.state_cache) > self.max_cache_size // 2:
            # Sort by cost + heuristic and keep the best half
            sorted_nodes = sorted(
                self.state_cache.items(),
                key=lambda x: x[1].cost + x[1].heuristic
            )
            
            keep_count = len(sorted_nodes) // 2
            self.state_cache = dict(sorted_nodes[:keep_count])
            cleanup_count += len(sorted_nodes) - keep_count
        
        self.cache_stats['cache_cleanup_count'] += cleanup_count
        
        if cleanup_count > 0:
            logger.debug(f"Cache cleanup removed {cleanup_count} items")
    
    def get_cache_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        stats = self.cache_stats.copy()
        
        # Compute hit rates
        total_state_accesses = stats['state_cache_hits'] + stats['state_cache_misses']
        total_program_accesses = stats['program_cache_hits'] + stats['program_cache_misses']
        total_pattern_accesses = stats['pattern_cache_hits'] + stats['pattern_cache_misses']
        
        stats['state_cache_hit_rate'] = (
            stats['state_cache_hits'] / max(total_state_accesses, 1)
        )
        stats['program_cache_hit_rate'] = (
            stats['program_cache_hits'] / max(total_program_accesses, 1)
        )
        stats['pattern_cache_hit_rate'] = (
            stats['pattern_cache_hits'] / max(total_pattern_accesses, 1)
        )
        
        # Overall cache efficiency
        total_hits = (stats['state_cache_hits'] + 
                     stats['program_cache_hits'] + 
                     stats['pattern_cache_hits'])
        total_accesses = (total_state_accesses + 
                         total_program_accesses + 
                         total_pattern_accesses)
        
        stats['overall_cache_hit_rate'] = total_hits / max(total_accesses, 1)
        
        return stats
    
    def search(self, initial_grid: np.ndarray, target_grid: np.ndarray, update_callback=None) -> SearchResult:
        """Search for optimal DSL program to transform initial_grid to target_grid.
        
        Args:
            initial_grid: Starting grid state
            target_grid: Goal grid state
            update_callback: Optional callback for real-time updates
            
        Returns:
            SearchResult with program and statistics
        """
        start_time = time.perf_counter()
        
        # Reset statistics
        self.statistics = SearchStatistics()
        self.beam_width_used = self.config.beam_width
        
        logger.info(f"Starting A* search: {initial_grid.shape} -> {target_grid.shape}")
        
        # Warm caches for better performance unless time budget is extremely tight
        if self.config.max_computation_time > 0.01:
            self._warm_pattern_cache(initial_grid)
            self._warm_pattern_cache(target_grid)
        
        try:
            def final_result_handler(result: SearchResult):
                if update_callback:
                    try:
                        # --- New logic for step-through ---
                        steps = []
                        if result.success and result.program:
                            current_grid = initial_grid.copy()
                            steps.append({
                                'operation': 'Initial State',
                                'grid': current_grid.tolist()
                            })
                            for op in result.program.operations:
                                next_grid = self.dsl_engine.apply_operation(current_grid, op)
                                steps.append({
                                    'operation': str(op),
                                    'grid': next_grid.tolist()
                                })
                                current_grid = next_grid

                        payload = {
                            'success': result.success,
                            'program_dict': result.program.to_dict() if result.program else None,
                            'steps': steps,
                            'final_grid': result.final_grid.tolist() if result.final_grid is not None else None,
                            'termination_reason': result.termination_reason,
                            'computation_time': result.computation_time,
                            'stats': {
                                'nodes_expanded': result.nodes_expanded,
                                'nodes_generated': result.nodes_generated,
                                'max_depth_reached': result.max_depth_reached
                            }
                        }
                        update_callback('solver_finished', payload)
                    except Exception as e:
                        logger.warning(f"Failed to send final result via callback: {e}")
                return result

            # Check if initial grid already matches target
            if np.array_equal(initial_grid, target_grid):
                computation_time = time.perf_counter() - start_time
                return final_result_handler(SearchResult(
                    success=True,
                    program=DSLProgram([]),  # Empty program
                    final_grid=initial_grid.copy(),
                    nodes_expanded=0,
                    nodes_generated=1,
                    computation_time=computation_time,
                    termination_reason="initial_match"
                ))
            
            # Initialize search
            # If the time budget is extremely tight, skip expensive heuristic init
            if self.config.max_computation_time <= 0.005:
                initial_h_value = 0.0
            else:
                initial_heuristic = self.heuristic_system.compute_heuristic(initial_grid, target_grid)
                initial_h_value = initial_heuristic.value
            
            root_node = SearchNode(
                grid=initial_grid.copy(),
                program=DSLProgram([]),
                cost=0.0,
                heuristic=initial_h_value,
                depth=0
            )
            
            # Compute version-space constraints from a single example when applicable
            # (In single-example mode, use (initial_grid, target_grid) as the only pair.)
            try:
                vs_constraints = compute_constraints([(initial_grid, target_grid)])
                allowed_ops = vs_constraints.get("allowed_op_names", set())
            except Exception:
                allowed_ops = set()

            # Priority queue for open nodes (min-heap)
            open_queue = [root_node]
            heapq.heapify(open_queue)
            deadline = start_time + self.config.max_computation_time
            
            # Closed set for duplicate detection
            closed_set: Set[int] = set() if self.config.duplicate_detection else set()
            
            # Best node found so far
            best_node = root_node
            
            while open_queue and self.statistics.nodes_expanded < self.config.max_nodes_expanded:
                # Check timeout
                if time.perf_counter() > deadline:
                    break
                
                # Adaptive beam scheduling
                new_beam_width = self._adaptive_beam_scheduling(open_queue)
                if new_beam_width != self.beam_width_used:
                    self.beam_width_used = new_beam_width
                    self.statistics.beam_reductions += 1
                    logger.debug(f"Adjusted beam width to {self.beam_width_used}")
                
                # Beam search pruning: keep only top beam_width nodes
                if len(open_queue) > self.beam_width_used:
                    open_queue = heapq.nsmallest(self.beam_width_used, open_queue)
                    heapq.heapify(open_queue)
                    self.statistics.nodes_pruned += len(open_queue) - self.beam_width_used
                
                # --- Real-time Update Callback ---
                if update_callback:
                    try:
                        # Prepare data for the frontend (top 5 candidates)
                        top_candidates = heapq.nsmallest(min(len(open_queue), 5), open_queue)
                        candidates_data = [{
                            'program': str(node.program),
                            'grid': node.grid.tolist(),
                            'cost': node.cost,
                            'heuristic': round(node.heuristic, 4),
                            'f_score': round(node.f_score, 4)
                        } for node in top_candidates]

                        # Prepare progress stats
                        progress_stats = {
                            'nodes_expanded': self.statistics.nodes_expanded,
                            'nodes_generated': self.statistics.nodes_generated,
                            'beam_width': self.beam_width_used,
                            'queue_size': len(open_queue)
                        }
                        update_callback('progress_update', {'candidates': candidates_data, 'stats': progress_stats})
                    except Exception as e:
                        logger.warning(f"Failed to execute update_callback: {e}")

                # Get best node from open queue
                current_node = heapq.heappop(open_queue)
                # Early timeout guard before any heavy work
                if time.perf_counter() - start_time > self.config.max_computation_time:
                    break
                
                # Check incremental cache
                cached_node = self._check_incremental_cache(current_node)
                if cached_node and cached_node.cost <= current_node.cost:
                    continue  # Skip this node, we have a better cached version
                
                # Check for goal state
                if np.array_equal(current_node.grid, target_grid):
                    computation_time = time.perf_counter() - start_time
                    return final_result_handler(self._create_success_result(
                        current_node, computation_time, "goal_reached"
                    ))
                
                # Skip if already explored (duplicate detection)
                if self.config.duplicate_detection:
                    if current_node.grid_hash in closed_set:
                        self.statistics.duplicate_states += 1
                        continue
                    closed_set.add(current_node.grid_hash)
                
                # Update incremental cache
                self._update_incremental_cache(current_node)
                
                # Update best node if this has lower heuristic
                if current_node.heuristic < best_node.heuristic:
                    best_node = current_node
                
                # Expand node if within depth limit and not trivially infeasible
                if current_node.depth < self.config.max_program_length:
                    try:
                        from arc_solver.reasoning.abstract_domains import is_infeasible
                        if is_infeasible(current_node.grid, target_grid, self.config.max_program_length - current_node.depth):
                            continue
                    except Exception:
                        pass
                    # Use parallel expansion if enabled
                    if self.config.parallel_expansion:
                        successors = self._expand_node_parallel(current_node, target_grid, deadline)
                    else:
                        successors = self._expand_node(current_node, target_grid, deadline)
                    
                    # Update statistics
                    self._update_statistics(current_node, successors)
                    # Synchronize public counters
                    self.nodes_expanded = self.statistics.nodes_expanded
                    self.nodes_generated = self.statistics.nodes_generated
                    self.max_depth_reached = self.statistics.max_depth_reached
                    
                    for successor in successors:
                        # Skip if already in closed set
                        if (self.config.duplicate_detection and 
                            successor.grid_hash in closed_set):
                            self.statistics.duplicate_states += 1
                            continue

                        # Version-space constraint: if inferred allowed set is non-empty,
                        # filter successors whose next operation is not allowed.
                        if allowed_ops:
                            try:
                                if successor.action and successor.action.primitive_name not in allowed_ops:
                                    continue
                            except Exception:
                                pass
                        
                        heapq.heappush(open_queue, successor)
                
                # Adaptive beam width reduction if search is taking too long
                if (self.config.adaptive_beam and 
                    self.statistics.nodes_expanded > self.config.max_nodes_expanded // 2 and
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
            
            # Compute final statistics
            self.statistics.compute_efficiency()
            
            # Determine termination reason
            if self.statistics.nodes_expanded >= self.config.max_nodes_expanded:
                termination_reason = "max_nodes_reached"
            elif time.perf_counter() > deadline:
                termination_reason = "timeout"
            elif not open_queue:
                termination_reason = "search_exhausted"
            else:
                termination_reason = "unknown"
            
            # Return best partial result
            return final_result_handler(self._create_partial_result(
                best_node, computation_time, termination_reason
            ))
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            logger.error(f"A* search failed with error: {e}")
            
            return final_result_handler(SearchResult(
                success=False,
                nodes_expanded=self.statistics.nodes_expanded,
                nodes_generated=self.nodes_generated,
                computation_time=computation_time,
                termination_reason=f"error: {str(e)}"
            ))
    
    def _expand_node(self, node: SearchNode, target_grid: np.ndarray, deadline: float) -> List[SearchNode]:
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

        # Invariants from input vs target to prune ops:
        # - If shapes match, avoid shape-changing ops
        # - If color sets are identical across train pair (single-example), de-prioritize MapColors
        shape_change_forbidden = True
        try:
            # Here, node holds only current grid; compare to target_grid
            shape_change_forbidden = (node.grid.shape == target_grid.shape)
        except Exception:
            pass

        # Enforce heavy-op budget: allow at most one heavy op per program
        heavy_ops = {"PaintIf", "FloodFill", "Repeat"}
        heavy_used = any(op.primitive_name in heavy_ops for op in node.program.operations)
        
        for operation in available_operations:
            if time.perf_counter() > deadline:
                break
            try:
                # Skip generating another heavy op if we've already used one
                if heavy_used and operation.primitive_name in heavy_ops:
                    continue
                # Skip shape-changing ops if target shape equals current shape
                if shape_change_forbidden and operation.primitive_name in {"Scale", "Extract", "Overlay"}:
                    continue
                # Cheap diff-driven gating for heavy ops
                if operation.primitive_name in {"PaintIf", "FloodFill"}:
                    try:
                        # If no difference between current and target, skip heavy ops
                        if np.array_equal(node.grid, target_grid):
                            continue
                    except Exception:
                        pass

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
    
    def _expand_node_multi_example(self, node: SearchNode, 
                                 training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[SearchNode]:
        """Expand a search node with multi-example validation and early pruning.
        
        Args:
            node: Node to expand
            training_examples: List of (input, expected_output) pairs for validation
            
        Returns:
            List of successor nodes that are valid for at least one example
        """
        successors = []
        
        # Get all available DSL operations
        available_operations = self.dsl_engine.get_available_operations(node.grid)

        # Enforce heavy-op budget for multi-example too
        heavy_ops = {"PaintIf", "FloodFill", "Repeat"}
        heavy_used = any(op.primitive_name in heavy_ops for op in node.program.operations)

        # Version-space constraints (multi-example): infer allowed ops once
        try:
            from arc_solver.reasoning.version_space import compute_constraints
            vs = compute_constraints(training_examples)
            allowed_ops = vs.get("allowed_op_names", set())
        except Exception:
            allowed_ops = set()
        
        for operation in available_operations:
            # Skip generating another heavy op if we've already used one
            if heavy_used and operation.primitive_name in heavy_ops:
                continue
            if allowed_ops and operation.primitive_name not in allowed_ops:
                continue
            try:
                # Apply operation to get new grid
                new_grid = self.dsl_engine.apply_operation(node.grid, operation)
                
                # Skip if operation had no effect
                if np.array_equal(new_grid, node.grid):
                    continue
                
                # Create new program with added operation
                new_program = DSLProgram(node.program.operations + [operation])
                
                # Validate against all training examples
                valid_examples = set()
                example_scores = {}
                
                for example_idx, (input_grid, expected_output) in enumerate(training_examples):
                    try:
                        # Execute program on this example
                        actual_output, _ = self.dsl_engine.execute_program(new_program, input_grid)
                        
                        # Compute similarity score
                        if np.array_equal(actual_output, expected_output):
                            score = 1.0  # Perfect match
                            valid_examples.add(example_idx)
                            example_scores[example_idx] = score
                        elif actual_output.shape == expected_output.shape:
                            # Partial similarity for debugging
                            matching_pixels = np.sum(actual_output == expected_output)
                            total_pixels = actual_output.size
                            score = matching_pixels / total_pixels
                            example_scores[example_idx] = score
                            # Only consider as valid if score is high enough
                            if score > 0.9:  # 90% similarity threshold
                                valid_examples.add(example_idx)
                        else:
                            example_scores[example_idx] = 0.0
                            
                    except Exception:
                        # Program execution failed on this example
                        example_scores[example_idx] = 0.0
                
                # Early pruning: skip nodes that don't work on any example
                if not valid_examples:
                    continue
                
                # Compute heuristic using the first example as reference
                primary_target = training_examples[0][1]
                heuristic_result = self.heuristic_system.compute_heuristic(new_grid, primary_target)
                
                # Create successor node with multi-example tracking
                successor = SearchNode(
                    grid=new_grid,
                    program=new_program,
                    cost=node.cost + 1.0,  # Uniform cost (each operation costs 1)
                    heuristic=heuristic_result.value,
                    parent=node,
                    action=operation,
                    depth=node.depth + 1,
                    valid_examples=valid_examples,
                    example_scores=example_scores
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
        
        # Add detailed search statistics
        search_stats = self.statistics.to_dict()
        search_stats.update(heuristic_stats)
        
        # Add cache performance statistics
        cache_stats = self.get_cache_performance_stats()
        search_stats.update(cache_stats)
        
        return SearchResult(
            success=True,
            program=node.program,
            final_grid=node.grid.copy(),
            nodes_expanded=self.statistics.nodes_expanded,
            nodes_generated=self.statistics.nodes_generated,
            computation_time=computation_time,
            max_depth_reached=self.statistics.max_depth_reached,
            beam_width_used=self.beam_width_used,
            termination_reason=termination_reason,
            heuristic_stats=search_stats
        )
    
    def _create_partial_result(self, best_node: SearchNode, computation_time: float,
                             termination_reason: str) -> SearchResult:
        """Create partial search result when exact solution not found."""
        heuristic_stats = self.heuristic_system.get_stats()
        
        # Add detailed search statistics
        search_stats = self.statistics.to_dict()
        search_stats.update(heuristic_stats)
        
        # Add cache performance statistics
        cache_stats = self.get_cache_performance_stats()
        search_stats.update(cache_stats)
        
        return SearchResult(
            success=False,
            program=best_node.program,
            final_grid=best_node.grid.copy(),
            nodes_expanded=self.statistics.nodes_expanded,
            nodes_generated=self.statistics.nodes_generated,
            computation_time=computation_time,
            max_depth_reached=self.statistics.max_depth_reached,
            beam_width_used=self.beam_width_used,
            termination_reason=termination_reason,
            heuristic_stats=search_stats
        )
    
    def search_multi_example(self, training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> SearchResult:
        """Search for optimal DSL program that works on ALL training examples.
        
        Args:
            training_examples: List of (input_grid, output_grid) pairs
            
        Returns:
            SearchResult with program that works on all examples
        """
        start_time = time.perf_counter()
        
        if not training_examples:
            return SearchResult(
                success=False,
                termination_reason="no_training_examples",
                computation_time=time.perf_counter() - start_time
            )
        
        logger.info(f"Starting multi-example A* search with {len(training_examples)} training pairs")
        
        # Use first example as primary target for search
        primary_input, primary_target = training_examples[0]
        
        # Bump K to 5 for multi-example searches only (keep single-example at default)
        try:
            self.dsl_engine.set_length_limit_for_mode('default', 5)
            # Ensure internal config also respects this during this run
            self.config.max_program_length = max(self.config.max_program_length, 5)
        except Exception:
            pass

        # Try enhanced multi-example search first
        enhanced_result = self._search_with_multi_example_nodes(training_examples)
        if enhanced_result.success:
            return enhanced_result
        
        # Fallback to candidate generation approach
        candidates = self._generate_candidates(primary_input, primary_target)
        
        # Track validation statistics
        validation_attempts = 0
        successful_validations = 0
        
        # Rank candidates by multi-example performance
        ranked_candidates = self._rank_candidates_by_performance(candidates, training_examples)
        
        # Validate each candidate against all training examples (in ranked order)
        for program, performance_score in ranked_candidates:
            validation_attempts += 1
            if self._validate_on_all_examples(program, training_examples):
                successful_validations += 1
                computation_time = time.perf_counter() - start_time
                
                # Execute program on primary input to get final grid
                try:
                    final_grid, _ = self.dsl_engine.execute_program(program, primary_input)
                except Exception:
                    final_grid = primary_input.copy()
                
                return SearchResult(
                    success=True,
                    program=program,
                    final_grid=final_grid,
                    nodes_expanded=self.nodes_expanded,
                    nodes_generated=self.nodes_generated,
                    computation_time=computation_time,
                    max_depth_reached=self.max_depth_reached,
                    beam_width_used=self.beam_width_used,
                    termination_reason="multi_example_success",
                    candidates_generated=len(candidates),
                    examples_validated=len(training_examples),
                    validation_success_rate=successful_validations / max(validation_attempts, 1)
                )
        
        # No program worked on all examples
        computation_time = time.perf_counter() - start_time
        return SearchResult(
            success=False,
            nodes_expanded=self.nodes_expanded,
            nodes_generated=self.nodes_generated,
            computation_time=computation_time,
            max_depth_reached=self.max_depth_reached,
            beam_width_used=self.beam_width_used,
            termination_reason="no_multi_example_solution",
            candidates_generated=len(candidates),
            examples_validated=len(training_examples),
            validation_success_rate=successful_validations / max(validation_attempts, 1) if validation_attempts > 0 else 0.0
        )
    
    def _generate_candidates(self, input_grid: np.ndarray, target_grid: np.ndarray, 
                           max_candidates: int = 50) -> List[DSLProgram]:
        """Generate candidate programs from single example using beam search.
        
        Args:
            input_grid: Input grid
            target_grid: Target grid
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            List of candidate DSL programs
        """
        candidates = []
        
        # Run standard A* search but collect multiple solutions
        original_beam_width = self.beam_width_used
        self.beam_width_used = max(original_beam_width, max_candidates)
        
        try:
            # Reset statistics
            self.nodes_expanded = 0
            self.nodes_generated = 0
            self.max_depth_reached = 0
            
            # Check if initial grid already matches target
            if np.array_equal(input_grid, target_grid):
                return [DSLProgram([])]
            
            # Initialize search
            initial_heuristic = self.heuristic_system.compute_heuristic(input_grid, target_grid)
            
            root_node = SearchNode(
                grid=input_grid.copy(),
                program=DSLProgram([]),
                cost=0.0,
                heuristic=initial_heuristic.value,
                depth=0
            )
            
            # Priority queue for open nodes
            open_queue = [root_node]
            heapq.heapify(open_queue)
            
            # Closed set for duplicate detection
            closed_set: Set[int] = set()
            
            # Collect all nodes that reach the target
            target_nodes = []
            
            search_start = time.perf_counter()
            
            while (open_queue and 
                   self.nodes_expanded < self.config.max_nodes_expanded and
                   len(target_nodes) < max_candidates):
                
                # Check timeout
                if time.perf_counter() - search_start > self.config.max_computation_time:
                    break
                
                # Beam search pruning
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
                    target_nodes.append(current_node)
                    continue  # Continue searching for more solutions
                
                # Skip if already explored
                if current_node.grid_hash in closed_set:
                    continue
                closed_set.add(current_node.grid_hash)
                
                # Expand node if within depth limit
                if current_node.depth < self.config.max_program_length:
                    successors = self._expand_node(
                        current_node, target_grid, deadline=search_start + self.config.max_computation_time
                    )
                    
                    for successor in successors:
                        if successor.grid_hash not in closed_set:
                            heapq.heappush(open_queue, successor)
                            self.nodes_generated += 1
            
            # Convert target nodes to programs
            for node in target_nodes:
                candidates.append(node.program)
            
            # If no exact solutions found, add best partial solutions
            if not candidates and open_queue:
                # Sort remaining nodes by heuristic and take best ones
                remaining_nodes = sorted(open_queue, key=lambda n: n.heuristic)
                for node in remaining_nodes[:max_candidates//2]:
                    candidates.append(node.program)
            
        finally:
            # Restore original beam width
            self.beam_width_used = original_beam_width
        
        logger.info(f"Generated {len(candidates)} candidate programs")
        return candidates
    
    def _validate_on_all_examples(self, program: DSLProgram, 
                                training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Validate that a program works on all training examples.
        
        Args:
            program: DSL program to validate
            training_examples: List of (input, expected_output) pairs
            
        Returns:
            True if program works on all examples, False otherwise
        """
        # Create cache key for this program + examples combination
        program_hash = hash(program)
        examples_hash = hash(tuple(
            (input_grid.tobytes(), expected_output.tobytes()) 
            for input_grid, expected_output in training_examples
        ))
        cache_key = (program_hash, examples_hash)
        
        # Check cache first
        if not hasattr(self, '_validation_cache'):
            self._validation_cache = {}
        
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        # Validate program on all examples
        is_valid = True
        for input_grid, expected_output in training_examples:
            try:
                # Execute program on this input
                actual_output, exec_info = self.dsl_engine.execute_program(program, input_grid)
                
                # Check if output matches expected
                if not np.array_equal(actual_output, expected_output):
                    is_valid = False
                    break
                    
            except Exception as e:
                # Program execution failed
                logger.debug(f"Program execution failed on validation: {e}")
                is_valid = False
                break
        
        # Cache the result
        self._validation_cache[cache_key] = is_valid
        return is_valid
    
    def _rank_candidates_by_performance(self, candidates: List[DSLProgram], 
                                      training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[DSLProgram, float]]:
        """Rank candidate programs by their multi-example performance scores.
        
        Args:
            candidates: List of candidate DSL programs
            training_examples: List of (input, expected_output) pairs
            
        Returns:
            List of (program, performance_score) tuples sorted by performance (best first)
        """
        ranked_candidates = []
        
        for program in candidates:
            performance_score = self._compute_performance_score(program, training_examples)
            ranked_candidates.append((program, performance_score))
        
        # Sort by performance score (higher is better)
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_candidates
    
    def _compute_performance_score(self, program: DSLProgram, 
                                 training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute performance score for a program across all training examples.
        
        Args:
            program: DSL program to evaluate
            training_examples: List of (input, expected_output) pairs
            
        Returns:
            Performance score (higher is better)
        """
        total_score = 0.0
        valid_examples = 0
        
        for input_grid, expected_output in training_examples:
            try:
                # Execute program on this input
                actual_output, exec_info = self.dsl_engine.execute_program(program, input_grid)
                
                # Compute similarity score (exact match = 1.0, partial match < 1.0)
                if np.array_equal(actual_output, expected_output):
                    example_score = 1.0  # Perfect match
                else:
                    # Compute partial similarity based on pixel accuracy
                    if actual_output.shape == expected_output.shape:
                        matching_pixels = np.sum(actual_output == expected_output)
                        total_pixels = actual_output.size
                        example_score = matching_pixels / total_pixels
                    else:
                        example_score = 0.0  # Shape mismatch
                
                total_score += example_score
                valid_examples += 1
                
            except Exception:
                # Program execution failed, score = 0 for this example
                valid_examples += 1
                # total_score += 0.0 (implicit)
        
        # Average score across all examples, with penalty for program length
        if valid_examples > 0:
            avg_score = total_score / valid_examples
            # Prefer shorter programs (slight penalty for length)
            length_penalty = 0.01 * len(program.operations)
            return max(0.0, avg_score - length_penalty)
        else:
            return 0.0
    
    def _search_with_multi_example_nodes(self, training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> SearchResult:
        """Enhanced A* search with multi-example node tracking and early pruning.
        
        Args:
            training_examples: List of (input, expected_output) pairs
            
        Returns:
            SearchResult with program that works on all examples
        """
        start_time = time.perf_counter()
        
        # Reset statistics
        nodes_expanded = 0
        nodes_generated = 0
        max_depth_reached = 0
        
        # Initialize search with first example
        primary_input, primary_target = training_examples[0]
        
        # Check if initial grid already matches target
        if np.array_equal(primary_input, primary_target):
            return SearchResult(
                success=True,
                program=DSLProgram([]),
                final_grid=primary_input.copy(),
                computation_time=time.perf_counter() - start_time,
                termination_reason="initial_match",
                candidates_generated=0,
                examples_validated=len(training_examples),
                validation_success_rate=1.0
            )
        
        # Initialize search
        initial_heuristic = self.heuristic_system.compute_heuristic(primary_input, primary_target)
        
        # Create root node and validate against all examples
        root_node = SearchNode(
            grid=primary_input.copy(),
            program=DSLProgram([]),
            cost=0.0,
            heuristic=initial_heuristic.value,
            depth=0
        )
        
        # Validate root node against all examples
        for example_idx, (input_grid, expected_output) in enumerate(training_examples):
            if np.array_equal(input_grid, expected_output):
                root_node.add_example_validation(example_idx, 1.0)
        
        # Priority queue for open nodes
        open_queue = [root_node]
        heapq.heapify(open_queue)
        
        # Closed set for duplicate detection
        closed_set: Set[int] = set()
        
        # Best node found so far
        best_node = root_node
        
        search_start = time.perf_counter()
        
        while (open_queue and 
               nodes_expanded < self.config.max_nodes_expanded):
            
            # Check timeout
            if time.perf_counter() - search_start > self.config.max_computation_time:
                break
            
            # Beam search pruning
            if len(open_queue) > self.beam_width_used:
                open_queue = heapq.nsmallest(self.beam_width_used, open_queue)
                heapq.heapify(open_queue)
            
            # Get best node from open queue
            current_node = heapq.heappop(open_queue)
            nodes_expanded += 1
            
            # Update statistics
            max_depth_reached = max(max_depth_reached, current_node.depth)
            
            # Check if this node works on all examples
            if current_node.is_valid_for_all_examples(len(training_examples)):
                computation_time = time.perf_counter() - start_time
                
                return SearchResult(
                    success=True,
                    program=current_node.program,
                    final_grid=current_node.grid.copy(),
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    computation_time=computation_time,
                    max_depth_reached=max_depth_reached,
                    beam_width_used=self.beam_width_used,
                    termination_reason="multi_example_success",
                    candidates_generated=nodes_generated,
                    examples_validated=len(training_examples),
                    validation_success_rate=1.0
                )
            
            # Skip if already explored
            if current_node.grid_hash in closed_set:
                continue
            closed_set.add(current_node.grid_hash)
            
            # Update best node based on multi-example performance
            if (current_node.get_average_example_score() > best_node.get_average_example_score() or
                (current_node.get_average_example_score() == best_node.get_average_example_score() and
                 len(current_node.valid_examples) > len(best_node.valid_examples))):
                best_node = current_node
            
            # Expand node if within depth limit
            if current_node.depth < self.config.max_program_length:
                successors = self._expand_node_multi_example(current_node, training_examples)
                
                for successor in successors:
                    if successor.grid_hash not in closed_set:
                        heapq.heappush(open_queue, successor)
                        nodes_generated += 1
        
        # Search completed without finding exact solution
        computation_time = time.perf_counter() - start_time
        
        # Determine termination reason
        if nodes_expanded >= self.config.max_nodes_expanded:
            termination_reason = "max_nodes_reached"
        elif time.perf_counter() - search_start > self.config.max_computation_time:
            termination_reason = "timeout"
        elif not open_queue:
            termination_reason = "search_exhausted"
        else:
            termination_reason = "unknown"
        
        return SearchResult(
            success=False,
            program=best_node.program,
            final_grid=best_node.grid.copy(),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            computation_time=computation_time,
            max_depth_reached=max_depth_reached,
            beam_width_used=self.beam_width_used,
            termination_reason=termination_reason,
            candidates_generated=nodes_generated,
            examples_validated=len(training_examples),
            validation_success_rate=best_node.get_average_example_score()
        )

    def get_search_stats(self) -> Dict[str, Any]:
        """Get detailed search statistics."""
        heuristic_stats = self.heuristic_system.get_stats()
        # Report from statistics for consistency
        nodes_expanded = self.statistics.nodes_expanded
        nodes_generated = self.statistics.nodes_generated
        max_depth_reached = self.statistics.max_depth_reached
        beam_width_used = self.beam_width_used
        
        return {
            'nodes_expanded': nodes_expanded,
            'nodes_generated': nodes_generated,
            'max_depth_reached': max_depth_reached,
            'beam_width_used': beam_width_used,
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
