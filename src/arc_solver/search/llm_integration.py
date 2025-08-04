"""Integration between LLM proposals and A* search.

This module provides the integration layer that combines LLM-generated
DSL program proposals with the A* search algorithm for enhanced performance.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from arc_solver.llm.llm_proposer import LLMProposer, create_llm_proposer, ProposalResult
from arc_solver.search.astar import AStarSearcher, SearchResult, SearchNode, SearchConfig
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLEngine
from arc_solver.perception.blob_labeling import create_blob_labeler
from arc_solver.core.data_models import Blob
import heapq

logger = logging.getLogger(__name__)


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM-A* integration."""
    
    # LLM configuration
    llm_enabled: bool = True
    llm_model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    num_proposals: int = 3
    
    # Search configuration
    original_beam_width: int = 64
    llm_beam_width: int = 8
    priority_boost: float = 2.0
    
    # Fallback configuration
    fallback_on_failure: bool = True
    fallback_on_timeout: bool = True
    max_llm_time: float = 10.0
    
    # Quality thresholds
    min_parsing_success_rate: float = 0.5
    max_failed_attempts: int = 3


@dataclass
class IntegratedSearchResult:
    """Result from LLM-integrated A* search."""
    
    success: bool
    program: Optional[DSLProgram] = None
    final_grid: Optional[np.ndarray] = None
    
    # Search statistics
    nodes_expanded: int = 0
    nodes_generated: int = 0
    computation_time: float = 0.0
    termination_reason: str = "unknown"
    
    # LLM statistics
    llm_used: bool = False
    llm_proposals_generated: int = 0
    llm_proposals_parsed: int = 0
    llm_generation_time: float = 0.0
    llm_parsing_success_rate: float = 0.0
    
    # Integration statistics
    beam_width_used: int = 0
    fallback_used: bool = False
    priority_boost_applied: bool = False


class LLMGuidedAStarSearcher:
    """A* searcher with LLM proposal priority boosting."""
    
    def __init__(self, base_searcher: AStarSearcher, llm_proposals: List[DSLProgram], 
                 priority_boost: float, dsl_engine: DSLEngine):
        """Initialize LLM-guided A* searcher.
        
        Args:
            base_searcher: Base A* searcher
            llm_proposals: LLM-generated program proposals
            priority_boost: Priority boost factor for LLM-guided paths
            dsl_engine: DSL engine for program execution
        """
        self.base_searcher = base_searcher
        self.llm_proposals = llm_proposals
        self.priority_boost = priority_boost
        self.dsl_engine = dsl_engine
        
        # Create set of LLM operations for fast lookup
        self.llm_operations = set()
        for proposal in llm_proposals:
            for op in proposal.operations:
                self.llm_operations.add((op.primitive_name, tuple(sorted(op.parameters.items()))))
        
        logger.info(f"LLM-guided searcher initialized with {len(llm_proposals)} proposals, "
                   f"boost factor {priority_boost}")
    
    def search(self, initial_grid: np.ndarray, target_grid: np.ndarray) -> SearchResult:
        """Perform LLM-guided A* search.
        
        Args:
            initial_grid: Initial grid
            target_grid: Target grid
            
        Returns:
            SearchResult with guided search statistics
        """
        start_time = time.perf_counter()
        
        # Check if initial grid already matches target
        if np.array_equal(initial_grid, target_grid):
            return SearchResult(
                success=True,
                program=None,
                final_grid=initial_grid.copy(),
                nodes_expanded=0,
                nodes_generated=1,
                computation_time=time.perf_counter() - start_time,
                termination_reason="initial_match"
            )
        
        # Initialize search with LLM-guided priority queue
        initial_heuristic = self.base_searcher.heuristic_system.compute_heuristic(
            initial_grid, target_grid
        )
        
        root_node = SearchNode(
            grid=initial_grid.copy(),
            program=DSLProgram([]),
            cost=0.0,
            heuristic=initial_heuristic.value,
            depth=0
        )
        
        # Priority queue with LLM guidance
        open_queue = [root_node]
        heapq.heapify(open_queue)
        
        # Closed set for duplicate detection
        closed_set = set()
        
        # Search statistics
        nodes_expanded = 0
        nodes_generated = 1
        max_depth_reached = 0
        
        while (open_queue and 
               nodes_expanded < self.base_searcher.config.max_nodes_expanded):
            
            # Check timeout
            if time.perf_counter() - start_time > self.base_searcher.config.max_computation_time:
                break
            
            # Beam search pruning
            if len(open_queue) > self.base_searcher.config.beam_width:
                open_queue = heapq.nsmallest(self.base_searcher.config.beam_width, open_queue)
                heapq.heapify(open_queue)
            
            # Get best node
            current_node = heapq.heappop(open_queue)
            nodes_expanded += 1
            max_depth_reached = max(max_depth_reached, current_node.depth)
            
            # Check for goal state
            if np.array_equal(current_node.grid, target_grid):
                computation_time = time.perf_counter() - start_time
                return SearchResult(
                    success=True,
                    program=current_node.program,
                    final_grid=current_node.grid.copy(),
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    computation_time=computation_time,
                    max_depth_reached=max_depth_reached,
                    termination_reason="goal_reached"
                )
            
            # Skip if already explored
            grid_hash = current_node.grid_hash
            if grid_hash in closed_set:
                continue
            closed_set.add(grid_hash)
            
            # Expand node if within depth limit
            if current_node.depth < self.base_searcher.config.max_program_length:
                successors = self._expand_node_with_guidance(current_node, target_grid)
                
                for successor in successors:
                    if successor.grid_hash not in closed_set:
                        heapq.heappush(open_queue, successor)
                        nodes_generated += 1
        
        # Search completed without finding exact solution
        computation_time = time.perf_counter() - start_time
        
        # Determine termination reason
        if nodes_expanded >= self.base_searcher.config.max_nodes_expanded:
            termination_reason = "max_nodes_reached"
        elif time.perf_counter() - start_time > self.base_searcher.config.max_computation_time:
            termination_reason = "timeout"
        elif not open_queue:
            termination_reason = "search_exhausted"
        else:
            termination_reason = "unknown"
        
        return SearchResult(
            success=False,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            computation_time=computation_time,
            max_depth_reached=max_depth_reached,
            termination_reason=termination_reason
        )
    
    def _expand_node_with_guidance(self, node: SearchNode, target_grid: np.ndarray) -> List[SearchNode]:
        """Expand node with LLM guidance priority boosting.
        
        Args:
            node: Node to expand
            target_grid: Target grid for heuristic computation
            
        Returns:
            List of successor nodes with adjusted priorities
        """
        successors = []
        
        # Get all available operations
        available_operations = self.dsl_engine.get_available_operations(node.grid)
        
        for operation in available_operations:
            try:
                # Apply operation
                new_grid = self.dsl_engine.apply_operation(node.grid, operation)
                
                # Skip if no effect
                if np.array_equal(new_grid, node.grid):
                    continue
                
                # Create new program
                new_program = DSLProgram(node.program.operations + [operation])
                
                # Compute base heuristic
                heuristic_result = self.base_searcher.heuristic_system.compute_heuristic(
                    new_grid, target_grid
                )
                base_heuristic = heuristic_result.value
                
                # Apply LLM guidance boost
                adjusted_heuristic = self._apply_llm_guidance(
                    operation, base_heuristic, node.depth + 1
                )
                
                # Create successor node
                successor = SearchNode(
                    grid=new_grid,
                    program=new_program,
                    cost=node.cost + 1.0,
                    heuristic=adjusted_heuristic,
                    parent=node,
                    action=operation,
                    depth=node.depth + 1
                )
                
                successors.append(successor)
                
            except Exception as e:
                logger.debug(f"Operation {operation} failed: {e}")
                continue
        
        return successors
    
    def _apply_llm_guidance(self, operation, base_heuristic: float, depth: int) -> float:
        """Apply LLM guidance to adjust heuristic values.
        
        Args:
            operation: DSL operation being considered
            base_heuristic: Base heuristic value
            depth: Current search depth
            
        Returns:
            Adjusted heuristic value
        """
        # Create operation signature for lookup
        op_signature = (operation.primitive_name, tuple(sorted(operation.parameters.items())))
        
        # Check if this operation is in LLM proposals
        if op_signature in self.llm_operations:
            # Apply priority boost (lower heuristic = higher priority)
            boost_factor = self.priority_boost * (1.0 / (depth + 1))  # Decay with depth
            adjusted_heuristic = base_heuristic / (1.0 + boost_factor)
            
            logger.debug(f"LLM boost applied to {operation}: {base_heuristic:.3f} -> {adjusted_heuristic:.3f}")
            return adjusted_heuristic
        
        return base_heuristic


class LLMIntegratedSearcher:
    """A* searcher with LLM proposal integration."""
    
    def __init__(self, config: Optional[LLMIntegrationConfig] = None):
        """Initialize LLM-integrated searcher.
        
        Args:
            config: Integration configuration
        """
        self.config = config or LLMIntegrationConfig()
        
        # Initialize components
        self.astar_searcher = AStarSearcher()
        self.llm_proposer = None  # Lazy initialization
        self.dsl_engine = DSLEngine()
        self.blob_labeler = create_blob_labeler(use_gpu=False)
        
        # Statistics
        self.total_searches = 0
        self.llm_successful_searches = 0
        self.fallback_searches = 0
        self.failed_llm_attempts = 0
        
        logger.info(f"LLM-integrated searcher initialized (LLM enabled: {self.config.llm_enabled})")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM proposer (lazy loading)."""
        if self.llm_proposer is not None or not self.config.llm_enabled:
            return
        
        try:
            self.llm_proposer = create_llm_proposer(
                model_name=self.config.llm_model_name,
                num_proposals=self.config.num_proposals
            )
            logger.info("LLM proposer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM proposer: {e}")
            logger.info("Falling back to vanilla A* search")
            self.config.llm_enabled = False
    
    def search(self, initial_grid: np.ndarray, target_grid: np.ndarray) -> IntegratedSearchResult:
        """Search for optimal DSL program with LLM integration.
        
        Args:
            initial_grid: Starting grid state
            target_grid: Goal grid state
            
        Returns:
            IntegratedSearchResult with program and statistics
        """
        start_time = time.perf_counter()
        self.total_searches += 1
        
        # Initialize result
        result = IntegratedSearchResult(success=False)
        
        try:
            # Try LLM-enhanced search first
            if self.config.llm_enabled:
                result = self._llm_enhanced_search(initial_grid, target_grid, start_time)
                
                # Check if LLM search was successful
                if result.success and not result.fallback_used:
                    self.llm_successful_searches += 1
                    return result
                
                # If LLM failed but we have partial results, continue with fallback
                if result.fallback_used:
                    self.fallback_searches += 1
            
            # Fallback to vanilla A* search
            if not result.success or result.fallback_used:
                logger.info("Using vanilla A* search")
                vanilla_result = self._vanilla_search(initial_grid, target_grid, start_time)
                
                # Merge results (keep LLM statistics if available)
                if result.llm_used:
                    vanilla_result.llm_used = result.llm_used
                    vanilla_result.llm_proposals_generated = result.llm_proposals_generated
                    vanilla_result.llm_proposals_parsed = result.llm_proposals_parsed
                    vanilla_result.llm_generation_time = result.llm_generation_time
                    vanilla_result.llm_parsing_success_rate = result.llm_parsing_success_rate
                    vanilla_result.fallback_used = True
                
                return vanilla_result
            
            return result
            
        except Exception as e:
            computation_time = time.perf_counter() - start_time
            logger.error(f"Integrated search failed: {e}")
            
            return IntegratedSearchResult(
                success=False,
                computation_time=computation_time,
                termination_reason=f"error: {str(e)}"
            )
    
    def _llm_enhanced_search(self, 
                           initial_grid: np.ndarray, 
                           target_grid: np.ndarray,
                           start_time: float) -> IntegratedSearchResult:
        """Perform LLM-enhanced A* search.
        
        Args:
            initial_grid: Initial grid
            target_grid: Target grid
            start_time: Search start time
            
        Returns:
            IntegratedSearchResult
        """
        try:
            # Initialize LLM if needed
            self._initialize_llm()
            
            if not self.config.llm_enabled or self.llm_proposer is None:
                return IntegratedSearchResult(
                    success=False,
                    fallback_used=True,
                    termination_reason="llm_unavailable"
                )
            
            # Extract blobs for LLM input
            input_blobs, _ = self.blob_labeler.label_blobs(initial_grid)
            target_blobs, _ = self.blob_labeler.label_blobs(target_grid)
            
            # Generate LLM proposals with timeout
            llm_start_time = time.perf_counter()
            try:
                proposal_result = self.llm_proposer.generate_proposals(
                    initial_grid, target_grid, input_blobs, target_blobs
                )
                llm_time = time.perf_counter() - llm_start_time
                
                # Check if LLM generation took too long
                if llm_time > self.config.max_llm_time:
                    logger.warning(f"LLM generation took {llm_time:.2f}s, exceeds {self.config.max_llm_time}s limit")
                    if self.config.fallback_on_timeout:
                        return IntegratedSearchResult(
                            success=False,
                            llm_used=True,
                            llm_generation_time=llm_time,
                            fallback_used=True,
                            termination_reason="llm_timeout"
                        )
                        
            except Exception as e:
                llm_time = time.perf_counter() - llm_start_time
                logger.error(f"LLM proposal generation failed with exception: {e}")
                self.failed_llm_attempts += 1
                
                return IntegratedSearchResult(
                    success=False,
                    llm_used=True,
                    llm_generation_time=llm_time,
                    fallback_used=True,
                    termination_reason=f"llm_exception: {str(e)}"
                )
            
            # Check if LLM generation was successful
            if not proposal_result.success:
                self.failed_llm_attempts += 1
                logger.warning(f"LLM proposal generation failed: {proposal_result.error}")
                
                return IntegratedSearchResult(
                    success=False,
                    llm_used=True,
                    llm_generation_time=llm_time,
                    fallback_used=True,
                    termination_reason="llm_generation_failed"
                )
            
            # Check parsing success rate
            if proposal_result.parsing_success_rate < self.config.min_parsing_success_rate:
                logger.warning(f"Low parsing success rate: {proposal_result.parsing_success_rate:.2%}")
                self.failed_llm_attempts += 1
                
                return IntegratedSearchResult(
                    success=False,
                    llm_used=True,
                    llm_proposals_generated=len(proposal_result.raw_responses),
                    llm_proposals_parsed=len(proposal_result.proposals),
                    llm_generation_time=llm_time,
                    llm_parsing_success_rate=proposal_result.parsing_success_rate,
                    fallback_used=True,
                    termination_reason="low_parsing_success"
                )
            
            # Check if we have any valid proposals
            if not proposal_result.proposals:
                logger.warning("No valid proposals generated by LLM")
                self.failed_llm_attempts += 1
                
                return IntegratedSearchResult(
                    success=False,
                    llm_used=True,
                    llm_proposals_generated=len(proposal_result.raw_responses),
                    llm_proposals_parsed=0,
                    llm_generation_time=llm_time,
                    llm_parsing_success_rate=proposal_result.parsing_success_rate,
                    fallback_used=True,
                    termination_reason="no_valid_proposals"
                )
            
            # Test proposals directly first
            for i, proposal in enumerate(proposal_result.proposals):
                try:
                    # Execute proposal on initial grid
                    result_grid, exec_info = self.dsl_engine.execute_program(proposal, initial_grid)
                    
                    # Check if proposal solves the task
                    if np.array_equal(result_grid, target_grid):
                        computation_time = time.perf_counter() - start_time
                        
                        logger.info(f"LLM proposal {i+1} solved the task directly!")
                        
                        return IntegratedSearchResult(
                            success=True,
                            program=proposal,
                            final_grid=result_grid,
                            computation_time=computation_time,
                            llm_used=True,
                            llm_proposals_generated=len(proposal_result.raw_responses),
                            llm_proposals_parsed=len(proposal_result.proposals),
                            llm_generation_time=llm_time,
                            llm_parsing_success_rate=proposal_result.parsing_success_rate,
                            beam_width_used=self.config.llm_beam_width,
                            termination_reason="llm_direct_solution"
                        )
                        
                except Exception as e:
                    logger.debug(f"LLM proposal {i+1} execution failed: {e}")
                    continue
            
            # Validate and filter proposals
            valid_proposals = self._validate_proposals(proposal_result.proposals)
            
            if not valid_proposals:
                logger.warning("No valid proposals after validation")
                self.failed_llm_attempts += 1
                
                return IntegratedSearchResult(
                    success=False,
                    llm_used=True,
                    llm_proposals_generated=len(proposal_result.raw_responses),
                    llm_proposals_parsed=len(proposal_result.proposals),
                    llm_generation_time=llm_time,
                    llm_parsing_success_rate=proposal_result.parsing_success_rate,
                    fallback_used=True,
                    termination_reason="no_valid_proposals_after_validation"
                )
            
            # If no direct solution, use proposals to guide search
            logger.info(f"Using {len(valid_proposals)} validated LLM proposals to guide search")
            
            # Perform guided A* search with reduced beam width
            guided_result = self._guided_astar_search(
                initial_grid, target_grid, valid_proposals, start_time
            )
            
            # Add LLM statistics to result
            guided_result.llm_used = True
            guided_result.llm_proposals_generated = len(proposal_result.raw_responses)
            guided_result.llm_proposals_parsed = len(proposal_result.proposals)
            guided_result.llm_generation_time = llm_time
            guided_result.llm_parsing_success_rate = proposal_result.parsing_success_rate
            guided_result.beam_width_used = self.config.llm_beam_width
            guided_result.priority_boost_applied = True
            
            return guided_result
            
        except Exception as e:
            logger.error(f"LLM-enhanced search failed: {e}")
            return IntegratedSearchResult(
                success=False,
                llm_used=True,
                fallback_used=True,
                termination_reason=f"llm_error: {str(e)}"
            )
    
    def _guided_astar_search(self,
                           initial_grid: np.ndarray,
                           target_grid: np.ndarray,
                           llm_proposals: List[DSLProgram],
                           start_time: float) -> IntegratedSearchResult:
        """Perform A* search guided by LLM proposals.
        
        Args:
            initial_grid: Initial grid
            target_grid: Target grid
            llm_proposals: LLM-generated program proposals
            start_time: Search start time
            
        Returns:
            IntegratedSearchResult
        """
        # Configure A* searcher with reduced beam width
        original_beam_width = self.astar_searcher.config.beam_width
        self.astar_searcher.config.beam_width = self.config.llm_beam_width
        
        try:
            # Create enhanced A* searcher with LLM proposal guidance
            enhanced_searcher = LLMGuidedAStarSearcher(
                base_searcher=self.astar_searcher,
                llm_proposals=llm_proposals,
                priority_boost=self.config.priority_boost,
                dsl_engine=self.dsl_engine
            )
            
            # Perform guided search
            search_result = enhanced_searcher.search(initial_grid, target_grid)
            
            # Convert to integrated result
            computation_time = time.perf_counter() - start_time
            
            return IntegratedSearchResult(
                success=search_result.success,
                program=search_result.program,
                final_grid=search_result.final_grid,
                nodes_expanded=search_result.nodes_expanded,
                nodes_generated=search_result.nodes_generated,
                computation_time=computation_time,
                termination_reason=search_result.termination_reason,
                beam_width_used=self.config.llm_beam_width
            )
            
        finally:
            # Restore original beam width
            self.astar_searcher.config.beam_width = original_beam_width
    
    def _vanilla_search(self,
                       initial_grid: np.ndarray,
                       target_grid: np.ndarray,
                       start_time: float) -> IntegratedSearchResult:
        """Perform vanilla A* search without LLM guidance.
        
        Args:
            initial_grid: Initial grid
            target_grid: Target grid
            start_time: Search start time
            
        Returns:
            IntegratedSearchResult
        """
        # Perform standard A* search
        search_result = self.astar_searcher.search(initial_grid, target_grid)
        
        # Convert to integrated result
        computation_time = time.perf_counter() - start_time
        
        return IntegratedSearchResult(
            success=search_result.success,
            program=search_result.program,
            final_grid=search_result.final_grid,
            nodes_expanded=search_result.nodes_expanded,
            nodes_generated=search_result.nodes_generated,
            computation_time=computation_time,
            termination_reason=search_result.termination_reason,
            beam_width_used=self.astar_searcher.config.beam_width,
            llm_used=False
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_searches': self.total_searches,
            'llm_successful_searches': self.llm_successful_searches,
            'fallback_searches': self.fallback_searches,
            'failed_llm_attempts': self.failed_llm_attempts,
            'llm_success_rate': (self.llm_successful_searches / max(1, self.total_searches)),
            'fallback_rate': (self.fallback_searches / max(1, self.total_searches)),
            'config': {
                'llm_enabled': self.config.llm_enabled,
                'llm_model_name': self.config.llm_model_name,
                'original_beam_width': self.config.original_beam_width,
                'llm_beam_width': self.config.llm_beam_width,
                'priority_boost': self.config.priority_boost
            }
        }
        
        # Add LLM proposer stats if available
        if self.llm_proposer is not None:
            stats['llm_proposer'] = self.llm_proposer.get_stats()
        
        # Add A* searcher stats
        stats['astar_searcher'] = self.astar_searcher.get_search_stats()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_searches = 0
        self.llm_successful_searches = 0
        self.fallback_searches = 0
        self.failed_llm_attempts = 0
        
        if self.llm_proposer is not None:
            self.llm_proposer.reset_stats()
    
    def _validate_proposals(self, proposals: List[DSLProgram]) -> List[DSLProgram]:
        """Validate LLM proposals for correctness and safety.
        
        Args:
            proposals: List of DSL programs to validate
            
        Returns:
            List of valid DSL programs
        """
        valid_proposals = []
        
        for i, proposal in enumerate(proposals):
            try:
                # Validate program structure
                is_valid, error = self.dsl_engine.validate_program(proposal)
                if not is_valid:
                    logger.debug(f"Proposal {i+1} invalid: {error}")
                    continue
                
                # Check program length
                if len(proposal.operations) > self.dsl_engine.max_program_length:
                    logger.debug(f"Proposal {i+1} too long: {len(proposal.operations)} operations")
                    continue
                
                # Check for duplicate operations (simple heuristic)
                if len(proposal.operations) > 1:
                    op_names = [op.primitive_name for op in proposal.operations]
                    if len(set(op_names)) < len(op_names) / 2:  # Too many duplicates
                        logger.debug(f"Proposal {i+1} has too many duplicate operations")
                        continue
                
                valid_proposals.append(proposal)
                
            except Exception as e:
                logger.debug(f"Error validating proposal {i+1}: {e}")
                continue
        
        logger.info(f"Validated {len(valid_proposals)}/{len(proposals)} LLM proposals")
        return valid_proposals


def create_llm_integrated_searcher(llm_enabled: bool = True,
                                 llm_model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                                 original_beam_width: int = 64,
                                 llm_beam_width: int = 8) -> LLMIntegratedSearcher:
    """Factory function to create LLM-integrated searcher.
    
    Args:
        llm_enabled: Whether to enable LLM integration
        llm_model_name: Name of the LLM model
        original_beam_width: Original beam width for vanilla search
        llm_beam_width: Reduced beam width for LLM-guided search
        
    Returns:
        Configured LLMIntegratedSearcher
    """
    config = LLMIntegrationConfig(
        llm_enabled=llm_enabled,
        llm_model_name=llm_model_name,
        original_beam_width=original_beam_width,
        llm_beam_width=llm_beam_width
    )
    
    return LLMIntegratedSearcher(config)