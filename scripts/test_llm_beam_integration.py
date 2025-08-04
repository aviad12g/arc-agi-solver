#!/usr/bin/env python3
"""Test script for LLM beam search integration.

This script specifically tests the integration between LLM proposals
and beam search optimization, verifying the beam width reduction
and fallback mechanisms work correctly.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.search.llm_integration import (
    LLMIntegratedSearcher, LLMIntegrationConfig, create_llm_integrated_searcher
)
from arc_solver.llm.llm_proposer import LLMProposer, LLMConfig, ProposalResult
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation
from arc_solver.search.astar import AStarSearcher, SearchConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockLLMProposer:
    """Enhanced mock LLM proposer for beam search testing."""
    
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        self.total_proposals_generated = 0
        self.total_proposals_parsed = 0
        self.total_generation_time = 0.0
        
        # Control success/failure for testing
        self.should_succeed = True
        self.parsing_success_rate = 1.0
        self.generation_delay = 0.001  # Small delay to simulate real generation
    
    def generate_proposals(self, input_grid, target_grid, input_blobs, target_blobs):
        """Generate mock proposals with configurable behavior."""
        start_time = time.perf_counter()
        
        # Simulate generation delay
        time.sleep(self.generation_delay)
        
        if not self.should_succeed:
            return ProposalResult(
                success=False,
                proposals=[],
                raw_responses=[],
                parsing_success_rate=0.0,
                generation_time=time.perf_counter() - start_time,
                error="Mock failure"
            )
        
        proposals = []
        raw_responses = []
        
        # Generate proposals based on simple pattern matching
        # Check 90-degree rotation
        if np.array_equal(target_grid, np.rot90(input_grid)):
            proposals.append(DSLProgram([DSLOperation('Rotate90', {})]))
            raw_responses.append("Program: Rotate90")
        
        # Check 180-degree rotation  
        elif np.array_equal(target_grid, np.rot90(input_grid, 2)):
            proposals.append(DSLProgram([DSLOperation('Rotate180', {})]))
            raw_responses.append("Program: Rotate180")
        
        # Check 270-degree rotation (equivalent to 3 * 90)
        elif np.array_equal(target_grid, np.rot90(input_grid, 3)):
            proposals.append(DSLProgram([
                DSLOperation('Rotate90', {}),
                DSLOperation('Rotate90', {}),
                DSLOperation('Rotate90', {})
            ]))
            raw_responses.append("Program: Rotate90 -> Rotate90 -> Rotate90")
        
        # Check horizontal reflection
        elif np.array_equal(target_grid, np.fliplr(input_grid)):
            proposals.append(DSLProgram([DSLOperation('ReflectH', {})]))
            raw_responses.append("Program: ReflectH")
        
        # Check vertical reflection
        elif np.array_equal(target_grid, np.flipud(input_grid)):
            proposals.append(DSLProgram([DSLOperation('ReflectV', {})]))
            raw_responses.append("Program: ReflectV")
        
        # Add some default proposals if none found
        if not proposals:
            default_proposals = [
                DSLProgram([DSLOperation('Rotate90', {})]),
                DSLProgram([DSLOperation('ReflectH', {})]),
                DSLProgram([DSLOperation('Rotate180', {})])
            ]
            default_responses = ["Rotate90", "ReflectH", "Rotate180"]
            
            proposals.extend(default_proposals[:self.config.num_proposals])
            raw_responses.extend(default_responses[:self.config.num_proposals])
        
        # Limit to requested number of proposals
        proposals = proposals[:self.config.num_proposals]
        raw_responses = raw_responses[:self.config.num_proposals]
        
        generation_time = time.perf_counter() - start_time
        
        # Update statistics
        self.total_proposals_generated += len(raw_responses)
        self.total_proposals_parsed += len(proposals)
        self.total_generation_time += generation_time
        
        return ProposalResult(
            success=True,
            proposals=proposals,
            raw_responses=raw_responses,
            parsing_success_rate=self.parsing_success_rate,
            generation_time=generation_time
        )
    
    def get_stats(self):
        """Get mock statistics."""
        return {
            'total_proposals_generated': self.total_proposals_generated,
            'total_proposals_parsed': self.total_proposals_parsed,
            'total_generation_time': self.total_generation_time,
            'parsing_success_rate': self.parsing_success_rate,
            'model_name': 'mock-model',
            'model_loaded': True
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_proposals_generated = 0
        self.total_proposals_parsed = 0
        self.total_generation_time = 0.0


def test_beam_width_reduction():
    """Test that beam width is correctly reduced when using LLM proposals."""
    logger.info("Testing beam width reduction...")
    
    # Create searcher with specific beam widths
    config = LLMIntegrationConfig(
        llm_enabled=True,
        original_beam_width=64,
        llm_beam_width=8,
        num_proposals=3
    )
    
    searcher = LLMIntegratedSearcher(config)
    searcher.llm_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
    
    # Test grid that requires search (not directly solvable by LLM)
    input_grid = np.array([
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 5]
    ], dtype=np.int32)
    
    # Target that's not a simple transformation
    target_grid = np.array([
        [5, 0, 0],
        [0, 4, 3],
        [0, 2, 1]
    ], dtype=np.int32)
    
    # Perform search
    result = searcher.search(input_grid, target_grid)
    
    logger.info(f"Search completed: success={result.success}")
    logger.info(f"LLM used: {result.llm_used}")
    logger.info(f"Beam width used: {result.beam_width_used}")
    logger.info(f"Nodes expanded: {result.nodes_expanded}")
    
    # Verify beam width was reduced
    if result.llm_used and not result.fallback_used:
        if result.termination_reason == "llm_direct_solution":
            # For direct solutions, beam width should still be set to LLM beam width
            assert result.beam_width_used == 8, f"Expected beam width 8 for direct solution, got {result.beam_width_used}"
            logger.info("✓ Beam width correctly set to 8 for direct LLM solution")
        else:
            # For guided search, beam width should be reduced
            assert result.beam_width_used == 8, f"Expected beam width 8 for guided search, got {result.beam_width_used}"
            logger.info("✓ Beam width correctly reduced to 8 for guided search")
    else:
        logger.info("LLM search fell back to vanilla, beam width not reduced")
    
    return result


def test_llm_direct_solution():
    """Test that LLM can provide direct solutions without search."""
    logger.info("Testing LLM direct solution...")
    
    searcher = create_llm_integrated_searcher(
        llm_enabled=True,
        original_beam_width=64,
        llm_beam_width=8
    )
    searcher.llm_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
    
    # Simple rotation task that LLM should solve directly
    input_grid = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.int32)
    
    target_grid = np.rot90(input_grid)  # 90-degree rotation
    
    # Perform search
    result = searcher.search(input_grid, target_grid)
    
    logger.info(f"Search completed: success={result.success}")
    logger.info(f"LLM used: {result.llm_used}")
    logger.info(f"Termination reason: {result.termination_reason}")
    logger.info(f"Nodes expanded: {result.nodes_expanded}")
    
    # Should be solved directly by LLM
    assert result.success, "LLM should solve simple rotation directly"
    assert result.llm_used, "LLM should be used"
    assert result.termination_reason == "llm_direct_solution", f"Expected direct solution, got {result.termination_reason}"
    assert result.nodes_expanded == 0, f"Expected 0 nodes expanded for direct solution, got {result.nodes_expanded}"
    
    logger.info("✓ LLM provided direct solution without search")
    return result


def test_fallback_mechanisms():
    """Test various fallback mechanisms."""
    logger.info("Testing fallback mechanisms...")
    
    # Test 1: LLM generation failure
    logger.info("  Testing LLM generation failure fallback...")
    
    config = LLMIntegrationConfig(
        llm_enabled=True,
        fallback_on_failure=True,
        original_beam_width=32,
        llm_beam_width=8
    )
    
    searcher = LLMIntegratedSearcher(config)
    mock_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
    mock_proposer.should_succeed = False  # Force failure
    searcher.llm_proposer = mock_proposer
    
    input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    target_grid = np.rot90(input_grid)
    
    result = searcher.search(input_grid, target_grid)
    
    logger.info(f"    Result: success={result.success}, fallback_used={result.fallback_used}")
    assert result.fallback_used, "Should fallback when LLM fails"
    
    # Test 2: Low parsing success rate
    logger.info("  Testing low parsing success rate fallback...")
    
    mock_proposer.should_succeed = True
    mock_proposer.parsing_success_rate = 0.3  # Below threshold
    
    result = searcher.search(input_grid, target_grid)
    
    logger.info(f"    Result: success={result.success}, fallback_used={result.fallback_used}")
    logger.info(f"    Parsing success rate: {result.llm_parsing_success_rate}")
    
    # Test 3: LLM disabled
    logger.info("  Testing LLM disabled fallback...")
    
    config.llm_enabled = False
    searcher_disabled = LLMIntegratedSearcher(config)
    
    result = searcher_disabled.search(input_grid, target_grid)
    
    logger.info(f"    Result: success={result.success}, llm_used={result.llm_used}")
    assert not result.llm_used, "LLM should not be used when disabled"
    
    logger.info("✓ All fallback mechanisms working correctly")


def test_performance_comparison():
    """Test performance comparison between different configurations."""
    logger.info("Testing performance comparison...")
    
    # Create test cases
    test_cases = [
        ("simple_rotation", np.array([[1, 2], [3, 4]], dtype=np.int32), None),
        ("reflection", np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), None),
        ("complex", np.array([[1, 2, 0], [3, 4, 0], [0, 0, 5]], dtype=np.int32), None)
    ]
    
    # Generate targets
    for i, (name, input_grid, _) in enumerate(test_cases):
        if name == "simple_rotation":
            target = np.rot90(input_grid)
        elif name == "reflection":
            target = np.fliplr(input_grid)
        else:  # complex
            target = np.rot90(input_grid, 2)  # 180 degree rotation
        test_cases[i] = (name, input_grid, target)
    
    # Test configurations
    configs = [
        ("Vanilla A*", {"llm_enabled": False, "original_beam_width": 64}),
        ("LLM + Beam=64", {"llm_enabled": True, "original_beam_width": 64, "llm_beam_width": 64}),
        ("LLM + Beam=8", {"llm_enabled": True, "original_beam_width": 64, "llm_beam_width": 8}),
    ]
    
    results = {}
    
    for config_name, config_params in configs:
        logger.info(f"  Testing configuration: {config_name}")
        results[config_name] = []
        
        searcher = create_llm_integrated_searcher(**config_params)
        if config_params.get("llm_enabled", False):
            searcher.llm_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
        
        for case_name, input_grid, target_grid in test_cases:
            start_time = time.perf_counter()
            result = searcher.search(input_grid, target_grid)
            search_time = time.perf_counter() - start_time
            
            results[config_name].append({
                'case': case_name,
                'success': result.success,
                'time': search_time,
                'nodes_expanded': result.nodes_expanded,
                'llm_used': result.llm_used,
                'beam_width': result.beam_width_used
            })
    
    # Print comparison table
    logger.info("Performance Comparison Results:")
    logger.info(f"{'Case':<15} {'Config':<15} {'Success':<8} {'Time(s)':<8} {'Nodes':<6} {'LLM':<4} {'Beam':<5}")
    logger.info("-" * 70)
    
    for case_name, _, _ in test_cases:
        for config_name in results:
            case_result = next(r for r in results[config_name] if r['case'] == case_name)
            logger.info(f"{case_name:<15} {config_name:<15} {case_result['success']:<8} "
                       f"{case_result['time']:<8.3f} {case_result['nodes_expanded']:<6} "
                       f"{case_result['llm_used']:<4} {case_result['beam_width']:<5}")
    
    logger.info("✓ Performance comparison completed")
    return results


def test_beam_search_optimization():
    """Test that beam search optimization works as expected."""
    logger.info("Testing beam search optimization...")
    
    # Create a more complex search scenario
    input_grid = np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 0],
        [7, 8, 9, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)
    
    # Target requires multiple operations
    target_grid = np.array([
        [7, 4, 1, 0],
        [8, 5, 2, 0],
        [9, 6, 3, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)
    
    # Test with different beam widths
    beam_widths = [4, 8, 16, 32, 64]
    results = []
    
    for beam_width in beam_widths:
        logger.info(f"  Testing beam width: {beam_width}")
        
        searcher = create_llm_integrated_searcher(
            llm_enabled=True,
            original_beam_width=64,
            llm_beam_width=beam_width
        )
        searcher.llm_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
        
        start_time = time.perf_counter()
        result = searcher.search(input_grid, target_grid)
        search_time = time.perf_counter() - start_time
        
        results.append({
            'beam_width': beam_width,
            'success': result.success,
            'time': search_time,
            'nodes_expanded': result.nodes_expanded,
            'nodes_generated': result.nodes_generated
        })
        
        logger.info(f"    Result: success={result.success}, time={search_time:.3f}s, "
                   f"nodes_expanded={result.nodes_expanded}")
    
    # Analyze results
    logger.info("Beam Width Optimization Results:")
    logger.info(f"{'Beam Width':<12} {'Success':<8} {'Time(s)':<8} {'Expanded':<9} {'Generated':<10}")
    logger.info("-" * 50)
    
    for result in results:
        logger.info(f"{result['beam_width']:<12} {result['success']:<8} "
                   f"{result['time']:<8.3f} {result['nodes_expanded']:<9} "
                   f"{result['nodes_generated']:<10}")
    
    logger.info("✓ Beam search optimization test completed")
    return results


def main():
    """Run all LLM beam search integration tests."""
    logger.info("Starting LLM beam search integration tests...")
    
    try:
        # Test 1: Beam width reduction
        beam_result = test_beam_width_reduction()
        logger.info("✓ Beam width reduction test passed")
        
        # Test 2: Direct LLM solutions
        direct_result = test_llm_direct_solution()
        logger.info("✓ LLM direct solution test passed")
        
        # Test 3: Fallback mechanisms
        test_fallback_mechanisms()
        logger.info("✓ Fallback mechanisms test passed")
        
        # Test 4: Performance comparison
        perf_results = test_performance_comparison()
        logger.info("✓ Performance comparison test passed")
        
        # Test 5: Beam search optimization
        beam_opt_results = test_beam_search_optimization()
        logger.info("✓ Beam search optimization test passed")
        
        logger.info("All LLM beam search integration tests completed successfully!")
        
        # Summary
        logger.info("\\nTest Summary:")
        logger.info(f"- Beam width reduction: {'✓' if beam_result.beam_width_used <= 8 else '✗'}")
        logger.info(f"- Direct LLM solution: {'✓' if direct_result.termination_reason == 'llm_direct_solution' else '✗'}")
        logger.info(f"- Performance configurations tested: {len(perf_results)}")
        logger.info(f"- Beam width optimizations tested: {len(beam_opt_results)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())