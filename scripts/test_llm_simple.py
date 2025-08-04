#!/usr/bin/env python3
"""Simple test for LLM integration functionality."""

import sys
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.search.llm_integration import create_llm_integrated_searcher
from arc_solver.llm.llm_proposer import LLMConfig, ProposalResult
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation

# Configure logging
logging.basicConfig(level=logging.ERROR)  # Reduce noise
logger = logging.getLogger(__name__)


class SimpleMockLLM:
    """Simple mock LLM for testing."""
    
    def __init__(self, config=None):
        self.config = config or LLMConfig()
    
    def generate_proposals(self, input_grid, target_grid, input_blobs, target_blobs):
        """Generate a single correct proposal."""
        # For 90-degree rotation
        if np.array_equal(target_grid, np.rot90(input_grid)):
            return ProposalResult(
                success=True,
                proposals=[DSLProgram([DSLOperation('Rotate90', {})])],
                raw_responses=["Rotate90"],
                parsing_success_rate=1.0,
                generation_time=0.001
            )
        
        # Default: no proposals
        return ProposalResult(
            success=True,
            proposals=[],
            raw_responses=[],
            parsing_success_rate=0.0,
            generation_time=0.001
        )
    
    def get_stats(self):
        return {'model_loaded': True}


def test_llm_integration():
    """Test basic LLM integration."""
    print("Testing LLM integration...")
    
    # Create searcher
    searcher = create_llm_integrated_searcher(
        llm_enabled=True,
        original_beam_width=64,
        llm_beam_width=8
    )
    
    # Replace with simple mock
    searcher.llm_proposer = SimpleMockLLM()
    
    # Test case: 90-degree rotation
    input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    target_grid = np.rot90(input_grid)
    
    # Perform search
    result = searcher.search(input_grid, target_grid)
    
    print(f"Success: {result.success}")
    print(f"LLM used: {result.llm_used}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Beam width: {result.beam_width_used}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    
    # Verify results
    assert result.success, "Search should succeed"
    assert result.llm_used, "LLM should be used"
    assert result.beam_width_used == 8, f"Expected beam width 8, got {result.beam_width_used}"
    
    if result.termination_reason == "llm_direct_solution":
        print("✓ LLM provided direct solution")
        assert result.nodes_expanded == 0, "Direct solution should not expand nodes"
    else:
        print("✓ LLM guided search to solution")
        assert result.nodes_expanded > 0, "Guided search should expand some nodes"
    
    print("✓ LLM integration test passed!")
    return result


def test_fallback():
    """Test fallback to vanilla search."""
    print("\nTesting fallback mechanism...")
    
    # Create searcher with LLM disabled
    searcher = create_llm_integrated_searcher(llm_enabled=False)
    
    input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    target_grid = np.rot90(input_grid)
    
    result = searcher.search(input_grid, target_grid)
    
    print(f"Success: {result.success}")
    print(f"LLM used: {result.llm_used}")
    print(f"Fallback used: {result.fallback_used}")
    
    assert not result.llm_used, "LLM should not be used when disabled"
    print("✓ Fallback test passed!")
    
    return result


def main():
    """Run simple LLM integration tests."""
    try:
        # Test 1: LLM integration
        llm_result = test_llm_integration()
        
        # Test 2: Fallback
        fallback_result = test_fallback()
        
        print("\n" + "="*50)
        print("All tests passed!")
        print(f"LLM integration: {'✓' if llm_result.llm_used else '✗'}")
        print(f"Fallback mechanism: {'✓' if not fallback_result.llm_used else '✗'}")
        print(f"Beam width reduction: {'✓' if llm_result.beam_width_used == 8 else '✗'}")
        
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())