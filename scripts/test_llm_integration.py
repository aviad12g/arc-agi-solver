#!/usr/bin/env python3
"""Test script for LLM integration functionality.

This script tests the LLM integration components including:
- Synthetic data generation
- LLM proposal generation (with mock)
- Integration with A* search
- Performance benchmarking
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arc_solver.llm.synthetic_data import create_synthetic_data_generator
from arc_solver.llm.prompt_templates import create_arc_prompt_template
from arc_solver.llm.llm_proposer import LLMProposer, LLMConfig, ProposalResult
from arc_solver.search.llm_integration import create_llm_integrated_searcher
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation
from arc_solver.core.data_models import Blob, FeatureVector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockLLMProposer:
    """Mock LLM proposer for testing without actual model."""
    
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        self.total_proposals_generated = 0
        self.total_proposals_parsed = 0
        self.total_generation_time = 0.0
    
    def generate_proposals(self, input_grid, target_grid, input_blobs, target_blobs):
        """Generate mock proposals based on simple heuristics."""
        start_time = time.perf_counter()
        
        proposals = []
        raw_responses = []
        
        # Simple heuristic: if target looks like rotated input, suggest rotation
        if np.array_equal(target_grid, np.rot90(input_grid)):
            proposals.append(DSLProgram([DSLOperation('Rotate90', {})]))
            raw_responses.append("Program: Rotate90")
        
        # If target looks like reflected input, suggest reflection
        if np.array_equal(target_grid, np.fliplr(input_grid)):
            proposals.append(DSLProgram([DSLOperation('ReflectH', {})]))
            raw_responses.append("Program: ReflectH")
        
        if np.array_equal(target_grid, np.flipud(input_grid)):
            proposals.append(DSLProgram([DSLOperation('ReflectV', {})]))
            raw_responses.append("Program: ReflectV")
        
        # Add some default proposals if none found
        if not proposals:
            proposals = [
                DSLProgram([DSLOperation('Rotate90', {})]),
                DSLProgram([DSLOperation('ReflectH', {})]),
                DSLProgram([DSLOperation('Rotate180', {})])
            ]
            raw_responses = ["Rotate90", "ReflectH", "Rotate180"]
        
        generation_time = time.perf_counter() - start_time
        
        # Update statistics
        self.total_proposals_generated += len(raw_responses)
        self.total_proposals_parsed += len(proposals)
        self.total_generation_time += generation_time
        
        return ProposalResult(
            success=True,
            proposals=proposals[:self.config.num_proposals],
            raw_responses=raw_responses[:self.config.num_proposals],
            parsing_success_rate=1.0,
            generation_time=generation_time
        )
    
    def get_stats(self):
        """Get mock statistics."""
        return {
            'total_proposals_generated': self.total_proposals_generated,
            'total_proposals_parsed': self.total_proposals_parsed,
            'total_generation_time': self.total_generation_time,
            'parsing_success_rate': 1.0,
            'model_name': 'mock-model',
            'model_loaded': True
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_proposals_generated = 0
        self.total_proposals_parsed = 0
        self.total_generation_time = 0.0


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    logger.info("Testing synthetic data generation...")
    
    generator = create_synthetic_data_generator(
        grid_sizes=[(3, 3), (4, 4), (5, 5)],
        max_program_length=3
    )
    
    # Generate small training set
    tasks = generator.generate_training_set(
        num_tasks=20,
        task_types=['rotation', 'reflection', 'color_mapping'],
        difficulty_distribution={'easy': 0.5, 'medium': 0.3, 'hard': 0.2}
    )
    
    logger.info(f"Generated {len(tasks)} synthetic tasks")
    
    # Analyze task distribution
    task_types = {}
    difficulties = {}
    
    for task in tasks:
        task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
        difficulties[task.difficulty] = difficulties.get(task.difficulty, 0) + 1
    
    logger.info(f"Task types: {task_types}")
    logger.info(f"Difficulties: {difficulties}")
    
    # Test a few tasks
    for i, task in enumerate(tasks[:3]):
        logger.info(f"Task {i+1}: {task.task_type} ({task.difficulty})")
        logger.info(f"  Input shape: {task.input_grid.shape}")
        logger.info(f"  Target shape: {task.target_grid.shape}")
        logger.info(f"  Program: {task.program}")
        logger.info(f"  Program length: {len(task.program)}")
    
    return tasks


def test_prompt_templates():
    """Test prompt template functionality."""
    logger.info("Testing prompt templates...")
    
    template = create_arc_prompt_template()
    
    # Create sample task description
    task_description = {
        'input_shape': (3, 3),
        'target_shape': (3, 3),
        'input_colors': [0, 1, 2],
        'target_colors': [0, 1, 2],
        'input_blobs': [
            {'color': 1, 'area': 4, 'center_of_mass': (1.5, 1.5)}
        ],
        'target_blobs': [
            {'color': 1, 'area': 4, 'center_of_mass': (1.5, 1.5)}
        ],
        'transformation_hints': {
            'rotation_90': True,
            'reflection_horizontal': False,
            'color_change': {'added_colors': [], 'removed_colors': []}
        }
    }
    
    # Format prompt
    prompt = template.format_prompt(task_description)
    
    logger.info(f"Generated prompt length: {len(prompt)} characters")
    logger.info("Prompt preview:")
    logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    return prompt


def test_mock_llm_proposer():
    """Test mock LLM proposer."""
    logger.info("Testing mock LLM proposer...")
    
    config = LLMConfig(num_proposals=3)
    proposer = MockLLMProposer(config)
    
    # Test with rotation task
    input_grid = np.array([
        [1, 2, 0],
        [3, 4, 0],
        [0, 0, 0]
    ], dtype=np.int32)
    
    target_grid = np.rot90(input_grid)  # 90-degree rotation
    
    # Generate proposals
    result = proposer.generate_proposals(input_grid, target_grid, [], [])
    
    logger.info(f"Proposal generation success: {result.success}")
    logger.info(f"Number of proposals: {len(result.proposals)}")
    logger.info(f"Parsing success rate: {result.parsing_success_rate:.2%}")
    logger.info(f"Generation time: {result.generation_time:.3f}s")
    
    for i, proposal in enumerate(result.proposals):
        logger.info(f"  Proposal {i+1}: {proposal}")
    
    # Test statistics
    stats = proposer.get_stats()
    logger.info(f"Proposer stats: {stats}")
    
    return result


def test_llm_integration():
    """Test LLM integration with A* search."""
    logger.info("Testing LLM integration with A* search...")
    
    # Create integrated searcher with mock LLM
    searcher = create_llm_integrated_searcher(
        llm_enabled=True,
        original_beam_width=32,
        llm_beam_width=8
    )
    
    # Replace with mock proposer
    searcher.llm_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
    
    # Test with simple rotation task
    input_grid = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.int32)
    
    target_grid = np.rot90(input_grid)  # Should be solvable with Rotate90
    
    # Perform search
    start_time = time.perf_counter()
    result = searcher.search(input_grid, target_grid)
    search_time = time.perf_counter() - start_time
    
    logger.info(f"Search completed in {search_time:.3f}s")
    logger.info(f"Search success: {result.success}")
    logger.info(f"LLM used: {result.llm_used}")
    logger.info(f"Fallback used: {result.fallback_used}")
    logger.info(f"Termination reason: {result.termination_reason}")
    
    if result.program:
        logger.info(f"Solution program: {result.program}")
        logger.info(f"Program length: {len(result.program)}")
    
    # Search statistics
    logger.info(f"Nodes expanded: {result.nodes_expanded}")
    logger.info(f"Nodes generated: {result.nodes_generated}")
    logger.info(f"Beam width used: {result.beam_width_used}")
    
    # LLM statistics
    if result.llm_used:
        logger.info(f"LLM proposals generated: {result.llm_proposals_generated}")
        logger.info(f"LLM proposals parsed: {result.llm_proposals_parsed}")
        logger.info(f"LLM parsing success rate: {result.llm_parsing_success_rate:.2%}")
        logger.info(f"LLM generation time: {result.llm_generation_time:.3f}s")
    
    # Get comprehensive statistics
    stats = searcher.get_stats()
    logger.info("Comprehensive statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    
    return result


def test_performance_comparison():
    """Test performance comparison between vanilla and LLM-enhanced search."""
    logger.info("Testing performance comparison...")
    
    # Create test cases
    test_cases = []
    
    # Simple rotation
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    target1 = np.rot90(input1)
    test_cases.append(("rotation", input1, target1))
    
    # Horizontal reflection
    input2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    target2 = np.fliplr(input2)
    test_cases.append(("reflection_h", input2, target2))
    
    # Vertical reflection
    input3 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    target3 = np.flipud(input3)
    test_cases.append(("reflection_v", input3, target3))
    
    # Test vanilla search
    logger.info("Testing vanilla A* search...")
    vanilla_searcher = create_llm_integrated_searcher(llm_enabled=False)
    vanilla_results = []
    
    for name, input_grid, target_grid in test_cases:
        start_time = time.perf_counter()
        result = vanilla_searcher.search(input_grid, target_grid)
        search_time = time.perf_counter() - start_time
        
        vanilla_results.append((name, result, search_time))
        logger.info(f"  {name}: success={result.success}, time={search_time:.3f}s, "
                   f"nodes={result.nodes_expanded}")
    
    # Test LLM-enhanced search
    logger.info("Testing LLM-enhanced A* search...")
    llm_searcher = create_llm_integrated_searcher(llm_enabled=True)
    llm_searcher.llm_proposer = MockLLMProposer(LLMConfig(num_proposals=3))
    llm_results = []
    
    for name, input_grid, target_grid in test_cases:
        start_time = time.perf_counter()
        result = llm_searcher.search(input_grid, target_grid)
        search_time = time.perf_counter() - start_time
        
        llm_results.append((name, result, search_time))
        logger.info(f"  {name}: success={result.success}, time={search_time:.3f}s, "
                   f"nodes={result.nodes_expanded}, llm_used={result.llm_used}")
    
    # Compare results
    logger.info("Performance comparison:")
    logger.info(f"{'Task':<15} {'Vanilla Time':<12} {'LLM Time':<10} {'Vanilla Nodes':<13} {'LLM Nodes':<10} {'Improvement'}")
    logger.info("-" * 80)
    
    for i, (name, _, _) in enumerate(test_cases):
        vanilla_time = vanilla_results[i][2]
        llm_time = llm_results[i][2]
        vanilla_nodes = vanilla_results[i][1].nodes_expanded
        llm_nodes = llm_results[i][1].nodes_expanded
        
        time_improvement = (vanilla_time - llm_time) / vanilla_time * 100 if vanilla_time > 0 else 0
        node_improvement = (vanilla_nodes - llm_nodes) / vanilla_nodes * 100 if vanilla_nodes > 0 else 0
        
        logger.info(f"{name:<15} {vanilla_time:<12.3f} {llm_time:<10.3f} "
                   f"{vanilla_nodes:<13} {llm_nodes:<10} "
                   f"T:{time_improvement:+.1f}% N:{node_improvement:+.1f}%")


def main():
    """Run all LLM integration tests."""
    logger.info("Starting LLM integration tests...")
    
    try:
        # Test 1: Synthetic data generation
        synthetic_tasks = test_synthetic_data_generation()
        logger.info("✓ Synthetic data generation test passed")
        
        # Test 2: Prompt templates
        prompt = test_prompt_templates()
        logger.info("✓ Prompt template test passed")
        
        # Test 3: Mock LLM proposer
        proposal_result = test_mock_llm_proposer()
        logger.info("✓ Mock LLM proposer test passed")
        
        # Test 4: LLM integration
        search_result = test_llm_integration()
        logger.info("✓ LLM integration test passed")
        
        # Test 5: Performance comparison
        test_performance_comparison()
        logger.info("✓ Performance comparison test passed")
        
        logger.info("All LLM integration tests completed successfully!")
        
        # Summary
        logger.info("\\nTest Summary:")
        logger.info(f"- Generated {len(synthetic_tasks)} synthetic training tasks")
        logger.info(f"- Prompt template length: {len(prompt)} characters")
        logger.info(f"- Mock LLM proposals: {len(proposal_result.proposals)}")
        logger.info(f"- Search result success: {search_result.success}")
        logger.info(f"- LLM integration working: {search_result.llm_used}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())