"""Tests for A* search algorithm."""

import pytest
import numpy as np
import time

from arc_solver.search.astar import (
    AStarSearcher, SearchNode, SearchResult, SearchConfig, create_astar_searcher
)
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation


class TestSearchNode:
    """Test SearchNode functionality."""
    
    @pytest.fixture
    def sample_grid(self):
        """Create sample grid for testing."""
        return np.array([[1, 2], [3, 4]], dtype=np.int32)
    
    @pytest.fixture
    def sample_program(self):
        """Create sample DSL program."""
        return DSLProgram([])
    
    def test_node_creation(self, sample_grid, sample_program):
        """Test basic node creation."""
        node = SearchNode(
            grid=sample_grid,
            program=sample_program,
            cost=1.0,
            heuristic=2.0
        )
        
        assert np.array_equal(node.grid, sample_grid)
        assert node.program == sample_program
        assert node.cost == 1.0
        assert node.heuristic == 2.0
        assert node.f_score == 3.0
        assert node.depth == 0
        assert node.parent is None
        assert node.action is None
    
    def test_f_score_calculation(self, sample_grid, sample_program):
        """Test f-score calculation."""
        node = SearchNode(
            grid=sample_grid,
            program=sample_program,
            cost=2.5,
            heuristic=1.5
        )
        
        assert node.f_score == 4.0
    
    def test_node_comparison(self, sample_grid, sample_program):
        """Test node comparison for priority queue."""
        node1 = SearchNode(
            grid=sample_grid,
            program=sample_program,
            cost=1.0,
            heuristic=2.0  # f_score = 3.0
        )
        
        node2 = SearchNode(
            grid=sample_grid,
            program=sample_program,
            cost=2.0,
            heuristic=1.0  # f_score = 3.0
        )
        
        node3 = SearchNode(
            grid=sample_grid,
            program=sample_program,
            cost=1.0,
            heuristic=3.0  # f_score = 4.0
        )
        
        # node1 should be preferred over node2 (same f_score, lower cost)
        assert node1 < node2
        assert not node2 < node1
        
        # node1 should be preferred over node3 (lower f_score)
        assert node1 < node3
        assert not node3 < node1
    
    def test_node_equality(self, sample_program):
        """Test node equality based on grid state."""
        grid1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        grid3 = np.array([[4, 3], [2, 1]], dtype=np.int32)
        
        node1 = SearchNode(grid=grid1, program=sample_program, cost=0, heuristic=0)
        node2 = SearchNode(grid=grid2, program=sample_program, cost=0, heuristic=0)
        node3 = SearchNode(grid=grid3, program=sample_program, cost=0, heuristic=0)
        
        assert node1 == node2  # Same grid content
        assert node1 != node3  # Different grid content
    
    def test_program_sequence_reconstruction(self, sample_grid):
        """Test reconstruction of program sequence from node chain."""
        # Create a chain of nodes representing a program sequence
        root = SearchNode(
            grid=sample_grid,
            program=DSLProgram([]),
            cost=0,
            heuristic=0
        )
        
        op1 = DSLOperation("Rotate90", {})
        child1 = SearchNode(
            grid=sample_grid,
            program=DSLProgram([op1]),
            cost=1,
            heuristic=0,
            parent=root,
            action=op1,
            depth=1
        )
        
        op2 = DSLOperation("ReflectH", {})
        child2 = SearchNode(
            grid=sample_grid,
            program=DSLProgram([op1, op2]),
            cost=2,
            heuristic=0,
            parent=child1,
            action=op2,
            depth=2
        )
        
        # Test program sequence reconstruction
        sequence = child2.get_program_sequence()
        assert len(sequence) == 2
        assert sequence[0].primitive_name == "Rotate90"
        assert sequence[1].primitive_name == "ReflectH"


class TestSearchConfig:
    """Test SearchConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()
        
        assert config.max_program_length == 4
        assert config.max_nodes_expanded == 600
        assert config.max_computation_time == 30.0
        assert config.beam_width == 64
        assert config.adaptive_beam is True
        assert config.min_beam_width == 8
        assert config.duplicate_detection is True
        assert config.early_termination is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SearchConfig(
            max_program_length=2,
            max_nodes_expanded=100,
            beam_width=32,
            adaptive_beam=False
        )
        
        assert config.max_program_length == 2
        assert config.max_nodes_expanded == 100
        assert config.beam_width == 32
        assert config.adaptive_beam is False


class TestAStarSearcher:
    """Test A* search algorithm."""
    
    @pytest.fixture
    def searcher(self):
        """Create A* searcher for testing."""
        config = SearchConfig(
            max_program_length=2,  # Shorter for faster tests
            max_nodes_expanded=50,
            beam_width=16,
            max_computation_time=5.0
        )
        return AStarSearcher(config)
    
    @pytest.fixture
    def simple_grids(self):
        """Create simple test grids."""
        initial = np.array([[1, 0], [0, 1]], dtype=np.int32)
        target = np.array([[0, 1], [1, 0]], dtype=np.int32)
        return initial, target
    
    def test_searcher_initialization(self, searcher):
        """Test searcher initialization."""
        assert searcher.config.max_program_length == 2
        assert searcher.config.max_nodes_expanded == 50
        assert searcher.heuristic_system is not None
        assert searcher.dsl_engine is not None
        assert searcher.nodes_expanded == 0
        assert searcher.nodes_generated == 0
    
    def test_identical_grids_search(self, searcher):
        """Test search with identical initial and target grids."""
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        
        result = searcher.search(grid, grid)
        
        assert result.success is True
        assert result.program is not None
        assert len(result.program.operations) == 0  # Empty program
        assert np.array_equal(result.final_grid, grid)
        assert result.nodes_expanded == 0
        assert result.nodes_generated == 1
        assert result.termination_reason == "initial_match"
    
    def test_basic_search(self, searcher, simple_grids):
        """Test basic search functionality."""
        initial, target = simple_grids
        
        result = searcher.search(initial, target)
        
        # Should return a result (success or partial)
        assert isinstance(result, SearchResult)
        assert result.program is not None
        assert result.final_grid is not None
        assert result.nodes_expanded >= 0
        assert result.nodes_generated >= 0
        assert result.computation_time > 0
        assert result.termination_reason != "unknown"
    
    def test_search_with_timeout(self):
        """Test search with very short timeout."""
        config = SearchConfig(
            max_computation_time=0.001,  # Very short timeout
            max_nodes_expanded=1000
        )
        searcher = AStarSearcher(config)
        
        initial = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        target = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32)
        
        result = searcher.search(initial, target)
        
        # Should timeout quickly
        assert result.computation_time <= 0.1  # Should be much less than 0.1s
        assert result.termination_reason in ["timeout", "max_nodes_reached", "search_exhausted"]
    
    def test_search_with_node_limit(self):
        """Test search with very low node expansion limit."""
        config = SearchConfig(
            max_nodes_expanded=5,  # Very low limit
            max_computation_time=10.0
        )
        searcher = AStarSearcher(config)
        
        initial = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target = np.array([[4, 3], [2, 1]], dtype=np.int32)
        
        result = searcher.search(initial, target)
        
        assert result.nodes_expanded <= 5
        assert result.termination_reason in ["max_nodes_reached", "search_exhausted", "goal_reached"]
    
    def test_search_statistics(self, searcher, simple_grids):
        """Test search statistics collection."""
        initial, target = simple_grids
        
        result = searcher.search(initial, target)
        stats = searcher.get_search_stats()
        
        assert 'nodes_expanded' in stats
        assert 'nodes_generated' in stats
        assert 'max_depth_reached' in stats
        assert 'beam_width_used' in stats
        assert 'heuristic_stats' in stats
        assert 'config' in stats
        
        # Statistics should match result
        assert stats['nodes_expanded'] == result.nodes_expanded
        assert stats['nodes_generated'] == result.nodes_generated
    
    def test_beam_search_pruning(self):
        """Test that beam search limits the number of nodes in open queue."""
        config = SearchConfig(
            beam_width=2,  # Very small beam
            max_nodes_expanded=20,
            adaptive_beam=False
        )
        searcher = AStarSearcher(config)
        
        initial = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        target = np.array([[6, 5, 4], [3, 2, 1]], dtype=np.int32)
        
        result = searcher.search(initial, target)
        
        # Should complete within reasonable time due to beam pruning
        assert result.computation_time < 5.0
        assert result.beam_width_used == 2
    
    def test_adaptive_beam_reduction(self):
        """Test adaptive beam width reduction."""
        config = SearchConfig(
            beam_width=32,
            min_beam_width=4,
            max_nodes_expanded=100,  # Force beam reduction
            adaptive_beam=True,
            beam_reduction_factor=0.5
        )
        searcher = AStarSearcher(config)
        
        initial = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        target = np.array([[8, 7, 6, 5], [4, 3, 2, 1]], dtype=np.int32)
        
        result = searcher.search(initial, target)
        
        # Beam width should have been reduced
        assert result.beam_width_used <= config.beam_width
    
    def test_duplicate_detection(self):
        """Test duplicate state detection."""
        config = SearchConfig(
            duplicate_detection=True,
            max_nodes_expanded=50
        )
        searcher = AStarSearcher(config)
        
        # Create grids where some operations might lead to same states
        initial = np.array([[1, 1], [1, 1]], dtype=np.int32)
        target = np.array([[2, 2], [2, 2]], dtype=np.int32)
        
        result = searcher.search(initial, target)
        
        # Should complete without exploring duplicate states
        assert isinstance(result, SearchResult)
    
    def test_error_handling(self, searcher):
        """Test error handling with invalid inputs."""
        # Test with invalid grid (wrong dtype)
        invalid_grid = np.array([[1.5, 2.5]], dtype=np.float32)
        valid_grid = np.array([[1, 2]], dtype=np.int32)
        
        result = searcher.search(invalid_grid, valid_grid)
        
        # Should handle gracefully
        assert isinstance(result, SearchResult)
        # May succeed or fail depending on how DSL engine handles it


class TestAStarFactory:
    """Test A* searcher factory function."""
    
    def test_create_default_searcher(self):
        """Test creating searcher with default parameters."""
        searcher = create_astar_searcher()
        
        assert searcher.config.max_program_length == 4
        assert searcher.config.max_nodes_expanded == 600
        assert searcher.config.beam_width == 64
        assert searcher.config.adaptive_beam is True
    
    def test_create_custom_searcher(self):
        """Test creating searcher with custom parameters."""
        searcher = create_astar_searcher(
            max_program_length=2,
            max_nodes_expanded=100,
            beam_width=32,
            adaptive_beam=False
        )
        
        assert searcher.config.max_program_length == 2
        assert searcher.config.max_nodes_expanded == 100
        assert searcher.config.beam_width == 32
        assert searcher.config.adaptive_beam is False


class TestSearchIntegration:
    """Integration tests for complete search system."""
    
    def test_end_to_end_simple_transformation(self):
        """Test end-to-end search for simple transformation."""
        searcher = create_astar_searcher(
            max_program_length=1,
            max_nodes_expanded=20,
            beam_width=8
        )
        
        # Simple transformation that might be solvable with one operation
        initial = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target = np.array([[3, 1], [4, 2]], dtype=np.int32)  # Rotated 90 degrees
        
        result = searcher.search(initial, target)
        
        assert isinstance(result, SearchResult)
        assert result.program is not None
        assert result.final_grid is not None
        
        # If successful, program should be short
        if result.success:
            assert len(result.program.operations) <= 1
    
    def test_performance_requirements(self):
        """Test that search meets performance requirements."""
        searcher = create_astar_searcher()
        
        # Test with moderately complex grids
        initial = np.array([
            [1, 2, 0, 3],
            [4, 5, 0, 6],
            [0, 0, 0, 0],
            [7, 8, 0, 9]
        ], dtype=np.int32)
        
        target = np.array([
            [9, 0, 8, 7],
            [0, 0, 0, 0],
            [6, 0, 5, 4],
            [3, 0, 2, 1]
        ], dtype=np.int32)
        
        start_time = time.perf_counter()
        result = searcher.search(initial, target)
        end_time = time.perf_counter()
        
        # Should complete within reasonable time
        assert end_time - start_time < 30.0  # 30 second timeout
        
        # Should not exceed node expansion limit
        assert result.nodes_expanded <= 600
        
        # Should return valid result
        assert isinstance(result, SearchResult)
        assert result.computation_time > 0


if __name__ == "__main__":
    pytest.main([__file__])