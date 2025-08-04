"""Tests for LLM integration module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from arc_solver.llm.llm_proposer import LLMProposer, LLMConfig, ProposalResult
from arc_solver.llm.prompt_templates import create_arc_prompt_template
from arc_solver.llm.synthetic_data import SyntheticDataGenerator, SyntheticTask
from arc_solver.search.llm_integration import LLMIntegratedSearcher, LLMIntegrationConfig
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation
from arc_solver.core.data_models import Blob, FeatureVector


class TestLLMProposer:
    """Test LLM proposer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock LLM configuration."""
        return LLMConfig(
            model_name="mock-model",
            max_tokens=128,
            num_proposals=2,
            use_4bit_quantization=False
        )
    
    @pytest.fixture
    def sample_grids(self):
        """Create sample grids for testing."""
        input_grid = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.int32)
        
        target_grid = np.array([
            [3, 1],
            [4, 2]
        ], dtype=np.int32)
        
        return input_grid, target_grid
    
    @pytest.fixture
    def sample_blobs(self):
        """Create sample blobs for testing."""
        feature_vector = FeatureVector(
            orbit_signature=np.zeros(8, dtype=np.float32),
            spectral_features=np.zeros(3, dtype=np.float32),
            persistence_landscape=np.zeros(32, dtype=np.float32),
            zernike_moments=np.zeros(7, dtype=np.float32)
        )
        
        blob = Blob(
            id=1,
            color=1,
            pixels=[(0, 0), (0, 1)],
            bounding_box=(0, 0, 1, 2),
            center_of_mass=(0.0, 0.5),
            area=2,
            holes=0,
            features=feature_vector
        )
        
        return [blob], [blob]
    
    def test_llm_proposer_initialization(self, mock_config):
        """Test LLM proposer initialization."""
        proposer = LLMProposer(mock_config)
        
        assert proposer.config == mock_config
        assert proposer.prompt_template is not None
        assert proposer.dsl_engine is not None
        assert proposer._model is None  # Lazy initialization
        assert proposer._tokenizer is None
    
    def test_structured_description_creation(self, mock_config, sample_grids, sample_blobs):
        """Test structured description creation."""
        proposer = LLMProposer(mock_config)
        input_grid, target_grid = sample_grids
        input_blobs, target_blobs = sample_blobs
        
        description = proposer._create_structured_description(
            input_grid, target_grid, input_blobs, target_blobs
        )
        
        assert 'input_shape' in description
        assert 'target_shape' in description
        assert 'input_colors' in description
        assert 'target_colors' in description
        assert 'input_blobs' in description
        assert 'target_blobs' in description
        assert 'transformation_hints' in description
        
        assert description['input_shape'] == (2, 2)
        assert description['target_shape'] == (2, 2)
    
    def test_symmetry_analysis(self, mock_config, sample_grids):
        """Test symmetry analysis."""
        proposer = LLMProposer(mock_config)
        input_grid, target_grid = sample_grids
        
        symmetry = proposer._analyze_symmetry(input_grid, target_grid)
        
        assert 'rotation_90' in symmetry
        assert 'rotation_180' in symmetry
        assert 'rotation_270' in symmetry
        assert 'reflection_horizontal' in symmetry
        assert 'reflection_vertical' in symmetry
        
        # Target is 90-degree rotation of input
        assert symmetry['rotation_90'] is True
    
    def test_response_parsing_valid(self, mock_config):
        """Test parsing of valid LLM responses."""
        proposer = LLMProposer(mock_config)
        
        # Test valid response
        response = "Program: Rotate90"
        program = proposer._parse_single_response(response)
        
        assert program is not None
        assert len(program.operations) == 1
        assert program.operations[0].primitive_name == "Rotate90"
    
    def test_response_parsing_with_parameters(self, mock_config):
        """Test parsing of responses with parameters."""
        proposer = LLMProposer(mock_config)
        
        # Test response with parameters
        response = "Solution: Paint(x=1, y=2, c=5)"
        program = proposer._parse_single_response(response)
        
        assert program is not None
        assert len(program.operations) == 1
        assert program.operations[0].primitive_name == "Paint"
        assert program.operations[0].parameters == {'x': 1, 'y': 2, 'c': 5}
    
    def test_response_parsing_multiple_operations(self, mock_config):
        """Test parsing of multi-operation responses."""
        proposer = LLMProposer(mock_config)
        
        # Test multi-operation response
        response = "Answer: Rotate90 -> ReflectH"
        program = proposer._parse_single_response(response)
        
        assert program is not None
        assert len(program.operations) == 2
        assert program.operations[0].primitive_name == "Rotate90"
        assert program.operations[1].primitive_name == "ReflectH"
    
    def test_response_parsing_invalid(self, mock_config):
        """Test parsing of invalid responses."""
        proposer = LLMProposer(mock_config)
        
        # Test invalid responses
        invalid_responses = [
            "This is not a valid program",
            "Program: InvalidOperation",
            "Solution: Rotate90 -> Rotate90 -> Rotate90 -> Rotate90 -> Rotate90",  # Too long
            ""
        ]
        
        for response in invalid_responses:
            program = proposer._parse_single_response(response)
            assert program is None
    
    @patch('arc_solver.llm.llm_proposer.AutoTokenizer')
    @patch('arc_solver.llm.llm_proposer.AutoModelForCausalLM')
    def test_model_initialization_success(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test successful model initialization."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        proposer = LLMProposer(mock_config)
        proposer._initialize_model()
        
        assert proposer._model is not None
        assert proposer._tokenizer is not None
        assert proposer._tokenizer.pad_token == "<eos>"
        mock_model.eval.assert_called_once()
    
    @patch('arc_solver.llm.llm_proposer.AutoTokenizer')
    def test_model_initialization_failure(self, mock_tokenizer_class, mock_config):
        """Test model initialization failure."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        proposer = LLMProposer(mock_config)
        
        with pytest.raises(Exception):
            proposer._initialize_model()
    
    def test_statistics_tracking(self, mock_config):
        """Test statistics tracking."""
        proposer = LLMProposer(mock_config)
        
        # Initial stats
        stats = proposer.get_stats()
        assert stats['total_proposals_generated'] == 0
        assert stats['total_proposals_parsed'] == 0
        assert stats['parsing_success_rate'] == 0.0
        
        # Update stats manually (simulating successful generation)
        proposer.total_proposals_generated = 10
        proposer.total_proposals_parsed = 8
        proposer.total_generation_time = 2.5
        
        stats = proposer.get_stats()
        assert stats['total_proposals_generated'] == 10
        assert stats['total_proposals_parsed'] == 8
        assert stats['parsing_success_rate'] == 0.8
        assert stats['average_generation_time'] == 0.25
        
        # Reset stats
        proposer.reset_stats()
        stats = proposer.get_stats()
        assert stats['total_proposals_generated'] == 0


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    def test_arc_prompt_template_creation(self):
        """Test ARC prompt template creation."""
        template = create_arc_prompt_template()
        
        assert template.system_prompt is not None
        assert template.user_template is not None
        assert len(template.examples) > 0
        
        # Check that system prompt contains key information
        assert "Rotate90" in template.system_prompt
        assert "ReflectH" in template.system_prompt
        assert "Paint" in template.system_prompt
    
    def test_prompt_formatting(self):
        """Test prompt formatting with task description."""
        template = create_arc_prompt_template()
        
        task_description = {
            'input_shape': (3, 3),
            'target_shape': (3, 3),
            'input_colors': [0, 1, 2],
            'target_colors': [0, 1, 2],
            'transformation_hints': {'rotation_90': True}
        }
        
        formatted_prompt = template.format_prompt(task_description)
        
        assert isinstance(formatted_prompt, str)
        assert len(formatted_prompt) > 0
        assert "input_shape" in formatted_prompt
        assert "transformation_hints" in formatted_prompt


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    @pytest.fixture
    def generator(self):
        """Create synthetic data generator."""
        return SyntheticDataGenerator(
            grid_sizes=[(3, 3), (4, 4)],
            colors=[0, 1, 2, 3],
            max_program_length=2
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert len(generator.grid_sizes) == 2
        assert len(generator.colors) == 4
        assert generator.max_program_length == 2
        assert len(generator.task_generators) > 0
    
    def test_rotation_task_generation(self, generator):
        """Test rotation task generation."""
        task = generator._generate_rotation_task('easy')
        
        assert task is not None
        assert isinstance(task, SyntheticTask)
        assert task.task_type == 'rotation'
        assert task.difficulty == 'easy'
        assert task.input_grid.shape == task.target_grid.shape
        assert len(task.program.operations) > 0
    
    def test_reflection_task_generation(self, generator):
        """Test reflection task generation."""
        task = generator._generate_reflection_task('medium')
        
        assert task is not None
        assert task.task_type == 'reflection'
        assert task.difficulty == 'medium'
        assert task.program.operations[0].primitive_name in ['ReflectH', 'ReflectV']
    
    def test_color_mapping_task_generation(self, generator):
        """Test color mapping task generation."""
        task = generator._generate_color_mapping_task('easy')
        
        if task is not None:  # May be None if not enough colors
            assert task.task_type == 'color_mapping'
            assert task.program.operations[0].primitive_name == 'MapColors'
            assert 'mapping' in task.program.operations[0].parameters
    
    def test_training_set_generation(self, generator):
        """Test training set generation."""
        tasks = generator.generate_training_set(
            num_tasks=10,
            task_types=['rotation', 'reflection']
        )
        
        assert len(tasks) <= 10  # May be fewer due to generation failures
        assert all(isinstance(task, SyntheticTask) for task in tasks)
        assert all(task.task_type in ['rotation', 'reflection'] for task in tasks)
    
    def test_grid_creation(self, generator):
        """Test grid creation with different difficulties."""
        for difficulty in ['easy', 'medium', 'hard']:
            grid = generator._create_random_grid(difficulty)
            
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert grid.dtype == np.int32
            assert np.all(np.isin(grid, generator.colors))


class TestLLMIntegration:
    """Test LLM integration with A* search."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock integration configuration."""
        return LLMIntegrationConfig(
            llm_enabled=True,
            llm_model_name="mock-model",
            original_beam_width=32,
            llm_beam_width=8
        )
    
    @pytest.fixture
    def sample_grids(self):
        """Create sample grids for testing."""
        input_grid = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.int32)
        
        target_grid = np.array([
            [3, 1],
            [4, 2]
        ], dtype=np.int32)
        
        return input_grid, target_grid
    
    def test_integrated_searcher_initialization(self, mock_config):
        """Test integrated searcher initialization."""
        searcher = LLMIntegratedSearcher(mock_config)
        
        assert searcher.config == mock_config
        assert searcher.astar_searcher is not None
        assert searcher.dsl_engine is not None
        assert searcher.llm_proposer is None  # Lazy initialization
    
    @patch('arc_solver.search.llm_integration.create_llm_proposer')
    def test_llm_initialization(self, mock_create_proposer, mock_config):
        """Test LLM initialization."""
        mock_proposer = Mock()
        mock_create_proposer.return_value = mock_proposer
        
        searcher = LLMIntegratedSearcher(mock_config)
        searcher._initialize_llm()
        
        assert searcher.llm_proposer is not None
        mock_create_proposer.assert_called_once()
    
    @patch('arc_solver.search.llm_integration.create_llm_proposer')
    def test_llm_initialization_failure(self, mock_create_proposer, mock_config):
        """Test LLM initialization failure."""
        mock_create_proposer.side_effect = Exception("Model loading failed")
        
        searcher = LLMIntegratedSearcher(mock_config)
        searcher._initialize_llm()
        
        assert searcher.config.llm_enabled is False
    
    def test_vanilla_search_fallback(self, mock_config, sample_grids):
        """Test fallback to vanilla search."""
        # Disable LLM
        mock_config.llm_enabled = False
        
        searcher = LLMIntegratedSearcher(mock_config)
        input_grid, target_grid = sample_grids
        
        # Mock the A* searcher to return a successful result
        mock_result = Mock()
        mock_result.success = True
        mock_result.program = DSLProgram([DSLOperation('Rotate90', {})])
        mock_result.final_grid = target_grid
        mock_result.nodes_expanded = 5
        mock_result.nodes_generated = 10
        mock_result.termination_reason = "goal_reached"
        
        searcher.astar_searcher.search = Mock(return_value=mock_result)
        
        result = searcher.search(input_grid, target_grid)
        
        assert result.success is True
        assert result.llm_used is False
        assert result.fallback_used is False
        assert result.nodes_expanded == 5
        assert result.nodes_generated == 10
    
    def test_statistics_tracking(self, mock_config):
        """Test statistics tracking."""
        searcher = LLMIntegratedSearcher(mock_config)
        
        # Initial stats
        stats = searcher.get_stats()
        assert stats['total_searches'] == 0
        assert stats['llm_successful_searches'] == 0
        assert stats['fallback_searches'] == 0
        assert stats['llm_success_rate'] == 0.0
        
        # Update stats manually
        searcher.total_searches = 10
        searcher.llm_successful_searches = 6
        searcher.fallback_searches = 4
        
        stats = searcher.get_stats()
        assert stats['total_searches'] == 10
        assert stats['llm_successful_searches'] == 6
        assert stats['fallback_searches'] == 4
        assert stats['llm_success_rate'] == 0.6
        assert stats['fallback_rate'] == 0.4
        
        # Reset stats
        searcher.reset_stats()
        stats = searcher.get_stats()
        assert stats['total_searches'] == 0


class TestMockLLMProposer:
    """Test mock LLM proposer for development."""
    
    def test_mock_proposal_generation(self):
        """Test mock proposal generation for testing."""
        # Create a mock proposer that returns dummy proposals
        mock_proposer = Mock()
        
        # Mock successful proposal result
        mock_result = ProposalResult(
            success=True,
            proposals=[
                DSLProgram([DSLOperation('Rotate90', {})]),
                DSLProgram([DSLOperation('ReflectH', {})])
            ],
            raw_responses=["Rotate90", "ReflectH"],
            parsing_success_rate=1.0,
            generation_time=0.1
        )
        
        mock_proposer.generate_proposals.return_value = mock_result
        
        # Test the mock
        input_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        target_grid = np.array([[3, 1], [4, 2]], dtype=np.int32)
        
        result = mock_proposer.generate_proposals(input_grid, target_grid, [], [])
        
        assert result.success is True
        assert len(result.proposals) == 2
        assert result.parsing_success_rate == 1.0
        assert result.generation_time == 0.1


if __name__ == "__main__":
    pytest.main([__file__])