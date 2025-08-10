"""LLM-based DSL program proposal generation.

This module implements the LLM proposer that generates candidate DSL programs
from structured feature descriptions using a soft-prompted language model.
"""

import logging
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation, DSLEngine
from arc_solver.core.data_models import FeatureVector, Blob
from .prompt_templates import PromptTemplate, create_arc_prompt_template

logger = logging.getLogger(__name__)

# Expose patchable symbols for tests (will be monkey-patched by unit tests)
# Default to None so test mocks can attach these attributes even if transformers
# is not installed in the local environment.
AutoTokenizer = None  # type: ignore
AutoModelForCausalLM = None  # type: ignore
BitsAndBytesConfig = None  # type: ignore


@dataclass
class LLMConfig:
    """Configuration for LLM proposer."""
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_proposals: int = 3
    timeout_seconds: float = 10.0
    use_4bit_quantization: bool = True
    gpu_memory_fraction: float = 0.8


@dataclass
class ProposalResult:
    """Result from LLM proposal generation."""
    success: bool
    proposals: List[DSLProgram]
    raw_responses: List[str]
    parsing_success_rate: float
    generation_time: float
    error: Optional[str] = None


class LLMProposer:
    """LLM-based DSL program proposer using soft-prompted language models."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM proposer.
        
        Args:
            config: LLM configuration parameters
        """
        self.config = config or LLMConfig()
        self.prompt_template = create_arc_prompt_template()
        self.dsl_engine = DSLEngine()
        
        # LLM model (lazy initialization)
        self._model = None
        self._tokenizer = None
        
        # Statistics
        self.total_proposals_generated = 0
        self.total_proposals_parsed = 0
        self.total_generation_time = 0.0
        
        logger.info(f"LLM proposer initialized with model: {self.config.model_name}")
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model and tokenizer (lazy loading)."""
        if self._model is not None:
            return
        
        try:
            # Try to import transformers and related libraries
            import torch
            global AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            if AutoTokenizer is None or AutoModelForCausalLM is None or BitsAndBytesConfig is None:
                from transformers import AutoTokenizer as HF_AutoTokenizer, AutoModelForCausalLM as HF_AutoModelForCausalLM, BitsAndBytesConfig as HF_BitsAndBytesConfig
                AutoTokenizer = HF_AutoTokenizer
                AutoModelForCausalLM = HF_AutoModelForCausalLM
                BitsAndBytesConfig = HF_BitsAndBytesConfig
            
            logger.info(f"Loading LLM model: {self.config.model_name}")
            
            # Configure quantization if requested
            quantization_config = None
            if self.config.use_4bit_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.config.use_4bit_quantization else None
            )
            
            # Set to evaluation mode
            self._model.eval()
            
            logger.info("LLM model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries for LLM: {e}")
            logger.error("Please install: pip install torch transformers bitsandbytes accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    def generate_proposals(self, 
                         input_grid: np.ndarray,
                         target_grid: np.ndarray,
                         input_blobs: List[Blob],
                         target_blobs: List[Blob]) -> ProposalResult:
        """Generate DSL program proposals from grid analysis.
        
        Args:
            input_grid: Input grid
            target_grid: Target grid
            input_blobs: Blobs extracted from input grid
            target_blobs: Blobs extracted from target grid
            
        Returns:
            ProposalResult with generated programs and statistics
        """
        start_time = time.perf_counter()
        
        try:
            # Initialize model if needed
            self._initialize_model()
            
            # Create structured description
            description = self._create_structured_description(
                input_grid, target_grid, input_blobs, target_blobs
            )
            
            # Generate prompt
            prompt = self.prompt_template.format_prompt(description)
            
            # Generate responses from LLM
            raw_responses = self._generate_llm_responses(prompt)
            
            # Parse responses into DSL programs
            proposals, parsing_success_rate = self._parse_responses(raw_responses)
            
            generation_time = time.perf_counter() - start_time
            
            # Update statistics
            self.total_proposals_generated += len(raw_responses)
            self.total_proposals_parsed += len(proposals)
            self.total_generation_time += generation_time
            
            logger.info(f"Generated {len(proposals)} valid proposals from {len(raw_responses)} responses "
                       f"(parsing success: {parsing_success_rate:.1%})")
            
            return ProposalResult(
                success=True,
                proposals=proposals,
                raw_responses=raw_responses,
                parsing_success_rate=parsing_success_rate,
                generation_time=generation_time
            )
            
        except Exception as e:
            generation_time = time.perf_counter() - start_time
            logger.error(f"LLM proposal generation failed: {e}")
            
            return ProposalResult(
                success=False,
                proposals=[],
                raw_responses=[],
                parsing_success_rate=0.0,
                generation_time=generation_time,
                error=str(e)
            )
    
    def _create_structured_description(self,
                                     input_grid: np.ndarray,
                                     target_grid: np.ndarray,
                                     input_blobs: List[Blob],
                                     target_blobs: List[Blob]) -> Dict[str, Any]:
        """Create structured description of the transformation task.
        
        Args:
            input_grid: Input grid
            target_grid: Target grid
            input_blobs: Input blobs
            target_blobs: Target blobs
            
        Returns:
            Structured description dictionary
        """
        # Grid properties
        input_shape = input_grid.shape
        target_shape = target_grid.shape
        
        # Color analysis
        input_colors = np.unique(input_grid).tolist()
        target_colors = np.unique(target_grid).tolist()
        
        # Blob analysis
        input_blob_summary = self._summarize_blobs(input_blobs)
        target_blob_summary = self._summarize_blobs(target_blobs)
        
        # Transformation hints
        transformation_hints = self._analyze_transformation(
            input_grid, target_grid, input_blobs, target_blobs
        )
        
        return {
            'input_shape': input_shape,
            'target_shape': target_shape,
            'input_colors': input_colors,
            'target_colors': target_colors,
            'input_blobs': input_blob_summary,
            'target_blobs': target_blob_summary,
            'transformation_hints': transformation_hints
        }
    
    def _summarize_blobs(self, blobs: List[Blob]) -> List[Dict[str, Any]]:
        """Summarize blob properties for LLM input.
        
        Args:
            blobs: List of blobs to summarize
            
        Returns:
            List of blob summaries
        """
        summaries = []
        
        for blob in blobs:
            summary = {
                'color': blob.color,
                'area': blob.area,
                'bounding_box': blob.bounding_box,
                'center_of_mass': blob.center_of_mass,
                'holes': blob.holes
            }
            
            # Add feature summary if available
            if blob.features is not None:
                feature_array = blob.features.to_array()
                summary['feature_stats'] = {
                    'mean': float(np.mean(feature_array)),
                    'std': float(np.std(feature_array)),
                    'max': float(np.max(feature_array)),
                    'min': float(np.min(feature_array))
                }
            
            summaries.append(summary)
        
        return summaries
    
    def _analyze_transformation(self,
                              input_grid: np.ndarray,
                              target_grid: np.ndarray,
                              input_blobs: List[Blob],
                              target_blobs: List[Blob]) -> Dict[str, Any]:
        """Analyze the transformation between input and target.
        
        Args:
            input_grid: Input grid
            target_grid: Target grid
            input_blobs: Input blobs
            target_blobs: Target blobs
            
        Returns:
            Dictionary of transformation hints
        """
        hints = {}
        
        # Shape change
        if input_grid.shape != target_grid.shape:
            hints['shape_change'] = {
                'from': input_grid.shape,
                'to': target_grid.shape,
                'type': 'resize' if np.prod(input_grid.shape) != np.prod(target_grid.shape) else 'reshape'
            }
        
        # Color changes
        input_colors = set(np.unique(input_grid))
        target_colors = set(np.unique(target_grid))
        
        if input_colors != target_colors:
            hints['color_change'] = {
                'added_colors': list(target_colors - input_colors),
                'removed_colors': list(input_colors - target_colors),
                'preserved_colors': list(input_colors & target_colors)
            }
        
        # Blob count change
        if len(input_blobs) != len(target_blobs):
            hints['blob_count_change'] = {
                'from': len(input_blobs),
                'to': len(target_blobs),
                'change': len(target_blobs) - len(input_blobs)
            }
        
        # Symmetry analysis (simplified)
        hints['symmetry'] = self._analyze_symmetry(input_grid, target_grid)
        
        return hints
    
    def _analyze_symmetry(self, input_grid: np.ndarray, target_grid: np.ndarray) -> Dict[str, bool]:
        """Analyze potential symmetry transformations.
        
        Args:
            input_grid: Input grid
            target_grid: Target grid
            
        Returns:
            Dictionary of symmetry analysis results
        """
        symmetry_analysis = {}
        
        # Check if target matches rotated input
        try:
            # 90-degree rotation
            rotated_90 = np.rot90(input_grid, -1)  # clockwise 90 to match Rotate90 primitive
            symmetry_analysis['rotation_90'] = np.array_equal(rotated_90, target_grid)
            
            # 180-degree rotation
            rotated_180 = np.rot90(input_grid, 2)
            symmetry_analysis['rotation_180'] = np.array_equal(rotated_180, target_grid)
            
            # 270-degree rotation
            rotated_270 = np.rot90(input_grid, 3)
            symmetry_analysis['rotation_270'] = np.array_equal(rotated_270, target_grid)
            
            # Horizontal reflection (left-right flip corresponds to ReflectH)
            reflected_h = np.fliplr(input_grid)
            symmetry_analysis['reflection_horizontal'] = np.array_equal(reflected_h, target_grid)
            
            # Vertical reflection (up-down flip corresponds to ReflectV)
            reflected_v = np.flipud(input_grid)
            symmetry_analysis['reflection_vertical'] = np.array_equal(reflected_v, target_grid)
            
        except Exception as e:
            logger.debug(f"Symmetry analysis failed: {e}")
            symmetry_analysis = {'error': str(e)}
        
        return symmetry_analysis
    
    def _generate_llm_responses(self, prompt: str) -> List[str]:
        """Generate responses from the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            List of generated responses
        """
        import torch
        
        responses = []
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate multiple responses
            for i in range(self.config.num_proposals):
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self._tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                responses.append(response.strip())
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        
        return responses
    
    def _parse_responses(self, responses: List[str]) -> Tuple[List[DSLProgram], float]:
        """Parse LLM responses into DSL programs.
        
        Args:
            responses: List of raw LLM responses
            
        Returns:
            Tuple of (parsed_programs, parsing_success_rate)
        """
        programs = []
        successful_parses = 0
        
        for response in responses:
            try:
                program = self._parse_single_response(response)
                if program is not None:
                    programs.append(program)
                    successful_parses += 1
            except Exception as e:
                logger.debug(f"Failed to parse response: {e}")
                continue
        
        parsing_success_rate = successful_parses / len(responses) if responses else 0.0
        
        return programs, parsing_success_rate
    
    def _parse_single_response(self, response: str) -> Optional[DSLProgram]:
        """Parse a single LLM response into a DSL program.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed DSL program or None if parsing failed
        """
        try:
            # Look for program structure in response
            # Expected format: operation1 -> operation2 -> operation3
            
            # Extract program section (look for arrow notation)
            program_match = re.search(r'(?:Program|Solution|Answer):\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if program_match:
                program_text = program_match.group(1).strip()
            else:
                # Try to find arrow notation directly
                arrow_match = re.search(r'([A-Za-z0-9_]+(?:\([^)]*\))?(?:\s*->\s*[A-Za-z0-9_]+(?:\([^)]*\))?)*)', response)
                if arrow_match:
                    program_text = arrow_match.group(1).strip()
                else:
                    # Look for individual operations
                    operations = re.findall(r'([A-Za-z0-9_]+(?:\([^)]*\))?)', response)
                    if operations:
                        program_text = ' -> '.join(operations)
                    else:
                        return None
            
            # Delegate to DSLEngine grammar parser for robust parsing
            program = self.dsl_engine.parse_dsl_program(program_text)
            is_valid, error = self.dsl_engine.validate_program(program)
            if is_valid:
                return program
            logger.debug(f"Invalid program from parsed text: {error}")
            return None
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse response '{response}': {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM proposer statistics.
        
        Returns:
            Dictionary of statistics
        """
        avg_generation_time = (self.total_generation_time / max(1, self.total_proposals_generated))
        parsing_success_rate = (self.total_proposals_parsed / max(1, self.total_proposals_generated))
        
        return {
            'total_proposals_generated': self.total_proposals_generated,
            'total_proposals_parsed': self.total_proposals_parsed,
            'total_generation_time': self.total_generation_time,
            'average_generation_time': avg_generation_time,
            'parsing_success_rate': parsing_success_rate,
            'model_name': self.config.model_name,
            'model_loaded': self._model is not None
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_proposals_generated = 0
        self.total_proposals_parsed = 0
        self.total_generation_time = 0.0


def create_llm_proposer(model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                       max_tokens: int = 256,
                       num_proposals: int = 3,
                       use_4bit_quantization: bool = True) -> LLMProposer:
    """Factory function to create LLM proposer.
    
    Args:
        model_name: Name of the LLM model to use
        max_tokens: Maximum tokens to generate
        num_proposals: Number of proposals to generate
        use_4bit_quantization: Whether to use 4-bit quantization
        
    Returns:
        Configured LLMProposer instance
    """
    config = LLMConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        num_proposals=num_proposals,
        use_4bit_quantization=use_4bit_quantization
    )
    
    return LLMProposer(config)