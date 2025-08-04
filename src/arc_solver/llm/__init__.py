"""LLM integration module for ARC-AGI solver.

This module provides LLM-based proposal generation for DSL programs,
using soft-prompted language models to suggest candidate transformations.
"""

from .llm_proposer import LLMProposer, create_llm_proposer
from .prompt_templates import PromptTemplate, create_arc_prompt_template
from .synthetic_data import SyntheticDataGenerator, create_synthetic_data_generator

__all__ = [
    'LLMProposer',
    'create_llm_proposer',
    'PromptTemplate',
    'create_arc_prompt_template',
    'SyntheticDataGenerator',
    'create_synthetic_data_generator'
]