"""Reasoning layer for ARC-AGI solver.

This module handles the Domain-Specific Language (DSL) for grid transformations,
program synthesis, and search algorithms.
"""

from .dsl_engine import DSLEngine, DSLProgram, DSLOperation, create_dsl_engine
from .dsl_wrapper import EnhancedDSLEngine, ExecutionResult, create_enhanced_dsl_engine
from .primitives import (
    DSLPrimitive, Rotate90, Rotate180, ReflectH, ReflectV, Crop, Paint, 
    MapColors, PaintIf, SizePredicate, ColorPredicate, create_all_primitives
)

__all__ = [
    'DSLEngine',
    'DSLProgram', 
    'DSLOperation',
    'DSLPrimitive',
    'create_dsl_engine',
    'EnhancedDSLEngine',
    'ExecutionResult',
    'create_enhanced_dsl_engine',
    'Rotate90',
    'Rotate180', 
    'ReflectH',
    'ReflectV',
    'Crop',
    'Paint',
    'MapColors',
    'PaintIf',
    'SizePredicate',
    'ColorPredicate',
    'create_all_primitives'
]