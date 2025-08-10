"""Deprecated heuristic module.

This module re-exports the canonical heuristic system from
`arc_solver.search.heuristics` to avoid duplication and divergence.
"""

from arc_solver.search.heuristics import (  # noqa: F401
    HeuristicResult,
    BaseHeuristic,
    Tier1Heuristic,
    Tier2Heuristic,
    LearnedHeuristicWeights,
    DualHeuristic,
    HeuristicSystem,
    create_heuristic_system,
)

__all__ = [
    'HeuristicResult',
    'BaseHeuristic',
    'Tier1Heuristic',
    'Tier2Heuristic',
    'LearnedHeuristicWeights',
    'DualHeuristic',
    'HeuristicSystem',
    'create_heuristic_system',
]


