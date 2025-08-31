"""Verification-gated PUCT search over DSL programs (scaffold).

This is a lightweight PUCT implementation that uses uniform policy priors by
default and the heuristic system as a value proxy. Every candidate program is
verified on all train examples before expansion to keep correctness.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation


class PUCTNode:
    def __init__(self, program: DSLProgram, grid: np.ndarray, parent: Optional['PUCTNode']=None, prior: float=1.0) -> None:
        self.program = program
        self.grid = grid
        self.parent = parent
        self.children: List[Tuple[DSLOperation, 'PUCTNode']] = []
        self.N = 0  # visit count
        self.W = 0.0  # total value
        self.Q = 0.0  # mean value
        self.P = float(prior)  # prior

    def ucb_score(self, c_puct: float) -> float:
        Np = self.parent.N if self.parent is not None else max(1, self.N)
        U = c_puct * self.P * math.sqrt(Np) / (1 + self.N)
        return self.Q + U


class PUCTSearcher:
    def __init__(self, dsl_engine: DSLEngine, heuristic_system, c_puct: float = 1.0, max_depth: int = 4, time_budget: float = 1.0,
                 op_prior_boost: Optional[Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], float]] = None) -> None:
        self.dsl_engine = dsl_engine
        self.heuristic = heuristic_system
        self.c_puct = float(c_puct)
        self.max_depth = int(max_depth)
        self.time_budget = float(time_budget)
        self.op_prior_boost = op_prior_boost or {}

    def _verify_program(self, program: DSLProgram, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for a, b in train_pairs:
            try:
                out, _ = self.dsl_engine.execute_program(program, a)
            except Exception:
                return False
            if not np.array_equal(out, b):
                return False
        return True

    def _value(self, grid: np.ndarray, targets: List[np.ndarray]) -> float:
        # Lower heuristic values across all targets; we use the worst-case (max) then negate
        try:
            values = []
            for t in targets:
                values.append(float(self.heuristic.compute_heuristic(grid, t).value))
            if not values:
                return 0.0
            return -max(values)
        except Exception:
            return 0.0

    def search(self, initial_grid: np.ndarray, target_grid: np.ndarray, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[DSLProgram]:
        root = PUCTNode(DSLProgram([]), initial_grid, None, prior=1.0)
        deadline = time.perf_counter() + self.time_budget

        if np.array_equal(initial_grid, target_grid):
            return DSLProgram([])

        # Prepare list of targets for multi-example value aggregation
        targets = [b for (_, b) in train_pairs] if train_pairs else [target_grid]

        while time.perf_counter() < deadline:
            # Selection
            node = root
            path: List[PUCTNode] = [node]
            while node.children:
                node = max((child for _, child in node.children), key=lambda n: n.ucb_score(self.c_puct))
                path.append(node)

            # Expansion if not terminal
            if len(node.program.operations) < self.max_depth:
                ops = self.dsl_engine.get_available_operations(node.grid)
                if ops:
                    # Compute priors by normalized boost weights
                    priors = []
                    total = 0.0
                    for op in ops:
                        key = (op.primitive_name, tuple(sorted(op.parameters.items())))
                        w = 1.0 + float(self.op_prior_boost.get(key, 0.0))
                        priors.append((op, w))
                        total += w
                    for op, w in priors:
                        try:
                            new_grid = self.dsl_engine.apply_operation(node.grid, op)
                            if np.array_equal(new_grid, node.grid):
                                continue
                            new_prog = node.program.append(op)
                            # Verify candidate fully before adding children
                            if self._verify_program(new_prog, train_pairs):
                                return new_prog
                            prior = (w / total) if total > 0 else (1.0 / max(1, len(priors)))
                            child = PUCTNode(new_prog, new_grid, node, prior=prior)
                            node.children.append((op, child))
                    if not node.children:
                        # No effective ops; backprop neutral
                        v = -0.0
                    else:
                        # Evaluate value for children as rollout proxy (worst-case target)
                        v = max(self._value(ch[1].grid, targets) for ch in node.children)
                else:
                    v = -0.0
            else:
                v = self._value(node.grid, targets)

            # Backpropagation
            for n in reversed(path):
                n.N += 1
                n.W += v
                n.Q = n.W / max(1, n.N)

        return None
