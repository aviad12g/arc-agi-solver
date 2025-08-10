"""Ensemble selection with MDL + leave-one-out (LOO) scoring.

Given a set of train-consistent programs, pick the best according to a
deterministic score that balances simplicity and generalization:

score = -|P| - lambda*DL(P) - mu*LOO_error(P)

Where:
- |P| is the length (number of operations)
- DL(P) is a simple description length: op_count + 0.1 * num_params_total
- LOO_error is the sum of pixel error when validating P on each example
  with the example included (conservative approximation of LOO).

This module assumes programs already pass all train pairs or are candidate
solutions to rank; it will still compute error to break ties.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram


def select_best_program(
    engine: DSLEngine,
    programs: List[DSLProgram],
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    lambda_dl: float = 0.1,
    mu_loo: float = 1.0,
) -> Optional[DSLProgram]:
    if not programs:
        return None
    best_prog = None
    best_score = -float("inf")
    for prog in programs:
        score = _score_program(engine, prog, train_pairs, lambda_dl, mu_loo)
        if score > best_score:
            best_score = score
            best_prog = prog
    return best_prog


def _score_program(
    engine: DSLEngine,
    program: DSLProgram,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    lambda_dl: float,
    mu_loo: float,
) -> float:
    # Length and simple DL prior
    op_count = len(program.operations)
    param_count = 0
    for op in program.operations:
        try:
            param_count += len(op.parameters)
        except Exception:
            pass
    dl = op_count + 0.1 * param_count

    # Conservative "LOO" error: sum over examples of (1 - pixel accuracy)
    # Using in-sample evaluation as a proxy since train is tiny.
    error_sum = 0.0
    for src, dst in train_pairs:
        try:
            out, _ = engine.execute_program(program, src)
            if out.shape != dst.shape:
                error_sum += 1.0
            else:
                matches = float(np.sum(out == dst))
                total = float(out.size) if out.size > 0 else 1.0
                error_sum += (1.0 - matches / total)
        except Exception:
            error_sum += 1.0

    return float(-op_count - lambda_dl * dl - mu_loo * error_sum)


