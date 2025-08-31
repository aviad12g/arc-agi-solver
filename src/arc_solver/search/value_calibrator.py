"""Safe combination of admissible heuristic with a calibrated value signal (scaffold).

This module exposes a tiny utility to form a dominated floor for ordering,
without ever altering the admissible f=g+h used by A*.

Usage: compute a secondary tie-break score from h_adm and a raw value
prediction. Never use this to replace h in f; only for queue ordering.
"""

from __future__ import annotations

from typing import Optional


def calibrated_floor_for_ordering(
    h_adm: float,
    v_raw: Optional[float],
    proofs_floor: float = 0.0,
    slack: float = 0.0,
) -> float:
    """Return a safe secondary score for ordering only.

    - h_adm: admissible heuristic value.
    - v_raw: raw value estimate (may be None).
    - proofs_floor: known lower bound from proofs (PDB/SMT) if available.
    - slack: small allowed slack above admissible bound for robustness.

    The returned score is max(h_adm, clamp(v_raw, proofs_floor, h_adm + slack)).
    If v_raw is None, returns h_adm.
    """
    if v_raw is None:
        return float(h_adm)
    lower = float(max(0.0, proofs_floor))
    upper = float(h_adm + max(0.0, slack))
    v_clamped = min(max(float(v_raw), lower), upper)
    return float(max(h_adm, v_clamped))

