"""Dead-end predictor scaffold (ordering/gating only).

This module provides a lightweight interface for predicting whether a node is a
dead end (no solution ahead) based on cheap invariants. The default predictor
is conservative and always returns False (never gates). A trained model can
replace `predict` later without changing caller code.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


class DeadEndPredictor:
    def __init__(self, threshold: float = 0.99, model_path: Optional[str] = None) -> None:
        # Probability threshold above which we gate expansion
        self.threshold = float(threshold)
        self.model = None
        if model_path:
            self.load_model(model_path)

    def features(self, current: np.ndarray, target: np.ndarray) -> Any:
        # Cheap invariants and counts
        c_nz = int(np.count_nonzero(current))
        t_nz = int(np.count_nonzero(target))
        c_shape = tuple(map(int, current.shape))
        t_shape = tuple(map(int, target.shape))
        # Number of colors (including background)
        c_colors = int(np.unique(current).size)
        t_colors = int(np.unique(target).size)
        return {
            'c_shape': c_shape,
            't_shape': t_shape,
            'c_nz': c_nz,
            't_nz': t_nz,
            'c_colors': c_colors,
            't_colors': t_colors,
        }

    def load_model(self, path: str) -> None:
        try:
            import json
            with open(path, 'r') as f:
                self.model = json.load(f)
            # Optionally override threshold from model
            thr = self.model.get('nz_diff_threshold', None)
            if thr is not None:
                # Convert to a probability gate in a simple way: deterministic rule
                self.threshold = 0.5  # any positive flag → gate
        except Exception:
            self.model = None

    def predict_proba(self, feats: Any) -> float:
        # Default conservative predictor: never predict dead-end
        if not self.model:
            return 0.0
        try:
            c_shape = feats.get('c_shape')
            t_shape = feats.get('t_shape')
            c_nz = int(feats.get('c_nz', 0))
            t_nz = int(feats.get('t_nz', 0))
            c_colors = int(feats.get('c_colors', 0))
            t_colors = int(feats.get('t_colors', 0))

            # Rule 1: nz diff threshold
            nz_thr = int(self.model.get('nz_diff_threshold', 10**9))
            if abs(c_nz - t_nz) > nz_thr:
                return 1.0

            # Rule 2: colors diff threshold
            col_thr = int(self.model.get('colors_diff_threshold', 10**9))
            if abs(c_colors - t_colors) > col_thr:
                return 1.0

            # Rule 3: shape mismatch rule (optional)
            shape_rule = False
            thresholds = self.model.get('thresholds', {})
            # Accept either top-level shape_rule or infer if thresholds indicate mismatch harms
            shape_rule = bool(self.model.get('shape_rule', False))
            if shape_rule and c_shape != t_shape:
                return 1.0

            return 0.0
        except Exception:
            return 0.0

    def should_gate(self, current: np.ndarray, target: np.ndarray) -> bool:
        p = self.predict_proba(self.features(current, target))
        return bool(p >= self.threshold)
