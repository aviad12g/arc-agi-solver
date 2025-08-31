"""Simple retrieval index for solved programs keyed by invariant signatures.

Signatures are conservative and cheap: shapes + color-count multisets per pair.
This enables cross-task transfer within and across runs.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import threading
import numpy as np
import json
from pathlib import Path

from arc_solver.reasoning.dsl_engine import DSLProgram


def _pair_signature(inp: np.ndarray, out: np.ndarray) -> Tuple:
    def hist_sig(g: np.ndarray):
        unique, counts = np.unique(g, return_counts=True)
        return tuple(sorted((int(c) for c in counts), reverse=True))
    return (
        tuple(map(int, inp.shape)),
        tuple(map(int, out.shape)),
        hist_sig(inp),
        hist_sig(out),
    )


def task_signature(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple:
    """Aggregate per-pair signatures into one task signature.

    Use a multiset of per-pair signatures to avoid ordering sensitivity.
    """
    sigs = [_pair_signature(a, b) for a, b in train_pairs]
    # Add blob-graph coarse signatures for extra discriminative power
    try:
        bg_sigs = [blob_graph_pair_signature(a, b) for a, b in train_pairs]
    except Exception:
        bg_sigs = []
    # Sort for determinism
    sigs_sorted = tuple(sorted(sigs))
    if bg_sigs:
        return (sigs_sorted, tuple(sorted(bg_sigs)))
    return sigs_sorted


def blob_graph_signature(grid: np.ndarray) -> Tuple:
    """Compute a coarse blob-graph signature: (#blobs, degree histogram, area histogram).

    This is cheap and robust; it ignores color labels but captures structure.
    """
    try:
        from arc_solver.perception.blob_labeling import create_blob_labeler
        bl = create_blob_labeler(use_gpu=False)
        blobs, _ = bl.label_blobs(grid)
        graph = bl.get_blob_adjacency_graph(blobs, grid.shape)
        n = len(blobs)
        degrees = [len(graph.get(i, [])) for i in range(n)]
        # Degree histogram capped at small max
        max_deg = 4
        deg_hist = [0] * (max_deg + 1)
        for d in degrees:
            deg_hist[min(d, max_deg)] += 1
        # Area histogram coarse buckets
        areas = [b.area for b in blobs]
        buckets = [0, 0, 0]  # small, medium, large
        if areas:
            amax = max(areas)
            for a in areas:
                if a <= max(1, amax // 3):
                    buckets[0] += 1
                elif a <= max(1, (2 * amax) // 3):
                    buckets[1] += 1
                else:
                    buckets[2] += 1
        return (int(n), tuple(deg_hist), tuple(buckets))
    except Exception:
        # Fallback signature when perception not available
        return (0, (0, 0, 0, 0, 0), (0, 0, 0))


def blob_graph_pair_signature(inp: np.ndarray, out: np.ndarray) -> Tuple:
    """Pair-level blob-graph signature combining input and output graphs."""
    return (blob_graph_signature(inp), blob_graph_signature(out))


class RetrievalIndex:
    """Thread-safe in-memory retrieval index with optional persistence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._map: Dict[Tuple, List[DSLProgram]] = {}
        self._store_path: Optional[Path] = None
        self._sig_vectors: Dict[Tuple, List[float]] = {}

    def add(self, sig: Tuple, program: DSLProgram) -> None:
        with self._lock:
            lst = self._map.setdefault(sig, [])
            if not any(hash(p) == hash(program) for p in lst):
                lst.append(program)
            # Cache signature vector for NN matching
            if sig not in self._sig_vectors:
                self._sig_vectors[sig] = signature_vector(sig)
            # Persist if attached
            if self._store_path is not None:
                try:
                    rec = {'signature': sig, 'program': program.to_dict()}
                    with open(self._store_path, 'a') as f:
                        json.dump(rec, f)
                        f.write('\n')
                except Exception:
                    pass

    def get(self, sig: Tuple, max_items: int = 5) -> List[DSLProgram]:
        with self._lock:
            lst = self._map.get(sig, [])
            return list(lst)[:max_items]

    # Persistence helpers
    def attach_store(self, path: str) -> None:
        """Attach a JSONL file to persist retrieved programs across runs."""
        with self._lock:
            self._store_path = Path(path)
            self._store_path.parent.mkdir(parents=True, exist_ok=True)

    def load_from_jsonl(self, path: str) -> int:
        """Load stored signatures and programs from JSONL.

        Each line: {"signature": <nested tuple-like>, "program": <DSLProgram dict>}.
        Returns count loaded.
        """
        p = Path(path)
        if not p.exists():
            return 0
        count = 0
        with self._lock:
            try:
                with open(p, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            sig = _to_tuple(obj.get('signature'))
                            prog_dict = obj.get('program')
                            if sig is None or prog_dict is None:
                                continue
                            prog = DSLProgram.from_dict(prog_dict)
                            lst = self._map.setdefault(sig, [])
                            if not any(hash(p) == hash(prog) for p in lst):
                                lst.append(prog)
                                count += 1
                            if sig not in self._sig_vectors:
                                self._sig_vectors[sig] = signature_vector(sig)
                        except Exception:
                            continue
            except Exception:
                return count
        return count

    def nearest(self, sig: Tuple, k: int = 5) -> List[Tuple[Tuple, float]]:
        """Return k nearest signatures by L1 distance over signature vectors."""
        with self._lock:
            if not self._sig_vectors:
                # Build cache on demand
                for s in self._map.keys():
                    if s not in self._sig_vectors:
                        self._sig_vectors[s] = signature_vector(s)
            target_vec = signature_vector(sig)
            distances: List[Tuple[Tuple, float]] = []
            for s, vec in self._sig_vectors.items():
                d = l1_distance(vec, target_vec)
                distances.append((s, d))
            distances.sort(key=lambda x: x[1])
            return distances[:k]

    def get_by_signature(self, sig: Tuple, max_items: int = 5) -> List[DSLProgram]:
        with self._lock:
            return list(self._map.get(sig, []))[:max_items]


def signature_vector(sig: Tuple) -> List[float]:
    """Flatten a (sigs, [bg_sigs]) tuple into a numeric vector for NN matching.

    We encode shapes, histograms sizes, blob graph features into a fixed-length
    coarse vector. Missing parts are zero-filled.
    """
    vec: List[float] = []
    def enc_shape(sh):
        try:
            return [float(sh[0]), float(sh[1])]
        except Exception:
            return [0.0, 0.0]
    def enc_hist(h):
        # histogram multiset tuple of counts
        out: List[float] = []
        for c in list(h)[:10]:
            out.append(float(c))
        while len(out) < 10:
            out.append(0.0)
        return out
    def enc_pair(p):
        # ((h,w),(h,w),(size,hist,nz),(size,hist,nz))
        out: List[float] = []
        try:
            out += enc_shape(p[0])
            out += enc_shape(p[1])
            out.append(float(p[2][0]))
            out += enc_hist(p[2][1])
            out.append(float(p[2][2]))
            out.append(float(p[3][0]))
            out += enc_hist(p[3][1])
            out.append(float(p[3][2]))
        except Exception:
            out += [0.0] * (2+2+1+10+1+1+10+1)
        return out
    def enc_bg(bg):
        # ((n,deg_hist,buckets), (n,deg_hist,buckets))
        out: List[float] = []
        try:
            for side in bg:
                out.append(float(side[0]))
                dh = side[1]
                for v in list(dh)[:5]:
                    out.append(float(v))
                while len(out) < 1+5:
                    out.append(0.0)
                buckets = side[2]
                for v in list(buckets)[:3]:
                    out.append(float(v))
        except Exception:
            out += [0.0] * (1+5+3)*2
        return out

    if isinstance(sig, tuple) and len(sig) == 2 and isinstance(sig[1], tuple):
        # (sigs_sorted, bg_sigs_sorted)
        sigs_sorted, bg_sigs_sorted = sig
        for p in list(sigs_sorted)[:4]:
            vec += enc_pair(p)
        for bg in list(bg_sigs_sorted)[:4]:
            vec += enc_bg(bg)
    else:
        for p in list(sig)[:4]:
            vec += enc_pair(p)
    # Pad to a fixed length
    if len(vec) < 200:
        vec += [0.0] * (200 - len(vec))
    else:
        vec = vec[:200]
    return vec


def l1_distance(a: List[float], b: List[float]) -> float:
    s = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        s += abs(a[i] - b[i])
    return s + abs(len(a) - len(b)) * 0.0


def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(e) for e in x)
    if isinstance(x, tuple):
        return tuple(_to_tuple(e) for e in x)
    return x
