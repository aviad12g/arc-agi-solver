# Engineering notes: performance

This file collects working notes from local experiments. Treat the numbers as rough guidance, not targets.

Setup: Python 3.9 on a laptop (CPU). SciPy present; GPU disabled.

What we observed on a small sample of ARC‑format tasks (n≈10):
- The searcher expands from a few dozen to a few hundred nodes on trivial transforms (rotations/reflections), but can blow up rapidly with crop/paint combinations.
- Tier‑1 heuristic time dominates when feature extraction triggers spectral/persistence code. Caching helps; reusing blob stats early is worth it.
- Blob labeling via SciPy is generally fast enough on small grids; BFS fallback is fine for tests.

Practical tweaks that helped in this repo:
- Keep beam width moderate (16–32). Larger beams don’t buy much on easy cases and slow everything else.
- Turn on heuristic result and feature caching early.
- Gate Tier‑2 (assignment) by blob counts and depth to avoid worst‑case paths.

Next steps (short list):
- Profile Tier‑1 end‑to‑end and remove obvious redundancies (don’t recompute features for identical grids).
- Tighten parameter generation for expensive primitives; prefer no‑ops pruning early.
- Add a simple wall‑clock budget to degrade gracefully.