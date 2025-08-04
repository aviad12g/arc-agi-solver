# ARC-AGI Solver

A sophisticated two-stage system for solving Abstract Reasoning Corpus (ARC) puzzles using mathematical perception and domain-specific language reasoning.

## Architecture

The solver implements a novel approach that separates visual perception from symbolic reasoning:

1. **Perception Layer**: Converts raw pixel grids into rich mathematical feature representations
2. **Reasoning Layer**: Applies constrained search over a domain-specific language to discover transformation rules

## Features

- GPU-accelerated blob labeling using CUDA union-find
- Mathematical invariant extraction (symmetries, topology, spectral features)
- Domain-specific language with geometric and spatial operations
- A* search with two-tier heuristic system
- Optional LLM integration for proposal generation
- Redis caching for performance optimization

## Performance Targets

- **Accuracy**: ≥35% on ARC-AGI-2 public split (>143 out of 410 tasks)
- **Runtime**: Median ≤0.5s per puzzle, 95th percentile ≤5s
- **Memory**: Peak GPU ≤2GB, peak RAM ≤6GB

## Installation

```bash
# Install with Poetry
poetry install

# Install with GPU support
poetry install --extras gpu

# Install pre-commit hooks
pre-commit install
```

## Usage

```bash
# Solve a single puzzle
arc solve task.json

# Process batch of puzzles
arc batch folder --timeout 30s --threads 8
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=arc_solver

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## Project Structure

```
src/arc_solver/
├── core/           # Core data models and types
├── perception/     # Mathematical feature extraction
├── reasoning/      # DSL engine and search algorithms
└── integration/    # Configuration, caching, CLI
```