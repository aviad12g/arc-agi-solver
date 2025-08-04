# Requirements Document

## Introduction

The ARC-AGI Solver is a mathematically-optimal, GPU-accelerated system designed to solve Abstract Reasoning Corpus (ARC) puzzles with ≥35% accuracy on the ARC-AGI-2 public split. The system converts each ARC grid pair into a minimal DSL program through a sophisticated pipeline: GPU blob labeling → invariant feature encoding (σ, λ, PH, μ) → palette canonicalization → two-tier heuristic A* search ↔ DSL kernel library (C++/CUDA). The solver targets median runtime ≤0.5s per puzzle with research-grade mathematical rigor and optimal search guarantees.

## Requirements

### Requirement 1: Core System Architecture

**User Story:** As a researcher, I want a two-stage solver architecture so that I can leverage both mathematical feature extraction and symbolic reasoning for optimal ARC puzzle solving.

#### Acceptance Criteria

1. WHEN the system receives an ARC puzzle THEN it SHALL process it through a perception layer followed by a reasoning layer
2. WHEN the perception layer completes THEN it SHALL output structured feature data (not raw solutions)
3. WHEN the reasoning layer receives structured data THEN it SHALL deduce transformation rules and generate solutions
4. IF either stage fails THEN the system SHALL provide meaningful error diagnostics

### Requirement 2: Mathematical Perception Layer

**User Story:** As a perception system, I want to transform raw pixels into mathematically rigorous 50-dimensional feature vectors so that the reasoning engine can leverage invariant properties for optimal search.

#### Acceptance Criteria

1. WHEN processing input grids THEN the system SHALL compute orbit signatures σ(S) = {χ(ρ(S))}ρ∈D4 for symmetry-invariant blob identification
2. WHEN analyzing connectivity THEN the system SHALL extract non-trivial Laplacian eigenvalues λ2,λ3,λ4 from adjacency graphs as connectivity fingerprints
3. WHEN computing topology THEN the system SHALL generate persistence landscapes D0,D1⇒L0:31 from cubical complexes for topological hole detection
4. WHEN describing shape THEN the system SHALL compute Zernike moment invariants μpq (p+q≤4) for fine geometric description
5. WHEN combining features THEN the system SHALL output f∈R50 feature vectors with distance metrics for heuristic computation
6. WHEN using GPU acceleration THEN blob labeling SHALL complete in ≤2ms per grid using CUDA union-find
7. WHEN ensuring invariance THEN all features SHALL maintain mathematical guarantees under D4 group transformations

### Requirement 3: Domain-Specific Language (DSL)

**User Story:** As a reasoning system, I want a constrained set of transformation primitives so that I can systematically explore solution spaces efficiently.

#### Acceptance Criteria

1. WHEN defining transformations THEN the DSL SHALL support Rotate90, Rotate180, ReflectH, ReflectV operations
2. WHEN manipulating grids THEN the DSL SHALL support Crop(r1,r2,c1,c2) and Paint(x,y,c) operations  
3. WHEN handling colors THEN the DSL SHALL support MapColors(perm[10]) for color remapping
4. WHEN applying conditional logic THEN the DSL SHALL support PaintIf(predicate, action) operations
5. WHEN limiting complexity THEN program length SHALL be capped at K=4 operations maximum
6. WHEN executing DSL operations THEN each primitive SHALL complete in ≤200µs

### Requirement 4: Two-Tier Heuristic A* Search

**User Story:** As a search engine, I want mathematically admissible heuristics so that I can guarantee optimal DSL programs while maintaining efficient search performance.

#### Acceptance Criteria

1. WHEN computing Tier-1 heuristic THEN the system SHALL use h1(G)=min⁡ρ∈D4∥f̄in−f̄ρ(G)∥2 for fast L₂ spectral-homology distance
2. WHEN Tier-1 pruning is insufficient THEN the system SHALL apply Hungarian assignment for tighter edit distance lower bounds
3. WHEN proving admissibility THEN each DSL primitive SHALL alter ≥1 feature component by ≥1, ensuring Manhattan/L₂ norms provide valid lower bounds
4. WHEN searching solution space THEN the system SHALL use admissible A* with worst-case space ∑i=048i=5461 states for K=4 programs
5. WHEN expanding search nodes THEN average expanded states SHALL be ≤600 per puzzle through effective heuristic pruning
6. WHEN search completes THEN the system SHALL guarantee globally optimal solutions within DSL constraints

### Requirement 5: Performance Benchmarks and Resource Constraints

**User Story:** As a researcher, I want mathematically-optimal performance with specific resource constraints so that the solver achieves state-of-the-art results within practical computational limits.

#### Acceptance Criteria

1. WHEN solving ARC-AGI-2 public split THEN accuracy SHALL be ≥35% (>143 out of 410 tasks) with mathematical optimality guarantees
2. WHEN processing individual puzzles THEN median runtime SHALL be ≤0.5 seconds with 95th percentile ≤5 seconds
3. WHEN using GPU resources THEN inference cost SHALL be ≪1 GPU-ms per puzzle with complete 410-task evaluation <15 GPU-seconds
4. WHEN caching operations THEN loader + palette canonicalization SHALL complete in <30 seconds for full dataset
5. WHEN performing GPU union-find THEN blob labeling SHALL complete in <2ms per grid
6. WHEN executing DSL operations THEN each primitive SHALL be bound via pybind11 at ≤200µs per operation

### Requirement 6: LLM Proposal Module (Optional Enhancement)

**User Story:** As an advanced reasoning component, I want soft-prompted LLM guidance so that I can leverage language model insights while maintaining mathematical rigor and fallback guarantees.

#### Acceptance Criteria

1. WHEN LLM integration is enabled THEN the system SHALL use soft-prompted 32B model with 256 tokens trained on 300 synthetic tasks
2. WHEN generating proposals THEN LLM SHALL emit top-3 candidate DSL programs with 95% parseability guarantee
3. WHEN integrating with search THEN beam search + LLM proposals SHALL achieve ≥95% parseable programs
4. WHEN optimizing search THEN beam width SHALL reduce from 64 to 8 without accuracy degradation
5. WHEN LLM fails THEN the system SHALL maintain full fallback to vanilla A* search methods with identical optimality guarantees

### Requirement 7: System Integration and Usability

**User Story:** As a user, I want a complete system with CLI interface and configuration management so that I can easily run experiments and reproduce results.

#### Acceptance Criteria

1. WHEN running single puzzles THEN CLI SHALL support `arc solve task.json` command
2. WHEN processing batches THEN CLI SHALL support `arc batch folder --timeout 30s --threads 8`
3. WHEN configuring system THEN all hyperparameters SHALL be exposed via config.yaml
4. WHEN caching results THEN Redis SHALL store grid_hash → program_str mappings
5. WHEN reproducing results THEN `bash scripts/reproduce.sh` SHALL generate identical outputs
6. WHEN system times out THEN hard-kill SHALL occur after 30s with partial results saved

### Requirement 8: Mathematical Correctness and Validation

**User Story:** As a mathematical system, I want rigorous validation of invariant properties so that feature extraction maintains theoretical guarantees.

#### Acceptance Criteria

1. WHEN computing orbit signatures THEN σ(S) SHALL be invariant under D₄ group transformations
2. WHEN calculating spectral features THEN Laplacian eigenvalues SHALL be stable under relabeling (±1e-3)
3. WHEN computing persistent homology THEN rotated objects SHALL produce identical landscapes
4. WHEN using Zernike moments THEN D₄ invariance SHALL hold within 1e-6 tolerance
5. WHEN testing symmetry detection THEN bitboard operations SHALL complete in ≤10µs
6. WHEN validating heuristics THEN admissibility proofs SHALL guarantee optimality