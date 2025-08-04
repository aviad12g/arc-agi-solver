# Implementation Plan

- [x] 1. Set up project structure and core data models
  - Create Python package structure with src/arc_solver directory
  - Implement core data classes: Blob, FeatureVector, GridState, SearchNode
  - Set up development environment with Poetry, pre-commit hooks, and testing framework
  - _Requirements: 1.1, 1.4_

- [x] 2. Implement data loading and I/O backbone
  - Create ARC-AGI dataset loader that converts JSON to numpy arrays
  - Implement task caching system using .npz format for performance
  - Build palette canonicalizer that normalizes color representations
  - Write unit tests for data loading and caching functionality
  - _Requirements: 7.5, 5.6_

- [x] 3. Build GPU blob labeling engine
  - Implement CPU baseline flood-fill algorithm for connected components
  - Create CUDA union-find kernels for GPU-accelerated blob labeling
  - Add automatic fallback from GPU to CPU when CUDA unavailable
  - Write performance tests to verify ≤2ms labeling time per 30×30 grid
  - _Requirements: 2.6, 5.4_

- [x] 4. Implement symmetry detection system
  - Create bitboard representation using uint64 masks for each color
  - Implement 8-way symmetry transforms (4 rotations + 4 reflections) using bit operations
  - Add early-exit global symmetry checking for performance optimization
  - Write unit tests to verify ≤10µs symmetry detection performance
  - _Requirements: 2.2, 8.5_

- [x] 5. Build invariant feature extraction system
- [x] 5.1 Implement orbit signature computation
  - Create D₄ group transformation functions for symmetry fingerprinting
  - Compute orbit signatures σ(S) that are invariant under group transformations
  - Write tests to verify D₄ invariance properties
  - _Requirements: 2.7, 8.1_

- [x] 5.2 Add spectral feature computation
  - Implement Laplacian matrix construction from blob adjacency graphs
  - Compute λ₂, λ₃, λ₄ eigenvalues using scipy sparse eigenvalue solver
  - Add numerical stability checks and fallback methods for edge cases
  - Write tests to verify eigenvalue stability under relabeling (±1e-3 tolerance)
  - _Requirements: 2.7, 8.2_

- [x] 5.3 Implement persistent homology analysis
  - Integrate Gudhi library for cubical complex computation
  - Generate 32-length persistence landscape vectors for topological features
  - Add error handling for topology computation failures
  - Write tests to verify rotational invariance of persistence landscapes
  - _Requirements: 2.3, 8.3_

- [x] 5.4 Add Zernike moment computation
  - Implement complex moment invariants up to order 4 using scikit-image
  - Ensure D₄ invariance within 1e-6 tolerance for shape descriptors
  - Combine all features into 50-dimensional feature vectors
  - Write comprehensive tests for feature vector generation and invariance
  - _Requirements: 2.4, 8.4_

- [x] 6. Create Domain-Specific Language (DSL) engine
- [x] 6.1 Implement core DSL primitives in C++
  - Write C++ kernels for geometric transforms: Rotate90, Rotate180, ReflectH, ReflectV
  - Implement spatial operations: Crop(r1,r2,c1,c2), Paint(x,y,c)
  - Add color manipulation: MapColors(perm[10]) for palette remapping
  - Create conditional operations: PaintIf(predicate, action) with blob predicates
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 6.2 Build Python DSL wrapper with pybind11
  - Create Python bindings for C++ DSL kernels using pybind11
  - Implement DSLEngine class with program execution and validation
  - Add program length constraints (K=4 maximum operations)
  - Write performance tests to verify ≤200µs execution per primitive
  - _Requirements: 3.5, 3.6_

- [x] 7. Implement two-tier heuristic system
- [x] 7.1 Create Tier-1 L₂ spectral-homology heuristic
  - Implement fast L₂ distance computation between feature vectors
  - Add D₄ group minimization: h₁(G) = min_{ρ∈D₄} ||f̄_in - f̄_ρ(G)||₂
  - Optimize for speed using precomputed feature representations
  - Write tests to verify heuristic admissibility properties
  - _Requirements: 4.3, 8.6_

- [x] 7.2 Add Tier-2 Hungarian assignment fallback
  - Implement optimal blob matching using Hungarian algorithm
  - Compute edit distance lower bounds from unmatched blobs and color mismatches
  - Add automatic fallback when Tier-1 pruning is insufficient (<70% pruning)
  - Write tests to verify tighter bounds and admissibility guarantees
  - _Requirements: 4.4, 8.6_

- [x] 8. Build A* search engine with beam search
  - Implement admissible A* algorithm with priority queue (min-heap)
  - Create GridState representation with grid, program, cost, and heuristic
  - Add beam search pruning with adaptive beam width based on heuristic quality
  - Implement search termination with ≤600 expanded states target
  - Write tests to verify optimality on toy problems and search efficiency
  - _Requirements: 4.1, 4.5, 4.6_

- [x] 9. Create configuration and caching systems
- [x] 9.1 Implement Hydra configuration management
  - Set up hierarchical configuration with conf/config.yaml
  - Add runtime parameter override capabilities for experiments
  - Create configuration groups for different system components
  - Write tests for configuration loading and parameter validation
  - _Requirements: 7.3_

- [x] 9.2 Build Redis caching system
  - Implement SHA-1 hash-based cache keys for canonical grid representations
  - Store successful DSL program strings as cache values
  - Add cache hit rate monitoring with 10% target on development split
  - Write tests for cache consistency and performance
  - _Requirements: 7.4_

- [x] 10. Implement CLI interface and batch processing
  - Create command-line interface supporting `arc solve task.json` for single puzzles
  - Add batch processing with `arc batch folder --timeout 30s --threads 8`
  - Implement hard timeout mechanism with 30-second limit per puzzle
  - Add partial result saving when timeouts occur
  - Write integration tests for CLI functionality and timeout handling
  - _Requirements: 7.1, 7.2, 7.6_

- [x] 11. Add LLM integration module (optional enhancement)
- [x] 11.1 Implement LLM proposal generation
  - Set up Qwen-32B-Instruct model with 4-bit quantization via vLLM
  - Create soft-prompt tuning system with 256 tokens on synthetic tasks
  - Generate top-3 candidate DSL programs from structured feature descriptions
  - Add schema validation to ensure 95% parseability of generated programs
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 11.2 Integrate LLM proposals with beam search
  - Feed LLM proposals as high-priority candidates into A* search
  - Implement fallback to vanilla search when LLM proposals fail
  - Optimize beam width reduction from 64 to 8 without accuracy loss
  - Write tests for LLM integration robustness and fallback mechanisms
  - _Requirements: 6.4, 6.5_

- [-] 12. Build comprehensive testing and validation framework
- [x] 12.1 Create unit tests for all components
  - Write perception layer tests: blob labeling, feature invariance, symmetry detection
  - Add reasoning layer tests: DSL correctness, A* optimality, heuristic admissibility
  - Implement integration tests: end-to-end pipeline, configuration, caching
  - Create performance benchmarks for all timing requirements
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 12.2 Add accuracy and performance validation
  - Implement continuous validation on ARC-AGI-2 development split
  - Create accuracy tracking with ≥35% target (>143 out of 410 tasks)
  - Add runtime monitoring with ≤0.5s median and ≤5s 95th percentile targets
  - Implement memory usage tracking with ≤2GB GPU and ≤6GB RAM limits
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 13. Create error handling and robustness systems
  - Implement GPU failure detection with automatic CPU fallback
  - Add search timeout handling with partial result preservation
  - Create feature computation error recovery with fallback methods
  - Build comprehensive logging and diagnostic systems for failure analysis
  - Write tests for all error conditions and recovery mechanisms
  - _Requirements: 1.4_

- [ ] 14. Add profiling and optimization tools
  - Integrate Nsight Compute for CUDA kernel profiling and optimization
  - Add cProfile integration for Python performance hotspot identification
  - Implement auto-batching for homology and Zernike computations
  - Create performance monitoring dashboard for continuous optimization
  - _Requirements: 5.2, 5.3_

- [ ] 15. Build final integration and deployment pipeline
  - Create reproduction script: `bash scripts/reproduce.sh` for result validation
  - Implement end-to-end system integration with all components working together
  - Add final accuracy and performance validation on complete system
  - Create technical documentation and usage examples
  - Write deployment guide and system requirements documentation
  - _Requirements: 7.5, 5.1_