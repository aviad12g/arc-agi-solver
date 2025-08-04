# 🧠 ARC AGI Solver

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-90%25%20coverage-green.svg)](tests/)

A comprehensive solver for the **ARC (Abstraction and Reasoning Corpus) AGI Challenge**, implementing multiple search strategies and reasoning approaches to tackle visual pattern recognition and transformation tasks.

## 🎯 **Project Overview**

This solver addresses the ARC-AGI challenge through a modular architecture combining:
- **Advanced search algorithms** (A*, beam search, LLM-guided)
- **Sophisticated perception systems** with feature extraction
- **Domain-specific language** for grid transformations
- **Performance optimization** targeting competition-level accuracy

### **Current Status**
- ✅ **Architecture**: Complete modular system with 90%+ test coverage
- ⚡ **Performance**: 2% accuracy on real ARC tasks → **targeting 35%**
- 🔧 **Optimization**: Active performance enhancement in progress
- 📊 **Validation**: Comprehensive testing on real ARC dataset

## 🚀 **Features**

### **🔍 Search Algorithms**
- **A* Search**: Optimal pathfinding with sophisticated heuristics
- **Beam Search**: Parallel exploration of promising solution paths
- **LLM-Guided Search**: AI-assisted solution generation and validation

### **👁️ Perception System**
- **Feature Extraction**: Zernike moments, symmetry detection, blob analysis
- **Pattern Recognition**: Advanced grid analysis and transformation detection
- **Robust Processing**: Error handling for diverse input patterns

### **🛠️ Domain-Specific Language (DSL)**
- **8 Core Primitives**: Fill, MapColors, PaintIf, Rotate, Reflect, etc.
- **Composable Operations**: Complex transformations through primitive combinations
- **Parameter Validation**: Robust input handling and error recovery

### **📈 Performance Monitoring**
- **Real Dataset Validation**: Testing on actual ARC competition data
- **Comprehensive Metrics**: Accuracy, runtime, memory usage tracking
- **Regression Testing**: Automated performance validation

## 📦 **Installation**

### **Prerequisites**
- Python 3.8+
- NumPy, SciPy for numerical computations
- OpenAI API key (for LLM integration)

### **Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/arc-agi-solver.git
cd arc-agi-solver

# Install dependencies
pip install -e .

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
```

## 🎮 **Usage**

### **Command Line Interface**
```bash
# Solve a single ARC task
python -m arc_solver.cli solve path/to/task.json

# Run validation on real dataset
python scripts/validate_real_arc_dataset.py

# Comprehensive testing
python scripts/run_comprehensive_tests.py
```

### **Python API**
```python
from arc_solver.core.data_models import ARCTask
from arc_solver.search.astar import AStarSolver

# Load and solve a task
task = ARCTask.from_file("task.json")
solver = AStarSolver()
solution = solver.solve(task)
```

## 📊 **Performance Targets**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Accuracy** | 2% | 35% | 🔄 In Progress |
| **Runtime** | 3.15s | ≤2s | 🔄 Optimizing |
| **Memory** | 104MB | ≤2GB | ✅ Excellent |
| **Reliability** | 95% | 99% | 🔄 Improving |

## 🏗️ **Architecture**

```
src/arc_solver/
├── core/           # Data models and interfaces
├── perception/     # Feature extraction and analysis
├── reasoning/      # DSL and heuristics
├── search/         # Search algorithms
├── llm/           # LLM integration
├── caching/       # Performance optimization
└── cli/           # Command-line interface
```

### **Key Components**
- **`core/data_models.py`**: ARC task representation and validation
- **`search/astar.py`**: A* search implementation with custom heuristics
- **`reasoning/primitives.py`**: DSL primitive operations
- **`perception/features.py`**: Advanced feature extraction pipeline

## 🧪 **Testing**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/arc_solver --cov-report=html

# Performance regression tests
python tests/test_performance_regression.py
```

### **Test Coverage**
- **Unit Tests**: 90%+ coverage across all modules
- **Integration Tests**: End-to-end solver validation
- **Performance Tests**: Runtime and accuracy benchmarks
- **Real Data Tests**: Validation on actual ARC tasks

## 📈 **Performance Optimization**

The project includes a comprehensive [Performance Optimization Specification](.kiro/specs/arc-agi-solver/performance-optimization.md) addressing:

### **Phase 1: Critical Fixes**
- ✅ DSL parameter validation errors
- ✅ Zernike moment computation failures
- 🔄 Basic search optimizations

### **Phase 2: Core Improvements**
- 🔄 Enhanced A* search strategy
- 🔄 Missing DSL operations
- 🔄 Feature extraction performance

### **Phase 3: Validation**
- 🔄 Large-scale real dataset testing
- 🔄 Competition readiness assessment

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Guidelines**
- Maintain 90%+ test coverage
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Validate performance impact

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **ARC Challenge**: François Chollet and the ARC research community
- **Competition**: ARC Prize 2024/2025 organizers
- **Libraries**: NumPy, SciPy, OpenAI, and the Python ecosystem

## 📞 **Contact**

- **Issues**: [GitHub Issues](https://github.com/yourusername/arc-agi-solver/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/arc-agi-solver/discussions)

---

**🎯 Goal**: Achieve 35% accuracy on real ARC tasks and compete effectively in the ARC-AGI Challenge!