"""Tests for configuration management system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from arc_solver.config import (
    ConfigManager, load_config, get_config, validate_config, ConfigValidationError
)
from arc_solver.config.config_manager import ConfigContext


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "conf"
        config_dir.mkdir()
        
        # Create minimal config file
        config_content = """
solver:
  name: "test-solver"
  max_program_length: 4
  timeout_seconds: 30.0

performance:
  target_accuracy: 0.35
  median_runtime: 0.5

perception:
  blob_labeling:
    use_gpu: true
    connectivity: 4
  features:
    total_dimension: 50

reasoning:
  dsl_engine:
    max_program_length: 4
    max_execution_time: 0.001

search:
  astar:
    max_nodes_expanded: 600
    max_computation_time: 30.0
  beam_search:
    initial_beam_width: 64
    min_beam_width: 8

system:
  hardware:
    gpu:
      enabled: true
      device_id: 0
    cpu:
      num_threads: -1
"""
        
        config_file = config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        yield config_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_config_manager_initialization(self, temp_config_dir):
        """Test ConfigManager initialization."""
        manager = ConfigManager(temp_config_dir)
        assert manager.config_dir.resolve() == temp_config_dir.resolve()
        assert manager.config is None
    
    def test_load_config_basic(self, temp_config_dir):
        """Test basic configuration loading."""
        manager = ConfigManager(temp_config_dir)
        config = manager.load_config()
        
        assert isinstance(config, DictConfig)
        assert config.solver.name == "test-solver"
        assert config.solver.max_program_length == 4
        assert manager.config is config
    
    def test_load_config_with_overrides(self, temp_config_dir):
        """Test configuration loading with overrides."""
        manager = ConfigManager(temp_config_dir)
        overrides = [
            "solver.max_program_length=2",
            "search.beam_search.initial_beam_width=32"
        ]
        
        config = manager.load_config(overrides=overrides)
        
        assert config.solver.max_program_length == 2
        assert config.search.beam_search.initial_beam_width == 32
    
    def test_get_parameter(self, temp_config_dir):
        """Test parameter retrieval."""
        manager = ConfigManager(temp_config_dir)
        config = manager.load_config()
        
        # Test existing parameter
        assert manager.get_parameter("solver.name") == "test-solver"
        assert manager.get_parameter("solver.max_program_length") == 4
        
        # Test nested parameter
        assert manager.get_parameter("search.astar.max_nodes_expanded") == 600
        
        # Test non-existent parameter with default
        assert manager.get_parameter("nonexistent.param", "default") == "default"
    
    def test_set_parameter(self, temp_config_dir):
        """Test parameter setting."""
        manager = ConfigManager(temp_config_dir)
        config = manager.load_config()
        
        # Set existing parameter
        manager.set_parameter("solver.max_program_length", 3)
        assert manager.get_parameter("solver.max_program_length") == 3
        
        # Set new parameter
        manager.set_parameter("new.parameter", "test_value")
        assert manager.get_parameter("new.parameter") == "test_value"
    
    def test_update_config(self, temp_config_dir):
        """Test configuration updates."""
        manager = ConfigManager(temp_config_dir)
        config = manager.load_config()
        
        updates = {
            "solver.max_program_length": 3,
            "search.beam_search.initial_beam_width": 32
        }
        
        manager.update_config(updates)
        
        assert manager.get_parameter("solver.max_program_length") == 3
        assert manager.get_parameter("search.beam_search.initial_beam_width") == 32
    
    def test_save_config(self, temp_config_dir):
        """Test configuration saving."""
        manager = ConfigManager(temp_config_dir)
        config = manager.load_config()
        
        # Modify config
        manager.set_parameter("solver.name", "modified-solver")
        
        # Save config
        output_file = temp_config_dir / "saved_config.yaml"
        manager.save_config(output_file)
        
        assert output_file.exists()
        
        # Load saved config and verify
        saved_config = OmegaConf.load(output_file)
        assert saved_config.solver.name == "modified-solver"
    
    def test_config_without_loading(self, temp_config_dir):
        """Test operations without loading config first."""
        manager = ConfigManager(temp_config_dir)
        
        with pytest.raises(RuntimeError, match="No configuration loaded"):
            manager.get_parameter("solver.name")
        
        with pytest.raises(RuntimeError, match="No configuration loaded"):
            manager.set_parameter("solver.name", "test")
        
        with pytest.raises(RuntimeError, match="No configuration loaded"):
            manager.update_config({"solver.name": "test"})
        
        with pytest.raises(RuntimeError, match="No configuration loaded"):
            manager.save_config("test.yaml")


class TestGlobalConfigFunctions:
    """Test global configuration functions."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "conf"
        config_dir.mkdir()
        
        # Create minimal config file
        config_content = """
solver:
  name: "global-test-solver"
  max_program_length: 4

performance:
  target_accuracy: 0.35
"""
        
        config_file = config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        yield config_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_load_config_global(self, temp_config_dir):
        """Test global load_config function."""
        config = load_config(config_dir=temp_config_dir)
        
        assert isinstance(config, DictConfig)
        assert config.solver.name == "global-test-solver"
        
        # Test get_config
        global_config = get_config()
        assert global_config is config
    
    def test_load_config_with_overrides_global(self, temp_config_dir):
        """Test global load_config with overrides."""
        overrides = ["solver.max_program_length=2"]
        config = load_config(overrides=overrides, config_dir=temp_config_dir)
        
        assert config.solver.max_program_length == 2


class TestConfigContext:
    """Test ConfigContext context manager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "conf"
        config_dir.mkdir()
        
        # Create minimal config file
        config_content = """
solver:
  name: "context-test-solver"
  max_program_length: 4

search:
  beam_search:
    initial_beam_width: 64
"""
        
        config_file = config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        yield config_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_config_context(self, temp_config_dir):
        """Test ConfigContext functionality."""
        # Load initial config
        config = load_config(config_dir=temp_config_dir)
        assert config.solver.max_program_length == 4
        assert config.search.beam_search.initial_beam_width == 64
        
        # Use context manager to temporarily change values
        with ConfigContext(
            **{
                "solver.max_program_length": 2,
                "search.beam_search.initial_beam_width": 32
            }
        ) as ctx_config:
            assert ctx_config.solver.max_program_length == 2
            assert ctx_config.search.beam_search.initial_beam_width == 32
        
        # Values should be restored after context
        final_config = get_config()
        assert final_config.solver.max_program_length == 4
        assert final_config.search.beam_search.initial_beam_width == 64


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = OmegaConf.create({
            "solver": {
                "max_program_length": 4,
                "timeout_seconds": 30.0
            },
            "performance": {
                "target_accuracy": 0.35,
                "median_runtime": 0.5,
                "max_gpu_memory": 2.0,
                "max_ram_memory": 6.0
            },
            "perception": {
                "blob_labeling": {
                    "connectivity": 4
                },
                "features": {
                    "total_dimension": 50
                }
            },
            "reasoning": {
                "dsl_engine": {
                    "max_program_length": 4,
                    "max_execution_time": 0.001
                }
            },
            "search": {
                "astar": {
                    "max_nodes_expanded": 600,
                    "max_computation_time": 30.0
                },
                "beam_search": {
                    "initial_beam_width": 64,
                    "min_beam_width": 8
                },
                "heuristics": {
                    "tier2_threshold": 5.0
                }
            },
            "system": {
                "hardware": {
                    "gpu": {
                        "device_id": 0,
                        "memory_limit": 2.0
                    },
                    "cpu": {
                        "num_threads": -1
                    }
                },
                "caching": {
                    "redis": {
                        "port": 6379
                    },
                    "file_cache": {
                        "max_cache_size": 1.0
                    }
                }
            }
        })
        
        # Should not raise exception
        validate_config(config)
    
    def test_invalid_solver_config(self):
        """Test validation of invalid solver configuration."""
        config = OmegaConf.create({
            "solver": {
                "max_program_length": -1,  # Invalid
                "timeout_seconds": -5.0    # Invalid
            }
        })
        
        with pytest.raises(ConfigValidationError):
            validate_config(config)
    
    def test_invalid_performance_config(self):
        """Test validation of invalid performance configuration."""
        config = OmegaConf.create({
            "performance": {
                "target_accuracy": 1.5,  # Invalid (>1)
                "median_runtime": -1.0   # Invalid (negative)
            }
        })
        
        with pytest.raises(ConfigValidationError):
            validate_config(config)
    
    def test_invalid_search_config(self):
        """Test validation of invalid search configuration."""
        config = OmegaConf.create({
            "search": {
                "astar": {
                    "max_nodes_expanded": -100  # Invalid
                },
                "beam_search": {
                    "initial_beam_width": 32,
                    "min_beam_width": 64  # Invalid (> initial)
                }
            }
        })
        
        with pytest.raises(ConfigValidationError):
            validate_config(config)
    
    def test_empty_config_sections(self):
        """Test validation with empty configuration sections."""
        config = OmegaConf.create({})
        
        # Should not raise exception for empty config
        validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__])