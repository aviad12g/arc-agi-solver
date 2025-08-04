"""Configuration manager using Hydra for hierarchical configuration."""

import os
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from .validators import validate_config

logger = logging.getLogger(__name__)

# Global configuration instance
_global_config: Optional[DictConfig] = None


class ConfigManager:
    """Manages configuration loading and validation using Hydra."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            # Default to conf directory relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_dir = project_root / "conf"
        
        self.config_dir = Path(config_dir).resolve()
        self.config: Optional[DictConfig] = None
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
        logger.info(f"Configuration manager initialized with config_dir: {self.config_dir}")
    
    def load_config(self, 
                   config_name: str = "config",
                   overrides: Optional[list] = None,
                   validate: bool = True) -> DictConfig:
        """Load configuration with optional overrides.
        
        Args:
            config_name: Name of the main config file (without .yaml)
            overrides: List of configuration overrides
            validate: Whether to validate the configuration
            
        Returns:
            Loaded and validated configuration
        """
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        try:
            # Initialize Hydra with config directory
            with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
                # Compose configuration with overrides
                cfg = compose(config_name=config_name, overrides=overrides or [])
                
                # Validate configuration if requested
                if validate:
                    validate_config(cfg)
                
                self.config = cfg
                
                # Set global config
                global _global_config
                _global_config = cfg
                
                logger.info(f"Configuration loaded successfully: {config_name}")
                if overrides:
                    logger.info(f"Applied overrides: {overrides}")
                
                return cfg
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_config(self) -> Optional[DictConfig]:
        """Get the currently loaded configuration.
        
        Returns:
            Current configuration or None if not loaded
        """
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        # Apply updates using OmegaConf
        with OmegaConf.open_dict(self.config):
            for key, value in updates.items():
                OmegaConf.set(self.config, key, value)
        
        logger.info(f"Configuration updated with: {updates}")
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            OmegaConf.save(self.config, f)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a specific parameter from configuration.
        
        Args:
            key: Parameter key (supports dot notation, e.g., 'search.beam_width')
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception as e:
            logger.warning(f"Failed to get parameter '{key}': {e}")
            return default
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set a specific parameter in configuration.
        
        Args:
            key: Parameter key (supports dot notation)
            value: Parameter value
        """
        if self.config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        
        with OmegaConf.open_dict(self.config):
            OmegaConf.set(self.config, key, value)
        
        logger.debug(f"Parameter set: {key} = {value}")
    
    def print_config(self, resolve: bool = True) -> None:
        """Print current configuration.
        
        Args:
            resolve: Whether to resolve interpolations
        """
        if self.config is None:
            print("No configuration loaded.")
            return
        
        print("Current Configuration:")
        print("=" * 50)
        print(OmegaConf.to_yaml(self.config, resolve=resolve))


# Global configuration functions
def load_config(config_name: str = "config",
               overrides: Optional[list] = None,
               config_dir: Optional[Union[str, Path]] = None,
               validate: bool = True) -> DictConfig:
    """Load configuration using global config manager.
    
    Args:
        config_name: Name of the main config file
        overrides: List of configuration overrides
        config_dir: Path to configuration directory
        validate: Whether to validate the configuration
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager(config_dir)
    return manager.load_config(config_name, overrides, validate)


def get_config() -> Optional[DictConfig]:
    """Get the global configuration.
    
    Returns:
        Global configuration or None if not loaded
    """
    return _global_config


def get_parameter(key: str, default: Any = None) -> Any:
    """Get a parameter from global configuration.
    
    Args:
        key: Parameter key (supports dot notation)
        default: Default value if key not found
        
    Returns:
        Parameter value or default
    """
    config = get_config()
    if config is None:
        logger.warning("No global configuration loaded")
        return default
    
    try:
        return OmegaConf.select(config, key, default=default)
    except Exception as e:
        logger.warning(f"Failed to get parameter '{key}': {e}")
        return default


def create_experiment_config(base_config: str = "config",
                           experiment_name: str = "default",
                           overrides: Optional[list] = None) -> DictConfig:
    """Create configuration for an experiment.
    
    Args:
        base_config: Base configuration name
        experiment_name: Name of the experiment
        overrides: Additional configuration overrides
        
    Returns:
        Experiment configuration
    """
    experiment_overrides = [
        f"experiment.name={experiment_name}",
        f"hydra.run.dir=outputs/{experiment_name}/${{now:%Y-%m-%d_%H-%M-%S}}"
    ]
    
    if overrides:
        experiment_overrides.extend(overrides)
    
    return load_config(base_config, experiment_overrides)


# Configuration context manager
class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, **kwargs):
        """Initialize with temporary configuration changes.
        
        Args:
            **kwargs: Configuration parameters to temporarily change
        """
        self.changes = kwargs
        self.original_values = {}
        self.config = get_config()
    
    def __enter__(self):
        """Apply temporary configuration changes."""
        if self.config is None:
            raise RuntimeError("No global configuration loaded")
        
        # Save original values
        for key in self.changes:
            self.original_values[key] = OmegaConf.select(self.config, key)
        
        # Apply changes
        with OmegaConf.open_dict(self.config):
            for key, value in self.changes.items():
                OmegaConf.set(self.config, key, value)
        
        return self.config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original configuration values."""
        if self.config is None:
            return
        
        # Restore original values
        with OmegaConf.open_dict(self.config):
            for key, value in self.original_values.items():
                if value is not None:
                    OmegaConf.set(self.config, key, value)
                else:
                    # Remove key if it didn't exist originally
                    try:
                        OmegaConf.set(self.config, key, None)
                    except Exception:
                        pass