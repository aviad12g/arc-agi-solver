"""Configuration management for ARC-AGI solver.

This module provides Hydra-based configuration management with hierarchical
parameter groups and runtime override capabilities.
"""

from .config_manager import ConfigManager, load_config, get_config
from .validators import validate_config, ConfigValidationError

__all__ = [
    'ConfigManager',
    'load_config', 
    'get_config',
    'validate_config',
    'ConfigValidationError'
]