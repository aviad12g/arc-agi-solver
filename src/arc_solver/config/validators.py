"""Configuration validation for ARC-AGI solver."""

import logging
from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


def validate_config(config: DictConfig) -> None:
    """Validate the complete configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        # Validate each configuration section
        validate_solver_config(config.get('solver', {}))
        validate_performance_config(config.get('performance', {}))
        validate_perception_config(config.get('perception', {}))
        validate_reasoning_config(config.get('reasoning', {}))
        validate_search_config(config.get('search', {}))
        validate_system_config(config.get('system', {}))
        
        logger.info("Configuration validation passed")
        
    except Exception as e:
        raise ConfigValidationError(f"Configuration validation failed: {e}")


def validate_solver_config(solver_config: DictConfig) -> None:
    """Validate solver configuration section.
    
    Args:
        solver_config: Solver configuration section
    """
    if not solver_config:
        return
    
    # Validate max_program_length
    max_length = solver_config.get('max_program_length', 4)
    if not isinstance(max_length, int) or max_length < 1 or max_length > 10:
        raise ConfigValidationError(
            f"max_program_length must be integer between 1 and 10, got {max_length}"
        )
    
    # Validate timeout
    timeout = solver_config.get('timeout_seconds', 30.0)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigValidationError(
            f"timeout_seconds must be positive number, got {timeout}"
        )


def validate_performance_config(perf_config: DictConfig) -> None:
    """Validate performance configuration section.
    
    Args:
        perf_config: Performance configuration section
    """
    if not perf_config:
        return
    
    # Validate accuracy target
    accuracy = perf_config.get('target_accuracy', 0.35)
    if not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
        raise ConfigValidationError(
            f"target_accuracy must be between 0 and 1, got {accuracy}"
        )
    
    # Validate runtime targets
    for key in ['median_runtime', 'max_runtime_95th']:
        runtime = perf_config.get(key, 1.0)
        if not isinstance(runtime, (int, float)) or runtime <= 0:
            raise ConfigValidationError(
                f"{key} must be positive number, got {runtime}"
            )
    
    # Validate memory limits
    for key in ['max_gpu_memory', 'max_ram_memory']:
        memory = perf_config.get(key, 1.0)
        if not isinstance(memory, (int, float)) or memory <= 0:
            raise ConfigValidationError(
                f"{key} must be positive number, got {memory}"
            )


def validate_perception_config(perception_config: DictConfig) -> None:
    """Validate perception configuration section.
    
    Args:
        perception_config: Perception configuration section
    """
    if not perception_config:
        return
    
    # Validate blob labeling config
    blob_config = perception_config.get('blob_labeling', {})
    if blob_config:
        connectivity = blob_config.get('connectivity', 4)
        if connectivity not in [4, 8]:
            raise ConfigValidationError(
                f"blob_labeling.connectivity must be 4 or 8, got {connectivity}"
            )
    
    # Validate feature dimensions
    features_config = perception_config.get('features', {})
    if features_config:
        total_dim = features_config.get('total_dimension', 50)
        if not isinstance(total_dim, int) or total_dim <= 0:
            raise ConfigValidationError(
                f"features.total_dimension must be positive integer, got {total_dim}"
            )
        
        # Validate individual feature dimensions
        expected_dims = {
            'orbit_signature': 8,
            'spectral': 3,
            'persistence': 32,
            'zernike': 7
        }
        
        actual_total = 0
        for feature_name, expected_dim in expected_dims.items():
            feature_config = features_config.get(feature_name, {})
            if feature_config.get('enabled', True):
                dim = feature_config.get('dimension', expected_dim)
                if dim != expected_dim:
                    logger.warning(
                        f"Feature {feature_name} dimension {dim} != expected {expected_dim}"
                    )
                actual_total += dim
        
        if actual_total != total_dim:
            logger.warning(
                f"Sum of feature dimensions ({actual_total}) != total_dimension ({total_dim})"
            )


def validate_reasoning_config(reasoning_config: DictConfig) -> None:
    """Validate reasoning configuration section.
    
    Args:
        reasoning_config: Reasoning configuration section
    """
    if not reasoning_config:
        return
    
    # Validate DSL engine config
    dsl_config = reasoning_config.get('dsl_engine', {})
    if dsl_config:
        max_length = dsl_config.get('max_program_length', 4)
        if not isinstance(max_length, int) or max_length < 1:
            raise ConfigValidationError(
                f"dsl_engine.max_program_length must be positive integer, got {max_length}"
            )
        
        max_time = dsl_config.get('max_execution_time', 0.001)
        if not isinstance(max_time, (int, float)) or max_time <= 0:
            raise ConfigValidationError(
                f"dsl_engine.max_execution_time must be positive number, got {max_time}"
            )
    
    # Validate primitives config
    primitives_config = reasoning_config.get('primitives', {})
    if primitives_config:
        for category_name, primitives in primitives_config.items():
            if isinstance(primitives, list):
                for primitive in primitives:
                    if not isinstance(primitive, dict) or 'name' not in primitive:
                        raise ConfigValidationError(
                            f"Invalid primitive config in {category_name}: {primitive}"
                        )


def validate_search_config(search_config: DictConfig) -> None:
    """Validate search configuration section.
    
    Args:
        search_config: Search configuration section
    """
    if not search_config:
        return
    
    # Validate A* config
    astar_config = search_config.get('astar', {})
    if astar_config:
        max_nodes = astar_config.get('max_nodes_expanded', 600)
        if not isinstance(max_nodes, int) or max_nodes <= 0:
            raise ConfigValidationError(
                f"astar.max_nodes_expanded must be positive integer, got {max_nodes}"
            )
        
        max_time = astar_config.get('max_computation_time', 30.0)
        if not isinstance(max_time, (int, float)) or max_time <= 0:
            raise ConfigValidationError(
                f"astar.max_computation_time must be positive number, got {max_time}"
            )
    
    # Validate beam search config
    beam_config = search_config.get('beam_search', {})
    if beam_config:
        beam_width = beam_config.get('initial_beam_width', 64)
        if not isinstance(beam_width, int) or beam_width <= 0:
            raise ConfigValidationError(
                f"beam_search.initial_beam_width must be positive integer, got {beam_width}"
            )
        
        min_beam = beam_config.get('min_beam_width', 8)
        if not isinstance(min_beam, int) or min_beam <= 0 or min_beam > beam_width:
            raise ConfigValidationError(
                f"beam_search.min_beam_width must be positive integer ≤ initial_beam_width, got {min_beam}"
            )
    
    # Validate heuristics config
    heuristics_config = search_config.get('heuristics', {})
    if heuristics_config:
        threshold = heuristics_config.get('tier2_threshold', 5.0)
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ConfigValidationError(
                f"heuristics.tier2_threshold must be non-negative number, got {threshold}"
            )


def validate_system_config(system_config: DictConfig) -> None:
    """Validate system configuration section.
    
    Args:
        system_config: System configuration section
    """
    if not system_config:
        return
    
    # Validate hardware config
    hardware_config = system_config.get('hardware', {})
    if hardware_config:
        # Validate GPU config
        gpu_config = hardware_config.get('gpu', {})
        if gpu_config:
            device_id = gpu_config.get('device_id', 0)
            if not isinstance(device_id, int) or device_id < 0:
                raise ConfigValidationError(
                    f"hardware.gpu.device_id must be non-negative integer, got {device_id}"
                )
            
            memory_limit = gpu_config.get('memory_limit', 2.0)
            if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
                raise ConfigValidationError(
                    f"hardware.gpu.memory_limit must be positive number, got {memory_limit}"
                )
        
        # Validate CPU config
        cpu_config = hardware_config.get('cpu', {})
        if cpu_config:
            num_threads = cpu_config.get('num_threads', -1)
            if not isinstance(num_threads, int) or (num_threads < 1 and num_threads != -1):
                raise ConfigValidationError(
                    f"hardware.cpu.num_threads must be positive integer or -1, got {num_threads}"
                )
    
    # Validate caching config
    caching_config = system_config.get('caching', {})
    if caching_config:
        # Validate Redis config
        redis_config = caching_config.get('redis', {})
        if redis_config and redis_config.get('enabled', False):
            port = redis_config.get('port', 6379)
            if not isinstance(port, int) or not 1 <= port <= 65535:
                raise ConfigValidationError(
                    f"caching.redis.port must be integer between 1 and 65535, got {port}"
                )
        
        # Validate file cache config
        file_cache_config = caching_config.get('file_cache', {})
        if file_cache_config:
            max_size = file_cache_config.get('max_cache_size', 1.0)
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                raise ConfigValidationError(
                    f"caching.file_cache.max_cache_size must be positive number, got {max_size}"
                )


def validate_parameter_ranges(config: DictConfig) -> List[str]:
    """Validate parameter ranges and return warnings.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check performance targets
    perf_config = config.get('performance', {})
    if perf_config:
        accuracy = perf_config.get('target_accuracy', 0.35)
        if accuracy < 0.35:
            warnings.append(f"target_accuracy {accuracy} is below requirement (≥0.35)")
        
        median_runtime = perf_config.get('median_runtime', 0.5)
        if median_runtime > 0.5:
            warnings.append(f"median_runtime {median_runtime}s exceeds requirement (≤0.5s)")
    
    # Check search parameters
    search_config = config.get('search', {})
    if search_config:
        astar_config = search_config.get('astar', {})
        max_nodes = astar_config.get('max_nodes_expanded', 600)
        if max_nodes > 600:
            warnings.append(f"max_nodes_expanded {max_nodes} exceeds target (≤600)")
    
    return warnings


def check_config_consistency(config: DictConfig) -> List[str]:
    """Check configuration consistency and return issues.
    
    Args:
        config: Configuration to check
        
    Returns:
        List of consistency issues
    """
    issues = []
    
    # Check that max_program_length is consistent across components
    solver_length = config.get('solver', {}).get('max_program_length', 4)
    reasoning_length = config.get('reasoning', {}).get('dsl_engine', {}).get('max_program_length', 4)
    search_length = config.get('search', {}).get('astar', {}).get('max_program_length', 4)
    
    if not (solver_length == reasoning_length == search_length):
        issues.append(
            f"Inconsistent max_program_length: solver={solver_length}, "
            f"reasoning={reasoning_length}, search={search_length}"
        )
    
    # Check timeout consistency
    solver_timeout = config.get('solver', {}).get('timeout_seconds', 30.0)
    search_timeout = config.get('search', {}).get('astar', {}).get('max_computation_time', 30.0)
    
    if abs(solver_timeout - search_timeout) > 0.1:
        issues.append(
            f"Inconsistent timeout: solver={solver_timeout}s, search={search_timeout}s"
        )
    
    return issues