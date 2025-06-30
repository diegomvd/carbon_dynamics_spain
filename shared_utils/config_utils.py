"""
Configuration utilities for the Iberian Carbon Assessment Pipeline.

This module provides standardized configuration loading and validation
across all pipeline components.

Author: Diego Bengochea
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    component_name: Optional[str] = None,
    default_config_name: str = "config.yml"
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with standardized search patterns.
    
    Search order:
    1. Explicit config_path if provided
    2. Component directory + default_config_name
    3. Current directory + default_config_name
    4. Environment variable IBERIAN_CARBON_CONFIG
    
    Args:
        config_path: Explicit path to configuration file
        component_name: Name of component (for automatic config discovery)
        default_config_name: Default config filename to search for
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If no configuration file is found
        yaml.YAMLError: If configuration file is invalid YAML
        
    Examples:
        >>> config = load_config()  # Search for config.yaml
        >>> config = load_config("custom_config.yaml")
        >>> config = load_config(component_name="biomass_estimation")
    """
    logger = logging.getLogger(__name__)
    
    # Build search paths
    search_paths = []
    
    # 1. Explicit path
    if config_path:
        search_paths.append(Path(config_path))
    
    # 2. Component directory
    if component_name:
        # Try to find component directory
        component_dirs = [
            Path(f"{component_name}") / default_config_name,
            Path(f"{component_name}_model") / default_config_name,
            Path(f"{component_name}_analysis") / default_config_name,
        ]
        search_paths.extend(component_dirs)
    
    # 3. Current directory
    search_paths.append(Path(default_config_name))
    
    # 4. Environment variable
    env_config = os.environ.get('IBERIAN_CARBON_CONFIG')
    if env_config:
        search_paths.append(Path(env_config))
    
    # Search for config file
    config_file = None
    for path in search_paths:
        if path.exists():
            config_file = path
            logger.debug(f"Found configuration file: {config_file}")
            break
    
    if not config_file:
        searched_paths = [str(p) for p in search_paths]
        raise FileNotFoundError(
            f"Configuration file not found. Searched paths: {searched_paths}"
        )
    
    # Load and parse YAML
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Store config metadata
        config['_meta'] = {
            'config_file': str(config_file.absolute()),
            'component_name': component_name,
            'loaded_at': Path.cwd()
        }
        
        logger.info(f"Loaded configuration from: {config_file}")
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_file}: {e}")


def validate_config(config: Dict[str, Any], required_sections: list = None) -> bool:
    """
    Validate configuration dictionary structure.
    
    Args:
        config: Configuration dictionary to validate
        required_sections: List of required top-level sections
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
        
    Examples:
        >>> validate_config(config, ['data', 'processing', 'output'])
    """
    logger = logging.getLogger(__name__)
    
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Check required sections
    if required_sections:
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Validate paths exist if they're specified
    path_sections = ['data', 'paths', 'directories']
    for section_name in path_sections:
        if section_name in config:
            section = config[section_name]
            if isinstance(section, dict):
                _validate_paths_in_section(section, section_name, logger)
    
    logger.debug("Configuration validation passed")
    return True


def _validate_paths_in_section(section: dict, section_name: str, logger: logging.Logger) -> None:
    """
    Validate paths in a configuration section.
    
    Args:
        section: Configuration section to validate
        section_name: Name of the section for error reporting
        logger: Logger instance
    """
    for key, value in section.items():
        if isinstance(value, str):
            # Check if this looks like a path
            if any(indicator in key.lower() for indicator in ['dir', 'path', 'file']):
                path = Path(value)
                
                # For input paths, warn if they don't exist
                if 'input' in key.lower() or 'data' in key.lower():
                    if not path.exists():
                        logger.warning(f"Input path does not exist: {section_name}.{key} = {value}")
                
                # For output paths, create directory if needed
                elif 'output' in key.lower():
                    if key.lower().endswith(('dir', 'directory')):
                        path.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"Created output directory: {value}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'data.input_dir')
        default: Default value if key is not found
        
    Returns:
        Any: Configuration value or default
        
    Examples:
        >>> input_dir = get_config_value(config, 'data.input_dir', '/default/path')
        >>> batch_size = get_config_value(config, 'training.batch_size', 32)
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def update_config_paths(config: Dict[str, Any], base_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Update relative paths in configuration to be absolute from base_path.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        Dict[str, Any]: Updated configuration dictionary
        
    Examples:
        >>> config = update_config_paths(config, Path.cwd())
    """
    base_path = Path(base_path)
    updated_config = config.copy()
    
    def _update_paths_recursive(obj, current_path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, str) and any(indicator in key.lower() for indicator in ['dir', 'path', 'file']):
                    # Convert relative paths to absolute
                    path = Path(value)
                    if not path.is_absolute():
                        obj[key] = str(base_path / path)
                elif isinstance(value, dict):
                    _update_paths_recursive(value, new_path)
    
    _update_paths_recursive(updated_config)
    return updated_config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Output file path
        
    Examples:
        >>> save_config(config, "backup_config.yaml")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove metadata before saving
    config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration saved to: {output_path}")
