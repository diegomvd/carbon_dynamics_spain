"""
Configuration utilities for canopy height regression pipeline.

This module provides configuration management following the pattern
from the S2 mosaics component.

Author: Diego Bengochea
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to configuration file.
            If None, loads default config from config/default_config.yaml
            
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file is not found
        yaml.YAMLError: If configuration file is invalid YAML
    """
    if config_path is None:
        # Default config path relative to this file
        config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert paths to Path objects for easier handling
        config['data']['data_dir'] = Path(config['data']['data_dir'])
        config['data']['checkpoint_dir'] = Path(config['data']['checkpoint_dir'])
        if config['data']['checkpoint_path']:
            config['data']['checkpoint_path'] = Path(config['data']['checkpoint_path'])
            
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Setup standardized logging configuration.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('canopy_height_dl')
    return logger


def create_output_directory(path: Path) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path (Path): Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)