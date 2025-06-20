"""
Configuration management for biomass estimation pipeline.

This module loads configuration parameters from YAML file and provides
backwards-compatible access to configuration values throughout the pipeline.

Author: Diego Bengochea
"""

import yaml
import os
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {str(e)}")


# Load configuration
_config = load_config()

# Data paths (backwards compatibility)
ALLOMETRIES_DIR = _config['data']['allometries_dir']
FOREST_TYPE_DIR = _config['data']['forest_type_dir']
BGB_COEFFS_DIR = _config['data']['bgb_coeffs_dir']
INPUT_DATA_DIR = _config['data']['input_data_dir']
MFE_DIR = _config['data']['mfe_dir']
MASKS_DIR = _config['data']['masks_dir']

# Processing parameters
HEIGHT_THRESHOLD = _config['processing']['height_threshold']
COMPUTE_NO_ARBOLADO = _config['processing']['compute_no_arbolado']

# Compute parameters
NUM_WORKERS = _config['compute']['num_workers']
MEMORY_LIMIT = _config['compute']['memory_limit']
CHUNK_SIZE = _config['compute']['chunk_size']

# Forest type hierarchy
TIER_NAMES = _config['forest_types']['tier_names']

# Monte Carlo configuration
NUM_MC_SAMPLES = _config['monte_carlo']['num_samples']

# Output configuration
OUTPUT_TYPES = _config['output']['types']


def get_output_directory(output_type):
    """
    Get output directory path for a specific biomass type.
    
    Args:
        output_type (str): Type of biomass output ('agbd', 'bgbd', 'total')
        
    Returns:
        str: Full path to output directory
    """
    base_dir = _config['data']['output_base_dir']
    subdir = _config['data']['biomass_no_masking_dir']
    
    if output_type == 'agbd':
        type_dir = _config['data']['agbd_dir']
    elif output_type == 'bgbd':
        type_dir = _config['data']['bgbd_dir']
    else:  # total
        type_dir = _config['data']['tbd_dir']
    
    return os.path.join(base_dir, subdir, type_dir)


def get_config():
    """
    Get the full configuration dictionary.
    
    Returns:
        dict: Complete configuration dictionary
    """
    return _config
