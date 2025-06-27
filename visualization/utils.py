#!/usr/bin/env python3
"""
Visualization Utilities

Common utilities for the visualization component of the Iberian Carbon Assessment Pipeline.
Provides standardized functions for configuration loading, matplotlib styling, 
figure saving, and common data operations.

Author: Diego Bengochea
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Add repo root for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared_utils.central_data_paths import CentralDataPaths
from shared_utils.config_utils import load_config

def setup_logging(script_name: str) -> logging.Logger:
    """
    Set up consistent logging for visualization scripts.
    
    Args:
        script_name: Name of the script for logger identification
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(script_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[{script_name}] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def load_visualization_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load visualization configuration file.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Visualization config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")

def apply_style_config(config: Dict[str, Any]) -> None:
    """
    Apply matplotlib style configuration.
    
    Args:
        config: Configuration dictionary containing style parameters
    """
    if 'style' in config:
        plt.rcParams.update(config['style'])

def create_output_directory(config: Dict[str, Any], subfolder: str = 'main_text') -> Path:
    """
    Create and return output directory for figures.
    
    Args:
        config: Configuration dictionary
        subfolder: Subfolder name ('main_text' or 'supporting_info')
        
    Returns:
        Path to output directory
    """
    figures_dir = Path(config['output']['figures_dir'])
    output_dir = figures_dir / config['output'][f'{subfolder}_subdir']
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_figure_multiple_formats(
    fig: plt.Figure, 
    output_path: Union[str, Path], 
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Save figure in multiple formats specified in config.
    
    Args:
        fig: Matplotlib figure object
        output_path: Base output path (without extension)
        config: Configuration dictionary with export settings
        logger: Optional logger for output messages
        
    Returns:
        List of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    base_path = Path(output_path).with_suffix('')
    export_config = config['export']
    saved_files = []
    
    for fmt in export_config['formats']:
        # Get format-specific kwargs
        format_kwargs = export_config.get(f'{fmt}_kwargs', {})
        
        # Common export parameters
        save_kwargs = {
            'format': fmt,
            'dpi': export_config['dpi'],
            'bbox_inches': export_config['bbox_inches'],
            **format_kwargs
        }
        
        # Save file
        output_file = f"{base_path}.{fmt}"
        fig.savefig(output_file, **save_kwargs)
        saved_files.append(output_file)
        
        if logger:
            logger.info(f"Saved figure: {output_file}")
    
    return saved_files

def format_ticks_no_decimals(x: float, pos: int) -> str:
    """
    Custom formatter for tick labels without decimal points.
    
    Args:
        x: Tick value
        pos: Tick position
        
    Returns:
        Formatted tick label
    """
    return f"{int(x)}" if x == int(x) else f"{x:.1f}"

def load_biomass_data(
    data_paths: CentralDataPaths, 
    years: List[int], 
    biomass_type: str = 'TBD',
    measure: str = 'mean',
    with_mask: bool = True
) -> Dict[int, Path]:
    """
    Load biomass data file paths for multiple years.
    
    Args:
        data_paths: CentralDataPaths instance
        years: List of years to load
        biomass_type: Type of biomass (TBD, AGBD, BGBD)
        measure: Measure type (mean, uncertainty)
        with_mask: Whether to use masked biomass files
        
    Returns:
        Dictionary mapping years to file paths
    """
    biomass_files = {}
    for year in years:
        file_path = data_paths.get_biomass_path(
            biomass_type=biomass_type,
            year=year,
            measure=measure,
            with_mask=with_mask
        )
        biomass_files[year] = file_path
    return biomass_files

def load_trend_data(
    data_paths: CentralDataPaths,
    filename: str = 'biomass_national_by_year.csv',
    biomass_to_carbon: float = 0.5
) -> pd.DataFrame:
    """
    Load and process temporal trend data.
    
    Args:
        data_paths: CentralDataPaths instance
        filename: Name of the trend data file
        biomass_to_carbon: Conversion factor from biomass to carbon
        
    Returns:
        Processed DataFrame with carbon values and midyear points
    """
    try:
        # Look for trend data in analysis outputs
        trend_path = data_paths.get_path('analysis_outputs') / filename
        
        if not trend_path.exists():
            # Fallback to results directory
            trend_path = data_paths.get_path('results') / filename
            
        if not trend_path.exists():
            raise FileNotFoundError(f"Trend data file not found: {filename}")
        
        trend_data = pd.read_csv(trend_path)
        
        # Convert biomass columns to carbon
        biomass_cols = [col for col in trend_data.columns if 'biomass' in col.lower()]
        for col in biomass_cols:
            carbon_col = col.replace('biomass', 'carbon').replace('Biomass', 'Carbon')
            trend_data[carbon_col] = trend_data[col] * biomass_to_carbon
        
        # Create midyear points for summer measurement representation
        if 'year' in trend_data.columns:
            trend_data['year_mid'] = trend_data['year'] + 0.5
        
        return trend_data
        
    except Exception as e:
        raise RuntimeError(f"Error loading trend data: {e}")

def load_climate_analysis_results(data_paths: CentralDataPaths) -> Dict[str, Any]:
    """
    Load pre-computed climate analysis results for Figure 3.
    
    Args:
        data_paths: CentralDataPaths instance
        
    Returns:
        Dictionary containing SHAP analysis results
    """
    # Look for SHAP analysis results in multiple possible locations
    possible_locations = [
        data_paths.get_path('ml_outputs') / 'shap_analysis',
        data_paths.get_path('analysis_outputs') / 'shap_analysis',
        data_paths.get_path('results') / 'climate_biomass_analysis' / 'shap_analysis'
    ]
    
    results_dir = None
    for location in possible_locations:
        if location.exists():
            results_dir = location
            break
    
    if results_dir is None:
        raise FileNotFoundError(
            "SHAP analysis results not found. Expected locations:\n" +
            "\n".join(f"  - {loc}" for loc in possible_locations)
        )
    
    results = {}
    required_files = {
        'feature_frequencies': 'feature_frequencies_df.csv',
        'pdp_lowess_data': 'pdp_lowess_data.pkl',
        'interaction_results': 'interaction_results.pkl',
        'avg_shap_importance': 'avg_shap_importance.pkl'
    }
    
    # Load required files
    for key, filename in required_files.items():
        file_path = results_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required SHAP analysis file not found: {file_path}")
        
        if filename.endswith('.csv'):
            results[key] = pd.read_csv(file_path)
        elif filename.endswith('.pkl'):
            import pickle
            with open(file_path, 'rb') as f:
                results[key] = pickle.load(f)
    
    return results

def validate_required_files(file_paths: List[Union[str, Path]], script_name: str) -> None:
    """
    Validate that all required input files exist.
    
    Args:
        file_paths: List of file paths to validate
        script_name: Name of the calling script for error messages
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    missing_files = []
    for file_path in file_paths:
        if not Path(file_path).exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        raise FileNotFoundError(
            f"[{script_name}] Required input files not found:\n" +
            "\n".join(f"  - {file}" for file in missing_files)
        )

def get_figure_output_path(
    config: Dict[str, Any], 
    figure_name: str, 
    subfolder: str = 'main_text'
) -> Path:
    """
    Generate standardized output path for a figure.
    
    Args:
        config: Configuration dictionary
        figure_name: Name of the figure (without extension)
        subfolder: Subfolder name ('main_text' or 'supporting_info')
        
    Returns:
        Path for figure output (without extension)
    """
    output_dir = create_output_directory(config, subfolder)
    return output_dir / figure_name

def format_variable_name(var_name: str, variable_mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Format variable names for display in figures.
    
    Args:
        var_name: Raw variable name
        variable_mapping: Optional mapping dictionary for variable names
        
    Returns:
        Formatted variable name
    """
    if variable_mapping and var_name in variable_mapping:
        return variable_mapping[var_name]
    
    # Default formatting
    formatted = var_name.replace('_', ' ').capitalize()
    return formatted