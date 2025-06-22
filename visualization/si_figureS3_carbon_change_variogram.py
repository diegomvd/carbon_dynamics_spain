#!/usr/bin/env python3
"""
Generate Figure S3: Semivariogram Analysis

Creates semivariogram plot for multiple years showing spatial autocorrelation patterns
in biomass change data.

Author: Diego Bengochea
Component: Visualization Pipeline - Supporting Information
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib import rcParams
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, save_figure_multiple_formats,
    get_figure_output_path, validate_required_files
)

# Set up logging
logger = setup_logging('figure_si_03')

def find_semivariogram_data(data_paths) -> Path:
    """
    Auto-detect semivariogram data file.
    
    Args:
        data_paths: CentralDataPaths instance
        
    Returns:
        Path to semivariogram CSV file
    """
    # Expected filename patterns
    patterns = [
        "semivariogram_data*.csv",
        "semivariogram_*.csv", 
        "*semivariogram*.csv"
    ]
    
    # Search locations in order of preference
    search_locations = [
        data_paths.get_path('analysis_outputs'),
        data_paths.get_path('results'),
        data_paths.get_path('tables'),
        data_paths.data_root / "results" / "biomass_analysis",
        data_paths.data_root / "spatial_analysis"
    ]
    
    for location in search_locations:
        if location.exists():
            for pattern in patterns:
                files = list(location.glob(pattern))
                if files:
                    # Return the most recent file if multiple matches
                    semivariogram_file = max(files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"Found semivariogram data: {semivariogram_file}")
                    return semivariogram_file
    
    # If not found, provide helpful error message
    searched_paths = []
    for location in search_locations:
        for pattern in patterns:
            searched_paths.append(str(location / pattern))
    
    raise FileNotFoundError(
        f"Semivariogram data not found. Searched for patterns:\n" +
        "\n".join(f"  - {path}" for path in searched_paths) +
        f"\n\nPlease ensure spatial analysis has been run and produced semivariogram outputs."
    )

def extract_year_period(filename):
    """Extract year period from filename like TBD_S2_relative_change_symmetric_2018Sep-2019Aug_100m.tif - EXACT SAME LOGIC"""
    match = re.search(r'(\d{4})Sep-(\d{4})Aug', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None

def load_and_process_data(csv_file: Path):
    """Load CSV data and extract year periods - EXACT SAME LOGIC"""
    logger.info(f"Loading semivariogram data: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Extract year periods from filenames
    df['year_period'] = df['file'].apply(extract_year_period)
    
    # Remove any rows with missing year periods
    df = df.dropna(subset=['year_period'])
    
    logger.info(f"Loaded {len(df)} semivariogram data points")
    return df

def create_variogram_plot(df: pd.DataFrame, config: dict) -> plt.Figure:
    """
    Create semivariogram plot for multiple years - EXACT SAME LOGIC
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with semivariogram data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    # Figure size: 140mm x 100mm converted to inches - EXACT SAME LOGIC
    figsize = (140/25.4, 100/25.4)
    
    # Get unique year periods and sort them - EXACT SAME LOGIC
    year_periods = sorted(df['year_period'].unique())
    n_years = len(year_periods)
    
    logger.info(f"Found {n_years} year periods: {year_periods}")
    
    # Print data summary - EXACT SAME LOGIC
    logger.info("Data summary by year:")
    for period in year_periods:
        period_data = df[df['year_period'] == period]
        logger.info(f"{period}: {len(period_data)} points, "
                   f"semivariance range: {period_data['semivariance'].min():.1f} - {period_data['semivariance'].max():.1f}")
    
    # Create the figure - EXACT SAME LOGIC
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Generate colors using Dark2 palette - EXACT SAME LOGIC
    colors = sns.color_palette("Dark2", n_colors=n_years)
    
    # Plot each year period - EXACT SAME LOGIC
    for i, period in enumerate(year_periods):
        period_data = df[df['year_period'] == period].copy()
        
        # Sort by distance for proper line connection
        period_data = period_data.sort_values('distance_km')
        
        # Plot scatter points and connecting line
        ax.scatter(period_data['distance_km'], period_data['semivariance'], 
                  color=colors[i], alpha=0.7, s=15, zorder=3, label=period)
        ax.plot(period_data['distance_km'], period_data['semivariance'], 
               color=colors[i], linewidth=1.5, alpha=0.8, zorder=2)
    
    # Customize axes - EXACT SAME LOGIC
    ax.set_xlabel('Distance (km)', fontsize=10, fontweight='normal')
    ax.set_ylabel('Semivariance', fontsize=10, fontweight='normal')
    
    # Set tick parameters - EXACT SAME LOGIC
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Add grid - EXACT SAME LOGIC
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Despine the plot (remove top and right spines) - EXACT SAME LOGIC
    sns.despine(ax=ax, top=True, right=True)
    
    # Customize legend - position starting around x=80km with 2 columns - EXACT SAME LOGIC
    legend = ax.legend(fontsize=8, frameon=False, ncol=2,
                      bbox_to_anchor=(0.55, 0.65), loc='center left')
    
    # Adjust layout to prevent clipping - EXACT SAME LOGIC
    plt.tight_layout()
    
    logger.info("Semivariogram plot created successfully")
    logger.info("Using Dark2 color palette for distinct year periods")
    
    return fig

def main():
    """Main function to create Figure S3."""
    logger.info("Starting Figure S3 creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up the plotting style and fonts - EXACT SAME LOGIC as original
    plt.style.use('default')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    rcParams['pdf.fonttype'] = 42  # Ensure fonts are embedded properly in PDFs
    rcParams['ps.fonttype'] = 42
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    try:
        # Auto-detect semivariogram data
        semivariogram_path = find_semivariogram_data(data_paths)
        
        # Validate required files exist
        validate_required_files([semivariogram_path], 'figure_si_03')
        
        # Load and process data
        df = load_and_process_data(semivariogram_path)
        
        if df.empty:
            raise ValueError("No valid semivariogram data found in CSV file")
        
        # Create figure
        fig = create_variogram_plot(df, config)
        
        # Save figure
        output_path = get_figure_output_path(config, "Figure_S3_Semivariogram", 'supporting_info')
        save_figure_multiple_formats(fig, output_path, config, logger)
        
        plt.close()
        logger.info("Figure S3 created successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Required data not found: {e}")
        logger.error("Please ensure spatial analysis has been run and produced semivariogram outputs")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating Figure S3: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()