#!/usr/bin/env python3
"""
Generate Figure 4: Model Validation

This script creates a comprehensive validation figure with two panels:
- Panel A: Regression scatter plot with density coloring
- Panel B: Two aligned plots showing residual distributions (boxplot) and biomass histogram

TODO: currently there's no script extracting the predicted biomass values with the NFI point layers. Must check which created point layer can be used to sample. 

Author: Diego Bengochea
Component: Visualization Pipeline
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as stats
import geopandas as gpd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.ticker as ticker
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, save_figure_multiple_formats,
    get_figure_output_path, validate_required_files,
    format_ticks_no_decimals
)

from shared_utils.central_data_paths_constants import  *

# Set up logging
logger = setup_logging('figure_04')

def find_validation_data(data_paths) -> Path:
    """
    Find validation data file in the expected locations.
    
    Args:
        data_paths: CentralDataPaths instance
        
    Returns:
        Path to validation data file
    """
    # Possible validation file names and locations
    filepath = BIOMASS_VALIDATION_FILE

    if filepath.exists():
        logger.info(f"Found validation data: {filepath}")
        return filepath

    raise FileNotFoundError(
        f"Validation data file not found. Searched location: {filepath}" 
    )

def prepare_data(data_path: Path) -> tuple:
    """
    Prepare the data for analysis, including calculating residuals and statistics.
    
    Args:
        data_path: Path to the validation data
        
    Returns:
        Tuple of (processed_data, statistics_dict)
    """
    logger.info(f"Loading validation data: {data_path}")
    
    data = gpd.read_file(data_path)
    
    # Drop rows with missing values - 
    data = data.dropna()
    
    # Create TBD column as the sum of AGB and BGB - 
    data['TBD'] = data['AGB'] + data['BGB']
    
    # Column names for measured and estimated biomass - 
    measured_col = 'TBD'  # Using TBD instead of AGB
    predicted_col = 'predicted_'  # Assuming 'estimated_' is the field name for predictions
    
    # Calculate residuals - 
    data['residual'] = data[estimated_col] - data[measured_col]
    data['rel_residual'] = (data['residual'] / data[measured_col]) * 100  # percentage error
    
    # Create linear binning scheme with more resolution in lower values TODO: CHECK BINNING
    linear_bins = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
    linear_labels = ['0-40', '40-80', '80-120', '120-160', '160-200', '200-240', '240-280', '280-320', '320-360']
    
    # Apply linear binning to measured biomass - 
    data['biomass_bin'] = pd.cut(data[measured_col], bins=linear_bins, labels=linear_labels, right=False)
    
    # Add bin midpoint for continuous x-axis in boxplot - 
    bin_midpoints = {
        '0-40': 20,
        '40-80': 60,
        '80-120': 100,
        '120-160': 140,
        '160-200': 180,
        '200-240': 220,
        '240-280': 260,
        '280-320': 300,
        '320-360': 340
    }
    data['bin_midpoint'] = data['biomass_bin'].map(bin_midpoints)
    
    # Calculate validation statistics - 
    stats_dict = calculate_statistics(data, measured_col, estimated_col)
    
    return data, stats_dict

def calculate_statistics(data: pd.DataFrame, measured_col: str = 'TBD', estimated_col: str = 'predicted_') -> dict:
    """
    Calculate validation statistics
    
    Args:
        data: DataFrame containing measured and predicted biomass
        measured_col: Column name for measured values (default: 'TBD')
        estimated_col: Column name for predicted values (default: 'estimated_')
        
    Returns:
        Dictionary with statistics
    """
    # Filter out NaN values - 
    valid_data = data.dropna(subset=[measured_col, estimated_col])
    
    # Calculate metrics - 
    r = stats.pearsonr(valid_data[measured_col], valid_data[estimated_col])[0]
    
    # Use sklearn's r2_score instead of square of Pearson - 
    r2 = r2_score(valid_data[measured_col], valid_data[estimated_col])
    
    rmse = np.sqrt(mean_squared_error(valid_data[measured_col], valid_data[estimated_col]))
    mae = mean_absolute_error(valid_data[measured_col], valid_data[estimated_col])
    bias = np.mean(valid_data[estimated_col] - valid_data[measured_col])
    rel_rmse = (rmse / np.mean(valid_data[measured_col])) * 100  # RMSE as percentage of mean
    
    # Get sample size - 
    n = len(valid_data)
    
    # Calculate median values for distribution plots - 
    median_measured = np.median(valid_data[measured_col])
    median_predicted = np.median(valid_data[estimated_col])
    
    return {
        'r': r,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'rel_rmse': rel_rmse,
        'n': n,
        'mean_measured': np.mean(valid_data[measured_col]),
        'mean_predicted': np.mean(valid_data[estimated_col]),
        'median_measured': median_measured,
        'median_predicted': median_predicted
    }

def create_regression_plot(ax, data: pd.DataFrame, stats_dict: dict):
    """
    Create enhanced regression scatter plot with density coloring
    
    Args:
        ax: Matplotlib axes object
        data: Validation data
        stats_dict: Statistics dictionary
    """
    # Filter data for plotting (remove zeros and NaNs) - 
    plot_data = data.dropna(subset=['TBD', 'estimated_'])
    plot_data = plot_data[(plot_data['TBD'] > 0) & (plot_data['estimated_'] > 0)]
    
    # No background color set - use default - 
    
    # Create hexbin plot with magma colormap - 
    hb = ax.hexbin(plot_data['TBD'], plot_data['estimated_'], 
                  gridsize=100, cmap='magma', mincnt=1, bins='log',
                  xscale='log', yscale='log', alpha=1.0)  # No transparency
    
    # Add 1:1 line with white color and black outline for better visibility (narrower) - 
    min_val = np.min([plot_data['TBD'].min(), plot_data['estimated_'].min()])
    max_val = np.max([plot_data['TBD'].max(), plot_data['estimated_'].max()])
    lims = [max(1, min_val*0.9), min(500, max_val*1.1)]
    
    # Create 1:1 line with path effect for outline - narrower as requested - 
    line = ax.plot(lims, lims, 'w-', linewidth=1.5, zorder=10)[0]
    line.set_path_effects([
        path_effects.Stroke(linewidth=2.5, foreground='black'),
        path_effects.Normal()
    ])
    
    # Set axis limits - with 10 as lower limit for y-axis - 
    ax.set_xlim(lims)
    ax.set_ylim(10, lims[1])
    
    # Format axes with split labels for better spacing - 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('NFI biomass\n(Mg.ha$^{-1}$)', fontsize=10)
    ax.set_ylabel('Predicted biomass\n(Mg.ha$^{-1}$)', fontsize=10)
    
    # Remove all gridlines as requested - 
    ax.grid(False)
    
    # Apply despine - 
    sns.despine(ax=ax)
    
    # Add colorbar - positioned slightly higher - 
    cax = ax.inset_axes([0.825, 0.03, 0.03, 0.22])
    cbar = plt.colorbar(hb, cax=cax, orientation='vertical')
    cbar.set_label('Count', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    # Add statistics - only R², MAE and Bias as requested - 
    stats_text = (f"R² = {stats_dict['r2']:.3f}\n"
                 f"MAE = {stats_dict['mae']:.2f} Mg.ha"r"$^{-1}$""\n"
                 f"Bias = {stats_dict['bias']:.2f} Mg.ha"r"$^{-1}$")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.4'),
           fontsize=8)  # Reduced stats text font size for better fit

def create_residual_boxplot(ax, data: pd.DataFrame):
    """
    Create boxplot of absolute residuals with proper distribution along linear x-axis
    
    Args:
        ax: Matplotlib axes object
        data: Validation data
    """
    # Filter data for plotting - 
    plot_data = data.dropna(subset=['TBD', 'residual', 'biomass_bin', 'bin_midpoint'])
    plot_data = plot_data[plot_data['TBD'] > 0]
    
    # No background color set - use default - 
    
    # Define x-axis limits for linear scale - limited to 400 as requested - 
    xlim = [0, 400]
    
    # Set x-axis to linear scale with limited range - 
    ax.set_xlim(xlim)
    
    # Define a lighter gray color for boxes - 
    box_color = '#D0D0D0'  # Light gray
    
    # Create boxplot for absolute residuals at the bin midpoints - 
    all_positions = []
    all_residuals = []
    
    for bin_name, bin_data in plot_data.groupby('biomass_bin'):
        if len(bin_data) < 3:  # Skip bins with too few samples
            continue
            
        # Get bin midpoint for linear scale - 
        pos = bin_midpoints = {
            '0-40': 20,
            '40-80': 60,
            '80-120': 100,
            '120-160': 140,
            '160-200': 180,
            '200-240': 220,
            '240-280': 260,
            '280-320': 300,
            '320-360': 340
        }[bin_name]
        
        all_positions.append(pos)
        all_residuals.append(bin_data['residual'].values)
    
    # Create a single boxplot with all positions at once for consistent styling - 
    # Make boxes narrower and add notches
    bp = ax.boxplot(all_residuals, positions=all_positions, 
                   widths=[15] * len(all_positions),  # Much narrower boxes (15 units)
                   patch_artist=True, showfliers=False, 
                   notch=True)  # Add notches for confidence interval of median
    
    # Style the boxes with the chosen color - no transparency - 
    for box in bp['boxes']:
        box.set(facecolor=box_color)  # Light gray
        box.set(edgecolor='black')  # Dark edges as requested
        
    # Style the whiskers and caps - 
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    
    # Add zero line for residuals - 
    ax.axhline(y=0, color='darkgrey', linestyle='--', zorder=0)
    
    # Set y-axis to linear scale with split label to avoid overlap - 
    ax.set_ylabel('Residual\n(Mg.ha$^{-1}$)', fontsize=10)
    
    # Add gridlines as requested - 
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Apply despine - 
    sns.despine(ax=ax)

def create_distribution_histogram(ax, data: pd.DataFrame):
    """
    Create histogram of measured biomass with linear bins
    
    Args:
        ax: Matplotlib axes object
        data: Validation data
    """
    # Filter data for plotting - 
    plot_data = data.dropna(subset=['TBD'])
    plot_data = plot_data[plot_data['TBD'] > 0]
    
    # No background color set - use default - 
    
    # Create linear bins with 10-unit width for the histogram - 
    # This ensures bars align with tick marks at multiples of 10
    linear_bins = np.arange(0, 401, 10)  # 0, 10, 20, 30, ..., 400
    
    # Define a light, pleasant green color - 
    hist_color = '#BFDBC5'  # Current light green color looks good
    
    # Plot histogram as percentage of samples with visible edges - no transparency - 
    n, bins, patches = ax.hist(plot_data['TBD'], bins=linear_bins, 
                              color=hist_color,
                              edgecolor='black', linewidth=0.5,
                              weights=np.ones(len(plot_data)) / len(plot_data) * 100)
    
    # Set y-axis label for distribution with consistent formatting - 
    ax.set_ylabel('Percentage of\nsamples (%)', fontsize=10)
    
    # Set x-axis label with split formatting to match other labels - 
    ax.set_xlabel('Biomass\n(Mg.ha$^{-1}$)', fontsize=10)
    
    # Add gridlines as requested - 
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Set major ticks at multiples of 40 to avoid overcrowding with 10-unit bins - 
    ax.set_xticks(np.arange(0, 401, 40))
    
    # Apply despine - 
    sns.despine(ax=ax)

def create_validation_figure(data: pd.DataFrame, stats_dict: dict, config: dict) -> plt.Figure:
    """
    Create improved validation figure with two main panels - EXACT SAME LAYOUT LOGIC AS ORIGINAL
    
    Args:
        data: Processed validation data
        stats_dict: Dictionary with validation statistics
        config: Configuration dictionary
        
    Returns:
        Figure object
    """
    # Set up figure with journal-compliant size (169mm width) - 
    fig_params = config['figure_params']['figure4']
    fig = plt.figure(figsize=fig_params['figsize'])  # 169mm × 140mm
    
    # Create main grid with 2 columns - increased spacing to reduce label overlap - 
    gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[0.85, 1.15])
    
    # Set style for all plots - remove spines - 
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Create a nested gridspec for the left panel - less constrained height - 
    gs_left = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[0, 0], 
                                             height_ratios=[0.5, 0.9, 0.5], hspace=0)
    
    # Panel A: Regression scatter plot (centered in left panel, taller) - 
    ax_reg = fig.add_subplot(gs_left[1, 0])
    create_regression_plot(ax_reg, data, stats_dict)
    # Add panel label with consistent font size (10) - 
    ax_reg.set_title('a) Predicted vs. NFI biomass', fontsize=10, loc='left', y=1.05)

    # Create nested gridspec for right panel with 2 rows - more width, less height - 
    gs_right = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[0, 1], 
                                              height_ratios=[0.5,0.45,0.45,0.5], hspace=0.05)
    
    # Panel B-1: Residual boxplot (top right) - 
    ax_res = fig.add_subplot(gs_right[1, 0])
    
    # Panel B-2: Distribution histogram (bottom right, shares x-axis with boxplot) - 
    ax_hist = fig.add_subplot(gs_right[2, 0], sharex=ax_res)
    
    # Create the combined right panel plots - 
    create_residual_boxplot(ax_res, data)
    create_distribution_histogram(ax_hist, data)
    
    # Hide x tick labels on the top plot since they share an axis - 
    plt.setp(ax_res.get_xticklabels(), visible=False)
    
    # Only add title to the top plot (consistent font size of 10, split into two lines) - 
    ax_res.set_title('b) Model residuals and distribution\nof NFI biomass samples', fontsize=10, loc='left', y=1.11)
    
    # Apply custom tick formatter to remove decimal places - 
    for ax in [ax_reg, ax_res, ax_hist]:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks_no_decimals))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks_no_decimals))
        ax.tick_params(axis='both', labelsize=8)  # Consistent tick label size
    
    # Align y-labels in panel B - 
    fig.align_ylabels([ax_res, ax_hist])
    
    return fig

def main():
    """Main function to run validation analysis and create figure."""
    logger.info("Starting Figure 2 creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    try:
        # Find validation data
        data_path = find_validation_data(data_paths)
        
        # Prepare data and calculate statistics
        data, stats_dict = prepare_data(data_path)
        
        # Print summary statistics
        logger.info("Validation Statistics:")
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Create validation figure
        fig = create_validation_figure(data, stats_dict, config)
        
        # Save figure
        output_path = FIGURE_MAIN_DIR / 'figure_2_biomass_model_validation'
        save_figure_multiple_formats(fig, output_path, config, logger)
        
        plt.close()
        logger.info("Figure 2 created successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Validation data not found: {e}")
        logger.error("Please ensure biomass model validation has been run and produced validation results")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating Figure 4: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()