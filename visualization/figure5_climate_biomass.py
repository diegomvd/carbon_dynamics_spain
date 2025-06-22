#!/usr/bin/env python3
"""
Generate Figure 3: Climate-Biomass Relationships

This script creates a figure with two panels:
- Panel A: Individual feature effects using pre-computed LOWESS smoothed SHAP values
- Panel B: Interaction effect heatmap for current and accumulated precipitation

Author: Diego Bengochea
Component: Visualization Pipeline
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, save_figure_multiple_formats,
    get_figure_output_path, load_climate_analysis_results,
    format_variable_name
)

# Set up logging
logger = setup_logging('figure_03')

def determine_top_features(freq_df: pd.DataFrame, top_n: int = 6) -> list:
    """
    Determine top N features based on selection frequency.
    
    Args:
        freq_df: DataFrame with feature frequencies
        top_n: Number of top features to return
        
    Returns:
        List of top feature names
    """
    sorted_features = freq_df.sort_values('frequency', ascending=False).head(top_n)
    
    logger.info(f"Top {top_n} features by selection frequency:")
    for idx, row in sorted_features.iterrows():
        logger.info(f"  {row['feature']}: {row['frequency']*100:.1f}%")
    
    return sorted_features['feature'].tolist()

def create_lowess_panel(ax, results: dict, top_features: list, config: dict):
    """
    Create the LOWESS SHAP effects panel using pre-computed data.
    
    Args:
        ax: Matplotlib axes object
        results: Dictionary containing SHAP analysis results
        top_features: List of top features to display
        config: Configuration dictionary
    """
    variable_mapping = config.get('variable_mapping', {})
    fig_params = config['figure_params']['figure3']
    
    if 'pdp_lowess_data' not in results:
        ax.text(0.5, 0.5, 'LOWESS data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('a) Individual feature effects', fontsize=10)
        sns.despine(ax=ax)
        return
    
    pdp_lowess_data = results['pdp_lowess_data']
    available_features = [f for f in top_features if f in pdp_lowess_data]
    
    if not available_features:
        ax.text(0.5, 0.5, 'No LOWESS data for top features', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('a) Individual feature effects', fontsize=10)
        sns.despine(ax=ax)
        return
    
    logger.info(f"Creating LOWESS panel for {len(available_features)} features")
    
    # Collect all SHAP values for color normalization
    all_shap_values = []
    feature_curves = {}
    
    for feature in available_features:
        if feature in pdp_lowess_data:
            lowess_data = pdp_lowess_data[feature]
            if lowess_data and len(lowess_data) > 0:
                # Extract LOWESS smoothed values
                smooth_x = lowess_data['smooth_x']
                smooth_y = lowess_data['smooth_y']
                
                if smooth_x is not None and smooth_y is not None:
                    feature_curves[feature] = {'x': smooth_x, 'y': smooth_y}
                    all_shap_values.extend(smooth_y)
    
    if not feature_curves:
        ax.text(0.5, 0.5, 'No valid LOWESS curves', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('a) Individual feature effects', fontsize=10)
        sns.despine(ax=ax)
        return
    
    # Create color normalization using 95th percentile
    percentile_95 = np.percentile(np.abs(all_shap_values), 95)
    norm = Normalize(vmin=-percentile_95, vmax=percentile_95)
    cmap = plt.colormaps[fig_params['colormap']]
    
    # Plot colored horizontal lines
    for i, feature in enumerate(available_features):
        if feature not in feature_curves:
            continue
            
        y_pos = len(available_features) - 1 - i
        
        smooth_x = feature_curves[feature]['x']
        smooth_y = feature_curves[feature]['y']
        
        # Plot colored line segments
        for j in range(len(smooth_x) - 1):
            x_start, x_end = smooth_x[j], smooth_x[j + 1]
            shap_value = smooth_y[j]
            
            color = cmap(norm(shap_value))
            
            ax.plot([x_start, x_end], [y_pos, y_pos], 
                    color=color, linewidth=8, solid_capstyle='butt')
    
    # Format the plot
    formatted_names = [format_variable_name(feature, variable_mapping) 
                      for feature in available_features]
    
    ax.set_yticks(range(len(available_features)))
    ax.set_yticklabels(reversed(formatted_names), fontsize=8)
    ax.set_xlabel('Standardized anomaly', fontsize=10)
    ax.set_xlim(fig_params['xlim_range'])
    ax.set_ylim(-0.5, len(available_features) - 0.5)
    ax.set_title('a) Individual feature effects', fontsize=10, y=1.1)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    sns.despine(ax=ax)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('SHAP value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

def create_interaction_heatmap_panel(ax, results: dict, config: dict):
    """
    Create the 2D interaction heatmap panel using pre-computed data.
    
    Args:
        ax: Matplotlib axes object
        results: Dictionary containing interaction results
        config: Configuration dictionary
    """
    variable_mapping = config.get('variable_mapping', {})
    fig_params = config['figure_params']['figure3']
    
    if 'interaction_results' not in results:
        ax.text(0.5, 0.5, 'Interaction data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('b) Interaction effect', fontsize=10)
        sns.despine(ax=ax)
        return
    
    interaction_results = results['interaction_results']
    
    # Get interaction features from config
    feature1, feature2 = fig_params['interaction_features']
    
    logger.info(f"Creating interaction heatmap for {feature1} × {feature2}")
    
    # Check if we have the required interaction data
    required_keys = ['features', 'all_interactions_dict', 'all_feature_values_dict', 'interaction_counts']
    if not all(key in interaction_results for key in required_keys):
        ax.text(0.5, 0.5, 'Incomplete interaction data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('b) Interaction effect', fontsize=10)
        sns.despine(ax=ax)
        return
    
    features = interaction_results['features']
    all_interactions_dict = interaction_results['all_interactions_dict']
    all_feature_values_dict = interaction_results['all_feature_values_dict']
    interaction_counts = interaction_results['interaction_counts']
    
    # Get data for this feature pair
    pair = (feature1, feature2)
    
    if pair not in all_interactions_dict:
        ax.text(0.5, 0.5, f'No interaction data for {feature1} × {feature2}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('b) Interaction effect', fontsize=10)
        sns.despine(ax=ax)
        return
    
    # Extract data for these features
    interaction_values = all_interactions_dict[pair]
    feature_value_pairs = all_feature_values_dict[pair]
    
    feat1_values = feature_value_pairs[:, 0]
    feat2_values = feature_value_pairs[:, 1]
    
    n_models_contributing = interaction_counts[pair]
    logger.info(f"Using {len(interaction_values)} samples from {n_models_contributing} models")
    
    # Create 2D bins for heatmap
    n_bins = 60
    
    # Make plot square by using max range of both features
    feat1_abs_max = np.percentile(np.abs(feat1_values), 98)
    feat2_abs_max = np.percentile(np.abs(feat2_values), 98)
    
    # Use the maximum range to make square plot
    max_range = max(feat1_abs_max, feat2_abs_max)
    plot_min, plot_max = -max_range, max_range
    
    feat1_bins = np.linspace(plot_min, plot_max, n_bins + 1)
    feat2_bins = np.linspace(plot_min, plot_max, n_bins + 1)
    
    # Create 2D histogram of interaction values
    interaction_grid = np.full((n_bins, n_bins), np.nan)
    
    for i in range(n_bins):
        for j in range(n_bins):
            # Find points in this bin
            in_bin = ((feat1_values >= feat1_bins[i]) & (feat1_values < feat1_bins[i+1]) &
                      (feat2_values >= feat2_bins[j]) & (feat2_values < feat2_bins[j+1]))
            
            if np.sum(in_bin) > 5:  # Need minimum points in bin
                interaction_grid[j, i] = np.mean(interaction_values[in_bin])  # Note: j,i for correct orientation
    
    # Use 95th percentile for color scale normalization
    vmax = np.nanpercentile(np.abs(interaction_grid), 95)
    vmin = -vmax  # Symmetric around zero
    
    # Create heatmap
    im = ax.imshow(interaction_grid, 
                   aspect='equal',
                   origin='lower',
                   cmap=fig_params['colormap'],
                   vmin=vmin, vmax=vmax,
                   extent=[plot_min, plot_max, plot_min, plot_max])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('SHAP interaction value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    
    # Add 1:1 line
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 
            color='black', linestyle='-', linewidth=2, alpha=0.8)
    
    # Labels and title
    feat1_formatted = format_variable_name(feature1, variable_mapping)
    feat2_formatted = format_variable_name(feature2, variable_mapping)
    
    ax.set_xlabel(f'{feat1_formatted} anomaly', fontsize=10)
    ax.set_ylabel(f'{feat2_formatted} anomaly', fontsize=10)
    ax.set_title('b) Interaction effect of current\nand accumulated precipitation', fontsize=10, y=1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

def create_figure3(results: dict, config: dict) -> plt.Figure:
    """
    Create Figure 3 with LOWESS effects and interaction heatmap.
    
    Args:
        results: Dictionary containing SHAP analysis results
        config: Configuration dictionary
        
    Returns:
        Figure object
    """
    logger.info("Creating Figure 3: Climate-Biomass Relationships")
    
    # Get top features from frequency data
    freq_df = results['feature_frequencies']
    fig_params = config['figure_params']['figure3']
    top_features = determine_top_features(freq_df, fig_params['top_n_features'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_params['figsize'], 
                                   gridspec_kw={'width_ratios': [0.4, 0.6]})
    
    # Panel A: LOWESS SHAP effects
    create_lowess_panel(ax1, results, top_features, config)
    
    # Panel B: Interaction heatmap
    create_interaction_heatmap_panel(ax2, results, config)
    
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    
    return fig

def main():
    """Main function to create Figure 3."""
    logger.info("Starting Figure 3 creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    try:
        # Load pre-computed climate analysis results
        results = load_climate_analysis_results(data_paths)
        logger.info("Successfully loaded climate analysis results")
        
        # Create figure
        fig = create_figure3(results, config)
        
        # Save figure
        output_path = get_figure_output_path(config, "Figure3_Climate_Biomass_Relationships")
        save_figure_multiple_formats(fig, output_path, config, logger)
        
        plt.close()
        logger.info("Figure 3 created successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Required data files not found: {e}")
        logger.error("Please ensure climate_biomass_analysis component has been run and produced SHAP results")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating Figure 3: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()