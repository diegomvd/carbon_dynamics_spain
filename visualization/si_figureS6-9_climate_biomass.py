#!/usr/bin/env python3
"""
Generate Figures S6-S9: Climate-Biomass Analysis Supporting Information

Comprehensive SHAP visualization from pre-computed analysis results:
- Figure S6: Feature importance barplots (selection frequency, SHAP, permutation)  
- Figure S7: R² distribution histogram
- Figure S8: SHAP vs permutation importance scatter plot
- Figure S9: Partial dependence plots with LOWESS smoothing

Author: Diego Bengochea
Component: Visualization Pipeline - Supporting Information
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import pickle
import json
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.ticker as ticker
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler
import warnings
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, get_figure_output_path
)

warnings.filterwarnings('ignore')

# Set up logging
logger = setup_logging('climate_biomass_si')

def find_shap_analysis_results(data_paths):
    """
    Auto-detect SHAP analysis results directory.
    
    Args:
        data_paths: CentralDataPaths instance
        
    Returns:
        Path to SHAP analysis results directory
    """
    # Possible locations for SHAP results
    possible_locations = [
        data_paths.get_path('ml_outputs') / 'shap_analysis',
        data_paths.get_path('analysis_outputs') / 'shap_analysis', 
        data_paths.get_path('results') / 'climate_biomass_analysis' / 'shap_analysis',
        data_paths.data_root / "results" / "climate_biomass_analysis" / "shap_analysis"
    ]
    
    for location in possible_locations:
        if location.exists():
            # Check if it contains expected files
            expected_files = [
                'feature_frequencies_df.csv',
                'avg_shap_importance.pkl',
                'pdp_lowess_data.pkl'
            ]
            if all((location / file).exists() for file in expected_files):
                logger.info(f"Found SHAP analysis results: {location}")
                return location
    
    raise FileNotFoundError(
        "SHAP analysis results not found. Expected files:\n" +
        "  - feature_frequencies_df.csv\n" +
        "  - avg_shap_importance.pkl\n" +
        "  - pdp_lowess_data.pkl\n" +
        f"Searched locations:\n" +
        "\n".join(f"  - {loc}" for loc in possible_locations) +
        "\nPlease ensure climate_biomass_analysis component has generated SHAP results."
    )

def load_shap_results(results_dir: Path):
    """Load all SHAP analysis results."""
    logger.info(f"Loading SHAP analysis results from {results_dir}")
    
    results = {}
    
    # Load feature frequencies
    results['freq_df'] = pd.read_csv(results_dir / 'feature_frequencies_df.csv')
    
    # Load SHAP importance
    with open(results_dir / 'avg_shap_importance.pkl', 'rb') as f:
        results['avg_shap_importance'] = pickle.load(f)
    
    # Load permutation importance
    with open(results_dir / 'avg_permutation_importance.pkl', 'rb') as f:
        results['avg_permutation_importance'] = pickle.load(f)
    
    # Load PDP data
    with open(results_dir / 'pdp_data.pkl', 'rb') as f:
        results['pdp_data'] = pickle.load(f)
    
    # Load PDP LOWESS data
    with open(results_dir / 'pdp_lowess_data.pkl', 'rb') as f:
        results['pdp_lowess_data'] = pickle.load(f)
    
    # Load interaction results (if available)
    interaction_file = results_dir / 'interaction_results.pkl'
    if interaction_file.exists():
        with open(interaction_file, 'rb') as f:
            results['interaction_results'] = pickle.load(f)
    
    # Load analysis summary (if available)
    summary_file = results_dir / 'analysis_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['analysis_summary'] = json.load(f)
    
    logger.info("✅ All SHAP results loaded successfully")
    return results

def load_r2_data(results_dir):
    """Load R² data from the original analysis results."""
    # Try to load from original analysis directory - check multiple possible locations
    search_dirs = [
        results_dir.parent / 'analysis_results_enhanced_20250515_215513',
        results_dir.parent / 'analysis_results_r2_0.2',
        Path('results/climate_biomass_analysis/analysis_results_enhanced_20250515_215513'),
        Path('results/climate_biomass_analysis/analysis_results_r2_0.2'),
        Path('analysis_results_enhanced_20250515_215513'),
        Path('analysis_results_r2_0.2')
    ]
    
    for analysis_dir in search_dirs:
        r2_file = analysis_dir / 'test_r2_values.csv'
        if r2_file.exists():
            logger.info(f"Loading R² data from {r2_file}")
            return pd.read_csv(r2_file)
    
    logger.warning("⚠️ R² data not found, skipping R² histogram")
    return None

def format_variable_name(var_name: str, variable_mapping: dict = None) -> str:
    """Format variable names for display in figures."""
    if var_name == 'biomass_rel_change':
        return 'Biomass relative change'
    
    # Extract accumulation period if present
    accumulation_period = ''
    if '_' in var_name:
        parts = var_name.split('_')
        base_var = parts[0]
        if any(period in parts[-1] for period in ['yr', 'month', 'day', 'season']):
            accumulation_period = f" ({parts[-1]})"
        else:
            base_var = var_name
    else:
        base_var = var_name
    
    # Check if we have a mapping for this variable
    if variable_mapping and base_var in variable_mapping:
        return f"{variable_mapping[base_var]}{accumulation_period}"
    
    # If no mapping exists, try to make it more readable
    formatted = base_var.replace('_', ' ').capitalize()
    return f"{formatted}{accumulation_period}"

def plot_importance_barplots(results, config, output_path):
    """Create barplots for selection frequency, SHAP importance, and permutation importance."""
    freq_df = results['freq_df']
    avg_shap = results['avg_shap_importance']
    avg_perm = results['avg_permutation_importance']
    
    top_n = 10  # Default value
    
    # Get top features by selection frequency
    top_features = freq_df.sort_values('frequency', ascending=False).head(top_n)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Selection frequency
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars1 = ax1.bar(range(len(top_features)), top_features['frequency'] * 100, color=colors)
    ax1.set_xlabel('Features', fontsize=10)
    ax1.set_ylabel('Selection Frequency (%)', fontsize=10)
    ax1.set_title('Feature Selection Frequency', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(top_features)))
    ax1.set_xticklabels(top_features['feature'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. SHAP importance
    shap_values = [avg_shap.get(feature, 0) for feature in top_features['feature']]
    bars2 = ax2.bar(range(len(top_features)), shap_values, color=colors)
    ax2.set_xlabel('Features', fontsize=10)
    ax2.set_ylabel('Average |SHAP Value|', fontsize=10)
    ax2.set_title('SHAP Importance', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(top_features)))
    ax2.set_xticklabels(top_features['feature'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Permutation importance
    perm_values = [avg_perm.get(feature, 0) for feature in top_features['feature']]
    bars3 = ax3.bar(range(len(top_features)), perm_values, color=colors)
    ax3.set_xlabel('Features', fontsize=10)
    ax3.set_ylabel('Permutation Importance', fontsize=10)
    ax3.set_title('Permutation Importance', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(top_features)))
    ax3.set_xticklabels(top_features['feature'], rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved importance barplots to {output_path}")

def plot_r2_histogram(r2_df, config, r2_threshold, output_path):
    """Create R² distribution histogram."""
    if r2_df is None:
        logger.warning("No R² data available, skipping histogram")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot histogram
    n_bins = 50
    ax.hist(r2_df['r2'], bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add threshold line
    ax.axvline(r2_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold (R² = {r2_threshold})')
    
    # Statistics
    above_threshold = (r2_df['r2'] >= r2_threshold).sum()
    total_models = len(r2_df)
    percentage = (above_threshold / total_models) * 100
    
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_ylabel('Number of Models', fontsize=12)
    ax.set_title(f'Distribution of Model Performance\n{above_threshold}/{total_models} models ({percentage:.1f}%) above threshold', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved R² histogram to {output_path}")

def plot_importance_scatter(results, config, output_path):
    """Create scatter plot of SHAP vs permutation importance."""
    freq_df = results['freq_df']
    avg_shap = results['avg_shap_importance']
    avg_perm = results['avg_permutation_importance']
    
    # Get all features present in both importance measures
    common_features = set(avg_shap.keys()) & set(avg_perm.keys())
    
    shap_vals = [avg_shap[feature] for feature in common_features]
    perm_vals = [avg_perm[feature] for feature in common_features]
    
    # Get selection frequencies for coloring
    freq_dict = dict(zip(freq_df['feature'], freq_df['frequency']))
    frequencies = [freq_dict.get(feature, 0) for feature in common_features]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot colored by frequency
    scatter = ax.scatter(shap_vals, perm_vals, c=frequencies, cmap='viridis', 
                        s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Selection Frequency', fontsize=12)
    
    # Add diagonal line
    max_val = max(max(shap_vals), max(perm_vals))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Average |SHAP Value|', fontsize=12)
    ax.set_ylabel('Permutation Importance', fontsize=12)
    ax.set_title('SHAP vs Permutation Importance\n(colored by selection frequency)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved importance scatter to {output_path}")

def plot_pdp_panel(results, config, variable_mapping, output_path):
    """Create PDP panel with LOWESS smoothing."""
    pdp_data = results['pdp_data']
    pdp_lowess_data = results['pdp_lowess_data']
    
    # Get top 6 features for PDP
    freq_df = results['freq_df']
    top_features = freq_df.sort_values('frequency', ascending=False).head(6)['feature'].tolist()
    
    # Filter features that have PDP data
    available_features = [f for f in top_features if f in pdp_data]
    
    if not available_features:
        logger.warning("No PDP data available for top features")
        return
    
    # Set up subplot grid
    n_features = len(available_features)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, feature in enumerate(available_features):
        ax = axes_flat[i]
        
        # Get data for this feature across all models
        feature_values = []
        shap_values = []
        
        for model_id, model_pdp in pdp_data.items():
            if feature in model_pdp:
                feature_values.extend(model_pdp[feature]['values'])
                shap_values.extend(model_pdp[feature]['shap_values'])
        
        if not feature_values:
            continue
        
        # Plot raw points
        ax.scatter(feature_values, shap_values, alpha=0.3, s=1, color='lightblue')
        
        # Plot LOWESS smoothing if available
        if feature in pdp_lowess_data:
            lowess_x = pdp_lowess_data[feature]['x']
            lowess_y = pdp_lowess_data[feature]['y']
            ax.plot(lowess_x, lowess_y, color='red', linewidth=2, label='LOWESS')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Calculate linear slope for title
        if len(feature_values) > 1:
            slope, intercept = np.polyfit(feature_values, shap_values, 1)
        else:
            slope = 0
        
        # Set labels and title
        formatted_name = format_variable_name(feature, variable_mapping)
        ax.set_xlabel(f'{formatted_name} (Std)', fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel('SHAP Value', fontsize=9)
        ax.set_title(f'{formatted_name} (slope={slope:.3f})', fontsize=9)
        
        # Add model count
        n_models = len(pdp_data)
        ax.text(0.05, 0.95, f"n={n_models}", transform=ax.transAxes, 
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(axis='both', linestyle='--', alpha=0.3)
        sns.despine(ax=ax)
    
    # Hide empty subplots
    for i in range(len(available_features), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved PDP panel to {output_path}")

def main():
    """Main execution function."""
    logger.info("Starting climate-biomass SI figures creation (S6-S9)")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up plotting parameters for publication-quality figures
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 10,
        'figure.dpi': 300
    })
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    try:
        # Auto-detect SHAP analysis results
        results_dir = find_shap_analysis_results(data_paths)
        
        # Load SHAP results
        results = load_shap_results(results_dir)
        
        # Load R² data
        r2_df = load_r2_data(results_dir)
        
        # Variable mapping from config
        variable_mapping = config.get('variable_mapping', {})
        
        logger.info("Creating climate-biomass SI visualizations...")
        
        # Figure S6: Importance barplots
        logger.info("Creating Figure S6: Feature importance barplots...")
        s6_path = get_figure_output_path(config, "Figure_S6_Feature_Importance_Barplots", 'supporting_info')
        plot_importance_barplots(results, config, f"{s6_path}.png")
        
        # Figure S7: R² histogram  
        logger.info("Creating Figure S7: R² distribution histogram...")
        s7_path = get_figure_output_path(config, "Figure_S7_R2_Distribution", 'supporting_info')
        plot_r2_histogram(r2_df, config, r2_threshold=0.2, output_path=f"{s7_path}.png")
        
        # Figure S8: Importance scatter
        logger.info("Creating Figure S8: Importance methods comparison...")
        s8_path = get_figure_output_path(config, "Figure_S8_Importance_Scatter", 'supporting_info')
        plot_importance_scatter(results, config, f"{s8_path}.png")
        
        # Figure S9: PDP panel
        logger.info("Creating Figure S9: Partial dependence analysis...")
        s9_path = get_figure_output_path(config, "Figure_S9_Partial_Dependence", 'supporting_info')
        plot_pdp_panel(results, config, variable_mapping, f"{s9_path}.png")
        
        logger.info("✅ All climate-biomass SI figures created successfully!")
        logger.info("Generated files:")
        logger.info("   📊 Figure S6: Feature importance barplots")
        logger.info("   📈 Figure S7: R² distribution histogram") 
        logger.info("   🎯 Figure S8: Importance methods scatter")
        logger.info("   📉 Figure S9: Partial dependence plots")
        
    except FileNotFoundError as e:
        logger.error(f"Required data not found: {e}")
        logger.error("Please ensure climate_biomass_analysis component has generated SHAP results")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating climate-biomass SI figures: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()