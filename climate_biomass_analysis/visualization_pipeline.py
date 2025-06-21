"""
Comprehensive SHAP Visualization Pipeline

Creates all manuscript figures from pre-computed SHAP analysis results:
1. Barplots of selection frequency, SHAP importance, permutation importance
2. Histogram of R² distribution
3. Scatterplot of SHAP vs permutation importance colored by frequency
4. 1D PDP SHAP with LOWESS for top 6 features
5. Main figure with colored SHAP effects + 2D interaction heatmap

Author: Diego Bengochea (refactored)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import pickle
import json
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.ticker as ticker
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

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

def load_config(config_path="shap_analysis_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        raise

def format_variable_name(var_name, variable_mapping):
    """Format variable names to be more readable."""
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
    if base_var in variable_mapping:
        return f"{variable_mapping[base_var]}{accumulation_period}"
    
    # If no mapping exists, try to make it more readable
    formatted = base_var.replace('_', ' ').capitalize()
    return f"{formatted}{accumulation_period}"

def load_shap_results(results_dir):
    """Load all SHAP analysis results."""
    print(f"Loading SHAP analysis results from {results_dir}")
    
    results = {}
    
    # Load feature frequencies
    results['freq_df'] = pd.read_csv(os.path.join(results_dir, 'feature_frequencies_df.csv'))
    
    # Load SHAP importance
    with open(os.path.join(results_dir, 'avg_shap_importance.pkl'), 'rb') as f:
        results['avg_shap_importance'] = pickle.load(f)
    
    # Load permutation importance
    with open(os.path.join(results_dir, 'avg_permutation_importance.pkl'), 'rb') as f:
        results['avg_permutation_importance'] = pickle.load(f)
    
    # Load PDP data
    with open(os.path.join(results_dir, 'pdp_data.pkl'), 'rb') as f:
        results['pdp_data'] = pickle.load(f)
    
    # Load PDP LOWESS data
    with open(os.path.join(results_dir, 'pdp_lowess_data.pkl'), 'rb') as f:
        results['pdp_lowess_data'] = pickle.load(f)
    
    # Load interaction results
    with open(os.path.join(results_dir, 'interaction_results.pkl'), 'rb') as f:
        results['interaction_results'] = pickle.load(f)
    
    # Load analysis summary
    with open(os.path.join(results_dir, 'analysis_summary.json'), 'r') as f:
        results['analysis_summary'] = json.load(f)
    
    print("✅ All SHAP results loaded successfully")
    return results

def load_r2_data(results_dir):
    """Load R² data from the original analysis results."""
    # Try to load from original analysis directory
    original_analysis_dirs = [
        'results/climate_biomass_analysis/analysis_results_enhanced_20250515_215513',
        'results/climate_biomass_analysis/analysis_results_r2_0.2',
        'analysis_results_enhanced_20250515_215513',
        'analysis_results_r2_0.2'
    ]
    
    for analysis_dir in original_analysis_dirs:
        r2_file = os.path.join(analysis_dir, 'test_r2_values.csv')
        if os.path.exists(r2_file):
            print(f"Loading R² data from {r2_file}")
            return pd.read_csv(r2_file)
    
    print("⚠️ R² data not found, skipping R² histogram")
    return None

def plot_importance_barplots(results, config, save_path=None):
    """Create barplots for selection frequency, SHAP importance, and permutation importance."""
    freq_df = results['freq_df']
    avg_shap = results['avg_shap_importance']
    avg_perm = results['avg_permutation_importance']
    
    top_n = config['visualization']['top_n_features_barplots']
    
    # Get top features by selection frequency
    top_features = freq_df.sort_values('frequency', ascending=False).head(top_n)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Selection frequency
    top_features_sorted = top_features.sort_values('frequency', ascending=True)
    bars1 = ax1.barh(top_features_sorted['feature'], top_features_sorted['frequency'] * 100, 
                    color='steelblue', height=0.6)
    ax1.set_xlabel('Selection Frequency (%)', fontsize=10)
    ax1.set_ylabel('Feature', fontsize=10)
    ax1.set_title('a) Selection Frequency', fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f"{width:.1f}%", ha='left', va='center', fontsize=8)
    
    # 2. SHAP importance
    shap_data = []
    for feature in top_features['feature']:
        if feature in avg_shap:
            shap_data.append({'feature': feature, 'importance': avg_shap[feature]})
    
    shap_df = pd.DataFrame(shap_data).sort_values('importance', ascending=True)
    bars2 = ax2.barh(shap_df['feature'], shap_df['importance'], 
                    color='mediumseagreen', height=0.6)
    ax2.set_xlabel('SHAP Importance', fontsize=10)
    ax2.set_title('b) SHAP Importance', fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
               f"{width:.3f}", ha='left', va='center', fontsize=8)
    
    # 3. Permutation importance
    perm_data = []
    for feature in top_features['feature']:
        if feature in avg_perm:
            perm_data.append({'feature': feature, 'importance': avg_perm[feature]})
    
    perm_df = pd.DataFrame(perm_data).sort_values('importance', ascending=True)
    bars3 = ax3.barh(perm_df['feature'], perm_df['importance'], 
                    color='darkorange', height=0.6)
    ax3.set_xlabel('Permutation Importance', fontsize=10)
    ax3.set_title('c) Permutation Importance', fontsize=10)
    ax3.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        width = bar.get_width()
        ax3.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
               f"{width:.3f}", ha='left', va='center', fontsize=8)
    
    # Remove spines
    for ax in [ax1, ax2, ax3]:
        sns.despine(ax=ax)
        ax.set_yticklabels([])  # Remove y-labels from middle and right plots
    
    ax1.set_yticklabels(top_features_sorted['feature'])  # Only show on left plot
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
        print(f"✅ Saved importance barplots to {save_path}")
    
    return fig, (ax1, ax2, ax3)

def plot_r2_histogram(r2_df, config, r2_threshold=0.2, save_path=None):
    """Plot histogram of R² distribution."""
    if r2_df is None:
        print("⚠️ Skipping R² histogram - no data available")
        return None, None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create histogram
    sns.histplot(r2_df['test_r2'], bins=config['visualization']['r2_histogram_bins'], 
                kde=True, color='steelblue', ax=ax)
    
    # Add shaded area for excluded models
    ax.axvspan(-1, r2_threshold, alpha=0.2, color='gray')
    
    # Add vertical line for R² threshold
    ax.axvline(x=r2_threshold, color='red', linestyle='--', 
              linewidth=1.5, label=f'Threshold (R² = {r2_threshold})')
    
    # Add median line
    median_r2 = r2_df['test_r2'].median()
    ax.axvline(x=median_r2, color='darkgreen', linestyle='-', 
              linewidth=1.5, label=f'Median (R² = {median_r2:.3f})')
    
    # Calculate percentage of excluded models
    n_total = len(r2_df)
    n_excluded = len(r2_df[r2_df['test_r2'] < r2_threshold])
    pct_excluded = (n_excluded / n_total) * 100
    
    ax.text(r2_threshold - 0.05, ax.get_ylim()[1] * 0.9, 
           f"Excluded:\n{n_excluded}/{n_total}\n({pct_excluded:.1f}%)", 
           ha='right', va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Test R²', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Distribution of Test R² Values', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
        print(f"✅ Saved R² histogram to {save_path}")
    
    return fig, ax

def plot_importance_scatter(results, config, save_path=None):
    """Create scatterplot of SHAP vs permutation importance colored by frequency."""
    avg_shap = results['avg_shap_importance']
    avg_perm = results['avg_permutation_importance']
    freq_df = results['freq_df']
    
    # Convert frequency to dictionary
    feature_freq = {row['feature']: row['frequency'] for _, row in freq_df.iterrows()}
    
    # Get top features for highlighting
    top_features = freq_df.sort_values('frequency', ascending=False).head(6)['feature'].tolist()
    
    # Create dataframe for plotting
    plot_data = []
    for feature in set(avg_shap.keys()) & set(avg_perm.keys()):
        plot_data.append({
            'feature': feature,
            'shap_importance': avg_shap[feature],
            'perm_importance': avg_perm[feature],
            'frequency': feature_freq.get(feature, 0),
            'is_top': feature in top_features
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize frequencies for color mapping
    freq_values = plot_df['frequency'].values
    scaler = MinMaxScaler()
    normalized_frequencies = scaler.fit_transform(freq_values.reshape(-1, 1)).flatten()
    plot_df['norm_freq'] = normalized_frequencies
    
    # Create colormap
    cmap = plt.colormaps['YlOrBr']
    
    # Plot all points
    sizes = np.where(plot_df['is_top'], 120, 60)
    edge_colors = np.where(plot_df['is_top'], 'black', 'white')
    edge_widths = np.where(plot_df['is_top'], 1.0, 0.5)
    
    scatter = ax.scatter(
        plot_df['perm_importance'], 
        plot_df['shap_importance'],
        c=plot_df['norm_freq'],
        s=sizes,
        alpha=0.8,
        cmap=cmap,
        edgecolors=edge_colors,
        linewidths=edge_widths
    )
    
    # Add annotations for top features
    variable_mapping = config['variable_mapping']
    top_df = plot_df[plot_df['is_top']].copy()
    
    for _, row in top_df.iterrows():
        formatted_name = format_variable_name(row['feature'], variable_mapping)
        
        # Smart positioning for annotations
        x_offset = 15 if row['perm_importance'] < 0.15 else -15
        ha = 'left' if row['perm_importance'] < 0.15 else 'right'
        y_offset = 5
        
        # Handle specific overlapping cases
        if formatted_name == 'Annual prec. (3yr)':
            y_offset = 15
        elif formatted_name == 'Temp. seasonality (3yr)':
            y_offset = -10
        
        ax.annotate(
            formatted_name,
            (row['perm_importance'], row['shap_importance']),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            fontsize=7,
            ha=ha,
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=0.5)
        )
    
    # Add reference lines
    median_x = plot_df['perm_importance'].median()
    median_y = plot_df['shap_importance'].median()
    ax.axhline(y=median_y, color='black', linestyle='--', alpha=0.7, linewidth=1.0)
    ax.axvline(x=median_x, color='black', linestyle='--', alpha=0.7, linewidth=1.0)
    
    # Add quadrant labels
    quadrant_labels = {
        'top_right': "Important\nnon-redundant",
        'top_left': "Important but\nredundant", 
        'bottom_right': "Complementary",
        'bottom_left': "Low\nimportance"
    }
    
    positions = {
        'top_right': (0.72, 0.95),
        'top_left': (0.05, 0.95),
        'bottom_right': (0.72, 0.05),
        'bottom_left': (0.05, 0.05)
    }
    
    ha_values = {'top_right': 'right', 'top_left': 'left', 'bottom_right': 'right', 'bottom_left': 'left'}
    va_values = {'top_right': 'top', 'top_left': 'top', 'bottom_right': 'bottom', 'bottom_left': 'bottom'}
    
    for pos, label in quadrant_labels.items():
        ax.text(
            positions[pos][0], positions[pos][1], 
            label, 
            transform=ax.transAxes,
            ha=ha_values[pos], 
            va=va_values[pos], 
            fontsize=7,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3)
        )
    
    ax.set_xlabel('Permutation importance', fontsize=10)
    ax.set_ylabel('SHAP importance', fontsize=10)
    ax.set_title('Feature importance and redundancy', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Selection frequency', fontsize=9)
    
    # Format colorbar ticks as percentages
    freq_min, freq_max = plot_df['frequency'].min(), plot_df['frequency'].max()
    n_ticks = 5
    tick_positions = np.linspace(0, 1, n_ticks)
    tick_labels = [f'{(freq_min + (freq_max - freq_min) * pos)*100:.0f}%' 
                   for pos in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    
    sns.despine(ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
        print(f"✅ Saved importance scatter to {save_path}")
    
    return fig, ax

def plot_pdp_panel(results, config, save_path=None):
    """Create 1D PDP SHAP with LOWESS panel for top 6 features."""
    pdp_data = results['pdp_data']
    
    # Get top 6 features
    freq_df = results['freq_df']
    top_features = freq_df.head(6)['feature'].tolist()
    
    # Filter to available features
    available_features = [f for f in top_features if f in pdp_data and pdp_data[f]]
    
    if not available_features:
        print("⚠️ No PDP data available for visualization")
        return None, None
    
    ncols = config['visualization']['pdp_panel_ncols']
    nrows = (len(available_features) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=(5*ncols, 4*nrows), squeeze=False)
    axes_flat = axes.flatten()
    
    variable_mapping = config['variable_mapping']
    lowess_frac = config['visualization']['pdp_lowess_frac_viz']
    
    for i, feature in enumerate(available_features):
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i]
        feature_data = pdp_data[feature]
        
        if not feature_data:
            continue
        
        # Sort models by R² (highest first)
        sorted_data = sorted(feature_data, key=lambda x: x['model_r2'], reverse=True)
        
        # Collect all points for ensemble analysis
        all_feature_values = []
        all_shap_values = []
        
        # Plot individual model PDPs with alpha based on R²
        min_r2 = min(model['model_r2'] for model in sorted_data)
        max_r2 = max(model['model_r2'] for model in sorted_data)
        
        for model_data in sorted_data:
            feature_values = model_data['feature_values']
            shap_values = model_data['shap_values']
            model_r2 = model_data['model_r2']
            
            all_feature_values.extend(feature_values)
            all_shap_values.extend(shap_values)
            
            # Alpha based on R²
            r2_norm = (model_r2 - min_r2) / (max_r2 - min_r2) if max_r2 > min_r2 else 0.5
            alpha = 0.05 + 0.15 * r2_norm
            
            ax.scatter(feature_values, shap_values, color='steelblue', 
                      alpha=alpha, s=3, linewidths=0)
        
        # Apply LOWESS smoothing
        all_feature_values = np.array(all_feature_values)
        all_shap_values = np.array(all_shap_values)
        
        if len(all_feature_values) > 10:
            try:
                smoothed = lowess(all_shap_values, all_feature_values, frac=lowess_frac)
                smoothed_x = smoothed[:, 0]
                smoothed_y = smoothed[:, 1]
                ax.plot(smoothed_x, smoothed_y, color='darkred', linewidth=2.0)
            except Exception as e:
                print(f"LOWESS failed for {feature}: {e}")
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Calculate linear slope for title
        slope, intercept = np.polyfit(all_feature_values, all_shap_values, 1)
        
        # Set labels and title
        formatted_name = format_variable_name(feature, variable_mapping)
        ax.set_xlabel(f'{formatted_name} (Std)', fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel('SHAP Value', fontsize=9)
        ax.set_title(f'{formatted_name} (slope={slope:.3f})', fontsize=9)
        
        # Add model count
        n_models = len(sorted_data)
        ax.text(0.05, 0.95, f"n={n_models}", transform=ax.transAxes, 
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(axis='both', linestyle='--', alpha=0.3)
        sns.despine(ax=ax)
    
    # Hide empty subplots
    for i in range(len(available_features), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
        print(f"✅ Saved PDP panel to {save_path}")
    
    return fig, axes

def plot_main_figure(results, config, save_path=None):
    """Create main figure with SHAP effects + 2D interaction heatmap."""
    freq_df = results['freq_df']
    pdp_lowess_data = results['pdp_lowess_data']
    interaction_results = results['interaction_results']
    
    # Get configuration
    fig_size = config['visualization']['main_fig_size']
    width_ratios = config['visualization']['main_fig_width_ratios']
    variable_mapping = config['variable_mapping']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size, gridspec_kw={'width_ratios': width_ratios})
    
    # Panel a: SHAP effects with colored lines
    top_features = freq_df.head(6)['feature'].tolist()
    available_features = [f for f in top_features if f in pdp_lowess_data and pdp_lowess_data[f]]
    
    if available_features:
        # Collect all SHAP values for color normalization
        all_shap_values = []
        feature_curves = {}
        
        for feature in available_features:
            lowess_data = pdp_lowess_data[feature]
            if 'smooth_y' in lowess_data:
                smooth_y = lowess_data['smooth_y']
                smooth_x = lowess_data['smooth_x']
                feature_curves[feature] = {'x': smooth_x, 'y': smooth_y}
                all_shap_values.extend(smooth_y)
        
        if feature_curves:
            # Create color normalization
            percentile_95 = np.percentile(np.abs(all_shap_values), 95)
            norm = Normalize(vmin=-percentile_95, vmax=percentile_95)
            cmap = plt.colormaps['RdBu']
            
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
                    ax1.plot([x_start, x_end], [y_pos, y_pos], 
                            color=color, linewidth=8, solid_capstyle='butt')
            
            # Format panel a
            formatted_names = [format_variable_name(feature, variable_mapping) 
                              for feature in available_features]
            
            ax1.set_yticks(range(len(available_features)))
            ax1.set_yticklabels(reversed(formatted_names), fontsize=8)
            ax1.set_xlabel('Standardized anomaly', fontsize=10)
            ax1.set_xlim(-2.5, 2.5)
            ax1.set_ylim(-0.5, len(available_features) - 0.5)
            ax1.set_title('a) Individual feature effects', fontsize=10, y=1.08)
            
            # Add vertical line and grid
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1.0)
            ax1.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, shrink=0.8, aspect=20)
            cbar.set_label('SHAP value', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
    
    # Panel b: 2D interaction heatmap
    if interaction_results:
        feature1 = config['visualization']['interaction_feature_1']
        feature2 = config['visualization']['interaction_feature_2']
        
        features = interaction_results['features']
        all_interactions_dict = interaction_results['all_interactions_dict']
        all_feature_values_dict = interaction_results['all_feature_values_dict']
        interaction_counts = interaction_results['interaction_counts']
        
        pair = (feature1, feature2)
        
        if pair in all_interactions_dict and len(all_interactions_dict[pair]) > 0:
            interaction_values = all_interactions_dict[pair]
            feature_value_pairs = all_feature_values_dict[pair]
            
            feat1_values = feature_value_pairs[:, 0]
            feat2_values = feature_value_pairs[:, 1]
            
            # Create 2D bins for heatmap
            n_bins = config['visualization']['interaction_n_bins']
            percentile_clip = config['visualization']['interaction_percentile_clip']
            
            feat1_abs_max = np.percentile(np.abs(feat1_values), percentile_clip)
            feat2_abs_max = np.percentile(np.abs(feat2_values), percentile_clip)
            
            max_range = max(feat1_abs_max, feat2_abs_max)
            plot_min, plot_max = -max_range, max_range
            
            feat1_bins = np.linspace(plot_min, plot_max, n_bins + 1)
            feat2_bins = np.linspace(plot_min, plot_max, n_bins + 1)
            
            # Create 2D histogram of interaction values
            interaction_grid = np.full((n_bins, n_bins), np.nan)
            
            for i in range(n_bins):
                for j in range(n_bins):
                    in_bin = ((feat1_values >= feat1_bins[i]) & (feat1_values < feat1_bins[i+1]) &
                              (feat2_values >= feat2_bins[j]) & (feat2_values < feat2_bins[j+1]))
                    
                    if np.sum(in_bin) > 5:
                        interaction_grid[j, i] = np.mean(interaction_values[in_bin])
            
            # Color scale
            vmax = np.nanpercentile(np.abs(interaction_grid), 95)
            vmin = -vmax
            
            # Create heatmap
            im = ax2.imshow(interaction_grid, 
                           aspect='equal',
                           origin='lower',
                           cmap='RdBu',
                           vmin=vmin, vmax=vmax,
                           extent=[plot_min, plot_max, plot_min, plot_max])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8, aspect=20)
            cbar.set_label('SHAP interaction value', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            # Add reference lines
            ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            ax2.plot([plot_min, plot_max], [plot_min, plot_max], 
                    color='black', linestyle='-', linewidth=2, alpha=0.8)
            
            # Labels
            feat1_formatted = format_variable_name(feature1, variable_mapping)
            feat2_formatted = format_variable_name(feature2, variable_mapping)
            
            ax2.set_xlabel(f'{feat1_formatted} anomaly (standardized)', fontsize=10)
            ax2.set_ylabel(f'{feat2_formatted} anomaly (standardized)', fontsize=10)
            ax2.set_title('b) Interaction effect of current and accumulated precipitation', fontsize=10, y=1.1)
            
            ax2.grid(True, alpha=0.3)
    
    # Remove spines
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
        print(f"✅ Saved main figure to {save_path}")
    
    return fig, (ax1, ax2)

def run_visualization_pipeline(config_path="shap_analysis_config.yaml"):
    """Run the complete visualization pipeline."""
    print("🎨 Starting SHAP Visualization Pipeline...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    figures_dir = config['paths']['shap_figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load SHAP results
    results_dir = config['paths']['shap_analysis_dir']
    results = load_shap_results(results_dir)
    
    # Load R² data
    r2_df = load_r2_data(results_dir)
    
    print("\n📊 Creating visualizations...")
    
    # 1. Importance barplots
    print("1. Creating importance barplots...")
    plot_importance_barplots(
        results, config, 
        save_path=os.path.join(figures_dir, 'importance_barplots.png')
    )
    
    # 2. R² histogram
    print("2. Creating R² histogram...")
    plot_r2_histogram(
        r2_df, config, r2_threshold=config['model_filtering']['r2_threshold'],
        save_path=os.path.join(figures_dir, 'r2_histogram.png')
    )
    
    # 3. Importance scatter
    print("3. Creating importance scatter...")
    plot_importance_scatter(
        results, config,
        save_path=os.path.join(figures_dir, 'importance_scatter.png')
    )
    
    # 4. PDP panel
    print("4. Creating PDP panel...")
    plot_pdp_panel(
        results, config,
        save_path=os.path.join(figures_dir, 'pdp_panel.png')
    )
    
    # 5. Main figure
    print("5. Creating main figure...")
    plot_main_figure(
        results, config,
        save_path=os.path.join(figures_dir, 'main_figure.png')
    )
    
    print(f"\n✅ All visualizations saved to: {figures_dir}")
    print("📁 Generated files:")
    print("   📊 importance_barplots.png")
    print("   📈 r2_histogram.png") 
    print("   🎯 importance_scatter.png")
    print("   📉 pdp_panel.png")
    print("   🎨 main_figure.png")

def main():
    """Main function to run the visualization pipeline."""
    try:
        run_visualization_pipeline()
        print("\n🎉 SHAP visualization pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Visualization pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
