#!/usr/bin/env python3
"""
Figure S2: Canopy Height Model Evaluation

Comprehensive evaluation plots for trained canopy height regression models:
- Scatter plot with density (predictions vs observations)
- Residual analysis plot
- Height distribution comparison
- Error by height bins

Loads real evaluation results from canopy_height_model component.

Author: Diego Bengochea
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import pickle
import warnings

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, get_figure_output_path
)

warnings.filterwarnings('ignore')

# Set up logging
logger = setup_logging('figure_si_02_height_evaluation')

def find_evaluation_results(data_paths):
    """Find model evaluation results from canopy height model."""
    # Try multiple possible locations for evaluation results
    possible_locations = [
        data_paths.data_root / "results" / "canopy_height_model" / "evaluation_results",
        data_paths.data_root / "results" / "canopy_height_model" / "evaluation",
        data_paths.data_root / "canopy_height_model" / "evaluation_results",
        data_paths.data_root / "canopy_height_model" / "evaluation",
        data_paths.data_root / "model_evaluation",
        Path("data/results/canopy_height_model/evaluation_results"),
        Path("data/results/canopy_height_model/evaluation")
    ]
    
    for location in possible_locations:
        if location.exists():
            # Look for evaluation_results.pkl file
            results_file = location / "evaluation_results.pkl"
            if results_file.exists():
                logger.info(f"Found evaluation results: {results_file}")
                return results_file
    
    raise FileNotFoundError(
        "Height model evaluation results not found. Expected file: evaluation_results.pkl\n" +
        f"Searched locations:\n" +
        "\n".join(f"  - {loc}/evaluation_results.pkl" for loc in possible_locations) +
        "\nPlease run canopy height model evaluation first: python scripts/run_evaluation.py"
    )

def load_evaluation_data(results_file):
    """Load evaluation data from pickle file."""
    logger.info(f"Loading evaluation data from {results_file}")
    
    try:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract predictions and targets
        predictions = data['predictions']
        targets = data['targets']
        metrics = data.get('metrics', {})
        
        logger.info(f"Loaded {len(predictions)} prediction-target pairs")
        logger.info(f"Existing metrics: {list(metrics.keys())}")
        
        return predictions, targets, metrics
        
    except Exception as e:
        raise ValueError(f"Error loading evaluation data from {results_file}: {e}")

def calculate_additional_metrics(predictions, targets):
    """Calculate additional metrics if not already present."""
    # Remove any NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_clean = predictions[valid_mask]
    target_clean = targets[valid_mask]
    
    if len(pred_clean) == 0:
        raise ValueError("No valid prediction-target pairs found")
    
    # Calculate metrics
    mae = mean_absolute_error(target_clean, pred_clean)
    rmse = np.sqrt(mean_squared_error(target_clean, pred_clean))
    r2 = r2_score(target_clean, pred_clean)
    r_pearson = stats.pearsonr(target_clean, pred_clean)[0]
    bias = np.mean(pred_clean - target_clean)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'r_pearson': r_pearson,
        'bias': bias,
        'n_samples': len(pred_clean)
    }
    
    logger.info(f"Calculated metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}, n={len(pred_clean):,}")
    
    return metrics, pred_clean, target_clean

def plot_scatter_with_density(predictions, targets, ax, metrics):
    """Create scatter plot with density information."""
    # Create 2D histogram for density
    h, xedges, yedges = np.histogram2d(targets, predictions, bins=50)
    h = h.T  # Transpose for correct orientation
    
    # Plot density
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h, extent=extent, origin='lower', aspect='auto', 
                  cmap='Blues', norm=LogNorm(vmin=1))
    
    # Add 1:1 line
    max_val = max(np.max(targets), np.max(predictions))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7, label='1:1 line')
    
    # Formatting
    ax.set_xlabel('Observed Height (m)', fontsize=12)
    ax.set_ylabel('Predicted Height (m)', fontsize=12)
    ax.set_title('Predicted vs Observed Heights', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add metrics text
    metrics_text = f"MAE: {metrics['mae']:.2f} m\n"
    metrics_text += f"RMSE: {metrics['rmse']:.2f} m\n"
    metrics_text += f"R²: {metrics['r2']:.3f}\n"
    metrics_text += f"r: {metrics['r_pearson']:.3f}\n"
    metrics_text += f"n: {metrics['n_samples']:,}"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_residuals(predictions, targets, ax):
    """Create residual plot."""
    residuals = predictions - targets
    
    # Scatter plot of residuals vs predictions
    ax.scatter(predictions, residuals, alpha=0.6, s=1)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Calculate and plot trend line
    z = np.polyfit(predictions, residuals, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(predictions.min(), predictions.max(), 100)
    ax.plot(x_trend, p(x_trend), "orange", linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    
    ax.set_xlabel('Predicted Height (m)', fontsize=12)
    ax.set_ylabel('Residuals (m)', fontsize=12)
    ax.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_height_distributions(predictions, targets, ax):
    """Create height distribution comparison."""
    # Plot histograms
    bins = np.linspace(0, max(np.max(targets), np.max(predictions)), 30)
    
    ax.hist(targets, bins=bins, alpha=0.7, label='Observed', color='blue', density=True)
    ax.hist(predictions, bins=bins, alpha=0.7, label='Predicted', color='orange', density=True)
    
    # Add mean lines
    ax.axvline(np.mean(targets), color='blue', linestyle='--', linewidth=2, 
               label=f'Obs. mean: {np.mean(targets):.1f}m')
    ax.axvline(np.mean(predictions), color='orange', linestyle='--', linewidth=2,
               label=f'Pred. mean: {np.mean(predictions):.1f}m')
    
    ax.set_xlabel('Height (m)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Height Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_error_by_height_bins(predictions, targets, ax):
    """Create error analysis by height bins."""
    # Create height bins
    n_bins = 10
    bin_edges = np.linspace(0, np.max(targets), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate metrics for each bin
    mae_by_bin = []
    rmse_by_bin = []
    counts_by_bin = []
    
    for i in range(n_bins):
        mask = (targets >= bin_edges[i]) & (targets < bin_edges[i + 1])
        if i == n_bins - 1:  # Include the last edge in the final bin
            mask = (targets >= bin_edges[i]) & (targets <= bin_edges[i + 1])
        
        if np.sum(mask) > 0:
            bin_targets = targets[mask]
            bin_predictions = predictions[mask]
            
            mae_by_bin.append(mean_absolute_error(bin_targets, bin_predictions))
            rmse_by_bin.append(np.sqrt(mean_squared_error(bin_targets, bin_predictions)))
            counts_by_bin.append(np.sum(mask))
        else:
            mae_by_bin.append(np.nan)
            rmse_by_bin.append(np.nan)
            counts_by_bin.append(0)
    
    # Plot MAE and RMSE by height bins
    ax2 = ax.twinx()
    
    bars = ax.bar(bin_centers, mae_by_bin, width=bin_edges[1] - bin_edges[0] * 0.8, 
                  alpha=0.7, label='MAE', color='skyblue')
    line = ax2.plot(bin_centers, counts_by_bin, 'ro-', linewidth=2, markersize=6, 
                    label='Sample count')
    
    ax.set_xlabel('Height Bins (m)', fontsize=12)
    ax.set_ylabel('MAE (m)', fontsize=12, color='blue')
    ax2.set_ylabel('Sample Count', fontsize=12, color='red')
    ax.set_title('Error Analysis by Height Bins', fontsize=14, fontweight='bold')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    ax.grid(True, alpha=0.3)

def create_evaluation_figure(predictions, targets, metrics):
    """Create the complete evaluation figure with 4 subplots."""
    # Set up matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Canopy Height Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with density
    plot_scatter_with_density(predictions, targets, axes[0, 0], metrics)
    
    # 2. Residual plot
    plot_residuals(predictions, targets, axes[0, 1])
    
    # 3. Height distribution comparison
    plot_height_distributions(predictions, targets, axes[1, 0])
    
    # 4. Error by height bins
    plot_error_by_height_bins(predictions, targets, axes[1, 1])
    
    plt.tight_layout()
    
    return fig

def main():
    """Main execution function."""
    logger.info("Starting Figure S2: Height Model Evaluation creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    try:
        # Find evaluation results
        results_file = find_evaluation_results(data_paths)
        
        # Load evaluation data
        predictions, targets, existing_metrics = load_evaluation_data(results_file)
        
        # Calculate additional metrics if needed
        metrics, pred_clean, target_clean = calculate_additional_metrics(predictions, targets)
        
        # Merge with existing metrics
        metrics.update(existing_metrics)
        
        # Create evaluation figure
        fig = create_evaluation_figure(pred_clean, target_clean, metrics)
        
        # Save figure
        output_path = get_figure_output_path(config, "Figure_S2_Height_Model_Evaluation", 'supporting_info')
        
        # Save as both PNG and PDF
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        
        logger.info(f"✅ Figure S2 saved:")
        logger.info(f"   🖼️  PNG: {output_path}.png")
        logger.info(f"   📄 PDF: {output_path}.pdf")
        logger.info(f"Model performance: MAE={metrics['mae']:.2f}m, RMSE={metrics['rmse']:.2f}m, R²={metrics['r2']:.3f}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating Figure S2: {e}")
        raise

if __name__ == "__main__":
    main()