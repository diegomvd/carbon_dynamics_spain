"""
Comprehensive evaluation pipeline for canopy height regression models.

This script provides detailed evaluation and visualization tools for trained
canopy height regression models. Includes statistical analysis, residual plots,
prediction density visualizations, and height distribution comparisons.

Generates publication-ready plots and comprehensive metrics for model
performance assessment across different height ranges.

Author: Diego Bengochea
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Standard library imports
import numpy as np
import pandas as pd

# Third-party imports
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy import stats
from sklearn.metrics import r2_score

# PyTorch Lightning imports
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer

# Shared utilities
from shared_utils import load_config, get_logger, ensure_directory

# Component imports
from .s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from .canopy_height_regression import CanopyHeightRegressionTask


class PredictionCollector(Callback):
    """
    Callback to collect predictions and targets during testing.
    
    This callback accumulates model predictions and ground truth targets
    throughout the testing process for subsequent analysis and visualization.
    """
    
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []
        
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Collect predictions and targets at the end of each test batch.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being tested
            outputs: Model outputs (unused)
            batch: Input batch containing image and mask
            batch_idx: Batch index
            dataloader_idx: Dataloader index (default=0)
        """
        x = batch["image"]
        y = batch['mask'].to(torch.float)
        
        # Handle NaN inputs and get mask using model's methods
        x_filled, input_mask = pl_module._handle_nan_inputs(x)
        
        # Get predictions
        with torch.no_grad():
            y_hat = pl_module(x_filled)
            if y_hat.ndim != y.ndim:
                y = y.unsqueeze(dim=1)
        
        # Convert from log space
        pred = torch.expm1(y_hat).cpu().numpy()
        target = torch.expm1(y).cpu().numpy()
        
        # Filter valid values
        valid_mask = (target != pl_module.nan_value_target) & input_mask.cpu().numpy()
        
        self.predictions.append(pred[valid_mask])
        self.targets.append(target[valid_mask])
        
    def get_predictions_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve concatenated predictions and targets.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Flattened predictions and targets
        """
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        return predictions, targets


class ModelEvaluationPipeline:
    """
    Comprehensive evaluation pipeline for canopy height regression models.
    
    This class provides detailed model evaluation including statistical metrics,
    visualization, and performance analysis across different height ranges.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        self.logger = get_logger('canopy_height_dl.evaluation')
        
        # Initialize components
        self.pred_collector = PredictionCollector()
        self.datamodule = None
        self.model = None
        self.trainer = None
        
        self.logger.info("ModelEvaluationPipeline initialized")
    
    def _setup_datamodule(self) -> None:
        """Initialize the data module."""
        self.datamodule = S2PNOAVegetationDataModule(
            data_dir=str(self.config['data']['data_dir']),
            config_path=None  # Will use component default
        )
        self.logger.info("Datamodule initialized for evaluation")
        
    def _setup_model(self) -> None:
        """Initialize the model."""
        self.model = CanopyHeightRegressionTask(
            nan_value_target=self.datamodule.hparams['nan_target'],
            nan_value_input=self.datamodule.hparams['nan_input'],
            config_path=None  # Will use component default
        )
        self.logger.info("Model initialized for evaluation")
        
    def _setup_trainer(self) -> None:
        """Initialize the PyTorch Lightning trainer."""
        # Determine accelerator
        if torch.cuda.is_available():
            accelerator = 'gpu'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerator = 'mps'
        else:
            accelerator = 'cpu'
        
        devices = self.config['compute'].get('devices', 1)
        
        self.trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            callbacks=[self.pred_collector],
            enable_checkpointing=False,
            enable_progress_bar=True,
            inference_mode=True,
            logger=False
        )
        self.logger.info(f"Trainer configured with {accelerator} accelerator")

    def _calculate_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics with optional confidence intervals.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            confidence_level (float, optional): Confidence level for bootstrap intervals
            
        Returns:
            Dict[str, float]: Dictionary containing calculated metrics
        """
        # Clean data
        predictions = np.where(predictions < 0, 0.0, predictions)
        predictions = np.where(predictions > 60, np.nan, predictions)
        targets = np.where(targets > 60, np.nan, targets)
        
        # Remove outliers
        residuals = predictions - targets
        cutoff = np.nanpercentile(residuals, 99)
        valid_idx = residuals <= cutoff   

        predictions = predictions[valid_idx]
        targets = targets[valid_idx]           
        residuals = residuals[valid_idx]

        # Calculate basic metrics
        mae = np.nanmean(np.abs(residuals))
        bias = np.nanmean(residuals)
        rmse = np.sqrt(np.nanmean(residuals**2))
        
        # Calculate correlation and R²
        valid_both = ~(np.isnan(predictions) | np.isnan(targets))
        if np.sum(valid_both) > 1:
            r_pearson = stats.pearsonr(targets[valid_both], predictions[valid_both])[0]
            r2 = r2_score(targets[valid_both], predictions[valid_both])
        else:
            r_pearson = np.nan
            r2 = np.nan

        metrics = {
            'mae': mae,
            'bias': bias,
            'rmse': rmse,
            'r_pearson': r_pearson,
            'r2': r2,
            'n_samples': len(predictions)
        }

        # Add confidence intervals if requested
        if confidence_level and np.sum(valid_both) > 100:
            metrics.update(self._bootstrap_confidence_intervals(
                predictions[valid_both], targets[valid_both], confidence_level
            ))

        return metrics
    
    def _bootstrap_confidence_intervals(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        confidence_level: float,
        n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with confidence intervals
        """
        n_samples = len(predictions)
        bootstrap_metrics = {'mae': [], 'rmse': [], 'r2': []}
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_pred = predictions[indices]
            boot_target = targets[indices]
            
            # Calculate metrics
            residuals = boot_pred - boot_target
            bootstrap_metrics['mae'].append(np.mean(np.abs(residuals)))
            bootstrap_metrics['rmse'].append(np.sqrt(np.mean(residuals**2)))
            
            if len(boot_pred) > 1:
                bootstrap_metrics['r2'].append(r2_score(boot_target, boot_pred))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_results = {}
        for metric_name, values in bootstrap_metrics.items():
            lower = np.percentile(values, lower_percentile)
            upper = np.percentile(values, upper_percentile)
            ci_results[f'{metric_name}_ci'] = (lower, upper)
        
        return ci_results

    def _create_evaluation_plots(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        output_dir: Path,
        metrics: Dict[str, float]
    ) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            output_dir: Output directory for plots
            metrics: Calculated metrics
        """
        self.logger.info("Creating evaluation plots...")
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Canopy Height Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot with density
        ax1 = axes[0, 0]
        self._plot_scatter_with_density(predictions, targets, ax1, metrics)
        
        # 2. Residual plot
        ax2 = axes[0, 1]
        self._plot_residuals(predictions, targets, ax2)
        
        # 3. Height distribution comparison
        ax3 = axes[1, 0]
        self._plot_height_distributions(predictions, targets, ax3)
        
        # 4. Error by height bins
        ax4 = axes[1, 1]
        self._plot_error_by_height_bins(predictions, targets, ax4)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / 'model_evaluation_summary.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation plots saved to: {plot_file}")
    
    def _plot_scatter_with_density(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        ax: plt.Axes,
        metrics: Dict[str, float]
    ) -> None:
        """Create scatter plot with density information."""
        # Remove invalid values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        pred_clean = predictions[valid_mask]
        target_clean = targets[valid_mask]
        
        # Create 2D histogram for density
        h, xedges, yedges = np.histogram2d(target_clean, pred_clean, bins=50)
        h = h.T  # Transpose for correct orientation
        
        # Plot density
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(h, extent=extent, origin='lower', aspect='auto', 
                      cmap='Blues', norm=LogNorm(vmin=1))
        
        # Add 1:1 line
        max_val = max(np.max(target_clean), np.max(pred_clean))
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
    
    def _plot_residuals(self, predictions: np.ndarray, targets: np.ndarray, ax: plt.Axes) -> None:
        """Create residual plot."""
        residuals = predictions - targets
        valid_mask = ~np.isnan(residuals)
        
        ax.scatter(targets[valid_mask], residuals[valid_mask], alpha=0.5, s=1)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Observed Height (m)', fontsize=12)
        ax.set_ylabel('Residuals (m)', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_height_distributions(self, predictions: np.ndarray, targets: np.ndarray, ax: plt.Axes) -> None:
        """Create height distribution comparison."""
        bins = np.linspace(0, 30, 31)
        
        ax.hist(targets[~np.isnan(targets)], bins=bins, alpha=0.6, label='Observed', density=True)
        ax.hist(predictions[~np.isnan(predictions)], bins=bins, alpha=0.6, label='Predicted', density=True)
        
        ax.set_xlabel('Height (m)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Height Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_error_by_height_bins(self, predictions: np.ndarray, targets: np.ndarray, ax: plt.Axes) -> None:
        """Create error analysis by height bins."""
        # Define height bins
        bin_edges = np.array([0, 1, 2, 4, 8, 12, 16, 20, 30])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        mae_by_bin = []
        bias_by_bin = []
        
        for i in range(len(bin_edges) - 1):
            mask = (targets >= bin_edges[i]) & (targets < bin_edges[i+1])
            mask = mask & ~np.isnan(predictions) & ~np.isnan(targets)
            
            if np.sum(mask) > 0:
                residuals = predictions[mask] - targets[mask]
                mae_by_bin.append(np.mean(np.abs(residuals)))
                bias_by_bin.append(np.mean(residuals))
            else:
                mae_by_bin.append(np.nan)
                bias_by_bin.append(np.nan)
        
        # Plot
        ax.plot(bin_centers, mae_by_bin, 'o-', label='MAE', linewidth=2, markersize=6)
        ax.plot(bin_centers, bias_by_bin, 's-', label='Bias', linewidth=2, markersize=6)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Height Bin (m)', fontsize=12)
        ax.set_ylabel('Error (m)', fontsize=12)
        ax.set_title('Error by Height Range', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def evaluate_model(
        self, 
        checkpoint_path: str, 
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete model evaluation pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_dir: Output directory for results
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Starting model evaluation with checkpoint: {checkpoint_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(self.config['data']['checkpoint_dir']).parent / 'evaluation_results'
        else:
            output_dir = Path(output_dir)
        
        ensure_directory(output_dir)
        
        try:
            # Setup components
            self._setup_datamodule()
            self._setup_model()
            self._setup_trainer()
            
            # Load model from checkpoint
            self.model = CanopyHeightRegressionTask.load_from_checkpoint(
                checkpoint_path,
                nan_value_target=self.datamodule.hparams['nan_target'],
                nan_value_input=self.datamodule.hparams['nan_input']
            )
            
            # Run evaluation
            self.logger.info("Running model testing...")
            self.trainer.test(self.model, datamodule=self.datamodule)
            
            # Get predictions and targets
            predictions, targets = self.pred_collector.get_predictions_and_targets()
            
            self.logger.info(f"Collected {len(predictions)} prediction-target pairs")
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, targets, confidence_level=0.95)
            
            # Create plots
            self._create_evaluation_plots(predictions, targets, output_dir, metrics)
            
            # Save metrics
            metrics_file = output_dir / 'evaluation_metrics.json'
            import json
            with open(metrics_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                              for k, v in metrics.items()}
                json.dump(json_metrics, f, indent=2)
            
            self.logger.info(f"Evaluation results saved to: {output_dir}")
            self.logger.info("Evaluation Summary:")
            self.logger.info(f"  MAE: {metrics['mae']:.3f} m")
            self.logger.info(f"  RMSE: {metrics['rmse']:.3f} m")
            self.logger.info(f"  R²: {metrics['r2']:.3f}")
            self.logger.info(f"  Bias: {metrics['bias']:.3f} m")
            
            return {
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets,
                'output_dir': str(output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise


def main():
    """Main entry point for model evaluation."""
    # This would be called from a script
    evaluator = ModelEvaluationPipeline()
    
    # Example usage
    checkpoint_path = "path/to/checkpoint.ckpt"
    results = evaluator.evaluate_model(checkpoint_path)
    
    return results


if __name__ == "__main__":
    main()
