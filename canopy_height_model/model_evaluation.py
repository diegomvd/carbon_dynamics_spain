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

# Local imports
from config import load_config, setup_logging, create_output_directory
from s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from canopy_height_regression import CanopyHeightRegressionTask


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
            Tuple[np.ndarray, np.ndarray]: Combined predictions and targets arrays
        """
        return (
            np.concatenate(self.predictions),
            np.concatenate(self.targets)
        )


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization pipeline.
    
    This class provides methods for statistical analysis, plotting, and 
    performance assessment of canopy height regression models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model evaluator with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.output_dir = Path(config['evaluation']['output_dir'])
        self.logger = setup_logging(config['logging']['level'])
        
        # Create output directory
        create_output_directory(self.output_dir)
        
        # Set style for publication-ready plots
        self._setup_plot_style()
        
        # Initialize components
        self.pred_collector = PredictionCollector()
        self._setup_datamodule()
        self._setup_model()
        self._setup_trainer()
        
        self.logger.info(f"Model evaluator initialized. Output directory: {self.output_dir}")

    def _setup_plot_style(self) -> None:
        """Configure matplotlib and seaborn for publication-ready plots."""
        sns.set_style("white")
        sns.set_context("paper")
        plt.rcParams.update({
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'legend.title_fontsize': 9
        })

    def _setup_datamodule(self) -> None:
        """Initialize the data module."""
        self.datamodule = S2PNOAVegetationDataModule(
            data_dir=str(self.config['data']['data_dir'])
        )
        
    def _setup_model(self) -> None:
        """Initialize the model."""
        self.model = CanopyHeightRegressionTask(
            nan_value_target=self.datamodule.hparams['nan_target'],
            nan_value_input=self.datamodule.hparams['nan_input']
        )
        
    def _setup_trainer(self) -> None:
        """Initialize the PyTorch Lightning trainer."""
        accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.trainer = Trainer(
            accelerator=accelerator,
            devices=self.config['compute']['devices'],
            callbacks=[self.pred_collector],
            enable_checkpointing=False,
            enable_progress_bar=True,
            inference_mode=True,
            logger=False
        )

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
        r = stats.pearsonr(targets, predictions)[0]
        r2 = r2_score(targets, predictions)
        
        metrics = {
            'MAE': mae,
            'Bias': bias,
            'RMSE': rmse,
            'R2': r**2,
            'r2': r2
        }
        
        # Calculate confidence intervals if requested
        if confidence_level:
            self.logger.info('Calculating confidence intervals...')
            n = len(predictions)
            confidence_metrics = {}
            
            # Bootstrap confidence intervals
            n_bootstrap = 1000
            rng = np.random.default_rng()
            
            for metric in ['MAE', 'Bias', 'RMSE']:
                bootstrap_values = []
                for _ in range(n_bootstrap):
                    # Resample with replacement
                    idx = rng.choice(n, size=n, replace=True)
                    boot_pred = predictions[idx]
                    boot_targ = targets[idx]
                    boot_resid = boot_pred - boot_targ
                    
                    if metric == 'MAE':
                        val = np.mean(np.abs(boot_resid))
                    elif metric == 'Bias':
                        val = np.mean(boot_resid)
                    else:  # RMSE
                        val = np.sqrt(np.mean(boot_resid**2))
                    bootstrap_values.append(val)
                
                # Calculate confidence intervals
                ci_lower = np.percentile(bootstrap_values, (1 - confidence_level) * 100 / 2)
                ci_upper = np.percentile(bootstrap_values, (1 + confidence_level) * 100 / 2)
                confidence_metrics[f'{metric}_CI'] = (ci_lower, ci_upper)
            
            metrics.update(confidence_metrics)
            self.logger.info('Confidence intervals calculated')
            
        return metrics
        
    def _calculate_bin_statistics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        bin_width: float = None,
        max_height: float = None
    ) -> pd.DataFrame:
        """
        Calculate statistics for each height bin.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            bin_width (float): Width of height bins
            max_height (float): Maximum height for binning
            
        Returns:
            pd.DataFrame: DataFrame containing bin statistics
        """
        if bin_width is None:
            bin_width = self.config['evaluation']['plot']['bin_width']
        if max_height is None:
            max_height = self.config['evaluation']['plot']['max_height']

        # Clean data
        predictions = np.where(predictions < 0, 0.0, predictions)
        predictions = np.where(predictions > 60, np.nan, predictions)
        targets = np.where(targets > 60, np.nan, targets)

        # Create bins
        bins_under_10 = np.arange(0, 10 + 2.0, 2.0)
        bins_over_10 = np.arange(10, max_height + bin_width, bin_width)
        bins = np.concatenate([bins_under_10, bins_over_10])
        bin_centers = bins[:-1] + bin_width/2
        
        stats_list = []
        for i in range(len(bins)-1):
            mask = (targets >= bins[i]) & (targets < bins[i+1])
            if mask.any():
                bin_predictions = predictions[mask]
                bin_targets = targets[mask]
                residuals = bin_predictions - bin_targets
                
                stats_list.append({
                    'Height_Bin': f'{bins[i]:.0f}-{bins[i+1]:.0f}m',
                    'Bin_Center': bin_centers[i],
                    'Count': np.nansum(mask),
                    'MAE': np.nanmean(np.abs(residuals)),
                    'Bias': np.nanmean(residuals),
                    'RMSE': np.sqrt(np.nanmean(residuals**2)),
                    'Residuals': residuals
                })
        
        return pd.DataFrame(stats_list)

    def plot_residual_boxplots(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        bin_width: float = None,
        max_height: float = None
    ) -> None:
        """
        Create boxplot of residuals binned by height using seaborn's catplot.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            bin_width (float): Width of height bins
            max_height (float): Maximum height for binning
        """
        if bin_width is None:
            bin_width = self.config['evaluation']['plot']['bin_width']
        if max_height is None:
            max_height = self.config['evaluation']['plot']['max_height']

        self.logger.info("Creating residual boxplots...")

        # Clean data
        predictions = np.where(predictions < 0, 0.0, predictions)
        predictions = np.where(predictions > 60, np.nan, predictions)
        targets = np.where(targets > 60, np.nan, targets)

        # Get bin statistics and prepare data
        bin_stats = self._calculate_bin_statistics(predictions, targets, bin_width, max_height)
        
        # Explode the residuals column into long format
        plot_data = []
        for _, row in bin_stats.iterrows():
            for residual in row['Residuals']:
                plot_data.append({
                    'Height Bin': row['Height_Bin'],
                    'Residual': residual,
                    'Bin Center': row['Bin_Center']
                })
        df = pd.DataFrame(plot_data).dropna()

        # Create boxplot using catplot with consistent color
        g = sns.catplot(
            data=df,
            x='Height Bin',
            y='Residual',
            kind='box',
            height=4,
            aspect=1.7,
            color='#FDA832',
            showfliers=False,
            width=0.3
        )
        
        # Add horizontal dotted line at y=0
        g.ax.axhline(y=0, color='gray', linestyle=':', zorder=0)
        
        # Customize the plot
        g.set_xticklabels(rotation=45, ha='right')
        g.ax.set_title('Model Residuals by Height Bin')
        
        # Add bias values for each bin
        for i, stats in enumerate(bin_stats.itertuples()):
            g.ax.text(
                i, 
                g.ax.get_ylim()[0],
                f'Bias: {stats.Bias:.2f}m\nn={stats.Count}',
                rotation=90,
                va='bottom',
                ha='center',
                fontsize=8
            )
        
        # Add overall metrics
        metrics = self._calculate_metrics(predictions, targets, confidence_level=None)
        g.ax.text(
            0.02, 0.98,
            f"Overall Metrics:\nMAE: {metrics['MAE']:.2f}m\nBias: {metrics['Bias']:.2f}m",
            transform=g.ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        output_path = self.output_dir / 'residual_boxplots.png'
        g.savefig(output_path, dpi=self.config['evaluation']['plot']['figure_dpi'], 
                 bbox_inches=self.config['evaluation']['plot']['bbox_inches'])
        plt.close()
        
        # Save data
        df.to_csv(self.output_dir / 'residual_data.csv', index=False)
        self.logger.info(f"Residual boxplots saved to {output_path}")
        
    def plot_prediction_density(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> None:
        """
        Create 2D histogram of predictions vs targets.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
        """
        self.logger.info("Creating prediction density plot...")

        # Clean data
        predictions = np.where(predictions < 0, 0.0, predictions)
        predictions = np.where(predictions > 60, np.nan, predictions)
        targets = np.where(targets > 60, np.nan, targets)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Predicted': predictions,
            'Measured': targets
        })

        df = df[(df.Predicted > 1.0) & (df.Measured > 1.0)].dropna()
        
        # Create figure
        plt.figure(figsize=(3.5, 3.5))
        
        # Create 2D histogram
        g = sns.histplot(
            data=df,
            x='Measured',
            y='Predicted',
            bins=150,
            cmap='magma',
            norm=LogNorm(),
            vmin=None,
            vmax=None,
            cbar=True,
            cbar_kws={'label': 'Count'}
        )
        
        # Add 1:1 line
        max_val = max(np.nanmax(predictions), np.nanmax(targets))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 line')
        
        # Calculate and add metrics
        metrics = self._calculate_metrics(predictions, targets)
        plt.text(
            0.02, 0.98,
            f"R² = {metrics['R2']:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=8
        )
        
        plt.xlabel('Measured Height (m)')
        plt.ylabel('Predicted Height (m)')
        
        # Despine
        sns.despine(trim=False)
        
        plt.tight_layout()
        output_path = self.output_dir / 'prediction_density.png'
        plt.savefig(output_path, dpi=self.config['evaluation']['plot']['figure_dpi'], 
                   bbox_inches=self.config['evaluation']['plot']['bbox_inches'])
        plt.close()
        
        # Save data
        df.to_csv(self.output_dir / 'density_data.csv', index=False)
        self.logger.info(f"Prediction density plot saved to {output_path}")
        
    def plot_height_distributions(
        self, 
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """
        Plot height distributions in log scale using displot.
        
        Args:
            targets (np.ndarray): Ground truth targets
            predictions (np.ndarray): Model predictions
        """
        self.logger.info("Creating height distribution plots...")

        # Clean data
        predictions = np.where(predictions < 0, 0.0, predictions)
        predictions = np.where(predictions > 60, np.nan, predictions)
        targets = np.where(targets > 60, np.nan, targets)        

        # Prepare data
        df = pd.DataFrame({
            'Height (m)': np.concatenate([targets, predictions]),
            'Type': ['Measured'] * len(targets) + ['Predicted'] * len(predictions)
        })

        df = df[df['Height (m)'] > 1].dropna()
        
        # Create figure with displot
        g = sns.displot(
            kind='hist',
            data=df,
            x='Height (m)',
            hue='Type',
            height=6,
            aspect=1.2,
            palette=['#FDA832', '#FFD082']
        )
        g.set(yscale='log')
                               
        g.ax.axvline(x=np.nanpercentile(targets, 95), color='k', linestyle=':', zorder=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'height_distributions.png'
        plt.savefig(output_path, dpi=self.config['evaluation']['plot']['figure_dpi'], 
                   bbox_inches=self.config['evaluation']['plot']['bbox_inches'])
        plt.close()
        
        # Save data
        pd.DataFrame({
            'Measured_Heights': targets,
            'Predicted_Heights': predictions
        }).to_csv(self.output_dir / 'height_distribution_data.csv', index=False)
        self.logger.info(f"Height distribution plots saved to {output_path}")

    def evaluate(self) -> Dict:
        """
        Run comprehensive evaluation pipeline.
        
        Returns:
            Dict: Test metrics dictionary
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        try:
            # Create base model - weights will be loaded by trainer
            model = CanopyHeightRegressionTask(
                nan_value_target=self.datamodule.hparams['nan_target'],
                nan_value_input=self.datamodule.hparams['nan_input']
            )
            
            # Run test metrics and collect predictions
            self.logger.info("Running test metrics and collecting predictions...")
            test_metrics = self.trainer.test(
                model,
                datamodule=self.datamodule,
                ckpt_path=self.config['evaluation']['checkpoint_path']
            )
            
            # Get collected predictions
            predictions, targets = self.pred_collector.get_predictions_and_targets()
            
            # Create visualizations
            self.logger.info("Creating visualizations...")
            self.plot_residual_boxplots(predictions, targets)
            self.plot_prediction_density(predictions, targets)
            self.plot_height_distributions(targets, predictions)
            
            self.logger.info(f"Evaluation completed successfully. Results saved to {self.output_dir}")
            return test_metrics[0] if test_metrics else {}
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise


def main():
    """Main entry point for the evaluation pipeline."""
    try:
        # Load configuration
        config = load_config()
        logger = setup_logging(config['logging']['level'])
        logger.info("Starting Canopy Height Model Evaluation Pipeline...")
        
        # Run evaluation
        evaluator = ModelEvaluator(config)
        results = evaluator.evaluate()
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()