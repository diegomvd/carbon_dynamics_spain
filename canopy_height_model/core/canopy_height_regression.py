"""
Range-specific canopy height regression task using PyTorch Lightning.

This module implements a comprehensive deep learning task for canopy height regression
with range-aware loss functions, balanced sampling, and detailed metrics across 
different height ranges. Supports multiple model architectures with optimized
training schedules and comprehensive evaluation.

The task handles NaN values, applies log-space transformations, and provides
detailed range-specific metrics for thorough model assessment.

Author: Diego Bengochea
"""

from typing import Any, Optional, Union, List, Tuple, Dict

# Standard library imports
import math

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

# PyTorch Lightning imports
from torchgeo.trainers import PixelwiseRegressionTask

# Shared utilities
from shared_utils import get_logger, load_config

from .height_regression_losses import FrequecnyBalancedL1Loss


def get_frequency_balanced_loss(config: Dict[str, Any]) -> Optional[Any]:
    """
    Create frequency-balanced loss function from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing loss parameters
        
    Returns:
        FrequecnyBalancedL1Loss: Configured loss function instance or None if not available
    """        
    loss_config = config['loss']
    return FrequecnyBalancedL1Loss(
        alpha=loss_config['alpha'],
        max_height=loss_config['max_height'],
        eps=loss_config['eps'],
        weights=loss_config['weights']
    )
        

class HeightRangeMetrics:
    """
    Height range specific metrics calculator for canopy height regression.
    
    This class calculates detailed metrics (MAE, std, bias) for predefined
    height ranges to assess model performance across different vegetation heights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with height ranges from configuration or defaults."""
        if config and 'evaluation' in config and 'height_ranges' in config['evaluation']:
            height_ranges_config = config['evaluation']['height_ranges']
        else:
            # Original default ranges
            height_ranges_config = [[0, 1], [1, 2], [2, 4], [4, 8], 
                                   [8, 12], [12, 16], [16, 20], [20, 25], 
                                   [25, float('inf')]]
        
        # Convert to tuples and handle infinity
        self.height_ranges = []
        for min_h, max_h in height_ranges_config:
            if max_h == '.inf' or max_h == 'inf':
                max_h = float('inf')
            self.height_ranges.append((min_h, max_h))
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        input_mask: torch.Tensor,
        nan_value: float
    ) -> Dict[str, float]:
        """
        Calculate metrics for each height range.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            input_mask (torch.Tensor): Valid input mask
            nan_value (float): Value representing NaN/missing data
            
        Returns:
            Dict[str, float]: Dictionary containing range-specific metrics
        """
        metrics = {}
            
        # Create valid mask
        valid_mask = (targets != nan_value) & input_mask
        
        for min_h, max_h in self.height_ranges:
            # Create range mask
            range_mask = valid_mask & (targets >= min_h) & (targets < max_h)
            
            if range_mask.any():
                range_preds = predictions[range_mask]
                range_targets = targets[range_mask]
                
                # Calculate metrics
                abs_errors = torch.abs(range_preds - range_targets)
                signed_diff = range_preds - range_targets
                
                range_name = f"h_{min_h}_{max_h}"
                metrics.update({
                    f"test_{range_name}_mae": abs_errors.mean(),
                    f"test_{range_name}_std": abs_errors.std(),
                    f"test_{range_name}_signed_mean": signed_diff.mean(),
                    f"test_{range_name}_signed_std": signed_diff.std(),
                    f"test_{range_name}_count": range_mask.sum()
                })
            else:
                range_name = f"h_{min_h}_{max_h}"
                metrics.update({
                    f"test_{range_name}_mae": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_std": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_signed_mean": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_signed_std": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_count": torch.tensor(0, device=predictions.device)
                })
        
        return metrics


class CanopyHeightRegressionTask(PixelwiseRegressionTask):
    """
    LightningModule for range-specific canopy height regression.
    
    This task implements a comprehensive training and evaluation pipeline for
    canopy height regression including range-aware loss functions, detailed
    metrics, and optimized training schedules.
    """
    
    target_key = 'mask'
    
    def __init__(
        self,
        model: str = 'unet',
        backbone: str = "efficientnet-b4",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 10,
        num_outputs: int = 1,
        nan_value_target: float = -1.0,
        nan_value_input: float = 0.0,
        lr: float = 1e-4,
        patience: int = 15,
        config_path: str = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the canopy height regression task.
        
        Args:
            model: Model architecture ('unet', 'deeplabv3+', etc.)
            backbone: Encoder backbone (e.g., 'efficientnet-b4')
            weights: Pre-trained weights to use
            in_channels: Number of input channels
            num_outputs: Number of output channels
            nan_value_target: NaN value for target data
            nan_value_input: NaN value for input data
            lr: Learning rate
            patience: Early stopping patience
            config_path: Path to configuration file
            **kwargs: Additional arguments
        """
        # Store parameters before calling super().__init__
        self.nan_value_target = nan_value_target
        self.nan_value_input = nan_value_input
        self.nan_counter = 0
        self.lr = lr
        self.patience = patience
        
        # Setup logging
        self.module_logger = get_logger('canopy_height_model')
        
        # Load configuration if provided
        self.config = None
        if not config_path:
            try:
                self.config = load_config(config_path, component_name="canopy_height_model")
            except Exception as e:
                self.module_logger.warning(f"Could not load config: {e}")
        
        # Initialize range metrics if config available
        self.range_metrics = None
        if self.config:
            try:
                self.range_metrics = HeightRangeMetrics(self.config)
            except Exception as e:
                self.module_logger.warning(f"Could not initialize range metrics: {e}")
        
        # Call parent constructor
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_outputs=num_outputs,
            lr=lr,
            patience=patience,
            **kwargs
        )
        
        # Setup custom loss if available
        self._setup_custom_loss()
        
        self.module_logger.info(f"CanopyHeightRegressionTask initialized:")
        self.module_logger.info(f"  Model: {model} with {backbone} backbone")
        self.module_logger.info(f"  Input channels: {in_channels}")
        self.module_logger.info(f"  Learning rate: {lr}")
        self.module_logger.info(f"  NaN values: target={nan_value_target}, input={nan_value_input}")
    
    def _setup_custom_loss(self) -> None:
        """Setup custom loss function if available."""
        if self.config:
            try:
                custom_loss = get_frequency_balanced_loss(self.config)
                if custom_loss:
                    self.loss = custom_loss
                    self.module_logger.info("Using custom FrequecnyBalancedL1Loss")
            except Exception as e:
                self.module_logger.warning(f"Could not setup custom loss: {e}")
    
    def _create_input_mask(self, x: Tensor) -> Tensor:
        """Create mask for valid input values across all bands"""
        # Returns True for valid pixels across all bands, False otherwise
        return ~(x == self.nan_value_input).any(dim=1, keepdim=True)  # Mask where any band has nodata

    def _handle_nan_inputs(self, x: Tensor) -> Tensor:
        """Handle NaN values in input tensor."""
        input_mask = self._create_input_mask(x)
        x_filled = x.clone()
        x_filled[x == self.nan_value_input] = 0.0  # Fill nodata with zeros
        return x_filled, input_mask

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step with NaN handling.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """

        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        
        # Handle NaN inputs
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        # Get predictions
        y_hat = self(x_filled)
        
        # Ensure consistent dimensions
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        
        # Calculate loss only on valid pixels
        valid_mask = (y != self.nan_value_target) & input_mask
        
        if valid_mask.any():
            loss = self.loss(y_hat[valid_mask], y[valid_mask])
        else:
            # No valid pixels, return zero loss
            self.nan_counter += 1
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("nan_count", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """
        Validation step with comprehensive metrics.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """

        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        
        # Handle NaN inputs
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        # Get predictions
        y_hat = self(x_filled)
        
        # Ensure consistent dimensions
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        
        # Calculate validation loss
        valid_mask = (y != self.nan_value_target) & input_mask
        
        if valid_mask.any():
            loss = self.loss(y_hat[valid_mask], y[valid_mask])
                
            # Calculate basic metrics on valid pixels
            y_hat_valid = y_hat[valid_mask]
            y_valid = y[valid_mask]
            
            mae = torch.mean(torch.abs(y_hat_valid - y_valid))
            mse = torch.mean((y_hat_valid - y_valid) ** 2)
            rmse = torch.sqrt(mse)
            
            # Log metrics
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_mae", mae, on_epoch=True, prog_bar=True)
            self.log("val_rmse", rmse, on_epoch=True)
    
    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """
        Test step with range-specific metrics.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        
        # Handle NaN inputs
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        # Get predictions
        y_hat = self(x_filled)
        
        # Ensure consistent dimensions
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        
        # Convert from log space if needed
        # (assuming log transformation was applied during preprocessing)
        pred = torch.expm1(y_hat)
        target = torch.expm1(y)
        
        # Calculate basic test metrics
        valid_mask = (y != self.nan_value_target) & input_mask
        
        if valid_mask.any():
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
            
            mae = torch.mean(torch.abs(pred_valid - target_valid))
            mse = torch.mean((pred_valid - target_valid) ** 2)
            rmse = torch.sqrt(mse)
            
            # Log basic metrics
            self.log("test_mae", mae, on_epoch=True)
            self.log("test_rmse", rmse, on_epoch=True)
            self.log("test_mse", mse, on_epoch=True)
            
            # Calculate range-specific metrics if available
            if self.range_metrics:
                range_metrics = self.range_metrics.calculate_metrics(
                    pred, target, input_mask, self.nan_value_target
                )
                
                # Log range-specific metrics
                for metric_name, metric_value in range_metrics.items():
                    self.log(metric_name, metric_value, on_epoch=True)
    
    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Prediction step for inference.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Predictions tensor
        """
        x = batch["image"]
        
        # # Handle NaN inputs
        # x_filled, _ = self._handle_nan_inputs(x)
        
        # Get predictions
        y_hat = self(x)
        
        # Convert from log space
        pred = torch.expm1(y_hat)
        
        return pred

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer and scheduler configuration
        """

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['optimizer']['max_lr'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['optimizer']['max_lr'],
            epochs=self.trainer.max_epochs,
            total_steps=total_steps,
            pct_start=self.config['scheduler']['pct_start'],
            div_factor=self.config['scheduler']['div_factor'],        
            final_div_factor=self.config['scheduler']['final_div_factor'], 
            three_phase=False,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    

