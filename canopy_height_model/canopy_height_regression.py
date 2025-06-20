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

# Local imports
from config import load_config
from height_regression_losses import RangeAwareL1Loss


def get_balanced_universal_loss(config: Dict[str, Any]) -> RangeAwareL1Loss:
    """
    Create a balanced universal loss function from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing loss parameters
        
    Returns:
        RangeAwareL1Loss: Configured loss function instance
    """
    loss_config = config['loss']
    return RangeAwareL1Loss(
        percentile_range=tuple(loss_config['percentile_range']),
        lambda_reg=loss_config['lambda_reg'],
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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with height ranges from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing height ranges
        """
        # Convert .inf string back to float('inf') for configuration compatibility
        self.height_ranges = []
        for min_h, max_h in config['model']['height_ranges']:
            if max_h == '.inf':
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
        target_range: str = "universal",  
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
            model (str): Model architecture type
            backbone (str): Backbone architecture
            weights: Pre-trained weights
            in_channels (int): Number of input channels
            num_outputs (int): Number of output channels
            target_range (str): Target height range for specialized training
            nan_value_target (float): NaN value for target data
            nan_value_input (float): NaN value for input data
            lr (float): Learning rate
            patience (int): Patience for learning rate scheduler
            config_path (str, optional): Path to configuration file
            **kwargs: Additional arguments
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Override config with provided parameters
        self.nan_value_target = nan_value_target
        self.nan_value_input = nan_value_input
        self.nan_counter = 0
        self.target_range = target_range
        
        # Configure range-specific parameters and metrics
        self.range_configs = {
            "universal": {
                "loss_fn": get_balanced_universal_loss(self.config),
                "metric_ranges": self.config['model']['metric_ranges']['universal']
            }
        }
        
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
        
    def _create_input_mask(self, x: Tensor) -> Tensor:
        """
        Create mask for valid input values across all bands.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Boolean mask where True indicates valid pixels
        """
        return ~(x == self.nan_value_input).any(dim=1, keepdim=True)

    def _handle_nan_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Handle NaN values in input tensor.
        
        Args:
            x (Tensor): Input tensor with potential NaN values
            
        Returns:
            Tuple[Tensor, Tensor]: Filled tensor and validity mask
        """
        input_mask = self._create_input_mask(x)
        x_filled = x.clone()
        x_filled[x == self.nan_value_input] = 0.0  # Fill nodata with zeros
        return x_filled, input_mask

    def configure_losses(self) -> None:
        """Configure the loss function based on target range."""
        if self.target_range not in self.range_configs:
            raise ValueError(f"Unknown target range: {self.target_range}")
        self.criterion = self.range_configs[self.target_range]["loss_fn"]

    def _nan_robust_loss_reduction(
        self, 
        loss: Tensor, 
        y: Tensor, 
        nan_value_target: float, 
        input_mask: Tensor
    ) -> Tensor:
        """
        Apply NaN-robust loss reduction.
        
        Args:
            loss (Tensor): Loss tensor
            y (Tensor): Target tensor
            nan_value_target (float): NaN value for targets
            input_mask (Tensor): Valid input mask
            
        Returns:
            Tensor: Reduced loss value
        """
        target_mask = y != nan_value_target
        valid_mask = input_mask & target_mask    
        loss = torch.masked_select(loss, valid_mask)
        return torch.mean(loss) if loss.numel() > 0 else torch.tensor(1e5, device=loss.device)

    def _calculate_range_metrics(
        self, 
        y_hat: Tensor, 
        y: Tensor, 
        input_mask: Tensor
    ) -> Dict[str, float]:
        """
        Calculate metrics for specific height ranges.
        
        Args:
            y_hat (Tensor): Model predictions
            y (Tensor): Ground truth targets
            input_mask (Tensor): Valid input mask
            
        Returns:
            Dict[str, float]: Dictionary containing range-specific metrics
        """
        metrics = {}
        current_ranges = self.range_configs[self.target_range]["metric_ranges"]

        valid_data_mask = y != self.nan_value_target

        y_hat = torch.clamp(torch.expm1(y_hat), max=100.0)
        y = torch.expm1(y)
 
        for min_val, max_val in current_ranges:
            # Define range name
            range_name = f"h_{min_val}_{max_val}".replace('.', '_')
            if max_val == float('inf'):
                range_mask = (y >= min_val)
            else:
                range_mask = (y >= min_val) & (y < max_val)
            
            # Combine masks
            valid_mask = range_mask & valid_data_mask & input_mask
            
            if valid_mask.any():
                range_preds = y_hat[valid_mask]
                range_targets = y[valid_mask]
                
                # Calculate metrics
                mae = torch.nn.functional.l1_loss(range_preds, range_targets).item()
                rmse = torch.sqrt(torch.nn.functional.mse_loss(range_preds, range_targets)).item()
                bias = (range_preds - range_targets).mean().item()
                
                metrics[f"{range_name}_mae"] = mae
                metrics[f"{range_name}_rmse"] = rmse
                metrics[f"{range_name}_bias"] = bias
                metrics[f"{range_name}_count"] = valid_mask.sum().item()
            else:
                metrics[f"{range_name}_mae"] = 0.0
                metrics[f"{range_name}_rmse"] = 0.0
                metrics[f"{range_name}_bias"] = 0.0
                metrics[f"{range_name}_count"] = 0
                
        return metrics

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """
        Execute training step.
        
        Args:
            batch: Input batch containing image and mask
            batch_idx: Batch index
            dataloader_idx: Dataloader index
            
        Returns:
            Tensor: Loss value
        """
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        y_hat = self(x_filled)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
            
        loss: Tensor = self.criterion(y_hat, y)
        
        # Handle potential NaN loss
        if torch.isnan(loss):
            self.nan_counter += 1
            loss = torch.tensor(1e5, device=loss.device, requires_grad=True)

        # Log current parameters
        self.log("train_loss", loss, prog_bar=True)
        self.log("nan_count", self.nan_counter)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Execute validation step.
        
        Args:
            batch: Input batch containing image and mask
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
   
        # Handle NaN inputs and get mask
        x_filled, input_mask = self._handle_nan_inputs(x)

        # Forward pass with filled inputs
        y_hat = self(x_filled)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.criterion(y_hat, y)

        # Handle potential NaN loss
        if torch.isnan(loss):
            loss = torch.tensor(1e5, device=loss.device)
    
        # Log overall loss
        self.log("val_loss", loss)
        
        metrics = self._calculate_range_metrics(y_hat * input_mask, y, input_mask)
        for name, value in metrics.items():
            self.log(f"val_{name}", value)
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Execute test step.
        
        Args:
            batch: Input batch containing image and mask
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)

        # Handle NaN inputs and get mask
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        # Forward pass with filled inputs
        y_hat = self(x_filled)

        return None

    def predict_step(
        self, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> Optional[Tensor]:
        """
        Execute prediction step.
        
        Args:
            batch: Input batch containing image
            batch_idx: Batch index
            dataloader_idx: Dataloader index
            
        Returns:
            Optional[Tensor]: Predictions in natural space
        """
        if batch_idx > -1:
            x = batch['image']
            y_hat: Tensor = self(x)
            return torch.expm1(y_hat)    
        return None

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers from configuration.
        
        Returns:
            Dict: Dictionary containing optimizer and scheduler configuration
        """
        optimizer_config = self.config['optimizer']
        scheduler_config = self.config['scheduler']
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scheduler_config['max_lr'],  
            weight_decay=optimizer_config['weight_decay']
        )
        
        total_steps = self.trainer.estimated_stepping_batches

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config['max_lr'],
            epochs=self.trainer.max_epochs,
            total_steps=total_steps,
            pct_start=scheduler_config['pct_start'],
            div_factor=scheduler_config['div_factor'],        
            final_div_factor=scheduler_config['final_div_factor'], 
            three_phase=scheduler_config['three_phase'],
            anneal_strategy=scheduler_config['anneal_strategy']
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }