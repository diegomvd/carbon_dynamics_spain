"""
Range-aware loss functions for canopy height regression in log-space.

This module provides specialized loss functions designed for canopy height regression
tasks that operate in log-space and incorporate range-specific weighting to handle
the natural imbalance in vegetation height distributions. The loss functions
encourage diverse height predictions while maintaining accuracy across all ranges.

The main loss function applies inverse frequency weighting to height ranges,
promoting better performance on underrepresented height classes and improving
overall model generalization across the full spectrum of vegetation heights.

Author: Diego Bengochea
"""

from typing import Optional, Dict, Tuple

# Third-party imports
import torch
import torch.nn as nn


class RangeAwareL1Loss(nn.Module):
    """
    Loss function for height prediction in log1p space with range diversity weighting.
    
    This loss function operates in log-space and applies inverse frequency weighting
    to different height ranges, encouraging the model to perform well across all
    vegetation height ranges rather than just the most common ones.
    
    The range calculation is performed in natural space (after expm1 transformation)
    which naturally handles the log space predictions while providing interpretable
    height-based weighting.
    """
    
    def __init__(
        self,
        percentile_range: Tuple[float, float] = (10.0, 90.0),
        lambda_reg: float = 0.25,
        max_height: float = 30.0,
        eps: float = 1e-6,
        nan_value: float = -1.0,
        alpha: float = 0.5,
        weights: bool = True
    ):
        """
        Initialize the range-aware L1 loss function.
        
        Args:
            percentile_range (Tuple[float, float]): Percentile range for diversity calculation
            lambda_reg (float): Weight for range diversity regularization term
            max_height (float): Maximum height for range definitions
            eps (float): Small constant for numerical stability
            nan_value (float): Value to be treated as no-data
            alpha (float): Exponent for inverse frequency weighting (0=no weighting, 1=full inverse)
            weights (bool): Whether to apply range-specific weighting
        """
        super().__init__()
        self.percentile_range = percentile_range
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.nan_value = nan_value
        self.weights = weights
        self.alpha = alpha

        if weights:
            # Define 1-meter height ranges for weighting
            self.height_ranges = [(float(i), float(i+1)) for i in range(int(max_height))]
            self.height_ranges.append((int(max_height), float('inf')))

    def _compute_range_frequencies(self, target: torch.Tensor) -> Dict[int, float]:
        """
        Compute frequency of samples in each height range for current batch.
        
        Args:
            target (torch.Tensor): Target tensor in log1p space
            
        Returns:
            Dict[int, float]: Dictionary mapping range indices to frequencies
        """
        # Convert from log space to natural space
        natural_target = torch.expm1(target)
        frequencies = {}
        valid_mask = target != self.nan_value
        total_valid = valid_mask.sum().item()
        
        # Handle empty batch case
        if total_valid == 0:
            return {i: 1.0/len(self.height_ranges) for i in range(len(self.height_ranges))}
        
        # Compute frequencies for each 1m bin
        for i, (min_h, max_h) in enumerate(self.height_ranges):
            range_mask = (natural_target >= min_h) & (natural_target < max_h) & valid_mask
            frequencies[i] = range_mask.sum().item() / total_valid
            
        return frequencies    
        
    def forward(
        self,
        pred: torch.Tensor,  # Predictions in log1p space
        target: torch.Tensor,  # Targets in log1p space
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the range-aware L1 loss.
        
        Args:
            pred (torch.Tensor): Model predictions in log1p space
            target (torch.Tensor): Ground truth targets in log1p space
            mask (torch.Tensor, optional): Additional mask for valid pixels
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Compute range-specific weights if enabled
        if self.weights:
            # Get batch frequencies
            frequencies = self._compute_range_frequencies(target)
            weights = {i: 1.0 / (freq**self.alpha + self.eps) for i, freq in frequencies.items()}
        
        # Create valid data mask
        if mask is None:
            mask = target != self.nan_value
            
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        # Handle empty batch
        if len(valid_pred) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Base L1 loss in log1p space
        base_loss = torch.abs(valid_pred - valid_target)
        
        # Apply range-specific weights if enabled
        if self.weights:
            # Convert to natural space for range assignment
            natural_target = torch.expm1(valid_target)
            
            # Apply range-specific weights
            weighted_loss = torch.zeros_like(base_loss)
            for i, (min_h, max_h) in enumerate(self.height_ranges):
                range_mask = (natural_target >= min_h) & (natural_target < max_h)
                if range_mask.any():
                    weighted_loss[range_mask] = base_loss[range_mask] * weights[i]
            
            mean_loss = torch.mean(weighted_loss)
        else:
            mean_loss = torch.mean(base_loss)
        
        return mean_loss