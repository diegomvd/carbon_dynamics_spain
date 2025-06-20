"""
Range-aware loss functions for canopy height regression in log-space.

This module provides specialized loss functions designed for canopy height regression
tasks that operate in log-space and incorporate range-specific weighting to handle
the natural imbalance in vegetation height distributions. The loss functions
encourage diverse height predictions while maintaining accuracy across all ranges.

The main loss function applies inverse frequency weighting to height ranges,
promoting better performance on underrepresented height classes and improving
overall model generalization across the full spectrum of vegetation heights.

Key features:
- Operates in log1p space for numerical stability
- Range-specific weighting based on height distribution
- Inverse frequency weighting to balance height classes
- Robust handling of NaN values and edge cases

Author: Diego Bengochea
"""

from typing import Optional, Dict, Tuple

# Third-party imports
import torch
import torch.nn as nn

# Shared utilities
from shared_utils import get_logger


# Initialize module logger
logger = get_logger(__name__)


class RangeAwareL1Loss(nn.Module):
    """
    Loss function for height prediction in log1p space with range diversity weighting.
    
    This loss function operates in log-space and applies inverse frequency weighting
    to different height ranges, encouraging the model to perform well across all
    vegetation height ranges rather than just the most common ones.
    
    The range calculation is performed in natural space (after expm1 transformation)
    which naturally handles the log space predictions while providing interpretable
    height-based weighting. This approach addresses the natural imbalance in vegetation
    height distributions where shorter vegetation is much more common than tall trees.
    
    Mathematical details:
    - Base loss: L1 loss in log1p space
    - Range weighting: w_i = 1 / (freq_i^alpha + eps)
    - Final loss: mean(w_i * |pred_i - target_i|) for pixels in range i
    
    Args:
        percentile_range: Percentile range for diversity calculation (currently unused)
        lambda_reg: Weight for range diversity regularization term (currently unused)
        max_height: Maximum height for range definitions in meters
        eps: Small constant for numerical stability in inverse frequency calculation
        nan_value: Value to be treated as no-data/invalid
        alpha: Exponent for inverse frequency weighting (0=no weighting, 1=full inverse)
        weights: Whether to apply range-specific weighting
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
            # Creates ranges: [0,1), [1,2), [2,3), ..., [29,30), [30,inf)
            self.height_ranges = [(float(i), float(i+1)) for i in range(int(max_height))]
            self.height_ranges.append((int(max_height), float('inf')))
            logger.debug(f"Initialized {len(self.height_ranges)} height ranges for weighting")
        
        logger.info(f"Initialized RangeAwareL1Loss with weights={weights}, alpha={alpha}, max_height={max_height}")

    def _compute_range_frequencies(self, target: torch.Tensor) -> Dict[int, float]:
        """
        Compute frequency of samples in each height range for current batch.
        
        This method analyzes the distribution of target heights within the current
        batch to compute range-specific frequencies. These frequencies are used
        to determine inverse weights for balancing the loss across height ranges.
        
        Args:
            target (torch.Tensor): Target tensor in log1p space
            
        Returns:
            Dict[int, float]: Dictionary mapping range indices to frequencies
        """
        try:
            # Convert from log space to natural space for range analysis
            natural_target = torch.expm1(target)
            frequencies = {}
            valid_mask = target != self.nan_value
            total_valid = valid_mask.sum().item()
            
            # Handle empty batch case - assign uniform frequencies
            if total_valid == 0:
                uniform_freq = 1.0 / len(self.height_ranges)
                frequencies = {i: uniform_freq for i in range(len(self.height_ranges))}
                logger.debug("Empty batch detected, using uniform frequencies")
                return frequencies
            
            # Compute frequencies for each height range
            for i, (min_h, max_h) in enumerate(self.height_ranges):
                if max_h == float('inf'):
                    range_mask = (natural_target >= min_h) & valid_mask
                else:
                    range_mask = (natural_target >= min_h) & (natural_target < max_h) & valid_mask
                
                range_count = range_mask.sum().item()
                frequencies[i] = range_count / total_valid
            
            # Log frequency distribution for debugging
            non_zero_ranges = sum(1 for freq in frequencies.values() if freq > 0)
            logger.debug(f"Computed frequencies for {non_zero_ranges}/{len(self.height_ranges)} ranges with data")
            
            return frequencies
            
        except Exception as e:
            logger.error(f"Failed to compute range frequencies: {e}")
            # Return uniform frequencies as fallback
            uniform_freq = 1.0 / len(self.height_ranges)
            return {i: uniform_freq for i in range(len(self.height_ranges))}
        
    def forward(
        self,
        pred: torch.Tensor,  # Predictions in log1p space
        target: torch.Tensor,  # Targets in log1p space
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the range-aware L1 loss.
        
        This method computes the L1 loss in log-space with optional range-specific
        weighting. The weighting is applied based on the inverse frequency of height
        ranges in the current batch, encouraging the model to pay attention to
        underrepresented height classes.
        
        Args:
            pred (torch.Tensor): Model predictions in log1p space
            target (torch.Tensor): Ground truth targets in log1p space
            mask (torch.Tensor, optional): Additional mask for valid pixels
            
        Returns:
            torch.Tensor: Computed loss value
            
        Raises:
            RuntimeError: If loss computation fails
        """
        try:
            # Compute range-specific weights if enabled
            if self.weights:
                # Get batch frequencies and compute inverse weights
                frequencies = self._compute_range_frequencies(target)
                weights = {i: 1.0 / (freq**self.alpha + self.eps) for i, freq in frequencies.items()}
                logger.debug(f"Computed inverse weights with alpha={self.alpha}")
            
            # Create valid data mask
            if mask is None:
                mask = target != self.nan_value
                
            valid_pred = pred[mask]
            valid_target = target[mask]
            
            # Handle empty batch case
            if len(valid_pred) == 0:
                logger.warning("No valid pixels in batch, returning zero loss")
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
            # Base L1 loss in log1p space
            base_loss = torch.abs(valid_pred - valid_target)
            
            # Apply range-specific weights if enabled
            if self.weights:
                # Convert to natural space for range assignment
                natural_target = torch.expm1(valid_target)
                
                # Initialize weighted loss tensor
                weighted_loss = torch.zeros_like(base_loss)
                
                # Apply range-specific weights
                for i, (min_h, max_h) in enumerate(self.height_ranges):
                    if max_h == float('inf'):
                        range_mask = natural_target >= min_h
                    else:
                        range_mask = (natural_target >= min_h) & (natural_target < max_h)
                    
                    if range_mask.any():
                        weight = weights[i]
                        weighted_loss[range_mask] = base_loss[range_mask] * weight
                
                mean_loss = torch.mean(weighted_loss)
                logger.debug(f"Applied range weighting to {len(valid_pred)} valid pixels")
            else:
                mean_loss = torch.mean(base_loss)
                logger.debug(f"Computed standard L1 loss for {len(valid_pred)} valid pixels")
            
            return mean_loss
            
        except Exception as e:
            logger.error(f"Failed to compute range-aware L1 loss: {e}")
            # Fallback to simple L1 loss
            try:
                if mask is None:
                    mask = target != self.nan_value
                valid_pred = pred[mask]
                valid_target = target[mask]
                if len(valid_pred) > 0:
                    fallback_loss = torch.mean(torch.abs(valid_pred - valid_target))
                    logger.warning("Using fallback L1 loss due to error")
                    return fallback_loss
                else:
                    return torch.tensor(0.0, device=pred.device, requires_grad=True)
            except Exception as fallback_error:
                logger.error(f"Fallback loss computation also failed: {fallback_error}")
                raise RuntimeError(f"Loss computation failed: {e}") from e


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    'RangeAwareL1Loss'
]
