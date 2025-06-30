"""
Data transformation pipeline for PNOA vegetation and Sentinel-2 preprocessing.

This module provides Kornia-based transformations for preprocessing vegetation
height data and satellite imagery in the canopy height regression pipeline.
Includes scaling, outlier removal, log transformations, and data cleaning
operations optimized for vegetation analysis.

The transformations are designed to work with Kornia's augmentation framework
and handle specific characteristics of Spanish PNOA vegetation data and
Sentinel-2 satellite imagery mosaics.

Key transformations:
- S2Scaling: Convert Sentinel-2 digital numbers to reflectance values
- PNOAVegetationRemoveAbnormalHeight: Filter unrealistic vegetation heights
- PNOAVegetationInput0InArtificialSurfaces: Handle artificial surface markers
- PNOAVegetationLogTransform: Apply log transformation for training stability

Author: Diego Bengochea
"""

from typing import Dict, Optional

# Third-party imports
import kornia.augmentation as K
import torch
from torch import Tensor

# Shared utilities
from shared_utils import get_logger


# Initialize module logger
logger = get_logger(__name__)


class S2Scaling(K.IntensityAugmentationBase2D):
    """
    Scale Sentinel-2 imagery from digital numbers to reflectance values.
    
    Applies the standard Sentinel-2 scaling factor (10000) to convert digital numbers
    to surface reflectance values for proper neural network input normalization.
    This transformation preserves NaN values specified in the input data.
    
    The scaling is essential for proper model training as it normalizes the input
    data to the expected reflectance range of 0-1 (or slightly above for bright
    surfaces).
    
    Args:
        nan_input (float): NaN value in input data to preserve during scaling.
                          Default is 0.0 for Sentinel-2 mosaics.
    """

    def __init__(self, nan_input: float = 0.0) -> None:
        """
        Initialize S2 scaling transformation.
        
        Args:
            nan_input (float): NaN value in input data to preserve during scaling
        """
        super().__init__(p=1)  # Always apply this transformation
        self.nan_input = nan_input
        logger.debug(f"Initialized S2Scaling with nan_input={nan_input}")

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply scaling transformation to input tensor.
        
        Converts Sentinel-2 digital numbers to reflectance values by dividing
        by 10000, while preserving NaN values in the input data.
        
        Args:
            input (Tensor): Input tensor to transform (Sentinel-2 digital numbers)
            params (Dict[str, Tensor]): Transformation parameters (unused)
            flags (Dict[str, int]): Transformation flags (unused)
            transform (Optional[Tensor]): Additional transformation matrix (unused)
            
        Returns:
            Tensor: Scaled tensor with reflectance values (0-1 range typically)
        """
        try:
            scaled = torch.where(input != self.nan_input, input / 10000, input)
            logger.debug(f"Applied S2 scaling to tensor with shape {input.shape}")
            return scaled
        except Exception as e:
            logger.error(f"Failed to apply S2 scaling: {e}")
            raise RuntimeError(f"S2 scaling transformation failed: {e}") from e
       

class PNOAVegetationRemoveAbnormalHeight(K.IntensityAugmentationBase2D):
    """
    Remove vegetation height outliers by setting them to NaN value.
    
    Filters out unrealistic vegetation heights that exceed biological limits
    to improve training stability and model performance. Heights above the
    specified threshold are considered outliers and replaced with the NaN value.
    
    This transformation is crucial for PNOA data as it can contain artifacts
    from buildings, towers, or processing errors that result in unrealistic
    vegetation height values.
    
    Args:
        hmax (float): Maximum realistic vegetation height threshold in meters.
                     Default is 60.0m which covers most natural vegetation.
        nan_target (float): NaN value to assign to outliers. Default is -1.0.
    """

    def __init__(self, hmax: float = 60.0, nan_target: float = -1.0) -> None:
        """
        Initialize height outlier removal transformation.
        
        Args:
            hmax (float): Maximum realistic vegetation height threshold
            nan_target (float): NaN value to assign to outliers
        """
        super().__init__(p=1)  # Always apply this transformation
        self.nan_target = nan_target
        self.flags = {"hmax": torch.tensor(hmax).view(-1, 1, 1)}
        logger.debug(f"Initialized height outlier removal with hmax={hmax}m, nan_target={nan_target}")

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply height outlier removal to input tensor.
        
        Sets all vegetation heights above the maximum threshold to the
        specified NaN value to remove unrealistic outliers.
        
        Args:
            input (Tensor): Input height tensor (vegetation heights in meters)
            params (Dict[str, Tensor]): Transformation parameters (unused)
            flags (Dict[str, int]): Transformation flags including hmax
            transform (Optional[Tensor]): Additional transformation matrix (unused)
            
        Returns:
            Tensor: Height tensor with outliers set to nan_target
        """
        try:
            # Apply outlier removal in-place
            hmax_tensor = flags['hmax'].to(device=input.device)
            outlier_mask = input > hmax_tensor
            input[outlier_mask] = self.nan_target
            
            num_outliers = outlier_mask.sum().item()
            if num_outliers > 0:
                logger.debug(f"Removed {num_outliers} height outliers above {hmax_tensor.item()}m")
            
            return input
        except Exception as e:
            logger.error(f"Failed to remove height outliers: {e}")
            raise RuntimeError(f"Height outlier removal failed: {e}") from e   


class PNOAVegetationInput0InArtificialSurfaces(K.IntensityAugmentationBase2D):
    """
    Handle artificial surfaces in PNOA vegetation data.
    
    Artificial surfaces (roads, buildings, etc.) are marked with specific no-data 
    values in PNOA datasets. This transform converts them to a standard NaN 
    representation for consistent handling throughout the pipeline.
    
    The PNOA vegetation model assigns specific values to artificial surfaces
    which need to be converted to the pipeline's standard NaN representation
    for proper masking during training and inference.
    
    Args:
        nan_value (float): Original NaN value marking artificial surfaces.
                          Default is -32767.0 as used in PNOA datasets.
    """

    def __init__(self, nan_value: float = -32767.0) -> None:
        """
        Initialize artificial surface handling transformation.
        
        Args:
            nan_value (float): Original NaN value marking artificial surfaces
        """
        super().__init__(p=1)  # Always apply this transformation
        self.flags = {"nodata": torch.tensor(nan_value).view(-1, 1, 1)}
        logger.debug(f"Initialized artificial surface handling with nan_value={nan_value}")

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply artificial surface handling to input tensor.
        
        Converts pixels marked as artificial surfaces (with the original NaN value)
        to the standard pipeline NaN value (-1.0) for consistent processing.
        
        Args:
            input (Tensor): Input vegetation height tensor
            params (Dict[str, Tensor]): Transformation parameters (unused)
            flags (Dict[str, int]): Transformation flags including nodata value
            transform (Optional[Tensor]): Additional transformation matrix (unused)
            
        Returns:
            Tensor: Tensor with artificial surfaces set to standard NaN value
        """
        try:
            nodata_tensor = flags['nodata'].to(device=input.device)
            artificial_mask = (input - nodata_tensor) == 0
            input[artificial_mask] = -1.0
            
            num_artificial = artificial_mask.sum().item()
            if num_artificial > 0:
                logger.debug(f"Converted {num_artificial} artificial surface pixels to standard NaN")
            
            return input
        except Exception as e:
            logger.error(f"Failed to handle artificial surfaces: {e}")
            raise RuntimeError(f"Artificial surface handling failed: {e}") from e


class PNOAVegetationLogTransform(K.IntensityAugmentationBase2D):
    """
    Apply log1p transformation to vegetation height data.
    
    Transforms vegetation heights to log space using log1p (log(1+x)) to
    stabilize training and handle the natural right-skewed distribution
    of vegetation heights while preserving zero values.
    
    The log1p transformation is particularly effective for vegetation height
    data because:
    - It handles the right-skewed distribution common in height data
    - It preserves zero values (ground level)
    - It compresses the dynamic range for better gradient flow
    - It's numerically stable for small values
    
    Args:
        nodata (float): NaN value to preserve during transformation.
                       Default is -1.0 for consistency with pipeline.
    """
    
    def __init__(self, nodata: float = -1.0):
        """
        Initialize log transformation.
        
        Args:
            nodata (float): NaN value to preserve during transformation
        """
        super().__init__(p=1.0)  # Always apply
        self.nodata = nodata
        logger.debug(f"Initialized log transform with nodata={nodata}")

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply log1p transformation to input tensor.
        
        Applies log(1+x) transformation to vegetation heights while preserving
        NaN values. This transformation stabilizes training by reducing the
        impact of very large height values.
        
        Args:
            input (Tensor): Input vegetation height tensor
            params (Dict[str, Tensor]): Transformation parameters (unused)
            flags (Dict[str, int]): Transformation flags (unused)
            transform (Optional[Tensor]): Additional transformation matrix (unused)
            
        Returns:
            Tensor: Log-transformed tensor with NaN values preserved
        """
        try:
            # Apply log1p while preserving nodata values
            valid_mask = input != self.nodata
            log_transformed = torch.where(valid_mask, torch.log1p(input), input)
            
            num_valid = valid_mask.sum().item()
            logger.debug(f"Applied log1p transformation to {num_valid} valid height pixels")
            
            return log_transformed
        except Exception as e:
            logger.error(f"Failed to apply log transformation: {e}")
            raise RuntimeError(f"Log transformation failed: {e}") from e


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    'S2Scaling',
    'PNOAVegetationRemoveAbnormalHeight', 
    'PNOAVegetationInput0InArtificialSurfaces',
    'PNOAVegetationLogTransform'
]
