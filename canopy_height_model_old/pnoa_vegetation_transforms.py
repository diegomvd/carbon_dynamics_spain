"""
Data transformation pipeline for PNOA vegetation and Sentinel-2 preprocessing.

This module provides Kornia-based transformations for preprocessing vegetation
height data and satellite imagery in the canopy height regression pipeline.
Includes scaling, outlier removal, log transformations, and data cleaning
operations optimized for vegetation analysis.

Author: Diego Bengochea
"""

from typing import Dict, Optional

# Third-party imports
import kornia.augmentation as K
import torch
from torch import Tensor


class S2Scaling(K.IntensityAugmentationBase2D):
    """
    Scale Sentinel-2 imagery from digital numbers to reflectance values.
    
    Applies standard Sentinel-2 scaling factor (10000) to convert digital numbers
    to surface reflectance values for proper neural network input normalization.
    """

    def __init__(self, nan_input: float = 0.0) -> None:
        """
        Initialize S2 scaling transformation.
        
        Args:
            nan_input (float): NaN value in input data to preserve during scaling
        """
        super().__init__(p=1)
        self.nan_input = nan_input

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply scaling transformation to input tensor.
        
        Args:
            input (Tensor): Input tensor to transform
            params (Dict[str, Tensor]): Transformation parameters
            flags (Dict[str, int]): Transformation flags
            transform (Optional[Tensor]): Additional transformation matrix
            
        Returns:
            Tensor: Scaled tensor with reflectance values
        """
        return torch.where(input != self.nan_input, input/10000, input)
       

class PNOAVegetationRemoveAbnormalHeight(K.IntensityAugmentationBase2D):
    """
    Remove vegetation height outliers by setting them to NaN value.
    
    Filters out unrealistic vegetation heights that exceed biological limits
    to improve training stability and model performance.
    """

    def __init__(self, hmax: float = 60.0, nan_target: float = -1.0) -> None:
        """
        Initialize height outlier removal transformation.
        
        Args:
            hmax (float): Maximum realistic vegetation height threshold
            nan_target (float): NaN value to assign to outliers
        """
        super().__init__(p=1)
        self.nan_target = nan_target
        self.flags = {"hmax": torch.tensor(hmax).view(-1, 1, 1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply height outlier removal to input tensor.
        
        Args:
            input (Tensor): Input height tensor
            params (Dict[str, Tensor]): Transformation parameters  
            flags (Dict[str, int]): Transformation flags including hmax
            transform (Optional[Tensor]): Additional transformation matrix
            
        Returns:
            Tensor: Height tensor with outliers set to nan_target
        """
        input[(input > flags['hmax'].to(torch.device(input.device)))] = self.nan_target
        return input   


class PNOAVegetationInput0InArtificialSurfaces(K.IntensityAugmentationBase2D):
    """
    Handle artificial surfaces in PNOA vegetation data.
    
    Artificial surfaces are marked with specific no-data values. This transform
    converts them to a standard NaN representation for consistent handling
    throughout the pipeline.
    """

    def __init__(self, nan_value: float = -32767.0) -> None:
        """
        Initialize artificial surface handling transformation.
        
        Args:
            nan_value (float): Original NaN value marking artificial surfaces
        """
        super().__init__(p=1)
        self.flags = {"nodata": torch.tensor(nan_value).view(-1, 1, 1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply artificial surface handling to input tensor.
        
        Args:
            input (Tensor): Input vegetation height tensor
            params (Dict[str, Tensor]): Transformation parameters
            flags (Dict[str, int]): Transformation flags including nodata value
            transform (Optional[Tensor]): Additional transformation matrix
            
        Returns:
            Tensor: Tensor with artificial surfaces set to standard NaN value
        """
        # Set artificial surfaces to standard nodata value
        input[(input - flags['nodata'].to(torch.device(input.device))) == 0] = -1.0
        return input


class PNOAVegetationLogTransform(K.IntensityAugmentationBase2D):
    """
    Apply log1p transformation to vegetation height data.
    
    Transforms vegetation heights to log space using log1p (log(1+x)) to
    stabilize training and handle the natural right-skewed distribution
    of vegetation heights while preserving zero values.
    """
    
    def __init__(self, nodata: float = -1.0):
        """
        Initialize log transformation.
        
        Args:
            nodata (float): NaN value to preserve during transformation
        """
        super().__init__(p=1.0)  # Always apply
        self.nodata = nodata

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply log1p transformation to input tensor.
        
        Args:
            input (Tensor): Input vegetation height tensor
            params (Dict[str, Tensor]): Transformation parameters
            flags (Dict[str, int]): Transformation flags
            transform (Optional[Tensor]): Additional transformation matrix
            
        Returns:
            Tensor: Log-transformed tensor with NaN values preserved
        """
        return torch.where(input != self.nodata, torch.log1p(input), input)
