"""
Dataset classes for Sentinel-2 and PNOA vegetation data integration.

This module provides TorchGeo-compatible dataset classes for loading and
processing Sentinel-2 satellite imagery and PNOA vegetation height models.
Includes specialized handling for Spanish vegetation data and satellite
imagery mosaics with proper metadata and coordinate reference systems.

Author: Diego Bengochea
"""

from typing import Any

# Third-party imports
import torch
from torch import Tensor

# TorchGeo imports
from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.datasets.utils import BoundingBox


# -----------------------------------------------------------------------------
# PNOA Vegetation Dataset
# -----------------------------------------------------------------------------

class PNOAVegetation(RasterDataset):
    """
    Spanish PNOA vegetation Normalized Surface Digital Model dataset.
    
    This dataset provides access to high-resolution vegetation height models
    derived from the Spanish National Aerial Orthophotography Plan (PNOA).
    The data represents normalized surface heights focused on vegetation
    structures across Spain.
    """
    
    is_image = False 
    filename_glob = "PNOA_*"
    filename_regex = r'PNOA_(?P<date>\d{4})'
    date_format = "%Y"
    nan_value = -32767.0

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """
        Retrieve data for a given bounding box query.
        
        Args:
            query (BoundingBox): Spatial bounding box for data retrieval
            
        Returns:
            Dict[str, Any]: Dictionary containing data and metadata
        """
        result = super().__getitem__(query)
        return result

    @property
    def dtype(self) -> torch.dtype:
        """
        Return the data type for vegetation height values.
        
        Override the default dtype to use float32 for regression tasks
        instead of the typical long dtype used for classification.
        
        Returns:
            torch.dtype: Float32 dtype for continuous height values
        """
        return torch.float32


# -----------------------------------------------------------------------------
# Sentinel-2 Mosaic Dataset  
# -----------------------------------------------------------------------------

class S2Mosaic(RasterDataset):
    """
    Sentinel-2 green-season composite mosaics for 2017 to 2024.
    
    This dataset provides access to summer composite mosaics of Sentinel-2
    imagery covering Spain. The mosaics are created from cloud-free scenes
    during the growing season to provide consistent vegetation observations.
    """
    
    is_image = True
    filename_regex = r'S2_summer_mosaic_(?P<date>\d{4})'
    date_format = "%Y"
    all_bands = [
        'red', 'green', 'blue', 'nir', 'swir16', 'swir22', 
        'rededge1', 'rededge2', 'rededge3', 'nir08'
    ]
    separate_files = False
    nan_value = 0


# -----------------------------------------------------------------------------
# Kornia Intersection Dataset
# -----------------------------------------------------------------------------

class KorniaIntersectionDataset(IntersectionDataset):
    """
    Intersection dataset wrapper compatible with Kornia transformations.
    
    This wrapper addresses compatibility issues between TorchGeo's intersection
    dataset and Kornia's augmentation pipeline by filtering data keys passed
    to transforms. Required until Kornia releases version 0.7.4.
    """

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """
        Retrieve intersected data for a given bounding box query.
        
        Args:
            query (BoundingBox): Spatial bounding box for data retrieval
            
        Returns:
            Dict[str, Any]: Dictionary containing intersected data and metadata
            
        Raises:
            IndexError: If query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        samples = [ds[query] for ds in self.datasets]
        sample = self.collate_fn(samples)

        # Only pass valid Kornia data keys to transforms
        # This won't be needed after Kornia's next release (>0.7.4)
        kornia_sample = {'image': sample['image'], 'mask': sample['mask']}

        if self.transforms is not None:
            kornia_sample = self.transforms(kornia_sample)

        # Reconstruct full sample with geo information
        sample = {
            'image': kornia_sample['image'], 
            'mask': kornia_sample['mask'], 
            'crs': sample['crs'], 
            'bbox': sample['bbox']
        }

        return sample
