"""
Dataset classes for Sentinel-2 and PNOA vegetation data integration.

This module provides TorchGeo-compatible dataset classes for loading and
processing Sentinel-2 satellite imagery and PNOA vegetation height models.
Includes specialized handling for Spanish vegetation data and satellite
imagery mosaics with proper metadata and coordinate reference systems.

The module implements three main dataset classes:
- PNOAVegetation: Spanish PNOA vegetation height models
- S2Mosaic: Sentinel-2 summer composite mosaics
- KorniaIntersectionDataset: Kornia-compatible intersection dataset wrapper

Author: Diego Bengochea
"""

from typing import Any, Dict

# Third-party imports
import torch
from torch import Tensor

# TorchGeo imports
from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.datasets.utils import BoundingBox

# Shared utilities
from shared_utils import get_logger


# Initialize module logger
logger = get_logger(__name__)


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
    
    The dataset expects files with the naming pattern 'PNOA_YYYY' where YYYY
    represents the year of data acquisition. Height values are provided in
    meters with a specific NaN value for invalid pixels.
    
    Attributes:
        is_image (bool): False, as this is height data not imagery
        filename_glob (str): Glob pattern for PNOA files
        filename_regex (str): Regex pattern to extract date from filenames
        date_format (str): Date format string for parsing
        nan_value (float): NaN value used in the dataset (-32767.0)
    """
    
    is_image = False 
    filename_glob = "PNOA_*"
    filename_regex = r'PNOA_(?P<date>\d{4})'
    date_format = "%Y"
    nan_value = -32767.0

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """
        Retrieve vegetation height data for a given bounding box query.
        
        Args:
            query (BoundingBox): Spatial bounding box for data retrieval
            
        Returns:
            Dict[str, Any]: Dictionary containing vegetation height data and metadata
            
        Raises:
            IndexError: If query is outside dataset bounds
            RuntimeError: If data retrieval fails
        """
        try:
            result = super().__getitem__(query)
            logger.debug(f"Retrieved PNOA vegetation data for query: {query}")
            return result
        except IndexError as e:
            logger.error(f"Query {query} is outside PNOA dataset bounds: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve PNOA vegetation data for query {query}: {e}")
            raise RuntimeError(f"Data retrieval failed: {e}") from e

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
    
    The dataset includes all 10 Sentinel-2 bands commonly used for vegetation
    analysis, including visible, near-infrared, red-edge, and short-wave
    infrared bands. Data is provided as digital numbers that need scaling
    to reflectance values.
    
    Attributes:
        is_image (bool): True, as this is satellite imagery
        filename_regex (str): Regex pattern to extract date from filenames
        date_format (str): Date format string for parsing
        all_bands (List[str]): List of all available Sentinel-2 bands
        separate_files (bool): False, all bands in single file
        nan_value (int): NaN value used in the dataset (0)
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

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """
        Retrieve Sentinel-2 imagery data for a given bounding box query.
        
        Args:
            query (BoundingBox): Spatial bounding box for data retrieval
            
        Returns:
            Dict[str, Any]: Dictionary containing Sentinel-2 imagery and metadata
            
        Raises:
            IndexError: If query is outside dataset bounds
            RuntimeError: If data retrieval fails
        """
        try:
            result = super().__getitem__(query)
            logger.debug(f"Retrieved S2 mosaic data for query: {query}")
            return result
        except IndexError as e:
            logger.error(f"Query {query} is outside S2 dataset bounds: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve S2 mosaic data for query {query}: {e}")
            raise RuntimeError(f"Data retrieval failed: {e}") from e


# -----------------------------------------------------------------------------
# Kornia Intersection Dataset
# -----------------------------------------------------------------------------

class KorniaIntersectionDataset(IntersectionDataset):
    """
    Intersection dataset wrapper compatible with Kornia transformations.
    
    This wrapper addresses compatibility issues between TorchGeo's intersection
    dataset and Kornia's augmentation pipeline by filtering data keys passed
    to transforms. This filtering ensures only image and mask data are sent
    to Kornia transforms while preserving geographic metadata.
    
    The wrapper is needed until Kornia releases version 0.7.4 which will
    properly handle additional metadata keys in transformation pipelines.
    
    Note:
        This class filters data to include only 'image' and 'mask' keys when
        applying Kornia transforms, then reconstructs the full sample with
        geographic information (CRS, bounding box) afterward.
    """

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """
        Retrieve intersected data for a given bounding box query.
        
        This method performs intersection of multiple datasets and filters
        data appropriately for Kornia compatibility while preserving all
        geographic metadata.
        
        Args:
            query (BoundingBox): Spatial bounding box for data retrieval
            
        Returns:
            Dict[str, Any]: Dictionary containing intersected data and metadata
            with keys: 'image', 'mask', 'crs', 'bbox'
            
        Raises:
            IndexError: If query is not within bounds of the index
            RuntimeError: If intersection or transformation fails
        """
        if not query.intersects(self.bounds):
            logger.error(f"Query {query} does not intersect dataset bounds {self.bounds}")
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        try:
            # All datasets are guaranteed to have a valid query
            samples = [ds[query] for ds in self.datasets]
            sample = self.collate_fn(samples)
            
            logger.debug(f"Successfully intersected {len(self.datasets)} datasets for query: {query}")

            # Only pass valid Kornia data keys to transforms
            # This won't be needed after Kornia's next release (>0.7.4)
            kornia_sample = {'image': sample['image'], 'mask': sample['mask']}

            if self.transforms is not None:
                try:
                    kornia_sample = self.transforms(kornia_sample)
                    logger.debug("Applied Kornia transformations successfully")
                except Exception as e:
                    logger.error(f"Failed to apply Kornia transformations: {e}")
                    raise RuntimeError(f"Transformation failed: {e}") from e

            # Reconstruct full sample with geo information
            sample = {
                'image': kornia_sample['image'], 
                'mask': kornia_sample['mask'], 
                'crs': sample['crs'], 
                'bbox': sample['bbox']
            }

            return sample
            
        except IndexError:
            # Re-raise IndexError as is
            raise
        except Exception as e:
            logger.error(f"Failed to process intersection dataset query {query}: {e}")
            raise RuntimeError(f"Intersection dataset processing failed: {e}") from e


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    'PNOAVegetation',
    'S2Mosaic', 
    'KorniaIntersectionDataset'
]
