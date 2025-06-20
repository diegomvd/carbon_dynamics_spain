"""
Prediction writer callback for canopy height regression inference.

This module provides a PyTorch Lightning callback for writing model predictions
to raster files during inference. Handles geographic filtering, memory management,
and optimized raster output with configurable compression and tiling.

The writer performs geographic intersection testing to only save predictions
within the study area boundaries and implements memory-efficient processing
for large-scale inference across Spain.

Key features:
- Geographic filtering using Spain boundary shapefile
- Memory-efficient batch processing with garbage collection
- Configurable raster output parameters (compression, tiling, etc.)
- Proper georeferencing and coordinate transformations
- Robust error handling and logging

Author: Diego Bengochea
"""

import os
import gc
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import torch
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely import geometry

# PyTorch Lightning imports
from lightning.pytorch.callbacks import BasePredictionWriter

# Shared utilities
from shared_utils import get_logger, load_config


# Initialize module logger
logger = get_logger(__name__)


def print_memory_status() -> None:
    """
    Print current system memory usage for debugging purposes.
    
    This utility function helps monitor memory consumption during large-scale
    prediction processing to identify potential memory leaks or excessive usage.
    """
    try:
        vm = psutil.virtual_memory()
        logger.debug(f"Memory usage: {vm.percent}% | Available: {vm.available / (1024 * 1024 * 1024):.2f} GB")
    except Exception as e:
        logger.warning(f"Could not retrieve memory status: {e}")


def intersects(geometries: gpd.GeoSeries, test_geometry: geometry.base.BaseGeometry) -> gpd.GeoSeries:
    """
    Test geometric intersection between GeoSeries and a geometry.
    
    This function performs spatial intersection testing to determine which
    geometries in a GeoSeries intersect with a test geometry. Used for
    filtering predictions to only include areas within Spain boundaries.
    
    Args:
        geometries (gpd.GeoSeries): Series of geometries to test (Spain boundaries)
        test_geometry (geometry.base.BaseGeometry): Geometry to test intersection against (prediction tile)
        
    Returns:
        gpd.GeoSeries: Boolean series indicating which geometries intersect
        
    Raises:
        ValueError: If geometries have invalid or missing CRS
        RuntimeError: If intersection test fails
    """
    try:
        if geometries.crs is None:
            raise ValueError("Geometries must have a defined coordinate reference system")
        
        result = geometries.intersects(test_geometry)
        logger.debug(f"Intersection test completed for geometry with bounds: {test_geometry.bounds}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to test geometric intersection: {e}")
        raise RuntimeError(f"Intersection test failed: {e}") from e


class CanopyHeightRasterWriter(BasePredictionWriter):
    """
    PyTorch Lightning callback for writing canopy height predictions to raster files.
    
    This callback handles the conversion of model predictions to georeferenced raster
    files with geographic filtering, memory management, and configurable output settings.
    The writer ensures that only predictions within Spain boundaries are saved and
    manages memory efficiently during large-scale inference operations.
    
    The callback operates on a per-batch basis, writing each prediction tile immediately
    to minimize memory usage. Each output raster is properly georeferenced with
    configurable compression and tiling options for optimal storage and access.
    
    Attributes:
        output_dir (Path): Directory where prediction rasters will be saved
        config (Dict): Configuration dictionary with raster and processing parameters
        spain (gpd.GeoDataFrame): Spain boundary geometries for spatial filtering
    """
    
    def __init__(
        self, 
        output_dir: str, 
        write_interval: str,
        config_path: str = None
    ):
        """
        Initialize the raster writer with configuration and output settings.
        
        Args:
            output_dir (str): Directory to save prediction rasters
            write_interval (str): When to write predictions ('batch' or 'epoch')
            config_path (str, optional): Path to configuration file
            
        Raises:
            FileNotFoundError: If Spain shapefile is not found
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails
        """
        super().__init__(write_interval)
        
        try:
            # Load configuration
            self.config = load_config(config_path)
            
            # Setup output directory
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup geographic mask
            self._setup_spain_mask()
            
            logger.info(f"Prediction writer initialized with output directory: {self.output_dir}")
            logger.info(f"Write interval: {write_interval}")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction writer: {e}")
            raise RuntimeError(f"Prediction writer initialization failed: {e}") from e
        
    def _setup_spain_mask(self) -> None:
        """
        Load Spain shapefile for geographic filtering.
        
        This method loads the Spain boundary shapefile and reprojects it to the
        appropriate coordinate reference system (EPSG:25830 - ETRS89 UTM Zone 30N)
        for spatial filtering of prediction tiles.
        
        Raises:
            FileNotFoundError: If Spain shapefile is not found
            ValueError: If shapefile cannot be processed
            RuntimeError: If CRS transformation fails
        """
        try:
            # Get shapefile path from configuration
            if 'prediction' not in self.config or 'spain_shapefile' not in self.config['prediction']:
                raise ValueError("Spain shapefile path not found in configuration")
            
            spain_shapefile = self.config['prediction']['spain_shapefile']
            
            # Check if shapefile exists
            if not Path(spain_shapefile).exists():
                raise FileNotFoundError(f"Spain shapefile not found: {spain_shapefile}")
            
            # Load and reproject Spain boundaries
            self.spain = gpd.read_file(spain_shapefile)
            
            # Ensure proper CRS (ETRS89 UTM Zone 30N for Spain)
            if self.spain.crs is None:
                logger.warning("Spain shapefile has no CRS, assuming EPSG:25830")
                self.spain.set_crs('epsg:25830', inplace=True)
            elif self.spain.crs.to_string() != 'EPSG:25830':
                logger.info(f"Reprojecting Spain boundaries from {self.spain.crs} to EPSG:25830")
                self.spain = self.spain.to_crs('epsg:25830')
            
            # Validate geometry column
            if 'geometry' not in self.spain.columns:
                raise ValueError("Spain shapefile must contain a 'geometry' column")
            
            num_geometries = len(self.spain)
            bounds = self.spain.total_bounds
            
            logger.info(f"Spain mask loaded successfully: {num_geometries} geometries")
            logger.info(f"Spain bounds: {bounds}")
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to setup Spain mask: {e}")
            raise RuntimeError(f"Spain mask setup failed: {e}") from e

    def write_on_batch_end(
        self, 
        trainer, 
        pl_module, 
        prediction, 
        batch_indices, 
        batch, 
        batch_idx, 
        dataloader_idx
    ) -> None:
        """
        Write predictions to raster files at the end of each batch.
        
        This method processes each prediction in the batch, performs geographic
        filtering, and writes valid predictions to georeferenced raster files.
        Memory management is handled through explicit garbage collection.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being used for prediction
            prediction: Model predictions (single tensor or list of tensors)
            batch_indices: Batch indices containing bounding box information
            batch: Input batch data (unused but required by interface)
            batch_idx: Current batch index for logging
            dataloader_idx: Dataloader index for organizing outputs
            
        Raises:
            RuntimeError: If prediction processing fails
        """
        logger.debug(f"Processing batch {batch_idx} predictions")
        print_memory_status()

        try: 
            # Normalize inputs to lists for consistent processing
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]
            if not isinstance(batch_indices, (list, tuple)):
                batch_indices = [batch_indices]  
        
            # Process each prediction with memory management
            with torch.no_grad():
                try:
                    for pred_idx, (pred, index) in enumerate(zip(prediction, batch_indices)):
                        logger.debug(f"Processing prediction {pred_idx + 1}/{len(prediction)} in batch {batch_idx}")
                        self._process_single_prediction(
                            pred, 
                            index,
                            dataloader_idx,
                            batch_idx
                        )
                        
                except Exception as e:
                    logger.error(f'Error processing prediction in batch {batch_idx}: {e}')
                    raise RuntimeError(f"Prediction processing failed: {e}") from e
                finally:
                    # Explicit memory cleanup for GPU/MPS memory management
                    if 'pred' in locals():
                        del pred
                    gc.collect()
                    
        except Exception as e:
            logger.error(f'Critical error in batch {batch_idx} processing: {e}')
            raise
        finally:
            # Final cleanup
            if 'prediction' in locals():
                del prediction 
            gc.collect()    

        logger.debug(f"Batch {batch_idx} processing completed")
        print_memory_status()        

    def _process_single_prediction(
        self, 
        predicted_patch: torch.Tensor, 
        index: Any, 
        dataloader_idx: int, 
        batch_idx: int
    ) -> None:
        """
        Process and save a single prediction patch to raster file.
        
        This method performs the core functionality of converting a model prediction
        tensor to a georeferenced raster file. It includes geographic filtering,
        coordinate transformation, and optimized raster writing.
        
        Args:
            predicted_patch (torch.Tensor): Model prediction tensor in log1p space
            index: Bounding box index with spatial coordinates (minx, maxx, miny, maxy, mint)
            dataloader_idx (int): Dataloader index for output organization
            batch_idx (int): Batch index for logging and debugging
            
        Raises:
            ValueError: If bounding box or prediction dimensions are invalid
            RuntimeError: If raster writing fails
        """
        try:
            # Create tile geometry for intersection testing
            tile = geometry.box(
                minx=index.minx, 
                maxx=index.maxx, 
                miny=index.miny, 
                maxy=index.maxy
            )

            # Check if tile intersects with Spain boundaries
            if not intersects(self.spain['geometry'], tile).any():
                logger.debug(f"Skipping tile outside Spain boundaries: {tile.bounds}")
                return

            # Setup output path with spatial and temporal identifiers
            filename = f"predicted_minx_{index.minx}_maxy_{index.maxy}_mint_{index.mint}.tif"
            savepath = self.output_dir / str(dataloader_idx) / filename
            savepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert prediction tensor to numpy array
            with torch.no_grad():
                if isinstance(predicted_patch, torch.Tensor):
                    # Handle tensor with batch dimension
                    if predicted_patch.dim() >= 3:
                        pred_data = predicted_patch[0].cpu().numpy()
                    else:
                        pred_data = predicted_patch.cpu().numpy()
                else:
                    # Handle nested structure (backward compatibility)
                    pred_data = predicted_patch[0][0].cpu().numpy()

            # Validate prediction data dimensions
            if pred_data.ndim != 2:
                raise ValueError(f"Prediction data must be 2D, got shape: {pred_data.shape}")

            # Calculate geospatial transform from bounding box
            transform = from_bounds(
                index.minx, 
                index.miny, 
                index.maxx, 
                index.maxy, 
                pred_data.shape[1],  # width (columns)
                pred_data.shape[0]   # height (rows)
            )

            # Get raster configuration from config
            raster_config = self.config.get('prediction', {}).get('raster', {})
            
            # Set default values if not in config
            default_config = {
                'dtype': 'float32',
                'crs': 'EPSG:25830',
                'nodata': -1.0,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512
            }
            
            for key, default_value in default_config.items():
                if key not in raster_config:
                    raster_config[key] = default_value

            # Write georeferenced raster with optimized settings
            with rasterio.open(
                savepath,
                mode="w",
                driver="GTiff",
                height=pred_data.shape[0],
                width=pred_data.shape[1],
                count=1,
                dtype=raster_config['dtype'],
                crs=raster_config['crs'],
                transform=transform,
                nodata=raster_config['nodata'],
                compress=raster_config['compress'],
                tiled=raster_config['tiled'],
                blockxsize=raster_config['blockxsize'],
                blockysize=raster_config['blockysize']
            ) as new_dataset:
                # Write prediction data to raster
                new_dataset.write(pred_data, 1)
                
                # Add temporal metadata
                new_dataset.update_tags(DATE=str(index.mint))
                
            logger.debug(f"Successfully saved prediction raster: {savepath}")
            
        except ValueError as e:
            logger.error(f"Invalid data in batch {batch_idx}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to write raster for batch {batch_idx}: {e}")
            raise RuntimeError(f"Raster writing failed: {e}") from e
        finally:
            # Memory cleanup
            if 'pred_data' in locals():
                del pred_data
            gc.collect()
        
    def write_on_epoch_end(
        self, 
        trainer, 
        pl_module, 
        predictions, 
        batch_indices
    ) -> None:
        """
        Write predictions at the end of epoch.
        
        Currently not implemented as batch-wise writing is more memory efficient
        for large-scale predictions. This method is provided for interface
        compatibility but performs no operations.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being used
            predictions: All epoch predictions (unused)
            batch_indices: All batch indices (unused)
        """
        logger.debug("Epoch-end prediction writing not implemented (using batch-wise writing)")
        return None


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    'CanopyHeightRasterWriter',
    'print_memory_status',
    'intersects'
]
