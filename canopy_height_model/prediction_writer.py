"""
Prediction writer callback for canopy height regression inference.

This module provides a PyTorch Lightning callback for writing model predictions
to raster files during inference. Handles geographic filtering, memory management,
and optimized raster output with configurable compression and tiling.

The writer performs geographic intersection testing to only save predictions
within the study area boundaries and implements memory-efficient processing
for large-scale inference.

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

# Local imports
from config import load_config, setup_logging


def print_memory_status() -> None:
    """Print current system memory usage for debugging purposes."""
    vm = psutil.virtual_memory()
    print(f"Memory usage: {vm.percent}%")
    print(f"Available memory: {vm.available / (1024 * 1024 * 1024):.2f} GB")


def intersects(geometries: gpd.GeoSeries, test_geometry: geometry.base.BaseGeometry) -> gpd.GeoSeries:
    """
    Test geometric intersection between GeoSeries and a geometry.
    
    Args:
        geometries (gpd.GeoSeries): Series of geometries to test
        test_geometry (geometry.base.BaseGeometry): Geometry to test intersection against
        
    Returns:
        gpd.GeoSeries: Boolean series indicating intersections
    """
    return geometries.intersects(test_geometry)


class CanopyHeightRasterWriter(BasePredictionWriter):
    """
    PyTorch Lightning callback for writing canopy height predictions to raster files.
    
    This callback handles the conversion of model predictions to georeferenced raster
    files with geographic filtering, memory management, and configurable output settings.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        write_interval: str,
        config_path: str = None
    ):
        """
        Initialize the raster writer with configuration.
        
        Args:
            output_dir (str): Directory to save prediction rasters
            write_interval (str): When to write predictions ('batch' or 'epoch')
            config_path (str, optional): Path to configuration file
        """
        super().__init__(write_interval)
        
        # Load configuration
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging']['level'])
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup geographic mask
        self._setup_spain_mask()
        
        self.logger.info(f"Prediction writer initialized. Output directory: {self.output_dir}")
        
    def _setup_spain_mask(self) -> None:
        """Load Spain shapefile for geographic filtering."""
        try:
            spain_shapefile = self.config['prediction']['spain_shapefile']
            self.spain = gpd.read_file(spain_shapefile)
            self.spain['geometry'] = self.spain['geometry'].to_crs('epsg:25830')
            self.logger.info(f"Spain mask loaded from: {spain_shapefile}")
        except Exception as e:
            self.logger.error(f"Failed to load Spain shapefile: {str(e)}")
            raise

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
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being used
            prediction: Model predictions
            batch_indices: Batch indices (bounding boxes)
            batch: Input batch data
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        self.logger.debug(f"Writing batch {batch_idx}, memory status:")
        print_memory_status()

        try: 
            # Handle both single and batch predictions
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]
            if not isinstance(batch_indices, (list, tuple)):
                batch_indices = [batch_indices]  
        
            with torch.no_grad():
                try:
                    for pred, index in zip(prediction, batch_indices):
                        self._process_single_prediction(
                            pred, 
                            index,
                            dataloader_idx,
                            batch_idx
                        )
                except Exception as e:
                    self.logger.error(f'Error processing prediction: {e}')
                    raise e                        
                finally:
                    # For MPS, explicit deletion is usually enough due to unified memory
                    if 'pred' in locals():
                        del pred
                        gc.collect()
        except Exception as e:
            self.logger.error(f'Error processing prediction: {e}')
            raise e    
        finally:
            if 'prediction' in locals():
                del prediction 
                gc.collect()    

        self.logger.debug(f"Batch {batch_idx} written, memory status:")
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
        
        Args:
            predicted_patch (torch.Tensor): Model prediction tensor
            index: Bounding box index with spatial coordinates
            dataloader_idx (int): Dataloader index
            batch_idx (int): Batch index
        """
        # Check Spain intersection
        tile = geometry.box(
            minx=index.minx, 
            maxx=index.maxx, 
            miny=index.miny, 
            maxy=index.maxy
        )

        if not intersects(self.spain['geometry'], tile).any():
            self.logger.debug(f"Skipping tile outside Spain boundaries: {tile.bounds}")
            return

        # Setup output path
        savepath = self.output_dir / str(dataloader_idx) / f"predicted_minx_{index.minx}_maxy_{index.maxy}_mint_{index.mint}.tif"
        savepath.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU and convert to numpy
        with torch.no_grad():
            if isinstance(predicted_patch, torch.Tensor):
                # For MPS, we just need to move to CPU
                pred_data = predicted_patch[0].cpu().numpy()
            else:
                pred_data = predicted_patch[0][0].cpu().numpy()

        # Calculate transform
        transform = from_bounds(
            index.minx, 
            index.miny, 
            index.maxx, 
            index.maxy, 
            pred_data.shape[0], 
            pred_data.shape[1]
        )

        # Get raster configuration
        raster_config = self.config['prediction']['raster']

        # Write to disk with configuration settings
        try:
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
                new_dataset.write(pred_data, 1)
                new_dataset.update_tags(DATE=index.mint)
                
            self.logger.debug(f"Saved prediction raster: {savepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to write raster {savepath}: {str(e)}")
            raise

        # Clear data
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
        Write predictions at the end of epoch (currently not implemented).
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: Lightning module being used
            predictions: All epoch predictions
            batch_indices: All batch indices
        """
        return None