"""
Land Use Masking Pipeline for Biomass Estimation

This module provides land cover masking functionality for biomass raster datasets
using Corine Land Cover data to exclude annual cropland areas. Refactored from
the original mask_annual_cropland.py script to fit the biomass_model component
architecture with centralized path constants.

Features:
- Annual cropland masking using Corine Land Cover values (12, 13, 14)
- Automatic CRS reprojection and spatial alignment
- Recursive directory processing
- Robust error handling and logging
- Integration with centralized path management

Author: Diego Bengochea
"""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from pathlib import Path
from typing import List, Optional, Union
import tempfile
import shutil

# Shared utilities
from shared_utils import get_logger, ensure_directory
from shared_utils.central_data_paths_constants import *


class LandUseMaskingPipeline:
    """
    Pipeline for masking biomass rasters to exclude annual cropland areas.
    
    Uses Corine Land Cover data to identify and mask annual crop pixels
    (values 12, 13, 14) from biomass estimation results. Handles spatial
    reprojection and alignment automatically.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the land use masking pipeline.
        
        Args:
            temp_dir: Optional custom temporary directory for intermediate files
        """
        # Store configuration and data paths
        self.config = load_config(config_path, component='biomass_estimation')

        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_estimation',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Default annual crop values to mask (from Corine Land Cover)
        self.annual_crop_values = self.config['processing']['annual_crop_values']  # Arable land for yearly crops
        
        self.temp_dir = BIOMASS_MASKING_TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
        # Target file extensions
        self.target_extensions = ['.tif', '.tiff']
        
        self.logger.info(f"LandUseMaskingPipeline initialized")
        self.logger.info(f"Temporary directory: {self.temp_dir}")
        self.logger.info(f"Annual crop values to mask: {self.annual_crop_values}")
    
    def run_full_pipeline(self) -> bool:
        """
        Process a directory of biomass rasters with annual cropland masking.
        """

        input_dir = BIOMASS_MAPS_PER_FOREST_TYPE_RAW_DIR
        output_dir = BIOMASS_MAPS_PER_FOREST_TYPE_MASKED_DIR
        land_cover_file = CORINE_LAND_COVER_FILE

        mask_values = self.annual_crop_values

        # Validate inputs
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not land_cover_file.exists():
            raise FileNotFoundError(f"Land cover file not found: {land_cover_file}")
        # Create output directory
        ensure_directory(output_dir)

        # Find raster files to process
        raster_files = self._find_raster_files(input_dir)

        if not raster_files:
                self.logger.warning("No raster files found to process")
                return True

        self.logger.info(f"Found {len(raster_files)} raster files to process")        

        # Process each raster file
        success_count = 0
        error_count = 0
        
        for raster_file in raster_files:
            try:
                # Calculate relative path to maintain directory structure
                relative_path = raster_file.relative_to(input_dir)
                output_file = output_dir / relative_path
                
                # Process the raster file
                success = self._process_single_raster(
                    raster_file, output_file, land_cover_file, mask_values
                )
                if success:
                    success_count += 1
                    self.logger.debug(f"Successfully processed: {relative_path}")
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing {raster_file}: {str(e)}")

        # Log completion
        self.logger.info(f"Masking completed: {success_count} succeeded, {error_count} failed")
        
        # Cleanup temporary files
        self._cleanup_temp_files()
        
        return error_count == 0

    
    def _find_raster_files(
        self,
        input_dir: Path,
    ) -> List[Path]:
        """
        Find raster files in input directory with optional filtering.
        
        Args:
            input_dir: Directory to search
            
        Returns:
            List of raster file paths
        """
        raster_files = []
        
        # Find all raster files recursively
        for ext in self.target_extensions:
            raster_files.extend(list(input_dir.rglob(f"*{ext}")))
        
        return sorted(raster_files)
    
    def _process_single_raster(
        self,
        input_file: Path,
        output_file: Path,
        land_cover_file: Path,
        mask_values: List[int]
    ) -> bool:
        """
        Process a single raster file with land cover masking.
        
        Args:
            input_file: Input biomass raster file
            output_file: Output masked raster file
            land_cover_file: Corine land cover raster
            mask_values: Values to mask in land cover data
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            self.logger.debug(f"Processing: {input_file.name}")
            
            # Ensure output directory exists
            ensure_directory(output_file.parent)
            
            # Open land cover raster for reuse
            with rasterio.open(land_cover_file) as mask_src:
                mask_crs = mask_src.crs
                
                # Open target biomass raster
                with rasterio.open(input_file) as target_src:
                    # Prepare output metadata
                    target_meta = target_src.meta.copy()
                    nodata_value = target_src.nodata
                    
                    # Set default nodata if not specified
                    if nodata_value is None:
                        nodata_value = -9999
                        target_meta.update(nodata=nodata_value)
                    
                    # Read target biomass data
                    target_data = target_src.read(1)
                    
                    # Get mask data aligned to target grid
                    mask_data = self._get_aligned_mask_data(
                        mask_src, target_src, mask_crs
                    )
                    
                    if mask_data is None:
                        self.logger.error(f"Failed to align mask data for {input_file.name}")
                        return False
                    
                    # Create boolean mask for annual crop values
                    crop_mask = np.isin(mask_data, mask_values)
                    
                    # Apply mask to biomass data
                    masked_data = target_data.copy()
                    masked_data[crop_mask] = nodata_value
                    
                    # Calculate masking statistics
                    total_pixels = target_data.size
                    masked_pixels = np.sum(crop_mask)
                    mask_percentage = (masked_pixels / total_pixels) * 100
                    
                    # Write masked result
                    with rasterio.open(output_file, 'w', **target_meta) as dst:
                        dst.write(masked_data, 1)
                    
                    self.logger.debug(
                        f"Masked {input_file.name}: {masked_pixels:,} pixels "
                        f"({mask_percentage:.1f}%) set to nodata"
                    )
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error processing {input_file.name}: {str(e)}")
            return False
    
    def _get_aligned_mask_data(
        self,
        mask_src: rasterio.DatasetReader,
        target_src: rasterio.DatasetReader,
        mask_crs: rasterio.crs.CRS
    ) -> Optional[np.ndarray]:
        """
        Get mask data aligned to target raster grid.
        
        Handles CRS reprojection and spatial resampling as needed.
        
        Args:
            mask_src: Open mask raster dataset
            target_src: Open target raster dataset
            mask_crs: Mask raster CRS
            
        Returns:
            Aligned mask data array or None if failed
        """
        try:
            target_data_shape = target_src.read(1).shape
            
            # Handle CRS differences
            if mask_crs != target_src.crs:
                self.logger.debug("CRS mismatch detected, reprojecting mask data...")
                
                # Create temporary file for reprojected mask
                temp_mask_file = self.temp_dir / f"reprojected_mask_{os.getpid()}.tif"
                
                try:
                    # Reproject mask to target CRS and grid
                    mask_reprojected = np.zeros(target_data_shape, dtype=rasterio.uint8)
                    
                    reproject(
                        source=rasterio.band(mask_src, 1),
                        destination=mask_reprojected,
                        src_transform=mask_src.transform,
                        src_crs=mask_crs,
                        dst_transform=target_src.transform,
                        dst_crs=target_src.crs,
                        dst_nodata=255,  # Temporary nodata for reprojection
                        resampling=Resampling.nearest
                    )
                    
                    # Clean up reprojection artifacts
                    mask_data = np.where(mask_reprojected == 255, 0, mask_reprojected)
                    
                finally:
                    # Clean up temporary file if created
                    if temp_mask_file.exists():
                        temp_mask_file.unlink()
                
            else:
                # Same CRS - handle spatial alignment
                target_bounds = target_src.bounds
                
                # Get mask window corresponding to target bounds
                window = from_bounds(*target_bounds, mask_src.transform)
                
                # Read windowed mask data
                mask_data = mask_src.read(1, window=window, boundless=True, fill_value=0)
                
                # Ensure exact shape matching through resampling if needed
                if mask_data.shape != target_data_shape:
                    self.logger.debug("Shape mismatch detected, resampling mask data...")
                    
                    mask_resized = np.zeros(target_data_shape, dtype=rasterio.uint8)
                    
                    # Calculate transform for windowed data
                    window_transform = rasterio.windows.transform(window, mask_src.transform)
                    
                    # Resample to exact target grid
                    reproject(
                        source=mask_data,
                        destination=mask_resized,
                        src_transform=window_transform,
                        src_crs=mask_crs,
                        dst_transform=target_src.transform,
                        dst_crs=target_src.crs,
                        dst_nodata=0,
                        resampling=Resampling.nearest
                    )
                    
                    mask_data = mask_resized
            
            return mask_data
            
        except Exception as e:
            self.logger.error(f"Failed to align mask data: {str(e)}")
            return None
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            if self.temp_dir.exists():
                # Remove temporary files but keep directory
                for temp_file in self.temp_dir.glob("*"):
                    if temp_file.is_file():
                        temp_file.unlink()
                        self.logger.debug(f"Cleaned up temporary file: {temp_file.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary files: {str(e)}")
    