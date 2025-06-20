#!/usr/bin/env python3
"""
Annual cropland masking script for biomass raster datasets.

This script masks biomass raster files using Corine Land Cover data to exclude
annual cropland areas (values 12, 13, 14 representing arable land for yearly crops).
Processes raster files recursively and handles spatial reprojection when needed.

The script applies masks by setting target raster values to nodata where the
mask raster contains specified annual crop values, effectively removing
biomass estimates from agricultural areas.

Author: Diego Bengochea
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from pathlib import Path

from config import get_config


def mask_raster_files(mask_raster_path, target_dir, values_to_mask=None, output_dir=None):
    """
    Mask a series of raster files based on values in a Corine Land Cover raster.
    
    Processes raster files recursively and masks pixels where the land cover
    raster contains specified annual crop values. Handles spatial reprojection
    and resampling when coordinate systems don't match.
    
    Args:
        mask_raster_path (str): Path to Corine Land Cover raster for masking
        target_dir (str): Directory containing biomass rasters to be masked
        values_to_mask (list, optional): Land cover values to mask. 
                                       Defaults to annual crop values from config
        output_dir (str, optional): Output directory. Defaults to target_dir
        
    Raises:
        FileNotFoundError: If mask raster or target directory doesn't exist
        ValueError: If required raster data cannot be processed
    """
    config = get_config()
    
    # Use default annual crop values if not specified
    if values_to_mask is None:
        values_to_mask = config['processing']['annual_crop_values']
    
    # Validate input paths
    if not os.path.exists(mask_raster_path):
        raise FileNotFoundError(f"Mask raster file not found: {mask_raster_path}")
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = target_dir
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Open the mask raster once for reuse
    with rasterio.open(mask_raster_path) as mask_src:
        mask_crs = mask_src.crs
        
        def process_raster_file(raster_file, relative_path=""):
            """
            Process a single raster file with land cover masking.
            
            Args:
                raster_file (str): Path to target raster file
                relative_path (str): Relative directory path for output structure
            """
            print(f"Processing: {os.path.relpath(raster_file, target_dir)}")
            
            # Create corresponding output directory structure
            current_output_dir = output_dir
            if relative_path:
                current_output_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(current_output_dir):
                    os.makedirs(current_output_dir)
                    print(f"Created subdirectory: {current_output_dir}")
            
            # Generate output filename
            output_file = os.path.join(
                current_output_dir,
                os.path.basename(raster_file)
            )
            
            # Skip processing if output already exists
            if os.path.exists(output_file):
                print(f"  Output file already exists: {os.path.basename(output_file)}")
                return
            
            try:
                with rasterio.open(raster_file) as target_src:
                    # Prepare target metadata and handle nodata values
                    target_meta = target_src.meta.copy()
                    nodata_value = target_src.nodata
                    
                    # Set default nodata if not specified
                    if nodata_value is None:
                        nodata_value = -9999
                        target_meta.update(nodata=nodata_value)
                    
                    # Read target biomass data
                    target_data = target_src.read(1)
                    
                    # Handle coordinate system differences
                    if mask_crs != target_src.crs:
                        print(f"  CRS mismatch detected. Reprojecting mask data...")
                        
                        # Reproject mask directly to target's grid
                        mask_reprojected = np.zeros(target_data.shape, dtype=rasterio.uint8)
                        
                        # Perform reprojection with nearest neighbor resampling
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
                        
                    else:
                        # Handle spatial alignment when projections match
                        target_bounds = target_src.bounds
                        
                        # Get mask window corresponding to target bounds
                        window = from_bounds(*target_bounds, mask_src.transform)
                        
                        # Read windowed mask data
                        mask_data = mask_src.read(1, window=window, boundless=True, fill_value=0)
                        
                        # Ensure exact shape matching through resampling if needed
                        if mask_data.shape != target_data.shape:
                            mask_resized = np.zeros(target_data.shape, dtype=rasterio.uint8)
                            
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
                    
                    # Create boolean mask for annual crop values
                    combined_mask = np.isin(mask_data, values_to_mask)
                    
                    # Apply mask to biomass data
                    masked_data = target_data.copy()
                    masked_data[combined_mask] = nodata_value
                    
                    # Write masked result
                    with rasterio.open(output_file, 'w', **target_meta) as dst:
                        dst.write(masked_data, 1)
                    
                    print(f"  Masked raster saved to: {os.path.basename(output_file)}")
            
            except Exception as e:
                print(f"  Error processing {os.path.basename(raster_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        def process_directory(current_dir, relative_path=""):
            """
            Recursively process all raster files in a directory.
            
            Args:
                current_dir (str): Current directory to process
                relative_path (str): Relative path for maintaining directory structure
            """
            target_extensions = config['masking']['target_extensions']
            
            # Find valid raster files in current directory
            raster_files = [
                os.path.join(current_dir, f) 
                for f in os.listdir(current_dir) 
                if os.path.isfile(os.path.join(current_dir, f)) and 
                   any(f.endswith(ext) for ext in target_extensions)
            ]
            
            # Process each raster file
            for raster_file in raster_files:
                process_raster_file(raster_file, relative_path)
            
            # Process subdirectories recursively if configured
            if config['masking']['recursive_processing']:
                for item in os.listdir(current_dir):
                    item_path = os.path.join(current_dir, item)
                    if os.path.isdir(item_path):
                        new_relative_path = os.path.join(relative_path, item) if relative_path else item
                        process_directory(item_path, new_relative_path)
        
        # Start recursive processing from target directory
        process_directory(target_dir)


def main():
    """
    Main function using centralized configuration for annual cropland masking.
    
    Loads paths and parameters from configuration and executes the masking
    workflow on biomass raster datasets.
    """
    try:
        config = get_config()
        
        # Get paths from centralized configuration
        mask_raster_path = config['data']['corine_land_cover']
        target_dir = config['masking']['target_input_dir']
        values_to_mask = config['processing']['annual_crop_values']
        
        # Create output directory path
        output_dir = os.path.join(
            os.path.dirname(target_dir), 
            config['masking']['masked_output_dir']
        )
        
        print(f"Mask raster: {mask_raster_path}")
        print(f"Target directory: {target_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Values to mask: {values_to_mask}")
        
        # Execute masking workflow
        mask_raster_files(mask_raster_path, target_dir, values_to_mask, output_dir)
        print("Annual cropland masking completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()