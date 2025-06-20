"""
File I/O utilities for biomass estimation pipeline.

This module provides functions for reading height and mask data, writing biomass results,
path building, and file management operations. Handles xarray datasets with dask chunking
for efficient processing of large raster datasets.

Author: Diego Bengochea
"""

import rasterio
import numpy as np
import re
from pathlib import Path
import glob
import os
from rasterio.windows import Window
import dask.array as da
import xarray as xr
import rioxarray as rxr

from logging_utils import logger, timer
from config import get_config, get_output_directory


def build_savepath(fname, output_type, kind, code):
    """
    Build output file path based on input filename, output type and statistical measure.
    
    Args:
        fname (Path): Input file path
        output_type (str): Type of output ('agbd', 'bgbd', 'total')
        kind (str): Statistical measure ('mean', 'uncertainty')
        code (str): Forest type code
        
    Returns:
        str: Full path to output file
        
    Raises:
        ValueError: If output_type is not recognized
    """
    config = get_config()
    stem = fname.stem

    # Extract specifications from canopy height filename
    try:
        specs = re.findall(r'canopy_height_(.*)', stem)[0]
    except IndexError:
        logger.error(f"Could not extract specifications from filename: {stem}")
        raise ValueError(f"Invalid filename format: {stem}")

    # Get base output directory for this output type
    output_dir = get_output_directory(output_type)
    
    # Build filename components
    prefix = config['output']['output_prefix']
    filename = f"{output_type.upper()}_{prefix}_{kind}_{specs}_code{code}.tif"
    
    return os.path.join(output_dir, filename)


def extract_tile_info(filepath):
    """
    Extract year and pattern information from a canopy height filename.
    
    Args:
        filepath (str): Path to canopy height file
        
    Returns:
        dict: Dictionary with year and base pattern, or None if pattern doesn't match
        
    Example:
        >>> extract_tile_info("canopy_height_2020_100m_tile1.tif")
        {'year': '2020', 'base_pattern': 'canopy_height_2020_100m'}
    """
    stem = Path(filepath).stem

    # Match pattern for canopy height files
    match_year = re.search(r'canopy_height_(\d{4})_100m', stem)
    if match_year:
        year = match_year.groups()[0]
        return {
            'year': year,
            'base_pattern': f"canopy_height_{year}_100m"
        }

    logger.error(f"Failed to extract tile info from: {stem}")
    return None


def check_existing_outputs(fname, code):
    """
    Check if output files for a specific tile and forest type already exist.
    
    Args:
        fname (Path): Input file path
        code (str): Forest type code
        
    Returns:
        bool: True if all outputs exist, False otherwise
    """
    config = get_config()
    
    # Check all combinations of output types and measures
    for output_type in config['output']['types']:
        for measure in config['output']['measures']:
            try:
                savepath = build_savepath(fname, output_type, measure, code)
                if not Path(savepath).exists():
                    return False
            except (ValueError, OSError) as e:
                logger.warning(f"Error checking output path for {output_type} {measure}: {e}")
                return False

    return True


def find_masks_for_tile(tile_info, masks_dir):
    """
    Find all mask files for a given tile.
    
    Args:
        tile_info (dict): Dictionary with tile information from extract_tile_info
        masks_dir (str): Directory containing mask files
        
    Returns:
        list: List of (mask_path, forest_type_code) tuples
        
    Raises:
        OSError: If masks directory doesn't exist or isn't accessible
    """
    if not os.path.exists(masks_dir):
        raise OSError(f"Masks directory not found: {masks_dir}")
        
    # Extract the year for pattern matching
    year = tile_info['year']

    # Construct pattern for masks
    mask_pattern = f"canopy_height_{year}_100m_*.tif"
    
    try:
        mask_paths = glob.glob(os.path.join(masks_dir, mask_pattern))
    except OSError as e:
        logger.error(f"Error searching for masks with pattern {mask_pattern}: {e}")
        return []

    results = []
    for mask_path in mask_paths:
        # Extract forest type code from filename
        match_code = re.search(r'_code(\d+)\.tif$', mask_path)
        if match_code:
            forest_type_code = match_code.group(1)
            results.append((mask_path, forest_type_code))
        else:
            logger.warning(f"Could not extract forest type code from: {mask_path}")
    
    return results


def read_height_and_mask_xarray(height_file, mask_file, chunk_size=750):
    """
    Read height and mask rasters into xarray with dask chunking.
    
    Args:
        height_file (str): Path to height raster
        mask_file (str): Path to forest type mask
        chunk_size (int): Size for dask chunks
        
    Returns:
        tuple: (height_xr, mask_xr, out_meta) where:
            - height_xr is an xarray DataArray of height values
            - mask_xr is an xarray DataArray of boolean mask values
            - out_meta is a dictionary with raster metadata for output
            
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If CRS mismatch between height and mask
        rasterio.errors.RasterioIOError: If files can't be read
    """
    # Validate input files exist
    if not os.path.exists(height_file):
        raise FileNotFoundError(f"Height file not found: {height_file}")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    try:
        # Read rasters with rioxarray (preserves geospatial metadata)
        height_xr = rxr.open_rasterio(height_file, chunks={'x': chunk_size, 'y': chunk_size})
        mask_xr = rxr.open_rasterio(mask_file, chunks={'x': chunk_size, 'y': chunk_size})
    except Exception as e:
        logger.error(f"Error reading raster files: {e}")
        raise
    
    # Verify CRS match
    if height_xr.rio.crs != mask_xr.rio.crs:
        raise ValueError(f"CRS mismatch between height ({height_xr.rio.crs}) and mask ({mask_xr.rio.crs})")
    
    # Convert to appropriate data types and squeeze to remove band dimension
    height_xr = height_xr.astype(np.float32).squeeze()
    mask_xr = (mask_xr > 0).astype(bool).squeeze()
   
    # Prepare metadata for output files
    out_meta = {
       'driver': 'GTiff',
       'height': height_xr.rio.height,
       'width': height_xr.rio.width,
       'count': 1,
       'dtype': 'float32',
       'crs': height_xr.rio.crs,
       'transform': height_xr.rio.transform(),
       'nodata': height_xr.rio.nodata
    }
    
    return height_xr, mask_xr, out_meta


def write_xarray_results(results, height_file, code, forest_type_name, mask_data, output_meta):
    """
    Write xarray Monte Carlo results using rioxarray with optimized GeoTIFF settings.
    
    Args:
        results (xarray.Dataset): Results from monte_carlo_biomass_optimized
        height_file (str): Path to input height file
        code (str): Forest type code
        forest_type_name (str): Name of forest type
        mask_data (xarray.DataArray): Boolean mask array
        output_meta (dict): Raster metadata for outputs
        
    Returns:
        bool: True if successful, False otherwise
    """
    config = get_config()
    
    # Get GeoTIFF optimization settings from config
    geotiff_options = config['output']['geotiff'].copy()
    geotiff_options['dtype'] = output_meta.get('dtype', 'float32')
    
    try:
        # Process each variable in the results dataset
        for variable in results.data_vars:
            # Parse variable name to determine output type and measure
            parts = variable.split('_')
            if len(parts) < 2:
                logger.warning(f"Unexpected variable name format: {variable}")
                continue
                
            output_type = parts[0]
            measure = parts[1]
            
            # Build save path using centralized configuration
            try:
                savepath = build_savepath(Path(height_file), output_type, measure, code)
            except ValueError as e:
                logger.error(f"Error building save path for {variable}: {e}")
                continue
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            
            logger.info(f"Writing {output_type} {measure} for {forest_type_name}")
            with timer(f"Writing {output_type} {measure}"):
                # Get this result variable
                result = results[variable]
                
                # Apply mask to the result and handle nodata values
                masked_result = result.where(mask_data, output_meta['nodata'])
                masked_result = masked_result.fillna(output_meta['nodata'])
                
                # Set CRS information for proper georeferencing
                if hasattr(mask_data, 'rio') and hasattr(mask_data.rio, 'crs'):
                    masked_result.rio.write_crs(mask_data.rio.crs, inplace=True)
                elif 'crs' in output_meta:
                    masked_result.rio.write_crs(output_meta['crs'], inplace=True)
                
                # Set nodata value
                masked_result.rio.write_nodata(output_meta['nodata'], inplace=True)
                
                # Write to optimized GeoTIFF
                try:
                    masked_result.rio.to_raster(
                        savepath,
                        driver='GTiff',
                        **geotiff_options
                    )
                    logger.info(f"Saved {output_type} {measure} to: {savepath}")
                except Exception as write_error:
                    logger.error(f"Error writing {savepath}: {write_error}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error writing results for {forest_type_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False