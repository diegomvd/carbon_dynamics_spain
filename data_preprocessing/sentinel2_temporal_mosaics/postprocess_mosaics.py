"""
Harmonized post-processing pipeline for Sentinel-2 mosaics.

This script combines downsampling and merging operations for Sentinel-2 mosaics:
1. Downsampling: Reduces spatial resolution by a given scale factor
2. Merging: Combines multiple tiles by year into single files

Author: Diego Bengochea
"""

import os
import re
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from s2_utils import setup_logging, load_config, create_output_directory


def downsample_rasters(input_dir, output_dir, scale_factor=10):
    """
    Downsample raster files matching pattern 'S2_summer_mosaic_YYYY_*.tif' in the input directory 
    by a given scale factor and save them to the output directory.
    
    Args:
        input_dir (str): Path to directory containing input raster files
        output_dir (str): Path to directory where downsampled files will be saved
        scale_factor (int): Factor by which to downsample the raster
        
    Returns:
        dict: Processing statistics with success/failure counts
    """
    logger = setup_logging()
    logger.info("Starting raster downsampling...")
    
    # Create output directory using shared function
    create_output_directory(output_dir)
    
    # Define the pattern to match
    pattern = re.compile(r'S2_summer_mosaic_\d{4}_.*\.tif$', re.IGNORECASE)
    
    # Get all files in input directory that match the pattern
    raster_files = [
        f for f in os.listdir(input_dir) 
        if pattern.match(f)
    ]
    
    if not raster_files:
        logger.warning("No files found matching the pattern 'S2_summer_mosaic_YYYY_*.tif'")
        return {'successful': 0, 'failed': 0}
    
    logger.info(f"Found {len(raster_files)} raster files to downsample")
    
    # Create progress bar
    pbar = tqdm(raster_files, desc="Processing rasters", unit="file")
    
    successful = 0
    failed = 0
    
    for raster_file in pbar:
        # Update progress bar description
        pbar.set_description(f"Processing {raster_file}")
        
        input_path = os.path.join(input_dir, raster_file)
        
        # Create output filename
        filename = Path(raster_file).stem
        output_filename = f"{filename}_downsampled.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        if not os.path.exists(output_path):
            try:
                with rasterio.open(input_path) as dataset:
                    # Calculate new dimensions
                    new_height = dataset.height // scale_factor
                    new_width = dataset.width // scale_factor
                    
                    # Calculate new transform
                    transform = dataset.transform * dataset.transform.scale(
                        (dataset.width / new_width),
                        (dataset.height / new_height)
                    )
                    
                    # Create output profile
                    profile = dataset.profile.copy()
                    profile.update({
                        'height': new_height,
                        'width': new_width,
                        'transform': transform
                    })
                    
                    # Read and resample data
                    data = dataset.read(
                        out_shape=(dataset.count, new_height, new_width),
                        resampling=Resampling.average
                    )
                    
                    # Write output file
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data)
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing {raster_file}: {str(e)}")
                failed += 1
                continue
        else:
            logger.info(f"Skipping {raster_file} (already processed)")
            successful += 1
    
    pbar.close()
    logger.info(f"Downsampling completed! Successful: {successful}, Failed: {failed}")
    
    return {'successful': successful, 'failed': failed}


def group_rasters_by_year(input_dir):
    """
    Group raster files by year based on filename pattern S2_summer_mosaic_YYYY_*.
    
    Args:
        input_dir (str): Directory containing the downsampled raster files
        
    Returns:
        dict: Dictionary with years as keys and lists of file paths as values
    """
    pattern = re.compile(r'S2_summer_mosaic_(\d{4})_.*_downsampled\.tif$', re.IGNORECASE)
    raster_groups = defaultdict(list)
    
    for file in os.listdir(input_dir):
        match = pattern.match(file)
        if match:
            year = match.group(1)
            raster_groups[year].append(os.path.join(input_dir, file))
            
    return raster_groups


def merge_rasters_by_year(input_dir, output_dir):
    """
    Merge raster files grouped by year.
    
    Args:
        input_dir (str): Directory containing the downsampled raster files
        output_dir (str): Directory where merged rasters will be saved
        
    Returns:
        dict: Processing statistics with success/failure counts
    """
    logger = setup_logging()
    logger.info("Starting raster merging by year...")
    
    # Create output directory using shared function
    create_output_directory(output_dir)
    
    # Group rasters by year
    raster_groups = group_rasters_by_year(input_dir)
    
    if not raster_groups:
        logger.warning("No raster files found matching the pattern 'S2_summer_mosaic_YYYY_*_downsampled.tif'")
        return {'successful': 0, 'failed': 0}
    
    logger.info(f"Found raster groups for years: {list(raster_groups.keys())}")
    
    successful = 0
    failed = 0
    
    # Process each year
    for year, raster_files in tqdm(raster_groups.items(), desc="Processing years"):
        output_path = os.path.join(output_dir, f'S2_summer_mosaic_{year}_merged.tif')
        
        if not os.path.exists(output_path):
            try:
                logger.info(f"Merging {len(raster_files)} rasters for year {year}")
                
                # Open all raster files
                src_files = []
                for raster_path in raster_files:
                    src = rasterio.open(raster_path)
                    src_files.append(src)
                
                # Merge rasters
                mosaic, out_transform = merge(src_files)
                
                # Get metadata from first raster
                out_meta = src_files[0].meta.copy()
                
                # Update metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "compress": "LZW"  # Add compression to save space
                })
                
                # Write merged raster
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                logger.info(f"Successfully merged {len(raster_files)} rasters for year {year}")
                successful += 1
                
            except Exception as e:
                logger.error(f"Error processing year {year}: {str(e)}")
                failed += 1
                continue
                
            finally:
                # Close all opened raster files
                for src in src_files:
                    src.close()
        else:
            logger.info(f"Skipping year {year} (already processed)")
            successful += 1
    
    logger.info(f"Merging completed! Successful: {successful}, Failed: {failed}")
    return {'successful': successful, 'failed': failed}


def main():
    """Main post-processing pipeline."""
    logger = setup_logging()
    logger.info("Starting Sentinel-2 mosaic post-processing pipeline...")
    
    try:
        # Load configuration using shared function
        config = load_config()
        
        # Get directories and parameters from config
        input_directory = config['paths']['output_dir']
        downsampled_directory = config['paths']['downsampled_dir']
        merged_directory = config['paths']['merged_dir']
        scale_factor = config.get('postprocessing', {}).get('downsample', {}).get('scale_factor', 10)
        
        logger.info(f"Input directory: {input_directory}")
        logger.info(f"Downsampled directory: {downsampled_directory}")
        logger.info(f"Merged directory: {merged_directory}")
        logger.info(f"Scale factor: {scale_factor}")
        
        # Step 1: Downsample rasters
        logger.info("\n" + "="*50)
        logger.info("STEP 1: DOWNSAMPLING RASTERS")
        logger.info("="*50)
        
        downsample_stats = downsample_rasters(input_directory, downsampled_directory, scale_factor)
        
        # Step 2: Merge rasters by year
        logger.info("\n" + "="*50)
        logger.info("STEP 2: MERGING RASTERS BY YEAR")
        logger.info("="*50)
        
        merge_stats = merge_rasters_by_year(downsampled_directory, merged_directory)
        
        # Final summary
        logger.info("\n" + "="*50)
        logger.info("POST-PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Downsampling: {downsample_stats['successful']} successful, {downsample_stats['failed']} failed")
        logger.info(f"Merging: {merge_stats['successful']} successful, {merge_stats['failed']} failed")
        
        logger.info("Post-processing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Post-processing pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
