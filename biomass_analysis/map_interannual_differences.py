#!/usr/bin/env python3
"""
Calculate interannual biomass differences between consecutive years.

This script generates raster maps of biomass changes between consecutive years,
calculating both raw signed differences and relative symmetric differences.
Outputs are saved to separate directories for different analysis purposes.

Author: Diego Bengochea
"""

import os
import re
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from glob import glob
import argparse
import yaml
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='analysis_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def find_biomass_files(input_dir):
    """
    Find and organize biomass files by year.
    
    Args:
        input_dir: Directory containing biomass files
        
    Returns:
        dict: Dictionary with years as keys and file paths as values
    """
    # Find all raster files matching the pattern
    raster_pattern = os.path.join(input_dir, "TBD_S2_mean_*_100m_TBD_merged.tif")
    raster_files = glob(raster_pattern)
    
    if not raster_files:
        logger.error(f"No biomass files found with pattern: {raster_pattern}")
        return {}
    
    # Extract years from filenames
    year_pattern = re.compile(r"TBD_S2_mean_(\d{4})_100m_TBD_merged\.tif")
    year_files = {}
    
    for file in raster_files:
        basename = os.path.basename(file)
        match = year_pattern.match(basename)
        if match:
            year = int(match.group(1))
            year_files[year] = file
    
    logger.info(f"Found biomass files for years: {sorted(year_files.keys())}")
    return year_files


def calculate_raw_difference(data1, data2, nodata_value):
    """
    Calculate raw signed difference (year2 - year1).
    
    Args:
        data1: Biomass data for first year
        data2: Biomass data for second year  
        nodata_value: NoData value to use
        
    Returns:
        numpy array: Raw difference
    """
    # Handle nodata and negative values
    data1_clean = np.where(data1 < 0, 0, data1)
    data2_clean = np.where(data2 < 0, 0, data2)
    
    # Calculate raw difference (year2 - year1)
    diff_data = data2_clean - data1_clean
    
    # Set nodata where either input was nodata
    mask_nodata = np.isnan(data1) | np.isnan(data2)
    diff_data = np.where(mask_nodata, nodata_value, diff_data)
    
    return diff_data


def calculate_relative_difference(data1, data2, nodata_value):
    """
    Calculate relative symmetric difference: 200*(year2-year1)/(year2+year1).
    
    Args:
        data1: Biomass data for first year
        data2: Biomass data for second year
        nodata_value: NoData value to use
        
    Returns:
        numpy array: Relative symmetric difference
    """
    # Handle nodata and negative values
    data1_clean = np.where(data1 < 0, 0, data1)
    data2_clean = np.where(data2 < 0, 0, data2)
    
    # Calculate relative symmetric difference
    # Formula: 200*(data2 - data1)/(data2 + data1)
    numerator = data2_clean - data1_clean
    denominator = data2_clean + data1_clean
    
    # Avoid division by zero
    diff_data = np.where(
        denominator != 0,
        200 * numerator / denominator,
        0  # Set to 0 where both values are 0
    )
    
    # Set nodata where either input was nodata
    mask_nodata = np.isnan(data1) | np.isnan(data2)
    diff_data = np.where(mask_nodata, nodata_value, diff_data)
    
    return diff_data


def generate_interannual_differences(year_files, config):
    """
    Generate both raw and relative difference rasters for consecutive years.
    
    Args:
        year_files: Dictionary with years as keys and file paths as values
        config: Configuration dictionary
    """
    base_dir = config['data']['base_dir']
    raw_output_dir = os.path.join(base_dir, config['interannual']['differences']['output_raw_dir'])
    relative_output_dir = os.path.join(base_dir, config['interannual']['differences']['output_relative_dir'])
    
    # Create output directories
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(relative_output_dir, exist_ok=True)
    
    # Sort years
    years = sorted(year_files.keys())
    
    # Process consecutive years
    differences_created = 0
    
    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]
        
        # Only process if years are consecutive
        if year2 == year1 + 1:
            file1 = year_files[year1]
            file2 = year_files[year2]
            
            logger.info(f"Processing difference: {year1} -> {year2}")
            
            # Output filenames
            raw_output = os.path.join(
                raw_output_dir, 
                f"TBD_S2_raw_change_{year1}Sep-{year2}Aug_100m.tif"
            )
            relative_output = os.path.join(
                relative_output_dir, 
                f"TBD_S2_relative_change_symmetric_{year1}Sep-{year2}Aug_100m.tif"
            )
            
            # Skip if both outputs already exist
            if os.path.exists(raw_output) and os.path.exists(relative_output):
                logger.info(f"  Both difference files already exist, skipping")
                continue
            
            # Calculate differences
            with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
                data1 = src1.read(1).astype(np.float32)
                data2 = src2.read(1).astype(np.float32)
                nodata_value = src1.nodata
                
                # Handle nodata values
                if nodata_value is not None:
                    data1 = np.where(data1 == nodata_value, np.nan, data1)
                    data2 = np.where(data2 == nodata_value, np.nan, data2)
                
                # Calculate raw difference
                if not os.path.exists(raw_output):
                    raw_diff = calculate_raw_difference(data1, data2, nodata_value)
                    
                    # Create output file with same metadata as input
                    meta = src1.meta.copy()
                    meta.update(dtype=rasterio.float32)
                    
                    with rasterio.open(raw_output, 'w', **meta) as dst:
                        dst.write(raw_diff.astype(np.float32), 1)
                    
                    logger.info(f"  Created raw difference: {os.path.basename(raw_output)}")
                
                # Calculate relative difference  
                if not os.path.exists(relative_output):
                    relative_diff = calculate_relative_difference(data1, data2, nodata_value)
                    
                    # Create output file with same metadata as input
                    meta = src1.meta.copy()
                    meta.update(dtype=rasterio.float32)
                    
                    with rasterio.open(relative_output, 'w', **meta) as dst:
                        dst.write(relative_diff.astype(np.float32), 1)
                    
                    logger.info(f"  Created relative difference: {os.path.basename(relative_output)}")
                
                differences_created += 1
        else:
            logger.warning(f"Years {year1} and {year2} are not consecutive, skipping")
    
    logger.info(f"Processing complete. Created differences for {differences_created} year pairs.")


def resample_to_10km(input_dir, create_resampled=False):
    """
    Resample difference rasters to 10km resolution.
    
    Args:
        input_dir: Directory containing difference rasters
        create_resampled: Whether to create resampled versions
    """
    if not create_resampled:
        logger.info("10km resampling disabled in configuration")
        return
    
    logger.info("Resampling difference rasters to 10km resolution...")
    
    # Find all difference rasters
    diff_pattern = os.path.join(input_dir, "*_100m.tif")
    diff_files = glob(diff_pattern)
    
    for file in diff_files:
        basename = os.path.basename(file)
        # Create output filename by replacing 100m with 10km
        output_file = os.path.join(input_dir, basename.replace("_100m.tif", "_10km.tif"))
        
        # Skip if output already exists
        if os.path.exists(output_file):
            logger.info(f"  10km file already exists: {os.path.basename(output_file)}")
            continue
        
        with rasterio.open(file) as src:
            # Calculate new dimensions (10km = 100 times larger than 100m)
            new_width = max(1, src.width // 100)
            new_height = max(1, src.height // 100)

            # Read and resample data
            dst_data = src.read(
                out_shape=(
                    src.count,
                    new_height,
                    new_width
                ),
                resampling=Resampling.average
            )

            # Calculate new transform
            dst_transform = src.transform * src.transform.scale(
                (src.width / dst_data.shape[-1]),
                (src.height / dst_data.shape[-2])
            )

            # Update profile
            dst_profile = src.profile.copy()
            dst_profile.update(
                transform=dst_transform,
                width=new_width,
                height=new_height
            )
            
            # Write the output raster
            with rasterio.open(output_file, 'w', **dst_profile) as dst:
                dst.write(dst_data)
            
            logger.info(f"  Created 10km resampled: {os.path.basename(output_file)}")


def main():
    """
    Main function to calculate interannual biomass differences.
    """
    parser = argparse.ArgumentParser(description="Calculate interannual biomass differences")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--years', nargs='+', type=int, default=None, help='Specific years to process')
    parser.add_argument('--create-10km', action='store_true', help='Force creation of 10km resampled versions')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Override 10km creation if specified
    if args.create_10km:
        config['interannual']['differences']['create_10km_resampled'] = True
    
    logger.info("Starting interannual biomass difference calculation...")
    
    # Build input directory path
    base_dir = config['data']['base_dir']
    input_dir = os.path.join(base_dir, config['interannual']['differences']['input_biomass_dir'])
    
    logger.info(f"Input directory: {input_dir}")
    
    # Find biomass files
    year_files = find_biomass_files(input_dir)
    
    if not year_files:
        logger.error("No biomass files found. Exiting.")
        return
    
    # Filter years if specified
    if args.years:
        filtered_year_files = {year: path for year, path in year_files.items() if year in args.years}
        if not filtered_year_files:
            logger.error(f"None of the specified years {args.years} found in available files")
            return
        year_files = filtered_year_files
        logger.info(f"Processing specific years: {sorted(year_files.keys())}")
    
    # Step 1: Generate interannual differences (both raw and relative)
    logger.info("Generating interannual differences...")
    generate_interannual_differences(year_files, config)
    
    # Step 2: Resample to 10km if configured
    if config['interannual']['differences']['create_10km_resampled']:
        raw_output_dir = os.path.join(base_dir, config['interannual']['differences']['output_raw_dir'])
        relative_output_dir = os.path.join(base_dir, config['interannual']['differences']['output_relative_dir'])
        
        resample_to_10km(raw_output_dir, True)
        resample_to_10km(relative_output_dir, True)
    
    logger.info("Interannual difference calculation complete!")


if __name__ == "__main__":
    main()
