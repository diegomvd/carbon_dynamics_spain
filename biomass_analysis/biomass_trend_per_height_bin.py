#!/usr/bin/env python3
"""
Calculate total biomass by height range and year.

This script creates height range masks (if they don't exist) and then calculates
total biomass for each height range across multiple years. Combines the functionality
of creating height masks and analyzing biomass trends in a single workflow.

Author: Diego Bengochea
"""

import os
import csv
import numpy as np
import rasterio
import argparse
import yaml
from datetime import datetime
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


def create_height_masks_for_year(year, config):
    """
    Creates binary masks for different vegetation height ranges for a specific year.
    
    Args:
        year: Year to process
        config: Configuration dictionary
        
    Returns:
        bool: True if masks were created successfully, False otherwise
    """
    base_dir = config['data']['base_dir']
    height_dir = os.path.join(base_dir, config['data']['canopy_height_dir'])
    mask_dir = os.path.join(base_dir, config['height_ranges']['mask_dir'])
    
    # Get height ranges from config
    height_bins = config['height_ranges']['bins']
    height_labels = config['height_ranges']['labels']
    
    # Build height file path
    height_file_pattern = config['height_ranges']['height_file_pattern']
    height_file = height_file_pattern.format(year=year)
    height_path = os.path.join(height_dir, height_file)
    
    # Check if height file exists
    if not os.path.exists(height_path):
        logger.error(f"Height file not found for year {year}: {height_path}")
        return False
    
    # Ensure mask output directory exists
    os.makedirs(mask_dir, exist_ok=True)
    
    logger.info(f"Creating height masks for year {year}")
    
    # Read input height raster
    with rasterio.open(height_path) as src:
        # Get metadata
        meta = src.meta.copy()
        height_data = src.read(1)
        nodata_value = src.nodata
        
        # Update metadata for binary masks
        meta.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=255  # Use 255 as NoData value for uint8
        )
        
        # Create a mask for NoData values
        if nodata_value is not None:
            nodata_mask = (height_data == nodata_value)
        else:
            nodata_mask = np.zeros_like(height_data, dtype=bool)
        
        # Process each height range
        for i, (label) in enumerate(height_labels):
            # Create output filename
            output_filename = f"height_{label}_{year}.tif"
            output_path = os.path.join(mask_dir, output_filename)
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                logger.info(f"  Height mask already exists: {output_filename}")
                continue
            
            # Determine height range bounds
            if i < len(height_bins) - 1:
                min_height = height_bins[i]
                max_height = height_bins[i + 1]
                mask = (height_data >= min_height) & (height_data < max_height) & ~nodata_mask
            else:
                # Last bin (e.g., 20m+)
                min_height = height_bins[i]
                mask = (height_data >= min_height) & ~nodata_mask
            
            # Convert to uint8 (1 for in range, 0 for out of range)
            mask_data = mask.astype(np.uint8)
            
            # Apply NoData value (255) to areas that were NoData in original
            mask_data[nodata_mask] = 255
            
            # Write the mask
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(mask_data, 1)
            
            logger.info(f"  Created height mask: {output_filename}")
    
    return True


def check_and_create_height_masks(target_years, config):
    """
    Check if height masks exist for target years and create them if needed.
    
    Args:
        target_years: List of years to process
        config: Configuration dictionary
        
    Returns:
        bool: True if all masks are available, False otherwise
    """
    base_dir = config['data']['base_dir']
    mask_dir = os.path.join(base_dir, config['height_ranges']['mask_dir'])
    height_labels = config['height_ranges']['labels']
    
    all_masks_available = True
    
    for year in target_years:
        year_masks_exist = True
        
        # Check if all height range masks exist for this year
        for label in height_labels:
            mask_file = f"height_{label}_{year}.tif"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if not os.path.exists(mask_path):
                year_masks_exist = False
                break
        
        # If masks don't exist for this year, create them
        if not year_masks_exist:
            logger.info(f"Height masks missing for year {year}, creating them...")
            success = create_height_masks_for_year(year, config)
            if not success:
                logger.error(f"Failed to create height masks for year {year}")
                all_masks_available = False
        else:
            logger.info(f"All height masks exist for year {year}")
    
    return all_masks_available


def calculate_biomass_by_height_ranges(target_years, config):
    """
    Calculate total biomass for each height range and year.
    
    Args:
        target_years: List of years to process
        config: Configuration dictionary
        
    Returns:
        List of result dictionaries
    """
    base_dir = config['data']['base_dir']
    biomass_dir = os.path.join(base_dir, config['data']['biomass_maps_dir'], 'mean')
    mask_dir = os.path.join(base_dir, config['height_ranges']['mask_dir'])
    height_labels = config['height_ranges']['labels']
    pixel_area_ha = config['analysis']['pixel_area_ha']
    
    results = []
    
    # Process each year
    for year in target_years:
        logger.info(f"Processing biomass by height ranges for year {year}")
        
        # Load biomass data
        biomass_file = f"TBD_S2_mean_{year}_100m_TBD_merged.tif"
        biomass_path = os.path.join(biomass_dir, biomass_file)
        
        if not os.path.exists(biomass_path):
            logger.error(f"Biomass file not found: {biomass_file}")
            continue
        
        with rasterio.open(biomass_path) as biomass_src:
            biomass_data = biomass_src.read(1)
            biomass_nodata = biomass_src.nodata
            
            # Create mask for NoData
            if biomass_nodata is not None:
                biomass_nodata_mask = (biomass_data == biomass_nodata)
            else:
                biomass_nodata_mask = np.zeros_like(biomass_data, dtype=bool)
            
            # Process each height range
            for height_range in height_labels:
                height_mask_file = f"height_{height_range}_{year}.tif"
                height_mask_path = os.path.join(mask_dir, height_mask_file)
                
                if not os.path.exists(height_mask_path):
                    logger.error(f"Height mask file not found: {height_mask_file}")
                    continue
                
                with rasterio.open(height_mask_path) as mask_src:
                    mask_data = mask_src.read(1)
                    mask_nodata = mask_src.nodata
                    
                    # Create mask for valid height data
                    if mask_nodata is not None:
                        height_mask = (mask_data == 1) & (mask_data != mask_nodata)
                    else:
                        height_mask = (mask_data == 1)
                    
                    # Apply both masks to get valid biomass values for this height range
                    combined_mask = height_mask & ~biomass_nodata_mask
                    
                    # Calculate total biomass (tonnes per hectare * hectares = tonnes)
                    # Since each pixel is 1 hectare, we can sum directly
                    total_biomass = np.sum(biomass_data[combined_mask]) * pixel_area_ha
                    
                    # Store result
                    results.append({
                        'height_bin': height_range,
                        'year': year,
                        'biomass': total_biomass
                    })
                    
                    logger.info(f"  Height range {height_range}: {total_biomass:.2f} tonnes")
    
    return results


def save_results(results, config):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        config: Configuration dictionary
        
    Returns:
        str: Path to saved CSV file
    """
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output CSV file (matching original naming)
    output_csv = os.path.join(output_dir, "biomass_by_height_year.csv")
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        if results:
            fieldnames = ['height_bin', 'year', 'biomass']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    logger.info(f"Results saved to: {output_csv}")
    return output_csv


def main():
    """
    Main function to process height range biomass analysis.
    """
    parser = argparse.ArgumentParser(description="Height range biomass trend analysis")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--years', nargs='+', type=int, default=None, help='Specific years to process')
    parser.add_argument('--skip-mask-creation', action='store_true', help='Skip mask creation and use existing masks only')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Determine target years
    if args.years:
        target_years = args.years
        logger.info(f"Processing specific years: {target_years}")
    else:
        target_years = config['analysis']['target_years']
        logger.info(f"Processing all configured years: {target_years}")
    
    logger.info("Starting height range biomass analysis...")
    logger.info(f"Height ranges: {config['height_ranges']['labels']}")
    
    # Check and create height masks if needed
    if not args.skip_mask_creation:
        masks_available = check_and_create_height_masks(target_years, config)
        if not masks_available:
            logger.error("Some height masks could not be created. Continuing with available masks...")
    else:
        logger.info("Skipping mask creation - using existing masks only")
    
    # Calculate biomass by height ranges
    results = calculate_biomass_by_height_ranges(target_years, config)
    
    if not results:
        logger.error("No results calculated. Check previous errors.")
        return
    
    # Save results
    output_file = save_results(results, config)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total records: {len(results)}")
    logger.info(f"Height ranges processed: {len(set(r['height_bin'] for r in results))}")
    logger.info(f"Years processed: {sorted(set(r['year'] for r in results))}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
