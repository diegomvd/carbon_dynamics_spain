#!/usr/bin/env python3
"""
Calculate total biomass by landcover group and year.

This script processes yearly rasters of total biomass density (TBD) and masks them
according to Corine Land Cover classes grouped into natural, agricultural and urban
categories. Handles spatial reprojection when coordinate systems don't match.

Author: Diego Bengochea
"""

import os
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
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


def calculate_biomass_by_landcover(target_years, config):
    """
    Calculate total biomass for each landcover group and year.
    
    Args:
        target_years: List of years to process
        config: Configuration dictionary
        
    Returns:
        List of result dictionaries
    """
    base_dir = config['data']['base_dir']
    biomass_dir = os.path.join(base_dir, config['data']['biomass_maps_dir'], 'mean')
    corine_raster_path = os.path.join(base_dir, config['data']['corine_raster_path'])
    
    # Get landcover groups from config
    landcover_groups = config['landcover']['groups']
    pixel_area_ha = config['analysis']['pixel_area_ha']
    
    logger.info(f"Processing biomass by landcover for years: {target_years}")
    logger.info(f"Landcover groups: {list(landcover_groups.keys())}")
    logger.info(f"Using Corine raster: {corine_raster_path}")
    
    # Check if Corine raster exists
    if not os.path.exists(corine_raster_path):
        logger.error(f"Corine raster file not found: {corine_raster_path}")
        return []
    
    results = []
    
    # Open Corine landcover raster
    with rasterio.open(corine_raster_path) as corine_src:
        corine_crs = corine_src.crs
        
        # Process each year
        for year in target_years:
            logger.info(f"Processing year: {year}")
            
            # Load biomass data
            biomass_file = f"TBD_S2_mean_{year}_100m_TBD_merged.tif"
            biomass_path = os.path.join(biomass_dir, biomass_file)
            
            if not os.path.exists(biomass_path):
                logger.error(f"Biomass file not found: {biomass_file}")
                continue
            
            with rasterio.open(biomass_path) as biomass_src:
                biomass_data = biomass_src.read(1)
                biomass_nodata = biomass_src.nodata
                biomass_transform = biomass_src.transform
                biomass_crs = biomass_src.crs
                
                # Create mask for NoData in biomass
                if biomass_nodata is not None:
                    biomass_nodata_mask = (biomass_data == biomass_nodata)
                else:
                    biomass_nodata_mask = np.zeros_like(biomass_data, dtype=bool)
                
                # Reproject Corine data to match biomass data if needed
                if corine_crs != biomass_crs:
                    logger.info("  Reprojecting Corine data to match biomass data...")
                    corine_reprojected = np.zeros(biomass_data.shape, dtype=rasterio.uint8)
                    
                    # Reproject
                    reproject(
                        source=rasterio.band(corine_src, 1),
                        destination=corine_reprojected,
                        src_transform=corine_src.transform,
                        src_crs=corine_crs,
                        dst_transform=biomass_transform,
                        dst_crs=biomass_crs,
                        dst_nodata=255,  # Temporary nodata value
                        resampling=Resampling.nearest
                    )
                    
                    # Clean up reprojection artifacts
                    corine_data = corine_reprojected
                    corine_nodata_mask = (corine_data == 255)
                else:
                    # If projections match, handle spatial alignment
                    logger.info("  Aligning Corine data with biomass grid...")
                    target_bounds = biomass_src.bounds
                    
                    # Get the window in the Corine that corresponds to the biomass bounds
                    window = rasterio.windows.from_bounds(*target_bounds, corine_src.transform)
                    
                    # Read the Corine data for this window
                    corine_data = corine_src.read(1, window=window, boundless=True, fill_value=0)
                    
                    # If the window read resulted in a different shape, resize it
                    if corine_data.shape != biomass_data.shape:
                        # Reproject the windowed data to match biomass grid
                        corine_resized = np.zeros(biomass_data.shape, dtype=rasterio.uint8)
                        
                        # Calculate the transform for the windowed data
                        window_transform = rasterio.windows.transform(window, corine_src.transform)
                        
                        # Reproject the windowed data to the exact biomass grid
                        reproject(
                            source=corine_data,
                            destination=corine_resized,
                            src_transform=window_transform,
                            src_crs=corine_crs,
                            dst_transform=biomass_transform,
                            dst_crs=biomass_crs,
                            dst_nodata=0,
                            resampling=Resampling.nearest
                        )
                        
                        corine_data = corine_resized
                    
                    corine_nodata = corine_src.nodata
                    corine_nodata_mask = (corine_data == corine_nodata) if corine_nodata is not None else np.zeros_like(corine_data, dtype=bool)
                
                # Process each landcover group
                for group_name, class_values in landcover_groups.items():
                    logger.info(f"  Processing landcover group: {group_name}")
                    
                    # Create mask for this landcover group
                    group_mask = np.isin(corine_data, class_values)
                    
                    # Apply both masks to get valid biomass values for this landcover group
                    combined_mask = group_mask & ~biomass_nodata_mask & ~corine_nodata_mask
                    
                    # Calculate total biomass (tonnes per hectare * hectares = tonnes)
                    # Since each pixel is 1 hectare (100m x 100m), we can sum directly
                    total_biomass = np.sum(biomass_data[combined_mask]) * pixel_area_ha
                    
                    # Store result
                    results.append({
                        'landcover_group': group_name,
                        'year': year,
                        'biomass': total_biomass
                    })
                    
                    logger.info(f"    Landcover group {group_name}: {total_biomass:.2f} tonnes")
    
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
    output_csv = os.path.join(output_dir, "biomass_by_landcover_year.csv")
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        if results:
            fieldnames = ['landcover_group', 'year', 'biomass']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    logger.info(f"Results saved to: {output_csv}")
    return output_csv


def main():
    """
    Main function to process landcover biomass analysis.
    """
    parser = argparse.ArgumentParser(description="Landcover biomass trend analysis")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--years', nargs='+', type=int, default=None, help='Specific years to process')
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
    
    logger.info("Starting landcover biomass analysis...")
    
    # Calculate biomass by landcover groups
    results = calculate_biomass_by_landcover(target_years, config)
    
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
    logger.info(f"Landcover groups processed: {len(set(r['landcover_group'] for r in results))}")
    logger.info(f"Years processed: {sorted(set(r['year'] for r in results))}")
    logger.info(f"Results saved to: {output_file}")
    
    # Show sample results
    logger.info("\nSample results:")
    for result in results[:6]:  # Show first 6 results
        logger.info(f"  {result['year']} - {result['landcover_group']}: {result['biomass']:.2f} tonnes")


if __name__ == "__main__":
    main()
