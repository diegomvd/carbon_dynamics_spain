#!/usr/bin/env python3
"""
Biomass raster merging script for multi-forest-type datasets.

This script merges individual forest type biomass raster files by year to create
comprehensive maps covering all forest types. Processes AGBD, BGBD, and TBD
(above-ground, below-ground, and total biomass density) separately for both
mean and uncertainty estimates.

The merging process combines tiles from multiple forest types into single
country-wide rasters for each year and biomass component, with optimized
GeoTIFF compression and tiling for efficient storage and access.

Author: Diego Bengochea
"""

import os
import glob
import re
import numpy as np
import rasterio
from rasterio.merge import merge

from config import get_config


def extract_year_from_filename(filename):
    """
    Extract the year from a biomass raster filename using regex pattern matching.
    
    Args:
        filename (str): Input filename containing year information
        
    Returns:
        str or None: Extracted 4-digit year string, or None if no year found
        
    Example:
        >>> extract_year_from_filename("AGBD_S2_mean_2020_100m_code12.tif")
        '2020'
    """
    match = re.search(r'_(\d{4})_', filename)
    if match:
        return match.group(1)
    return None


def get_base_filename(filename):
    """
    Extract the base filename without the forest type code suffix.
    
    Removes the "_codeXX" suffix from biomass raster filenames to get
    the common base name for merging files from different forest types.
    
    Args:
        filename (str): Input filename with forest type code
        
    Returns:
        str: Base filename without extension and code suffix
        
    Example:
        >>> get_base_filename("AGBD_S2_mean_2020_100m_code12.tif")
        'AGBD_S2_mean_2020_100m'
    """
    # Extract basename without path and extension
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Remove the forest type code suffix
    base_name = re.sub(r'_code\d+$', '', basename)
    return base_name


def is_mean_file(filename):
    """
    Check if filename represents a mean biomass estimate file.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if file contains mean estimates, False otherwise
    """
    return "_mean_" in filename


def is_uncertainty_file(filename):
    """
    Check if filename represents an uncertainty estimate file.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if file contains uncertainty estimates, False otherwise
    """
    return "_uncertainty_" in filename


def merge_rasters_by_year_and_type(input_dir, year, output_dir, is_mean=True):
    """
    Merge all raster files for a specific year and statistical measure.
    
    Combines raster files from multiple forest types into a single mosaic
    for the specified year and measure type (mean or uncertainty). Uses
    rasterio merge functionality with optimized output settings.
    
    Args:
        input_dir (str): Directory containing input raster files
        year (str): Target year for file filtering
        output_dir (str): Directory to save merged output file
        is_mean (bool): If True, process mean files; if False, process uncertainty files
        
    Returns:
        bool: True if merging succeeded, False if no files found or error occurred
    """
    config = get_config()
    
    # Determine file type and create search pattern
    file_type = "mean" if is_mean else "uncertainty"
    print(f"Processing {file_type} files for year {year} in {input_dir}...")
    
    # Build search pattern for target files
    pattern = f"*_{file_type}_{year}_*code*.tif"
    search_pattern = os.path.join(input_dir, pattern)
    raster_files = glob.glob(search_pattern)
    
    if not raster_files:
        print(f"No {file_type} raster files found for year {year} with pattern {pattern} in {input_dir}")
        return False
    
    print(f"Found {len(raster_files)} {file_type} raster files for year {year}")
    
    # Extract base filename and biomass type for output naming
    first_file = raster_files[0]
    base_name = get_base_filename(first_file)
    
    # Extract biomass type from directory name
    biomass_type = os.path.basename(input_dir).split('_')[0]
    
    # Generate output filename
    output_filename = f"{base_name}_{biomass_type}_merged.tif"

    output_file = os.path.join(output_dir, output_filename)
    
    try:
        # Open all source raster files for merging
        src_files_to_mosaic = []
        for raster in raster_files:
            src = rasterio.open(raster)
            src_files_to_mosaic.append(src)
        
        # Perform raster merging operation
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Copy metadata from first file and update for output
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Configure optimized GeoTIFF output settings
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Write merged raster to output file
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Clean up source file handles
        for src in src_files_to_mosaic:
            src.close()
        
        print(f"Merged {file_type} raster saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error merging {file_type} files for year {year}: {str(e)}")
        return False


def process_biomass_directories(base_dir, output_dir, years=None):
    """
    Process all biomass directories for both mean and uncertainty file merging.
    
    Searches for AGBD, BGBD, and TBD directories and processes each year
    separately for both statistical measures. Creates organized output
    structure with separate directories for mean and uncertainty results.
    
    Args:
        base_dir (str): Base directory containing biomass subdirectories
        output_dir (str): Root directory for merged output files
        years (list, optional): Specific years to process. If None, processes all available years
    """
    config = get_config()
    
    # Get biomass directory patterns from configuration
    biomass_patterns = config['masking']['biomass_patterns']
    
    # Create organized output directory structure
    mean_output_dir = os.path.join(output_dir, "mean")
    uncertainty_output_dir = os.path.join(output_dir, "uncertainty")
    os.makedirs(mean_output_dir, exist_ok=True)
    os.makedirs(uncertainty_output_dir, exist_ok=True)
    
    # Process each biomass type directory
    for pattern in biomass_patterns:
        biomass_dirs = glob.glob(os.path.join(base_dir, pattern))
        
        for biomass_dir in biomass_dirs:
            print(f"Processing directory: {biomass_dir}")
            
            # Auto-discover available years if not specified
            if years is None:
                all_years = set()
                tif_files = glob.glob(os.path.join(biomass_dir, "*.tif"))
                
                # Extract years from all files in directory
                for file in tif_files:
                    year = extract_year_from_filename(file)
                    if year:
                        all_years.add(year)
                
                years_to_process = sorted(list(all_years))
            else:
                years_to_process = years
            
            # Process each year for both mean and uncertainty files
            for year in years_to_process:
                # Process mean files
                merge_rasters_by_year_and_type(biomass_dir, year, mean_output_dir, is_mean=True)
                
                # Process uncertainty files
                merge_rasters_by_year_and_type(biomass_dir, year, uncertainty_output_dir, is_mean=False)


def main():
    """
    Main function for biomass raster merging using centralized configuration.
    
    Loads paths and parameters from configuration and executes the complete
    merging workflow for all biomass types and years.
    """
    config = get_config()
    
    # Load paths from centralized configuration
    base_dir = config['masking']['merge_base_dir']
    output_dir = os.path.join(base_dir, config['masking']['merged_output_dir'])
    
    # Get years to process from configuration
    years = config['masking']['merge_years']
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target years: {years}")
    
    try:
        # Execute biomass directory processing
        process_biomass_directories(base_dir, output_dir, years)
        print("Biomass raster merging completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()