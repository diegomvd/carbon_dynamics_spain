#!/usr/bin/env python3
"""
Calculate total biomass by forest type, genus, and clade for multiple years.

This script processes biomass raster files and forest type masks to calculate
total biomass at different hierarchical levels: individual forest types, 
genus level, and clade level. Outputs separate CSV files for each aggregation level.

Author: Diego Bengochea
"""

import os
import pandas as pd
import numpy as np
import rasterio
from glob import glob
import logging
from datetime import datetime
import argparse
import yaml

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


def load_and_merge_forest_type_mappings(config):
    """
    Load and merge the two CSV files containing forest type mappings.
    Returns a DataFrame with code, name, genus, and clade.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with forest type mappings
    """
    logger.info("Loading forest type mappings...")
    
    base_dir = config['data']['base_dir']
    
    # Load CSV files
    forest_types_path = os.path.join(base_dir, config['data']['forest_types_csv'])
    forest_codes_path = os.path.join(base_dir, config['data']['forest_type_codes_csv'])
    
    forest_types = pd.read_csv(forest_types_path)
    forest_codes = pd.read_csv(forest_codes_path)
    
    # Merge on the forest type name
    merged = pd.merge(
        forest_codes, 
        forest_types, 
        left_on='name', 
        right_on='ForestTypeMFE', 
        how='left'
    )
    
    # Handle non-forest code (0)
    if 0 not in merged['code'].values:
        non_forest_row = pd.DataFrame({
            'code': [0],
            'name': ['Non forest'],
            'ForestTypeMFE': ['Non forest'],
            'Species': ['Non forest'],
            'Genus': ['Non forest'],
            'Family': ['Non forest'],
            'Clade': ['Non forest']
        })
        merged = pd.concat([merged, non_forest_row], ignore_index=True)
    
    # Replace "Angiosperm" with "Broadleaved" in Clade column
    merged['Clade'] = merged['Clade'].replace('Angiosperm', 'Broadleaved')
    
    # Check for missing values and report
    missing_genus = merged[merged['Genus'].isna()]
    missing_clade = merged[merged['Clade'].isna()]
    
    if not missing_genus.empty or not missing_clade.empty:
        logger.warning(f"Found {len(missing_genus)} forest types without genus mapping")
        logger.warning(f"Found {len(missing_clade)} forest types without clade mapping")
        logger.warning(f"Forest types with missing mappings: {missing_genus['name'].tolist() if not missing_genus.empty else []}")
        
        # Fill missing values with the forest type name
        merged['Genus'] = merged['Genus'].fillna(merged['name'])
        merged['Clade'] = merged['Clade'].fillna('Unknown')
        merged['Clade'] = merged['Clade'].replace('Angiosperm', 'Broadleaved')  # Apply replacement again in case of filled values
    
    logger.info(f"Loaded and merged mapping for {len(merged)} forest types")
    return merged[['code', 'name', 'Genus', 'Clade']]


def get_forest_type_masks(config):
    """
    Get list of forest type mask files from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of forest type mask file paths
    """
    base_dir = config['data']['base_dir']
    mask_dir = os.path.join(base_dir, config['data']['forest_type_masks_dir'])
    
    mask_files = glob(os.path.join(mask_dir, "forest_type_mask_100m_code*.tif"))
    
    if not mask_files:
        logger.error(f"No forest type mask files found in {mask_dir}")
        return []
    
    logger.info(f"Found {len(mask_files)} forest type mask files")
    return mask_files


def calculate_biomass_by_forest_type(year, forest_type_masks, forest_type_df, config):
    """
    Calculate total biomass by forest type for a specific year.
    
    Args:
        year: Year to process
        forest_type_masks: List of forest type mask file paths
        forest_type_df: DataFrame with forest type mappings
        config: Configuration dictionary
        
    Returns:
        DataFrame with biomass by forest type with genus and clade info
    """
    logger.info(f"Processing biomass for year {year}...")
    
    # Build biomass file path
    base_dir = config['data']['base_dir']
    biomass_dir = os.path.join(base_dir, config['data']['biomass_maps_dir'], 'mean')
    biomass_file = os.path.join(biomass_dir, f"TBD_S2_mean_{year}_100m_TBD_merged.tif")
    
    if not os.path.exists(biomass_file):
        logger.error(f"Biomass file not found for year {year}: {biomass_file}")
        return None
    
    # Load biomass raster
    with rasterio.open(biomass_file) as src:
        biomass = src.read(1)
        # Check for no-data values and replace with 0
        nodata = src.nodata
        if nodata is not None:
            biomass = np.where(biomass == nodata, 0, biomass)
        
        # Convert to float if needed
        if biomass.dtype != np.float32 and biomass.dtype != np.float64:
            biomass = biomass.astype(np.float32)
            
        logger.info(f"Loaded biomass raster for {year}, shape: {biomass.shape}, dtype: {biomass.dtype}")
        
        # Initialize results container
        results = []
        
        # Process each forest type mask
        for mask_file in forest_type_masks:
            # Extract forest type code from filename
            try:
                code = int(os.path.basename(mask_file).split('_code')[1].split('.')[0])
            except (ValueError, IndexError) as e:
                logger.error(f"Could not extract code from mask filename {mask_file}: {e}")
                continue
                
            # Skip if code not in our mapping
            if code not in forest_type_df['code'].values:
                logger.warning(f"Forest type code {code} not found in mapping, skipping")
                continue
                
            # Get forest type information for this code
            forest_info = forest_type_df[forest_type_df['code'] == code].iloc[0]
            genus = forest_info['Genus']
            clade = forest_info['Clade']
            name = forest_info['name']
            
            logger.info(f"Processing forest type code {code}: {name} (Genus: {genus}, Clade: {clade})")
            
            # Open mask and calculate total biomass
            with rasterio.open(mask_file) as mask_src:
                mask = mask_src.read(1)
                # Make sure mask is boolean 
                valid_mask = (mask == 1)
                
                # Extract biomass for this forest type
                forest_biomass = biomass[valid_mask]
                
                # Calculate total biomass (sum of all pixel values)
                # Since each pixel is 1 hectare and biomass is in tonnes/hectare, 
                # the sum equals the total biomass in tonnes
                total_biomass = np.sum(forest_biomass)
                
                logger.info(f"  - Forest type {code}: {name}, total biomass: {total_biomass:.2f} tonnes")
                
                # Store the result with all hierarchical information
                results.append({
                    'Year': year,
                    'ForestType_Code': code,
                    'ForestType_Name': name,
                    'Genus': genus,
                    'Clade': clade,
                    'Total_Biomass_Tonnes': total_biomass
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            logger.info(f"Completed processing biomass for year {year}, found {len(results_df)} forest type entries")
            return results_df
        else:
            logger.warning(f"No results for year {year}")
            return None


def aggregate_results(all_forest_type_results):
    """
    Create aggregated results by genus and clade from forest type results.
    
    Args:
        all_forest_type_results: DataFrame with all forest type results
        
    Returns:
        Tuple of (genus_df, clade_df) with aggregated results
    """
    if all_forest_type_results.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    logger.info("Creating aggregated results by genus and clade...")
    
    # Aggregate by genus (keeping clade info)
    genus_df = all_forest_type_results.groupby(['Year', 'Genus', 'Clade'])['Total_Biomass_Tonnes'].sum().reset_index()
    genus_df = genus_df.sort_values(['Year', 'Genus'])
    
    # Aggregate by clade only
    clade_df = all_forest_type_results.groupby(['Year', 'Clade'])['Total_Biomass_Tonnes'].sum().reset_index()
    clade_df = clade_df.sort_values(['Year', 'Clade'])
    
    logger.info(f"Created genus aggregation: {len(genus_df)} rows")
    logger.info(f"Created clade aggregation: {len(clade_df)} rows")
    
    return genus_df, clade_df


def save_results(forest_type_df, genus_df, clade_df, config):
    """
    Save all results to CSV files.
    
    Args:
        forest_type_df: Forest type level results
        genus_df: Genus level aggregated results
        clade_df: Clade level aggregated results
        config: Configuration dictionary
    """
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Save forest type level results
    forest_type_file = os.path.join(output_dir, "biomass_by_forest_type_year.csv")
    forest_type_df.to_csv(forest_type_file, index=False)
    logger.info(f"Forest type results exported to {forest_type_file}")
    
    # Save genus level results (matching original naming)
    genus_file = os.path.join(output_dir, "biomass_by_genus_year.csv")
    genus_df.to_csv(genus_file, index=False)
    logger.info(f"Genus results exported to {genus_file}")
    
    # Save clade level results (matching original naming)
    clade_file = os.path.join(output_dir, "biomass_by_clade_year.csv")
    clade_df.to_csv(clade_file, index=False)
    logger.info(f"Clade results exported to {clade_file}")
    
    return forest_type_file, genus_file, clade_file


def main():
    """
    Main function to process all years and generate hierarchical biomass aggregations.
    """
    parser = argparse.ArgumentParser(description="Forest type biomass trend analysis")
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
    
    # Override years if specified
    if args.years:
        target_years = args.years
        logger.info(f"Processing specific years: {target_years}")
    else:
        target_years = config['analysis']['target_years']
        logger.info(f"Processing all configured years: {target_years}")
    
    logger.info("Starting biomass by forest type calculation...")
    
    # Load and merge forest type mappings
    forest_type_df = load_and_merge_forest_type_mappings(config)
    
    # Get list of forest type masks
    mask_files = get_forest_type_masks(config)
    if not mask_files:
        logger.error("No forest type mask files found. Exiting.")
        return
    
    # Initialize DataFrame to store all years' results
    all_forest_type_results = []
    
    # Process each year
    for year in target_years:
        forest_type_results = calculate_biomass_by_forest_type(year, mask_files, forest_type_df, config)
        
        if forest_type_results is not None:
            all_forest_type_results.append(forest_type_results)
    
    # Combine results from all years
    if all_forest_type_results:
        # Combine all forest type results
        combined_forest_type_df = pd.concat(all_forest_type_results, ignore_index=True)
        combined_forest_type_df = combined_forest_type_df.sort_values(['Year', 'ForestType_Code'])
        
        # Create aggregated results by genus and clade
        genus_df, clade_df = aggregate_results(combined_forest_type_df)
        
        # Save all results
        forest_type_file, genus_file, clade_file = save_results(
            combined_forest_type_df, genus_df, clade_df, config
        )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Forest type level results: {len(combined_forest_type_df)} rows")
        logger.info(f"Genus level results: {len(genus_df)} rows")
        logger.info(f"Clade level results: {len(clade_df)} rows")
        logger.info(f"Years processed: {sorted(combined_forest_type_df['Year'].unique())}")
        logger.info(f"Forest types found: {combined_forest_type_df['ForestType_Code'].nunique()}")
        logger.info(f"Genera found: {genus_df['Genus'].nunique()}")
        logger.info(f"Clades found: {clade_df['Clade'].nunique()}")
        
    else:
        logger.error("No results were calculated. Check previous errors.")


if __name__ == "__main__":
    main()
