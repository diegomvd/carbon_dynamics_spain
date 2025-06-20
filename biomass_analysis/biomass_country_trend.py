#!/usr/bin/env python3
"""
Country-level biomass time series analysis with Monte Carlo uncertainty quantification.

This script processes country-level biomass data across multiple years, calculates
uncertainties using Monte Carlo simulations with spatial correlation, and saves 
results for time series analysis.

Author: Diego Bengochea
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm
import warnings
from pathlib import Path
import argparse
from datetime import datetime
import concurrent.futures
import yaml

# Suppress warnings
warnings.filterwarnings('ignore')


def load_config(config_path='analysis_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def build_file_path(config, biomass_type, year, file_type='mean'):
    """
    Build file path using configuration and parameters.
    
    Args:
        config: Configuration dictionary
        biomass_type: Type of biomass (TBD, AGBD, BGBD)
        year: Target year
        file_type: 'mean' or 'uncertainty'
    
    Returns:
        Complete file path
    """
    base_dir = config['data']['base_dir']
    biomass_maps_dir = config['data']['biomass_maps_dir']
    
    if file_type == 'mean':
        pattern = config['file_patterns']['biomass_mean']
    else:
        pattern = config['file_patterns']['biomass_uncertainty']
    
    filename = pattern.format(
        biomass_type=biomass_type,
        year=year,
        suffix=biomass_type
    )
    
    return os.path.join(base_dir, biomass_maps_dir, file_type, filename)


def load_raster(file_path, bounds_path=None, config=None):
    """
    Load a raster file and optionally mask it with a shapefile.
    
    Args:
        file_path: Path to the raster file
        bounds_path: Optional path to shapefile for masking
        config: Configuration for quality control parameters
        
    Returns:
        Tuple of (data, transform, crs, bounds) or (None, None, None, None) if error
    """
    print(f"Loading raster: {file_path}")
    
    try:
        with rasterio.open(file_path) as src:
            # If shapefile provided, mask the raster
            if bounds_path:
                bounds_shape = gpd.read_file(bounds_path)
                
                # Reproject shapefile if needed
                if bounds_shape.crs != src.crs:
                    bounds_shape = bounds_shape.to_crs(src.crs)
                    
                # Mask raster with shapefile
                data, masked_transform = rasterio.mask.mask(
                    src,
                    bounds_shape.geometry,
                    crop=True,
                    nodata=np.nan
                )
                data = data[0]  # Get first band
                transform = masked_transform
            else:
                # Read the entire raster
                data = src.read(1)
                transform = src.transform
                
            # Handle NoData values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
                
            # Get CRS and bounds
            crs = src.crs
            bounds = src.bounds
        
        # Apply quality control filters if config provided
        if config and 'quality_control' in config:
            max_threshold = config['quality_control']['max_biomass_threshold']
            min_threshold = config['quality_control']['min_biomass_threshold']
            
            # Filter unrealistic values
            data = np.where(data > max_threshold, np.nan, data)
            data = np.where(data < min_threshold, 0, data)  # Preserve NaN
        
        return data, transform, crs, bounds
    
    except Exception as e:
        print(f"Error loading raster {file_path}: {e}")
        return None, None, None, None


def create_spatial_blocks(data_shape, block_size):
    """
    Create spatial blocks for Monte Carlo simulation with spatial correlation.
    
    Args:
        data_shape: Shape of the data array (rows, cols)
        block_size: Size of spatial blocks
        
    Returns:
        Array of block indices with the same shape as data, and total number of blocks
    """
    rows, cols = data_shape
    num_row_blocks = int(np.ceil(rows / block_size))
    num_col_blocks = int(np.ceil(cols / block_size))
    total_blocks = num_row_blocks * num_col_blocks
    
    blocks = np.zeros(data_shape, dtype=int)
    
    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            # Calculate block index
            block_idx = i * num_col_blocks + j
            
            # Calculate row and column ranges
            row_start = i * block_size
            row_end = min((i + 1) * block_size, rows)
            col_start = j * block_size
            col_end = min((j + 1) * block_size, cols)
            
            # Assign block index
            blocks[row_start:row_end, col_start:col_end] = block_idx
            
    return blocks, total_blocks


def process_monte_carlo_batch(batch_params):
    """
    Process a batch of Monte Carlo simulations for parallel execution.
    
    Args:
        batch_params: Tuple containing simulation parameters
    
    Returns:
        List of total biomass values for this batch of simulations
    """
    mean_data, std_data, valid_mask, blocks, total_blocks, rng_seed, batch_size, pixel_area_ha = batch_params
    
    # Initialize batch results
    batch_totals = np.zeros(batch_size)
    
    # Create random number generator with unique seed for this batch
    rng = np.random.default_rng(rng_seed)
    
    # For each Monte Carlo sample in this batch
    for i in range(batch_size):
        # Create a copy of the mean data for this iteration
        sample_values = mean_data.copy()
        
        # For each block, apply correlated errors
        for block_idx in range(total_blocks):
            # Find pixels in this block that have valid data
            block_mask = (blocks == block_idx) & valid_mask
            
            # If no valid pixels in this block, skip
            if np.sum(block_mask) == 0:
                continue
            
            # Generate one random standard normal value for the entire block
            z = rng.normal(0, 1)
            
            # Apply the random error to all pixels in this block
            sample_values[block_mask] += z * std_data[block_mask]
        
        # Ensure no negative values
        sample_values[valid_mask] = np.maximum(sample_values[valid_mask], 0)
        
        # Calculate total for this sample (in tonnes)
        batch_totals[i] = np.sum(sample_values[valid_mask]) * pixel_area_ha
    
    return batch_totals


def monte_carlo_uncertainty(mean_data, uncertainty_data, config, n_workers=None):
    """
    Perform Monte Carlo simulation with spatial blocks to estimate uncertainty.
    
    Args:
        mean_data: Array of mean values
        uncertainty_data: Array of uncertainty values (half-width of 95% CI = 1.96*std)
        config: Configuration dictionary
        n_workers: Number of parallel workers (set to None to disable parallelization)
        
    Returns:
        Tuple of (total_low, total_mean, total_high, all_samples) in megatonnes
    """
    # Extract parameters from config
    n_samples = config['monte_carlo']['num_samples']
    block_size = config['monte_carlo']['spatial_block_size']
    seed = config['monte_carlo']['random_seed']
    pixel_area_ha = config['analysis']['pixel_area_ha']
    percentile_low = config['monte_carlo']['confidence_interval']['low_percentile']
    percentile_high = config['monte_carlo']['confidence_interval']['high_percentile']
    
    print(f"Starting Monte Carlo simulation with {n_samples} samples and block size {block_size}")
    
    # Create mask for valid data (where both mean and uncertainty are available)
    valid_mask = ~np.isnan(mean_data) & ~np.isnan(uncertainty_data)
    
    # If no valid data, return zeros
    if np.sum(valid_mask) == 0:
        print("No valid data for Monte Carlo simulation")
        return 0, 0, 0, np.array([])
    
    # Convert uncertainty from 95% CI half-width to standard deviation
    # 95% CI corresponds to z-score of 1.96
    z_score_95 = 1.96
    std_data = uncertainty_data / z_score_95
    
    # Create spatial blocks
    blocks, total_blocks = create_spatial_blocks(mean_data.shape, block_size)
    
    # Initialize array for totals
    totals = np.zeros(n_samples)
    
    # If parallelization is disabled or only a few samples, use sequential approach
    if n_workers is None or n_samples <= 10:
        # Random number generator with seed
        rng = np.random.default_rng(seed)
        
        # For each Monte Carlo sample
        for i in tqdm(range(n_samples), desc="Running Monte Carlo iterations"):
            # Create a copy of the mean data for this iteration
            sample_values = mean_data.copy()
            
            # For each block, apply correlated errors
            for block_idx in range(total_blocks):
                # Find pixels in this block that have valid data
                block_mask = (blocks == block_idx) & valid_mask
                
                # If no valid pixels in this block, skip
                if np.sum(block_mask) == 0:
                    continue
                
                # Generate one random standard normal value for the entire block
                z = rng.normal(0, 1)
                
                # Apply the random error to all pixels in this block
                sample_values[block_mask] += z * std_data[block_mask]
            
            # Ensure no negative values
            sample_values[valid_mask] = np.maximum(sample_values[valid_mask], 0)
            
            # Calculate total for this sample (in tonnes)
            totals[i] = np.sum(sample_values[valid_mask]) * pixel_area_ha
    else:
        # Parallel implementation
        print(f"Using {n_workers} parallel workers for Monte Carlo simulation")
        
        # Calculate batch size
        batch_size = max(1, n_samples // n_workers)
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Prepare batch parameters
        batch_params = []
        for i in range(n_batches):
            actual_batch_size = min(batch_size, n_samples - i * batch_size)
            # Create unique seed for each batch
            batch_seed = seed + i
            batch_params.append((
                mean_data, std_data, valid_mask, blocks, total_blocks, 
                batch_seed, actual_batch_size, pixel_area_ha
            ))
        
        # Process batches in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(process_monte_carlo_batch, batch_params),
                total=n_batches,
                desc="Processing Monte Carlo batches"
            ))
        
        # Combine results
        totals = np.concatenate([r for r in results])[:n_samples]
    
    # Calculate statistics (convert to megatonnes)
    total_low = np.percentile(totals, percentile_low) / 1e6
    total_mean = np.nanmean(totals) / 1e6
    total_high = np.percentile(totals, percentile_high) / 1e6
    
    print(f"Monte Carlo results: Low: {total_low:.2f} Mt, Mean: {total_mean:.2f} Mt, High: {total_high:.2f} Mt")
    
    return total_low, total_mean, total_high, totals / 1e6


def process_country_biomass(biomass_type, year, config):
    """
    Process country-level biomass data for a specific type and year.
    
    Args:
        biomass_type: Type of biomass (TBD, AGBD, BGBD)
        year: Target year
        config: Configuration dictionary
        
    Returns:
        Dictionary with country-level statistics or None if failed
    """
    print(f"\nProcessing {biomass_type} for year: {year}")
    
    # Build file paths
    mean_file = build_file_path(config, biomass_type, year, 'mean')
    uncertainty_file = build_file_path(config, biomass_type, year, 'uncertainty')
    
    # Check if files exist
    if not os.path.exists(mean_file):
        print(f"Mean file not found: {mean_file}")
        return None
    
    if not os.path.exists(uncertainty_file):
        print(f"Uncertainty file not found: {uncertainty_file}")
        return None
    
    # Get country boundary path
    base_dir = config['data']['base_dir']
    country_bounds_path = os.path.join(base_dir, config['data']['country_bounds_path'])
    
    # Load raster files with country boundary masking
    mean_data, transform, crs, bounds = load_raster(mean_file, country_bounds_path, config)
    uncertainty_data, _, _, _ = load_raster(uncertainty_file, country_bounds_path, config)
    
    # Check if data loaded successfully
    if mean_data is None or uncertainty_data is None:
        print(f"Failed to load data for {biomass_type} {year}")
        return None
    
    # Run Monte Carlo simulation
    n_workers = None
    if config['monte_carlo']['parallel_processing']['enabled']:
        n_workers = config['monte_carlo']['parallel_processing']['num_workers']
    
    low, mean, high, all_samples = monte_carlo_uncertainty(
        mean_data, 
        uncertainty_data,
        config,
        n_workers=n_workers
    )
    
    return {
        'biomass_type': biomass_type,
        'year': year,
        'biomass_low': low,
        'biomass_mean': mean,
        'biomass_high': high,
        'samples': all_samples
    }


def main():
    """
    Main function to process all years and biomass types and save results.
    """
    parser = argparse.ArgumentParser(description="Country-level biomass time series analysis")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--biomass-types', nargs='+', default=None, help='Specific biomass types to process')
    parser.add_argument('--years', nargs='+', type=int, default=None, help='Specific years to process')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Override config with command line arguments if provided
    if args.biomass_types:
        config['file_patterns']['biomass_types'] = args.biomass_types
        print(f"Using specified biomass types: {args.biomass_types}")
    
    if args.years:
        config['analysis']['target_years'] = args.years
        print(f"Using specified years: {args.years}")
    
    # Create output directory
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all combinations
    target_years = config['analysis']['target_years']
    biomass_types = config['file_patterns']['biomass_types']
    
    results = []
    year_samples = {}
    
    # Process each biomass type and year
    for biomass_type in biomass_types:
        print(f"\n{'='*60}")
        print(f"Processing biomass type: {biomass_type}")
        print(f"{'='*60}")
        
        for year in target_years:
            result = process_country_biomass(biomass_type, str(year), config)
            
            if result:
                results.append({
                    'biomass_type': result['biomass_type'],
                    'year': result['year'],
                    'biomass_low': result['biomass_low'],
                    'biomass_mean': result['biomass_mean'],
                    'biomass_high': result['biomass_high']
                })
                
                # Store Monte Carlo samples
                key = f"{biomass_type}_{year}"
                year_samples[key] = result['samples']
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV results
        output_file = os.path.join(output_dir, f"country_biomass_timeseries_{timestamp}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(results_df)
        
        # Save Monte Carlo samples if configured
        if config['output']['save_monte_carlo_samples']:
            samples_file = os.path.join(output_dir, f"country_biomass_mc_samples_{timestamp}.npz")
            np.savez(samples_file, **year_samples)
            print(f"Monte Carlo samples saved to: {samples_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total records processed: {len(results_df)}")
        print(f"Biomass types: {results_df['biomass_type'].unique().tolist()}")
        print(f"Years: {sorted(results_df['year'].unique().tolist())}")
        
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
