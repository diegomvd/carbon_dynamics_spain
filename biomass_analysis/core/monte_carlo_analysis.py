"""
Monte Carlo Analysis Module

This module implements Monte Carlo uncertainty quantification for country-level 
biomass analysis using spatial correlation and parallel processing.

All algorithmic logic preserved exactly from original biomass_country_trend.py

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
from datetime import datetime
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Union

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config
from shared_utils.central_data_paths_constants import *

warnings.filterwarnings('ignore')


class MonteCarloAnalyzer:
    """
    Monte Carlo uncertainty quantification for country-level biomass analysis.
    
    Implements spatial correlation-aware Monte Carlo simulation with parallel 
    processing for biomass uncertainty estimation. All algorithms preserved 
    exactly from original implementation.
    """
    
    def __init__(self, config: Optional[Union[str, Path, Dict]] = None):
        """
        Initialize the Monte Carlo analyzer.
        
        Args:
            config: Configuration dictionary or path to config file
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config, component_name="biomass_analysis")
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = load_config(component_name="biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='monte_carlo_analysis'
        )
        
        self.logger.info("Initialized MonteCarloAnalyzer")

    def build_file_path(self, biomass_type: str, year: str, file_type: str = 'mean') -> str:
        """
        Build file path using configuration and parameters.
        
        Args:
            biomass_type: Type of biomass (TBD, AGBD, BGBD)
            year: Target year
            file_type: 'mean' or 'uncertainty'
        
        Returns:
            Complete file path
        """
        biomass_maps_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR / "mean"
        
        if file_type == 'mean':
            pattern = self.config['file_patterns']['biomass_mean']
        else:
            pattern = self.config['file_patterns']['biomass_uncertainty']
        
        filename = pattern.format(
            biomass_type=biomass_type,
            year=year,
            suffix=biomass_type
        )
        
        return os.path.join(base_dir, biomass_maps_dir, file_type, filename)

    def load_raster(self, file_path: str, bounds_path: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[Any], Optional[Any]]:
        """
        Load a raster file and optionally mask it with a shapefile.
        
        Args:
            file_path: Path to raster file
            bounds_path: Optional path to boundary shapefile for masking
            
        Returns:
            Tuple of (data, transform, crs, bounds) or (None, None, None, None) if error
        """
        try:
            if bounds_path and os.path.exists(bounds_path):
                # Load boundary for masking
                boundaries = gpd.read_file(bounds_path)
                
                with rasterio.open(file_path) as src:
                    # Get the geometry in the same CRS as the raster
                    boundaries_proj = boundaries.to_crs(src.crs)
                    
                    # Mask the raster
                    out_image, out_transform = mask(src, boundaries_proj.geometry, crop=True)
                    out_meta = src.meta.copy()
                    
                    # Extract the data (first band)
                    data = out_image[0]
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    return data, out_transform, src.crs, src.bounds
            else:
                # Load raster without masking
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    
                    # Handle nodata values  
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    return data, src.transform, src.crs, src.bounds
                    
        except Exception as e:
            self.logger.error(f"Error loading raster {file_path}: {e}")
            return None, None, None, None

    def create_spatial_blocks(self, data_shape: Tuple[int, int], block_size: int) -> Tuple[np.ndarray, int]:
        """
        Create spatial block indices for Monte Carlo correlation.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
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

    def process_monte_carlo_batch(self, batch_params: Tuple) -> np.ndarray:
        """
        Process a batch of Monte Carlo simulations for parallel execution.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            batch_params: Tuple containing simulation parameters
        
        Returns:
            Array of total biomass values for this batch of simulations
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

    def monte_carlo_uncertainty(self, mean_data: np.ndarray, uncertainty_data: np.ndarray, n_workers: Optional[int] = None) -> Tuple[float, float, float, np.ndarray]:
        """
        Perform Monte Carlo simulation with spatial blocks to estimate uncertainty.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            mean_data: Array of mean values
            uncertainty_data: Array of uncertainty values (half-width of 95% CI = 1.96*std)
            n_workers: Number of parallel workers (set to None to disable parallelization)
            
        Returns:
            Tuple of (total_low, total_mean, total_high, all_samples) in megatonnes
        """
        # Extract parameters from config
        n_samples = self.config['monte_carlo']['num_samples']
        block_size = self.config['monte_carlo']['spatial_block_size']
        seed = self.config['monte_carlo']['random_seed']
        pixel_area_ha = self.config['analysis']['pixel_area_ha']
        percentile_low = self.config['monte_carlo']['confidence_interval']['low_percentile']
        percentile_high = self.config['monte_carlo']['confidence_interval']['high_percentile']
        
        self.logger.info(f"Starting Monte Carlo simulation with {n_samples} samples and block size {block_size}")
        
        # Create mask for valid data (where both mean and uncertainty are available)
        valid_mask = ~np.isnan(mean_data) & ~np.isnan(uncertainty_data)
        
        # If no valid data, return zeros
        if np.sum(valid_mask) == 0:
            self.logger.warning("No valid data for Monte Carlo simulation")
            return 0, 0, 0, np.array([])
        
        # Convert uncertainty from 95% CI half-width to standard deviation
        # 95% CI corresponds to z-score of 1.96
        z_score_95 = 1.96
        std_data = uncertainty_data / z_score_95
        
        # Create spatial blocks
        blocks, total_blocks = self.create_spatial_blocks(mean_data.shape, block_size)
        
        # Initialize array for totals
        totals = np.zeros(n_samples)
        
        if n_workers and n_workers > 1:
            # Parallel processing
            self.logger.info(f"Using parallel processing with {n_workers} workers")
            
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
                    executor.map(self.process_monte_carlo_batch, batch_params),
                    total=n_batches,
                    desc="Processing Monte Carlo batches"
                ))
            
            # Combine results
            totals = np.concatenate([r for r in results])[:n_samples]
        else:
            # Sequential processing
            self.logger.info("Using sequential processing")
            rng = np.random.default_rng(seed)
            
            for sample in tqdm(range(n_samples), desc="Monte Carlo samples"):
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
                totals[sample] = np.sum(sample_values[valid_mask]) * pixel_area_ha
        
        # Calculate statistics (convert to megatonnes)
        total_low = np.percentile(totals, percentile_low) / 1e6
        total_mean = np.nanmean(totals) / 1e6
        total_high = np.percentile(totals, percentile_high) / 1e6
        
        self.logger.info(f"Monte Carlo results: Low: {total_low:.2f} Mt, Mean: {total_mean:.2f} Mt, High: {total_high:.2f} Mt")
        
        return total_low, total_mean, total_high, totals / 1e6

    def process_country_biomass(self, biomass_type: str, year: str) -> Optional[Dict[str, Any]]:
        """
        Process country-level biomass data for a specific type and year.
        
        Args:
            biomass_type: Type of biomass (TBD, AGBD, BGBD)
            year: Target year
            
        Returns:
            Dictionary with country-level statistics or None if failed
        """
        self.logger.info(f"Processing {biomass_type} for year: {year}")
        
        # Build file paths
        mean_file = self.build_file_path(biomass_type, year, 'mean')
        uncertainty_file = self.build_file_path(biomass_type, year, 'uncertainty')
        
        # Check if files exist
        if not os.path.exists(mean_file):
            self.logger.error(f"Mean file not found: {mean_file}")
            return None
        
        if not os.path.exists(uncertainty_file):
            self.logger.error(f"Uncertainty file not found: {uncertainty_file}")
            return None
        
        # Get country boundary path
        country_bounds_path = SPAIN_BOUNDARIES
        
        # Load raster files with country boundary masking
        mean_data, transform, crs, bounds = self.load_raster(mean_file, country_bounds_path)
        uncertainty_data, _, _, _ = self.load_raster(uncertainty_file, country_bounds_path)
        
        # Check if data loaded successfully
        if mean_data is None or uncertainty_data is None:
            self.logger.error(f"Failed to load data for {biomass_type} {year}")
            return None
        
        # Run Monte Carlo simulation
        n_workers = None
        if self.config['monte_carlo']['parallel_processing']['enabled']:
            n_workers = self.config['monte_carlo']['parallel_processing']['num_workers']
        
        low, mean, high, all_samples = self.monte_carlo_uncertainty(
            mean_data, 
            uncertainty_data,
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

    def run_country_analysis(self, biomass_types: Optional[List[str]] = None, years: Optional[List[int]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Run complete country-level biomass analysis for multiple types and years.
        
        Args:
            biomass_types: List of biomass types to process (default from config)
            years: List of years to process (default from config)
            
        Returns:
            Tuple of (results_list, year_samples_dict)
        """
        # Use config defaults if not provided
        if biomass_types is None:
            biomass_types = self.config['file_patterns']['biomass_types']
        if years is None:
            years = self.config['analysis']['target_years']
        
        results = []
        year_samples = {}
        
        # Process each biomass type and year
        for biomass_type in biomass_types:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Processing biomass type: {biomass_type}")
            self.logger.info(f"{'='*60}")
            
            for year in years:
                result = self.process_country_biomass(biomass_type, str(year))
                
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
        
        return results, year_samples

    def save_results(self, results: List[Dict[str, Any]], year_samples: Dict[str, np.ndarray]) -> Tuple[str, str]:
        """
        Save analysis results to files.
        
        Args:
            results: List of analysis results
            year_samples: Dictionary of Monte Carlo samples
            
        Returns:
            Tuple of (summary_csv_path, samples_npz_path)
        """
        # Create output directory
        output_dir = ANALYSIS_OUTPUTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary results
        if results:
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(output_dir, f"country_biomass_timeseries_{timestamp}.csv")
            df.to_csv(summary_file, index=False)
            self.logger.info(f"Summary results saved to: {summary_file}")
        else:
            summary_file = None
            self.logger.warning("No results to save")
        
        # Save Monte Carlo samples if requested
        samples_file = None
        if self.config['output']['save_monte_carlo_samples'] and year_samples:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            samples_file = os.path.join(output_dir, f"country_biomass_mc_samples_{timestamp}.npz")
            np.savez_compressed(samples_file, **year_samples)
            self.logger.info(f"Monte Carlo samples saved to: {samples_file}")
        
        return summary_file, samples_file
