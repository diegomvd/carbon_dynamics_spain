"""
Monte Carlo simulation functions for biomass estimation with uncertainty quantification.

This module implements vectorized Monte Carlo simulations to estimate above-ground biomass (AGB),
below-ground biomass (BGB), and total biomass from canopy height data. Uses statistical sampling
to propagate uncertainty through allometric relationships and generate confidence intervals.

The implementation leverages xarray and dask for memory-efficient processing of large raster
datasets with distributed computing capabilities.

Author: Diego Bengochea
"""

import numpy as np
import dask.array as da
from scipy.stats import norm
import xarray as xr
import rioxarray as rxr
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Shared utilities
from shared_utils import get_logger, ensure_directory


class MonteCarloEstimator:
    """
    Monte Carlo estimator - RESTORED original logic without overengineering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Monte Carlo estimator."""
        self.config = config
        self.logger = get_logger('biomass_estimation.monte_carlo')
        
        # Monte Carlo parameters
        self.num_samples = config['monte_carlo']['num_samples']
        self.random_seed = config['monte_carlo']['random_seed']
        self.distribution_type = config['monte_carlo']['distribution_type']
        
        # Processing parameters
        self.height_threshold = config['processing']['height_threshold']
        
        self.logger.info(f"MonteCarloEstimator initialized with {self.num_samples} samples")
    
    def run(self, heights, agb_params, bgb_params):
        """
        Vectorized Monte Carlo simulation for biomass estimation with uncertainty quantification.
        
        Performs statistical sampling to propagate uncertainty through allometric relationships,
        generating mean biomass estimates and confidence intervals for AGB, BGB, and total biomass fro one forest type.
        Uses normal distribution approximations and vectorized operations for computational efficiency.
        
        Args:
            heights (xarray.DataArray): Canopy height data array with spatial coordinates
            agb_params (dict): Above-ground biomass allometry parameters containing:
                - 'median': (intercept, slope, function_type) for median allometry
                - 'p15': (intercept, slope, function_type) for 15th percentile
                - 'p85': (intercept, slope, function_type) for 85th percentile
            bgb_params (dict): Below-ground biomass ratio parameters containing:
                - 'mean': Mean BGB/AGB ratio
                - 'p5': 5th percentile of ratio
                - 'p95': 95th percentile of ratio
            num_samples (int): Number of Monte Carlo samples per pixel
            seed (int): Random seed for reproducibility
            
        Returns:
            xarray.Dataset: Dataset containing mean and uncertainty estimates for:
                - agbd_mean, agbd_uncertainty: Above-ground biomass
                - bgbd_mean, bgbd_uncertainty: Below-ground biomass  
                - total_mean, total_uncertainty: Total biomass
        """
        self.logger.info(f"Starting vectorized Monte Carlo simulation with {num_samples} samples")
        
        num_samples = self.num_samples
        seed = self.random_seed

        # Apply height filtering based on configuration
        self.logger.info("Masking heights")
        masked_heights = heights.where(heights > 0, np.nan)  # Remove non-positive heights
        
        # Apply minimum height threshold for biomass calculation
        masked_heights = masked_heights.where(
            masked_heights >= self.height_threshold, 0.0
        )
            
        masked_heights = masked_heights.persist()  # Materialize in distributed memory
        
        # Apply allometric relationships to generate AGB percentile curves
        self.logger.info("Calculating AGB allometry curves")
        
        # Median allometry curve
        intercept, slope, function_type = agb_params['median']
        if function_type == 'power':
            agbd_p50 = intercept * (masked_heights ** slope)
        else:
            agbd_p50 = xr.zeros_like(masked_heights)
        
        # 15th percentile allometry curve
        intercept, slope, function_type = agb_params['p15']
        if function_type == 'power':
            agbd_p15 = intercept * (masked_heights ** slope)
        else:
            agbd_p15 = xr.zeros_like(masked_heights)
        
        # 85th percentile allometry curve
        intercept, slope, function_type = agb_params['p85']
        if function_type == 'power':
            agbd_p85 = intercept * (masked_heights ** slope)
        else:
            agbd_p85 = xr.zeros_like(masked_heights)
        
        # Persist allometry results in distributed memory
        agbd_p50 = agbd_p50.persist()
        agbd_p15 = agbd_p15.persist()
        agbd_p85 = agbd_p85.persist()
        
        # Calculate AGB standard deviation from percentile bounds using normal approximation
        z_15 = norm.ppf(0.15)  # Z-score for 15th percentile
        z_85 = norm.ppf(0.85)  # Z-score for 85th percentile
        agb_sd = ((agbd_p50 - agbd_p15) / abs(z_15) + (agbd_p85 - agbd_p50) / z_85) / 2
        agb_sd = agb_sd.persist()
        
        # Calculate BGB ratio standard deviation from percentile bounds
        bgb_mean_val = bgb_params['mean']
        bgb_p5_val = bgb_params['p5']
        bgb_p95_val = bgb_params['p95']
        
        z_5 = norm.ppf(0.05)   # Z-score for 5th percentile
        z_95 = norm.ppf(0.95)  # Z-score for 95th percentile
        bgb_sd = ((bgb_mean_val - bgb_p5_val) / abs(z_5) + (bgb_p95_val - bgb_mean_val) / z_95) / 2
        
        # Set up sample dimension for vectorized Monte Carlo
        sample_dim = np.arange(num_samples)
        
        # Configure chunking for memory-efficient processing
        if hasattr(masked_heights, 'chunks'):
            y_chunks, x_chunks = masked_heights.chunks
        else:
            # Use default chunking if not available
            y_chunks, x_chunks = (1500, 1500)
        
        # Initialize results dataset
        results = xr.Dataset()
        
        self.logger.info("Generating vectorized Monte Carlo samples")
        
        # Initialize random number generators with different seeds for independence
        rs = da.random.RandomState(seed)
        rs_bgb = da.random.RandomState(seed + 10000)
        
        # Configure sample chunking to balance memory usage and computational efficiency
        sample_chunks = (min(25, num_samples), y_chunks[0], x_chunks[0])
        
        # Generate AGB noise samples and persist for reuse
        self.logger.info("Generating AGB noise")
        agb_noise = rs.normal(0, 1, (num_samples,) + masked_heights.shape, chunks=sample_chunks)
        agb_noise = agb_noise.persist()
        
        # Generate BGB noise samples and persist for reuse
        self.logger.info("Generating BGB noise")
        bgb_noise = rs_bgb.normal(0, 1, (num_samples,) + masked_heights.shape, chunks=sample_chunks)
        bgb_noise = bgb_noise.persist()
        
        # Broadcast mean and standard deviation for vectorized sampling
        self.logger.info("Broadcasting values for vectorized operations")
        agbd_p50_expanded = agbd_p50.expand_dims(sample=sample_dim)
        agb_sd_expanded = agb_sd.expand_dims(sample=sample_dim)
        
        # Generate AGB samples using normal distribution around allometric predictions
        self.logger.info("Calculating AGB samples")
        agbd_samples = agbd_p50_expanded + (agb_noise * agb_sd_expanded)
        agbd_samples = agbd_samples.persist()  # Materialize samples
        
        # Clean up intermediate arrays to free memory
        del agb_noise, agbd_p50_expanded, agb_sd_expanded
        gc.collect()
        
        # Generate BGB coefficient samples using normal distribution around ratio means
        self.logger.info("Calculating BGB coefficients")
        bgb_coef = xr.ones_like(masked_heights).expand_dims(sample=sample_dim) * bgb_mean_val + (bgb_noise * bgb_sd)
        bgb_coef = bgb_coef.persist()  # Materialize coefficients
        
        # Clean up BGB noise to free memory
        del bgb_noise
        gc.collect()
        
        # Calculate BGB samples by applying ratios to AGB samples
        self.logger.info("Calculating BGBD samples")
        bgbd_samples = agbd_samples * bgb_coef
        bgbd_samples = bgbd_samples.persist()  # Materialize samples
        
        # Calculate total biomass samples
        self.logger.info("Calculating total biomass samples")
        total_samples = agbd_samples + bgbd_samples
        total_samples = total_samples.persist()  # Materialize samples
        
        # Clean up coefficient array
        del bgb_coef
        gc.collect()
        
        # Calculate statistical summaries using efficient moment-based approach
        self.logger.info("Calculating means and standard deviations")
        
        # Compute mean values across Monte Carlo samples
        agbd_mean = agbd_samples.mean(dim='sample').persist()
        agbd_std = agbd_samples.std(dim='sample').persist()
        
        bgbd_mean = bgbd_samples.mean(dim='sample').persist()
        bgbd_std = bgbd_samples.std(dim='sample').persist()
        
        total_mean = total_samples.mean(dim='sample').persist()
        total_std = total_samples.std(dim='sample').persist()
        
        # Free memory from large sample arrays
        del agbd_samples, bgbd_samples, total_samples
        gc.collect()
        
        # Calculate uncertainty as half-width of 95% confidence interval using normal approximation
        z_95_ci = 1.96  # Z-score for 95% confidence interval (±1.96σ covers 95%)
        
        agbd_uncertainty = z_95_ci * agbd_std
        bgbd_uncertainty = z_95_ci * bgbd_std
        total_uncertainty = z_95_ci * total_std    
        
        # Store final results in structured dataset
        self.logger.info("Storing results")
        
        # Mean biomass estimates
        results['agbd_mean'] = agbd_mean
        results['bgbd_mean'] = bgbd_mean
        results['total_mean'] = total_mean
        
        # Uncertainty estimates (half-width of 95% confidence intervals)
        results['agbd_uncertainty'] = agbd_uncertainty
        results['bgbd_uncertainty'] = bgbd_uncertainty
        results['total_uncertainty'] = total_uncertainty

        # Clean up standard deviation arrays
        del agbd_std, bgbd_std, total_std
        gc.collect()
        
        self.logger.info("Monte Carlo simulation completed - results ready for incremental computation")
        return results