"""
AGBD Direct Estimation using Quantile Allometries

This module provides direct AGBD estimation by applying quantile regression
allometries (p15, median, p85) to height data without Monte Carlo sampling.
Produces mean and uncertainty bounds (70% confidence interval) efficiently.

Author: Diego Bengochea
"""

import numpy as np
import xarray as xr
from typing import Dict, Any

# Shared utilities
from shared_utils import get_logger


class AGBDDirectEstimator:
    """
    Direct AGBD estimator using quantile allometries without Monte Carlo.
    
    Applies three calibrated allometric relationships (15th, 50th, 85th percentiles)
    directly to height data to produce AGBD estimates with uncertainty bounds.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AGBD direct estimator.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.agbd_direct')
        
        # Processing parameters
        self.height_threshold = config['processing']['height_threshold']
        
        self.logger.info("AGBDDirectEstimator initialized")
    
    def run(self, heights: xr.DataArray, agb_params: Dict) -> xr.Dataset:
        """
        Apply quantile allometries directly to height data for AGBD estimation.
        
        Uses three calibrated allometric relationships to produce AGBD estimates:
        - Median (50th percentile) for mean estimate
        - 15th percentile for lower bound
        - 85th percentile for upper bound
        
        This provides a 70% confidence interval without Monte Carlo sampling.
        
        Args:
            heights: Canopy height data array with spatial coordinates
            agb_params: Above-ground biomass allometry parameters containing:
                - 'median': (intercept, slope, function_type) for median allometry
                - 'p15': (intercept, slope, function_type) for 15th percentile
                - 'p85': (intercept, slope, function_type) for 85th percentile
                
        Returns:
            xr.Dataset: Dataset containing:
                - agbd_mean: Mean AGBD estimate (from median allometry)
                - agbd_lower: Lower bound (from 15th percentile, 70% CI)
                - agbd_upper: Upper bound (from 85th percentile, 70% CI)
        """
        self.logger.info("Starting direct AGBD estimation with quantile allometries")
        
        # Apply height filtering based on configuration
        self.logger.info("Masking heights")
        masked_heights = heights.where(heights > 0, np.nan)  # Remove non-positive heights
        
        # Apply minimum height threshold for biomass calculation
        masked_heights = masked_heights.where(
            masked_heights >= self.height_threshold, 0.0
        )
        
        masked_heights = masked_heights.persist()  # Materialize in distributed memory
        
        # Initialize results dataset
        results = xr.Dataset()
        
        # Apply median allometry for mean estimate
        self.logger.info("Applying mean robust allometry")
        intercept, slope, function_type = agb_params['median']
        
        if function_type == 'power':
            agbd_mean = intercept * (masked_heights ** slope)
        else:
            self.logger.warning(f"Unknown function type: {function_type}, using zeros")
            agbd_mean = xr.zeros_like(masked_heights)
        
        agbd_mean = agbd_mean.persist()
        
        # Apply 15th percentile allometry for lower bound
        self.logger.info("Applying 15th percentile allometry (lower bound)")
        intercept, slope, function_type = agb_params['p15']
        
        if function_type == 'power':
            agbd_lower = intercept * (masked_heights ** slope)
        else:
            agbd_lower = xr.zeros_like(masked_heights)
        
        agbd_lower = agbd_lower.persist()
        
        # Apply 85th percentile allometry for upper bound
        self.logger.info("Applying 85th percentile allometry (upper bound)")
        intercept, slope, function_type = agb_params['p85']
        
        if function_type == 'power':
            agbd_upper = intercept * (masked_heights ** slope)
        else:
            agbd_upper = xr.zeros_like(masked_heights)
        
        agbd_upper = agbd_upper.persist()
        
        # Store results in dataset
        self.logger.info("Storing AGBD results")
        results['agbd_mean'] = agbd_mean
        results['agbd_lower'] = agbd_lower
        results['agbd_upper'] = agbd_upper
        
        self.logger.info("Direct AGBD estimation completed")
        return results