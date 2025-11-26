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
        Apply allometries directly to height data.
        
        Args:
            heights: Canopy height data
            agb_params: Dict with 'mean', 'p10', 'p90' allometry tuples
            
        Returns:
            Dataset with agbd_median and agbd_uncertainty
        """
        self.logger.info("Starting direct AGBD estimation")
        
        # Mask heights
        masked_heights = heights.where(heights >= self.height_threshold, 0.0)
        masked_heights = masked_heights.persist()

        # Apply allometries
        intercept_med, slope_med, _ = agb_params['mean']
        intercept_low, slope_low, _ = agb_params['p10']
        intercept_up, slope_up, _ = agb_params['p90']

        agbd_mean = intercept_med * (masked_heights ** slope_med)
        agbd_lower = intercept_low * (masked_heights ** slope_low)
        agbd_upper = intercept_up * (masked_heights ** slope_up)
        
        agbd_mean = agbd_mean.persist()
        agbd_lower = agbd_lower.persist()
        agbd_upper = agbd_upper.persist()
        
        # Compute symmetric uncertainty (upper - median)
        agbd_uncertainty = (agbd_upper - agbd_lower)*0.5
        agbd_uncertainty = agbd_uncertainty.persist()
        
        results = xr.Dataset({
            'agbd_mean': agbd_mean,
            'agbd_uncertainty': agbd_uncertainty
        })
        
        self.logger.info("Direct AGBD estimation completed")
        return results
