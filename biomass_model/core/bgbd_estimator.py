"""
BGBD Direct Estimation using Analytical Error Propagation

This module provides direct BGBD estimation by applying BGB/AGB ratios to AGBD data.
Uses analytical error propagation (no Monte Carlo sampling) for efficient processing.

Author: Diego Bengochea
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Tuple

# Shared utilities
from shared_utils import get_logger


class BGBDDirectEstimator:
    """
    Direct BGBD estimator using analytical error propagation.
    
    Applies BGB/AGB ratios to AGBD data to produce BGBD estimates with uncertainty.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BGBD direct estimator.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.bgbd_direct')
        
        self.logger.info("BGBDDirectEstimator initialized")
    
    def run(self, agbd_mean: xr.DataArray, agbd_uncertainty: xr.DataArray, 
            ratio_mean: float, ratio_uncertainty: float) -> xr.Dataset:
        """
        Apply ratio to AGBD data with analytical error propagation.
        
        For BGBD = ratio × AGBD:
        σ²_BGBD = AGBD² × σ²_ratio + ratio² × σ²_AGBD
        
        Args:
            agbd_mean: AGBD mean values
            agbd_uncertainty: AGBD uncertainty (80% CI half-width)
            ratio_mean: BGB/AGB ratio mean
            ratio_uncertainty: BGB/AGB ratio uncertainty (80% CI half-width)
            
        Returns:
            Dataset with bgbd_mean and bgbd_uncertainty
        """
        self.logger.info("Starting direct BGBD estimation")
        
        # Convert CI to standard deviations
        z_80 = 1.28
        sigma_agbd = agbd_uncertainty / z_80
        sigma_ratio = ratio_uncertainty / z_80
        
        # BGBD mean
        bgbd_mean = ratio_mean * agbd_mean
        bgbd_mean = bgbd_mean.persist()
        
        # BGBD variance using error propagation
        variance_bgbd = (agbd_mean ** 2) * (sigma_ratio ** 2) + (ratio_mean ** 2) * (sigma_agbd ** 2)
        sigma_bgbd = np.sqrt(variance_bgbd)
        
        # Convert back to 80% CI half-width
        bgbd_uncertainty = sigma_bgbd * z_80
        bgbd_uncertainty = bgbd_uncertainty.persist()
        
        results = xr.Dataset({
            'bgbd_mean': bgbd_mean,
            'bgbd_uncertainty': bgbd_uncertainty
        })
        
        self.logger.info("Direct BGBD estimation completed")
        return results


class TotalBiomassEstimator:
    """
    Direct Total biomass estimator using analytical error propagation.
    
    Calculates Total = AGBD + BGBD = AGBD × (1 + ratio).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Total biomass estimator."""
        self.config = config
        self.logger = get_logger('biomass_estimation.total_direct')
        self.logger.info("TotalBiomassEstimator initialized")
    
    def run(self, agbd_mean: xr.DataArray, agbd_uncertainty: xr.DataArray,
            ratio_mean: float, ratio_uncertainty: float) -> xr.Dataset:
        """
        Calculate Total biomass with analytical error propagation.
        
        For Total = AGBD × (1 + ratio):
        σ²_total = (1 + ratio)² × σ²_AGBD + AGBD² × σ²_ratio
        
        Returns:
            Dataset with total_mean and total_uncertainty
        """
        self.logger.info("Starting direct Total biomass estimation")
        
        # Convert CI to standard deviations
        z_80 = 1.28
        sigma_agbd = agbd_uncertainty / z_80
        sigma_ratio = ratio_uncertainty / z_80
        
        # Total mean
        total_mean = agbd_mean * (1 + ratio_mean)
        total_mean = total_mean.persist()
        
        # Total variance using error propagation
        variance_total = ((1 + ratio_mean) ** 2) * (sigma_agbd ** 2) + (agbd_mean ** 2) * (sigma_ratio ** 2)
        sigma_total = np.sqrt(variance_total)
        
        # Convert back to 80% CI half-width
        total_uncertainty = sigma_total * z_80
        total_uncertainty = total_uncertainty.persist()
        
        results = xr.Dataset({
            'total_mean': total_mean,
            'total_uncertainty': total_uncertainty
        })
        
        self.logger.info("Direct Total biomass estimation completed")
        return results