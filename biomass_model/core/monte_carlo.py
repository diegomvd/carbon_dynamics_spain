"""
Monte Carlo Uncertainty Quantification

This module implements Monte Carlo estimation for biomass uncertainty
quantification using statistical sampling of allometric parameters.

Author: Diego Bengochea
"""

import numpy as np
import xarray as xr
import rasterio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time

# Shared utilities
from shared_utils import get_logger, ensure_directory

# Component imports
from .io_utils import RasterManager


class MonteCarloEstimator:
    """
    Monte Carlo estimator for biomass uncertainty quantification.
    
    Implements statistical sampling of allometric parameters to generate
    biomass estimates with uncertainty bounds.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Monte Carlo estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.monte_carlo')
        
        # Monte Carlo parameters
        self.num_samples = config['monte_carlo']['num_samples']
        self.random_seed = config['monte_carlo']['random_seed']
        self.distribution_type = config['monte_carlo']['distribution_type']
        
        # Processing parameters
        self.height_threshold = config['processing']['height_threshold']
        
        # Initialize random state
        self.rng = np.random.RandomState(self.random_seed)
        
        # Initialize raster manager
        self.raster_manager = RasterManager(config)
        
        self.logger.info(f"MonteCarloEstimator initialized with {self.num_samples} samples")
    
    def estimate_biomass(
        self,
        height_files: List[Path],
        mask_file: Path,
        allometry_params: Dict[str, Any],
        biomass_type: str,
        measure: str,
        output_file: Path
    ) -> bool:
        """
        Estimate biomass with uncertainty using Monte Carlo sampling.
        
        Args:
            height_files: List of height raster files
            mask_file: Forest type mask file
            allometry_params: Allometric parameters dictionary
            biomass_type: Type of biomass ('agbd', 'bgbd', 'total')
            measure: Statistical measure ('mean', 'uncertainty')
            output_file: Output file path
            
        Returns:
            bool: True if estimation succeeded
        """
        try:
            self.logger.info(f"Starting Monte Carlo estimation: {biomass_type} {measure}")
            start_time = time.time()
            
            # Load and prepare input data
            height_data, mask_data, profile = self._load_input_data(height_files, mask_file)
            
            if height_data is None:
                self.logger.error("Failed to load input data")
                return False
            
            # Generate parameter samples
            param_samples = self._generate_parameter_samples(allometry_params, biomass_type)
            
            if param_samples is None:
                self.logger.error("Failed to generate parameter samples")
                return False
            
            # Run Monte Carlo estimation
            result = self._run_monte_carlo_estimation(
                height_data, mask_data, param_samples, biomass_type, measure
            )
            
            if result is None:
                self.logger.error("Monte Carlo estimation failed")
                return False
            
            # Save result
            success = self._save_result(result, profile, output_file)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Monte Carlo estimation completed in {elapsed_time:.2f}s")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo estimation: {str(e)}")
            return False
    
    def _load_input_data(
        self, 
        height_files: List[Path], 
        mask_file: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Load and prepare height and mask data.
        
        Args:
            height_files: Height raster files
            mask_file: Mask file
            
        Returns:
            Tuple of (height_data, mask_data, raster_profile)
        """
        try:
            # Load height data (use first file for now, could be mosaicked)
            if not height_files:
                self.logger.error("No height files provided")
                return None, None, None
            
            height_file = height_files[0]  # TODO: Handle multiple files
            
            with rasterio.open(height_file) as src:
                height_data = src.read(1).astype(np.float32)
                height_profile = src.profile.copy()
            
            # Load mask data
            with rasterio.open(mask_file) as src:
                mask_data = src.read(1).astype(np.uint8)
                mask_profile = src.profile.copy()
            
            # Validate dimensions match
            if height_data.shape != mask_data.shape:
                self.logger.error(f"Shape mismatch: height {height_data.shape}, mask {mask_data.shape}")
                return None, None, None
            
            # Apply height threshold
            height_data[height_data < self.height_threshold] = np.nan
            
            # Apply forest type mask
            height_data[mask_data == 0] = np.nan
            
            # Update profile for output
            output_profile = height_profile.copy()
            output_profile.update({
                'dtype': 'float32',
                'nodata': self.config['output']['geotiff']['nodata_value'],
                'compress': self.config['output']['geotiff']['compress'],
                'tiled': self.config['output']['geotiff']['tiled'],
                'blockxsize': self.config['output']['geotiff']['blockxsize'],
                'blockysize': self.config['output']['geotiff']['blockysize']
            })
            
            self.logger.debug(f"Loaded data: {height_data.shape}, valid pixels: {np.sum(~np.isnan(height_data))}")
            
            return height_data, mask_data, output_profile
            
        except Exception as e:
            self.logger.error(f"Error loading input data: {str(e)}")
            return None, None, None
    
    def _generate_parameter_samples(
        self, 
        allometry_params: Dict[str, Any], 
        biomass_type: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate Monte Carlo samples of allometric parameters.
        
        Args:
            allometry_params: Allometric parameters
            biomass_type: Biomass type
            
        Returns:
            Dictionary with parameter samples
        """
        try:
            samples = {}
            
            # AGB parameters (always needed)
            agb_params = allometry_params.get('agb_params')
            if agb_params:
                samples['a_samples'] = self._sample_parameter(
                    agb_params['a_mean'], 
                    agb_params['a_std']
                )
                samples['b_samples'] = self._sample_parameter(
                    agb_params['b_mean'], 
                    agb_params['b_std']
                )
            else:
                self.logger.error("No AGB parameters available")
                return None
            
            # BGB parameters (for bgbd and total biomass)
            if biomass_type in ['bgbd', 'total']:
                bgb_params = allometry_params.get('bgb_params')
                if bgb_params:
                    samples['ratio_samples'] = self._sample_parameter(
                        bgb_params['ratio_mean'],
                        bgb_params['ratio_std']
                    )
                else:
                    # Use default ratio if no BGB parameters
                    self.logger.warning("No BGB parameters, using default ratio")
                    samples['ratio_samples'] = np.full(self.num_samples, 0.25)  # Default 25%
            
            self.logger.debug(f"Generated {self.num_samples} parameter samples for {biomass_type}")
            return samples
            
        except Exception as e:
            self.logger.error(f"Error generating parameter samples: {str(e)}")
            return None
    
    def _sample_parameter(self, mean: float, std: float) -> np.ndarray:
        """
        Sample parameter from statistical distribution.
        
        Args:
            mean: Parameter mean
            std: Parameter standard deviation
            
        Returns:
            Array of parameter samples
        """
        if self.distribution_type == 'normal':
            return self.rng.normal(mean, std, self.num_samples)
        elif self.distribution_type == 'lognormal':
            # Convert to log-normal parameters
            log_mean = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            log_std = np.sqrt(np.log(1 + (std**2 / mean**2)))
            return self.rng.lognormal(log_mean, log_std, self.num_samples)
        else:
            self.logger.warning(f"Unknown distribution type: {self.distribution_type}, using normal")
            return self.rng.normal(mean, std, self.num_samples)
    
    def _run_monte_carlo_estimation(
        self,
        height_data: np.ndarray,
        mask_data: np.ndarray,
        param_samples: Dict[str, np.ndarray],
        biomass_type: str,
        measure: str
    ) -> Optional[np.ndarray]:
        """
        Run Monte Carlo estimation over all parameter samples.
        
        Args:
            height_data: Height raster data
            mask_data: Mask raster data  
            param_samples: Parameter samples
            biomass_type: Biomass type
            measure: Statistical measure
            
        Returns:
            Result array or None if failed
        """
        try:
            # Initialize storage for all samples
            valid_mask = ~np.isnan(height_data)
            n_valid_pixels = np.sum(valid_mask)
            
            if n_valid_pixels == 0:
                self.logger.warning("No valid pixels for estimation")
                return np.full(height_data.shape, self.config['output']['geotiff']['nodata_value'], dtype=np.float32)
            
            # Storage for Monte Carlo samples
            mc_samples = np.zeros((self.num_samples, *height_data.shape), dtype=np.float32)
            
            self.logger.info(f"Running {self.num_samples} Monte Carlo iterations...")
            
            # Run Monte Carlo iterations
            for i in range(self.num_samples):
                if i % 50 == 0:  # Progress logging
                    self.logger.debug(f"Monte Carlo iteration {i+1}/{self.num_samples}")
                
                # Get parameters for this iteration
                a_param = param_samples['a_samples'][i]
                b_param = param_samples['b_samples'][i]
                
                # Calculate AGB using allometric equation: AGB = a * H^b
                agb = np.zeros_like(height_data)
                agb[valid_mask] = a_param * (height_data[valid_mask] ** b_param)
                
                # Calculate final biomass based on type
                if biomass_type == 'agbd':
                    biomass = agb
                elif biomass_type == 'bgbd':
                    ratio = param_samples['ratio_samples'][i]
                    biomass = agb * ratio
                elif biomass_type == 'total':
                    ratio = param_samples['ratio_samples'][i]
                    biomass = agb * (1 + ratio)  # AGB + BGB
                else:
                    raise ValueError(f"Unknown biomass type: {biomass_type}")
                
                # Store sample
                mc_samples[i] = biomass
            
            # Calculate requested statistical measure
            if measure == 'mean':
                result = np.mean(mc_samples, axis=0)
            elif measure == 'uncertainty':
                result = np.std(mc_samples, axis=0)
            elif measure == 'median':
                result = np.median(mc_samples, axis=0)
            elif measure.startswith('percentile_'):
                percentile = float(measure.split('_')[1])
                result = np.percentile(mc_samples, percentile, axis=0)
            else:
                raise ValueError(f"Unknown measure: {measure}")
            
            # Set no-data values
            result[~valid_mask] = self.config['output']['geotiff']['nodata_value']
            
            self.logger.info(f"Monte Carlo estimation complete. Valid pixels: {n_valid_pixels}")
            return result.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo estimation: {str(e)}")
            return None
    
    def _save_result(
        self, 
        result: np.ndarray, 
        profile: Dict, 
        output_file: Path
    ) -> bool:
        """
        Save estimation result to raster file.
        
        Args:
            result: Result array
            profile: Raster profile
            output_file: Output file path
            
        Returns:
            bool: True if save succeeded
        """
        try:
            # Ensure output directory exists
            ensure_directory(output_file.parent)
            
            # Write result
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(result, 1)
                
                # Add metadata
                dst.update_tags(
                    MONTE_CARLO_SAMPLES=str(self.num_samples),
                    RANDOM_SEED=str(self.random_seed),
                    DISTRIBUTION_TYPE=self.distribution_type
                )
            
            self.logger.info(f"Result saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving result: {str(e)}")
            return False
    
    def estimate_single_pixel(
        self,
        height_value: float,
        allometry_params: Dict[str, Any],
        biomass_type: str = 'total'
    ) -> Dict[str, float]:
        """
        Estimate biomass for a single pixel (useful for testing/validation).
        
        Args:
            height_value: Canopy height value
            allometry_params: Allometric parameters
            biomass_type: Biomass type
            
        Returns:
            Dictionary with statistics
        """
        try:
            if height_value < self.height_threshold or np.isnan(height_value):
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            
            # Generate parameter samples
            param_samples = self._generate_parameter_samples(allometry_params, biomass_type)
            
            if param_samples is None:
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            
            # Calculate biomass for each sample
            biomass_samples = []
            
            for i in range(self.num_samples):
                a_param = param_samples['a_samples'][i]
                b_param = param_samples['b_samples'][i]
                
                # AGB calculation
                agb = a_param * (height_value ** b_param)
                
                # Final biomass
                if biomass_type == 'agbd':
                    biomass = agb
                elif biomass_type == 'bgbd':
                    ratio = param_samples['ratio_samples'][i]
                    biomass = agb * ratio
                elif biomass_type == 'total':
                    ratio = param_samples['ratio_samples'][i]
                    biomass = agb * (1 + ratio)
                
                biomass_samples.append(biomass)
            
            # Calculate statistics
            biomass_samples = np.array(biomass_samples)
            
            return {
                'mean': float(np.mean(biomass_samples)),
                'std': float(np.std(biomass_samples)),
                'median': float(np.median(biomass_samples)),
                'percentile_5': float(np.percentile(biomass_samples, 5)),
                'percentile_95': float(np.percentile(biomass_samples, 95))
            }
            
        except Exception as e:
            self.logger.error(f"Error in single pixel estimation: {str(e)}")
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
    
    def validate_parameters(self, allometry_params: Dict[str, Any]) -> bool:
        """
        Validate allometric parameters for Monte Carlo estimation.
        
        Args:
            allometry_params: Parameters to validate
            
        Returns:
            bool: True if parameters are valid
        """
        try:
            # Check AGB parameters
            agb_params = allometry_params.get('agb_params')
            if not agb_params:
                self.logger.error("Missing AGB parameters")
                return False
            
            required_agb = ['a_mean', 'b_mean', 'a_std', 'b_std']
            for param in required_agb:
                if param not in agb_params or pd.isna(agb_params[param]):
                    self.logger.error(f"Missing or invalid AGB parameter: {param}")
                    return False
            
            # Check parameter ranges
            if agb_params['a_mean'] <= 0 or agb_params['b_mean'] <= 0:
                self.logger.error("AGB parameters must be positive")
                return False
            
            if agb_params['a_std'] < 0 or agb_params['b_std'] < 0:
                self.logger.error("Standard deviations must be non-negative")
                return False
            
            # Check BGB parameters if present
            bgb_params = allometry_params.get('bgb_params')
            if bgb_params:
                if 'ratio_mean' not in bgb_params or pd.isna(bgb_params['ratio_mean']):
                    self.logger.error("Invalid BGB ratio parameter")
                    return False
                
                if bgb_params['ratio_mean'] < 0:
                    self.logger.error("BGB ratio must be non-negative")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {str(e)}")
            return False
