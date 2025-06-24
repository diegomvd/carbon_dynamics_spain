"""
Allometry Fitting Pipeline

Complete allometry fitting pipeline class for harmonized component architecture.
Preserves exact algorithmic logic from original allometry_fitting module.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.linear_model import QuantileRegressor
import warnings

# Shared utilities
from shared_utils import setup_logging, load_config, get_logger
from shared_utils.central_data_paths_constants import *

# Component imports
from .allometry_utils import (
    AllometryResults, BGBRatioResults, HIERARCHY_LEVELS,
    validate_height_biomass_data, validate_ratio_data, prepare_ratio_data,
    remove_outliers, remove_ratio_outliers, process_hierarchy_levels,
    create_training_dataset, create_output_directory
)


class AllometryFittingPipeline:
    """
    Complete allometry fitting pipeline for harmonized component architecture.
    
    Wraps existing allometry fitting functions in class structure without
    changing any algorithmic logic.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the allometry fitting pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name="biomass_estimation")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='allometry_fitting',
            log_file=self.config['logging'].get('log_file')
        )
        
        self.logger.info("AllometryFittingPipeline initialized")
    
    def run_full_pipeline(self) -> bool:
        """
        Execute the complete allometry fitting pipeline.
        
        Returns:
            bool: True if fitting completed successfully
        """

        self.logger.info("Starting integrated allometry fitting pipeline...")

        try:

            # Load forest types hierarchy
            forest_types_file = FOREST_TYPES_TIERS_FILE
            if not forest_types_file.exists():
                raise FileNotFoundError(f"Forest types hierarchy file not found: {forest_types_file}")
            
            forest_types_df = pd.read_csv(forest_types_file)
            self.logger.info(f"Loaded forest types hierarchy with {len(forest_types_df)} entries")
            
            # Create training dataset by sampling height maps at NFI locations
            training_df = create_training_dataset(self.config)
            self.logger.info(f"Created training dataset with {len(training_df)} samples")

            allometry_df, ratio_df = self._process_hierarchical_allometries(
                training_df, forest_types_df, self.config
            )

            # Validation
            if len(allometry_df) == 0 and len(ratio_df) == 0:
                raise ValueError("No valid allometries or BGB ratios could be fitted!")
                        
            # Save results using existing function
            output_files = self._save_allometry_results(allometry_df, ratio_df)
            
            self.logger.info("Allometry fitting pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Allometry fitting pipeline failed: {str(e)}")
            return False

    def _fit_quantile_regressor(self, X: np.ndarray, y: np.ndarray, quantile: float, alpha: float) -> QuantileRegressor:
        """
        Fit quantile regression model.

        Args:
            X (np.ndarray): Log-transformed height data
            y (np.ndarray): Log-transformed biomass data
            quantile (float): Quantile level to fit
            alpha (float): Regularization parameter
            
        Returns:
            QuantileRegressor: Fitted quantile regression model
        """
        model = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver='highs',
            fit_intercept=True
        )
        model.fit(X.reshape(-1, 1), y)
        return model

    def _fit_power_law_quantile(self, height: np.ndarray, biomass: np.ndarray, quantile: float, alpha: float) -> tuple:
        """
        Fit power-law function for given quantile.

        Args:
            height (np.ndarray): Height data
            biomass (np.ndarray): Biomass data  
            quantile (float): Quantile level to fit
            alpha (float): Regularization parameter
            
        Returns:
            tuple: (a_param, b_param, r2, rmse) - Power law parameters and fit statistics
        """
        try:
            # Log transform for power law fitting: biomass = a * height^b
            log_height = np.log(height)
            log_biomass = np.log(biomass)
            
            # Fit quantile regression on log-transformed data
            model = self._fit_quantile_regressor(log_height, log_biomass, quantile, alpha)
            
            # Extract parameters: log(biomass) = log(a) + b * log(height)
            log_a = model.intercept_
            b_param = model.coef_[0]
            a_param = np.exp(log_a)
            
            # Calculate fit statistics
            log_biomass_pred = model.predict(log_height.reshape(-1, 1))
            
            # R-squared on log scale
            ss_res = np.sum((log_biomass - log_biomass_pred) ** 2)
            ss_tot = np.sum((log_biomass - np.mean(log_biomass)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # RMSE on log scale
            rmse = np.sqrt(np.mean((log_biomass - log_biomass_pred) ** 2))
            
            return a_param, b_param, r2, rmse
            
        except Exception as e:
            # Return default values if fitting fails
            return 1.0, 1.0, 0.0, float('inf')

    def _fit_height_agb_allometry(self, data: pd.DataFrame, forest_type: str, tier: int, config: dict) -> Optional[AllometryResults]:
        """
        Fit height-AGB allometry for specific forest type using quantile regression.

        Args:
            data (pd.DataFrame): Training data for this forest type
            forest_type (str): Forest type name
            tier (int): Hierarchy tier level
            config (dict): Configuration dictionary
            
        Returns:
            AllometryResults: Results object or None if fitting failed
        """
        logger = get_logger('biomass_estimation.allometry_fitting')
        
        try:
            # Get configuration parameters
            quantiles = config['fitting']['quantiles']
            alpha = config['fitting']['alpha']
            min_samples = config['fitting']['min_samples']
            max_height = config['fitting']['max_height']
            min_height = config['fitting']['min_height']
            
            # Validate input data
            valid_data = validate_height_biomass_data(data)
            if len(valid_data) < min_samples:
                logger.debug(f"Insufficient samples for {forest_type}: {len(valid_data)} < {min_samples}")
                return None
            
            # Apply height filters
            height_filtered = valid_data[
                (valid_data['height'] >= min_height) & 
                (valid_data['height'] <= max_height)
            ].copy()
            
            if len(height_filtered) < min_samples:
                logger.debug(f"Insufficient samples after height filtering for {forest_type}: {len(height_filtered)}")
                return None
            
            # Remove outliers if configured
            if config['fitting']['outlier_removal']:
                clean_data = remove_outliers(height_filtered, config['fitting']['outlier_contamination'])
            else:
                clean_data = height_filtered
            
            if len(clean_data) < min_samples:
                logger.debug(f"Insufficient samples after outlier removal for {forest_type}: {len(clean_data)}")
                return None
            
            # Extract height and AGB values
            height = clean_data['height'].values
            agb = clean_data['AGB'].values
            
            # Fit quantile regression models for specified quantiles
            fit_results = {}
            for q in quantiles:
                a_param, b_param, r2, rmse = self._fit_power_law_quantile(height, agb, q, alpha)
                fit_results[q] = {
                    'a_param': a_param,
                    'b_param': b_param,
                    'r2': r2,
                    'rmse': rmse
                }
            
            # Use median quantile (or middle quantile) as primary fit
            median_q = 0.5 if 0.5 in quantiles else quantiles[len(quantiles) // 2]
            if median_q not in fit_results:
                # Fit median separately if not in original quantiles
                a_param, b_param, r2, rmse = self._fit_power_law_quantile(height, agb, median_q, alpha)
                fit_results[median_q] = {
                    'a_param': a_param,
                    'b_param': b_param,
                    'r2': r2,
                    'rmse': rmse
                }
            
            primary_fit = fit_results[median_q]
            
            # Apply quality filters
            min_r2 = config['fitting']['min_r2']
            min_slope = config['fitting']['min_slope']
            
            if primary_fit['r2'] < min_r2:
                logger.debug(f"R² too low for {forest_type}: {primary_fit['r2']:.3f} < {min_r2}")
                return None
            
            if primary_fit['b_param'] < min_slope:
                logger.debug(f"Slope too low for {forest_type}: {primary_fit['b_param']:.3f} < {min_slope}")
                return None
            
            # Calculate parameter uncertainties (simplified approach)
            # In practice, this would use bootstrap or other methods
            a_std = 0.1 * primary_fit['a_param']  # 10% relative uncertainty
            b_std = 0.05  # Fixed uncertainty for exponent
            
            # Create results object
            result = AllometryResults(
                forest_type=forest_type,
                tier=tier,
                n_samples=len(clean_data),
                a_param=primary_fit['a_param'],
                b_param=primary_fit['b_param'],
                r2=primary_fit['r2'],
                rmse=primary_fit['rmse'],
                a_std=a_std,
                b_std=b_std
            )
            
            logger.debug(f"Successfully fitted allometry for {forest_type}: "
                        f"a={result.a_param:.3f}, b={result.b_param:.3f}, R²={result.r2:.3f}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to fit allometry for {forest_type} (tier {tier}): {str(e)}")
            return None

    def _calculate_bgb_ratios(self, data: pd.DataFrame, forest_type: str, tier: int, config: dict) -> Optional[BGBRatioResults]:
        """
        Calculate BGB-AGB ratios for specific forest type using percentile-based approach.

        Args:
            data (pd.DataFrame): Training data for this forest type
            forest_type (str): Forest type name  
            tier (int): Hierarchy tier level
            config (dict): Configuration dictionary
            
        Returns:
            BGBRatioResults: Results object or None if calculation failed
        """
        logger = get_logger('biomass_estimation.allometry_fitting')
        
        try:
            # Get configuration parameters
            min_samples = config['fitting']['min_bgb_samples']
            percentiles = config['fitting']['percentiles']
            
            # Validate and prepare ratio data
            ratio_data = validate_ratio_data(data)
            if len(ratio_data) < min_samples:
                logger.debug(f"Insufficient samples for BGB ratios {forest_type}: {len(ratio_data)} < {min_samples}")
                return None
            
            # Remove outliers if configured
            outlier_config = config.get('outlier_removal', {})
            if outlier_config.get('enabled', True):
                clean_data = remove_ratio_outliers(ratio_data)
            else:
                clean_data = ratio_data
            
            if len(clean_data) < min_samples:
                logger.debug(f"Insufficient samples after outlier removal for BGB ratios {forest_type}: {len(clean_data)}")
                return None
            
            # Calculate BGB/AGB ratios
            ratios = clean_data['BGB'] / clean_data['AGB']
            ratios = ratios.dropna()
            
            if len(ratios) < min_samples:
                logger.debug(f"Insufficient valid ratios for {forest_type}: {len(ratios)}")
                return None
            
            # Calculate statistics
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            
            # Calculate percentiles
            ratio_percentiles = {}
            for p in percentiles:
                ratio_percentiles[f'q{p:02d}'] = np.percentile(ratios, p)
            
            # Create results object
            result = BGBRatioResults(
                forest_type=forest_type,
                tier=tier,
                n_samples=len(clean_data),
                mean=mean_ratio,
                std=std_ratio,
                **ratio_percentiles
            )
            
            logger.debug(f"Successfully calculated BGB ratios for {forest_type}: "
                        f"mean={result.mean:.3f}, std={result.std:.3f}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to calculate BGB ratios for {forest_type} (tier {tier}): {str(e)}")
            return None

    def _process_hierarchical_allometries(self, training_df: pd.DataFrame, forest_types_df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process allometry fitting and BGB ratio calculation across forest type hierarchy.
        
        Args:
            training_df (pd.DataFrame): Training dataset with NFI and height data
            forest_types_df (pd.DataFrame): Forest type hierarchy mapping
            config (dict): Configuration dictionary
            
        Returns:
            tuple: (allometry_df, ratio_df) - DataFrames with fitted results
        """
        logger = get_logger('biomass_estimation.allometry_fitting')
        logger.info("Processing hierarchical allometries and BGB ratios...")
        
        # Initialize results lists
        allometry_results = []
        ratio_results = []
        
        # Process hierarchy levels in order (0=General, 1=Clade, 2=Family, 3=Genus, 4=ForestType)
        hierarchy_levels = process_hierarchy_levels()
        
        for tier_level, tier_name in hierarchy_levels.items():
            logger.info(f"Processing tier {tier_level}: {tier_name}")
            
            if tier_name not in forest_types_df.columns:
                logger.warning(f"Tier {tier_name} not found in forest types data")
                continue
            
            # Get unique forest types for this tier
            forest_types = forest_types_df[tier_name].dropna().unique()
            logger.info(f"Found {len(forest_types)} forest types in tier {tier_name}")
            
            # Process each forest type
            for forest_type in forest_types:
                if pd.isna(forest_type) or forest_type == '':
                    continue
                
                logger.debug(f"Processing {tier_name}: {forest_type}")
                
                # Filter training data for this forest type
                if tier_name in training_df.columns:
                    type_data = training_df[training_df[tier_name] == forest_type].copy()
                else:
                    logger.warning(f"Column {tier_name} not found in training data")
                    continue
                
                if len(type_data) == 0:
                    logger.debug(f"No training data for {tier_name}: {forest_type}")
                    continue
                
                # Fit height-AGB allometry
                try:
                    allometry_result = self._fit_height_agb_allometry(type_data, forest_type, tier_level, config)
                    if allometry_result:
                        allometry_results.append(allometry_result)
                        logger.debug(f"✓ Fitted allometry for {forest_type}")
                except Exception as e:
                    logger.warning(f"Failed to fit allometry for {forest_type}: {str(e)}")
                
                # Calculate BGB ratios
                try:
                    ratio_result = self._calculate_bgb_ratios(type_data, forest_type, tier_level, config)
                    if ratio_result:
                        ratio_results.append(ratio_result)
                        logger.debug(f"✓ Calculated BGB ratios for {forest_type}")
                except Exception as e:
                    logger.warning(f"Failed to calculate BGB ratios for {forest_type}: {str(e)}")
        
        # Convert results to DataFrames
        if allometry_results:
            allometry_df = pd.DataFrame()
            ratio_df = pd.DataFrame()
        else:
            allometry_df = pd.DataFrame([vars(r) for r in allometry_results])
            logger.info(f"Successfully fitted {len(allometry_df)} height-AGB allometries")
        
        if ratio_results:
            ratio_df = pd.DataFrame([vars(r) for r in ratio_results])
            logger.info(f"Successfully calculated {len(ratio_df)} BGB ratio relationships")
        else:
            ratio_df = pd.DataFrame()
        
        # VERIFICATION: Check independence of results
        logger.info("\n" + "="*60)
        logger.info("INDEPENDENCE VERIFICATION:")
        logger.info("="*60)
        
        if len(allometry_df) > 0:
            allometry_types = set(allometry_df['forest_type'])
            logger.info(f"Height-AGB allometries fitted for {len(allometry_types)} forest types:")
            for ftype in sorted(allometry_types):
                logger.info(f"  - {ftype}")
        else:
            allometry_types = set()
            logger.info("No height-AGB allometries fitted")
        
        if len(ratio_df) > 0:
            ratio_types = set(ratio_df['forest_type'])
            logger.info(f"BGB ratios calculated for {len(ratio_types)} forest types:")
            for ftype in sorted(ratio_types):
                logger.info(f"  - {ftype}")
        else:
            ratio_types = set()
            logger.info("No BGB ratios calculated")
        
        # Check for differences (proving independence)
        only_allometry = allometry_types - ratio_types
        only_ratios = ratio_types - allometry_types
        both = allometry_types & ratio_types
        
        logger.info(f"\nIndependence verification:")
        logger.info(f"  Both allometry & ratios: {len(both)} forest types")
        logger.info(f"  Only allometry fitted: {len(only_allometry)} forest types")
        logger.info(f"  Only ratios calculated: {len(only_ratios)} forest types")
        
        if only_allometry:
            logger.info(f"  Forest types with only allometry: {sorted(only_allometry)}")
        if only_ratios:
            logger.info(f"  Forest types with only ratios: {sorted(only_ratios)}")
        
        logger.info("="*60)
        
        return allometry_df, ratio_df

    def _save_allometry_results(self, allometry_df: pd.DataFrame, ratio_df: pd.DataFrame) -> Dict[str, Path]:
        """
        Save allometry fitting results to harmonized output locations.

        Args:
            allometry_df (pd.DataFrame): Height-AGB allometry results
            ratio_df (pd.DataFrame): BGB ratio results  
            
        Returns:
            Dict[str, Path]: Dictionary of output file paths
        """
        logger = get_logger('biomass_estimation.allometry_fitting')
        
        output_files = {}
        
        # Save allometry results
        if len(allometry_df) > 0:
            allometry_path = FITTED_PARAMETERS_FILE
            create_output_directory(allometry_path)
            allometry_df.to_csv(allometry_path, index=False)
            output_files['fitted_parameters'] = allometry_path
            logger.info(f"Height-AGB allometries saved to {allometry_path}")
            
            # Log summary statistics
            logger.info("Height-AGB allometry summary:")
            logger.info(f"  Fitted relationships: {len(allometry_df)}")
            logger.info(f"  Mean R²: {allometry_df['r2'].mean():.3f}")
            logger.info(f"  Mean RMSE: {allometry_df['rmse'].mean():.3f}")
        else:
            logger.error("No valid height-AGB allometries to save!")
        
        # Save BGB ratio results
        if len(ratio_df) > 0:
            ratio_path = BGB_RATIOS_FILE
            create_output_directory(ratio_path)
            ratio_df.to_csv(ratio_path, index=False)
            output_files['bgb_ratios'] = ratio_path
            logger.info(f"BGB ratios saved to {ratio_path}")
            
            # Log summary statistics
            logger.info("BGB ratio summary:")
            logger.info(f"  Calculated ratios: {len(ratio_df)}")
            logger.info(f"  Mean ratio: {ratio_df['mean'].mean():.3f}")
        else:
            logger.error("No valid BGB ratios to save!")
        
        # Create summary file
        if output_files:
            summary_path = FITTING_SUMMARY_FILE
            create_output_directory(summary_path)
            
            summary_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'allometry_relationships': len(allometry_df),
                'bgb_ratios': len(ratio_df),
                'files_created': list(output_files.keys())
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(summary_path, index=False)
            output_files['summary'] = summary_path
            logger.info(f"Fitting summary saved to {summary_path}")
        
        return output_files