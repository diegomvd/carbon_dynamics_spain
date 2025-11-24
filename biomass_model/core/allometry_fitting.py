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
    create_training_dataset, create_output_directory, fit_theil_sen_allometry, fit_conformal_allometry
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
        self.config = load_config(config_path, component_name="biomass_model")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='allometry_fitting',
            log_file=self.config['logging'].get('log_file')
        )
        
        self.training_data = {}

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
            self.training_df = create_training_dataset(self.config)
            self.logger.info(f"Created training dataset with {len(self.training_df)} samples")

            allometry_df, ratio_df = self._process_hierarchical_allometries(
                self.training_df, forest_types_df, self.config
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
        logger = get_logger('biomass_estimation.allometry_fitting')

        try:
            # Log transform for power law fitting: biomass = a * height^b
            log_height = np.log(height)
            log_biomass = np.log(biomass)

            # Fit quantile regression on log-transformed data
            model = self._fit_quantile_regressor(log_height, log_biomass, quantile, alpha)
            
            # Extract parameters: log(biomass) = log(a) + b * log(height)
            intercept = model.intercept_
            slope = model.coef_[0]
            r2 = model.score(log_height.reshape(-1, 1), log_biomass)

            log_predictions = model.predict(log_height.reshape(-1, 1))
            predictions = np.exp(log_predictions)
            rmse = np.sqrt(np.mean((biomass - predictions)**2))
            
            return (slope, intercept, r2), rmse 
            
        except Exception as e:
            logger.error(f'Power law fit failed: {e}')
            # Return default values if fitting fails
            return (1.0, 1.0, 0.0), float('inf')

    def _fit_height_agb_allometry(
        self, 
        data: pd.DataFrame, 
        forest_type: str, 
        tier: int, 
        config: dict
    ) -> Optional[AllometryResults]:
        """
        Fit height-AGB allometry using Theil-Sen with conformal prediction intervals.
        
        Uses split-conformal prediction for guaranteed coverage bounds.
        No outlier removal needed - Theil-Sen is already robust.
        
        Args:
            data: Training data for this forest type
            forest_type: Forest type name
            tier: Hierarchy tier level
            config: Configuration dictionary
            
        Returns:
            AllometryResults object or None if fitting failed
        """
        logger = get_logger('biomass_estimation.allometry_fitting')
        
        try:
            # Get configuration parameters
            min_samples = config['fitting']['min_samples']
            max_height = config['fitting']['max_height']
            min_height = config['fitting']['min_height']
            coverage = config['fitting'].get('conformal_coverage', 0.70)
            cal_ratio = config['fitting'].get('conformal_cal_ratio', 0.30)
            
            # Validate input data
            valid_data = validate_height_biomass_data(
                data, 
                min_samples=min_samples, 
                max_height=max_height, 
                min_height=min_height
            )
            if not valid_data:
                logger.debug(f"Insufficient samples for {forest_type}")
                return None
            
            # Apply height filters
            height_filtered = data[
                (data['Height'] >= min_height) & 
                (data['Height'] <= max_height)
            ].copy()
            
            # Check we still have enough samples after filtering
            if len(height_filtered) < min_samples:
                logger.debug(
                    f"Insufficient samples after height filtering for {forest_type}: "
                    f"{len(height_filtered)} < {min_samples}"
                )
                return None
            
            # Extract height and AGB values
            height = height_filtered['Height'].values
            agb = height_filtered['AGB'].values
            
            # Fit using conformal prediction
            median_params, lower_params, upper_params, metrics = fit_conformal_allometry(
                heights=height,
                agbd=agb,
                coverage=coverage,
                cal_ratio=cal_ratio,
                random_state=42
            )
            
            median_intercept, median_slope = median_params
            lower_intercept, lower_slope = lower_params
            upper_intercept, upper_slope = upper_params
            
            # Calculate quality metrics on full dataset
            log_h = np.log(height)
            log_agb = np.log(agb)
            predictions = np.exp(median_intercept) * height ** median_slope
            
            # R² calculation
            ss_res = np.sum((agb - predictions) ** 2)
            ss_tot = np.sum((agb - np.mean(agb)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # RMSE calculation
            rmse = np.sqrt(np.mean((agb - predictions) ** 2))
            
            # Apply quality filters
            min_r2 = config['fitting']['min_r2']
            min_slope = config['fitting']['min_slope']
            max_slope = config['fitting'].get('max_slope', np.inf)
            
            if r2 < min_r2:
                logger.debug(
                    f"R² too low for {forest_type}: {r2:.3f} < {min_r2}"
                )
                return None
            
            if median_slope < min_slope:
                logger.debug(
                    f"Slope too low for {forest_type}: {median_slope:.3f} < {min_slope}"
                )
                return None
            
            if median_slope > max_slope:
                logger.debug(
                    f"Slope too high for {forest_type}: {median_slope:.3f} > {max_slope}"
                )
                return None
            
            # Create results object
            result = AllometryResults(
                forest_type=forest_type,
                tier=tier,
                n_samples=len(height_filtered),
                function_type='power',
                median_intercept=median_intercept,
                median_slope=median_slope,
                low_bound_intercept=lower_intercept,
                low_bound_slope=lower_slope,
                upper_bound_intercept=upper_intercept,
                upper_bound_slope=upper_slope,
                r2=r2,
                rmse=rmse
            )
            
            # Store data for validation plots
            self.training_data[forest_type] = {
                'height': height,
                'agb': agb,
                'tier': tier,
                'conformal_q': metrics['q'],
                'n_train': metrics['n_train'],
                'n_cal': metrics['n_cal']
            }
            
            logger.info(
                f"✓ Fitted conformal allometry for {forest_type}: "
                f"slope={median_slope:.3f}, R²={r2:.3f}, "
                f"conformal_q={metrics['q']:.3f}, coverage={coverage*100:.0f}%"
            )
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Failed to fit allometry for {forest_type} (tier {tier}): {str(e)}"
            )
            return None

    def _save_allometry_validation_plot(
        self, 
        forest_type: str, 
        allometry_result: AllometryResults,
        output_dir: Path
    ) -> None:
        """
        Save validation plot showing Theil-Sen fit with conformal prediction bounds.
        """
        import matplotlib.pyplot as plt
        
        # Get stored training data
        if forest_type not in self.training_data:
            self.logger.warning(f"No training data stored for {forest_type}, skipping plot")
            return
        
        data = self.training_data[forest_type]
        height = data['height']
        agb = data['agb']
        tier = data['tier']
        conformal_q = data['conformal_q']
        n_train = data['n_train']
        n_cal = data['n_cal']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data points
        ax.scatter(height, agb, alpha=0.5, s=30, color='gray', label='Training data')
        
        # Generate prediction curves
        h_range = np.linspace(height.min(), height.max(), 200)
        
        # Median (Theil-Sen)
        agb_median = np.exp(allometry_result.median_intercept) * h_range ** allometry_result.median_slope
        
        # Conformal bounds
        agb_lower = np.exp(allometry_result.low_bound_intercept) * h_range ** allometry_result.low_bound_slope
        agb_upper = np.exp(allometry_result.upper_bound_intercept) * h_range ** allometry_result.upper_bound_slope
        
        # Plot curves
        ax.plot(h_range, agb_median, 'b-', linewidth=2, label='Theil-Sen median')
        ax.plot(h_range, agb_lower, 'r--', linewidth=1.5, label=f'Conformal bounds (±{conformal_q:.2f} log-units)')
        ax.plot(h_range, agb_upper, 'r--', linewidth=1.5)
        ax.fill_between(h_range, agb_lower, agb_upper, alpha=0.2, color='red')
        
        # Labels and title
        ax.set_xlabel('Height (m)', fontsize=12)
        ax.set_ylabel('AGB (Mg/ha)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        title = (
            f'{forest_type} (Tier {tier})\n'
            f'n={allometry_result.n_samples} (train={n_train}, cal={n_cal}), '
            f'R²={allometry_result.r2:.3f}, RMSE={allometry_result.rmse:.1f} Mg/ha\n'
            f'Slope={allometry_result.median_slope:.3f}, Conformal q={conformal_q:.3f}'
        )
        ax.set_title(title, fontsize=11)
        
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save
        plot_dir = output_dir / 'validation_plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = forest_type.replace(' ', '_').replace('/', '_')
        plot_file = plot_dir / f'allometry_tier{tier}_{safe_name}.png'
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.debug(f"Saved validation plot: {plot_file}")

    def _save_and_log_metrics(self, allometry_df: pd.DataFrame, output_dir: Path):
        """
        Print and save allometry performance metrics.
        """
        logger = self.logger
        
        # Print to console/log
        logger.info("\n" + "="*60)
        logger.info("ALLOMETRY FITTING SUMMARY")
        logger.info("="*60)
        
        for tier in sorted(allometry_df['tier'].unique()):
            tier_data = allometry_df[allometry_df['tier'] == tier]
            logger.info(f"\nTier {tier}:")
            logger.info(f"  Relationships fitted: {len(tier_data)}")
            logger.info(f"  Mean R²: {tier_data['r2'].mean():.3f} (±{tier_data['r2'].std():.3f})")
            logger.info(f"  Mean RMSE: {tier_data['rmse'].mean():.1f} Mg/ha (±{tier_data['rmse'].std():.1f})")
            logger.info(f"  Mean slope: {tier_data['median_slope'].mean():.3f}")
            logger.info(f"  Mean samples: {tier_data['n_samples'].mean():.0f}")
        
        logger.info("\n" + "="*60)
        
        # Save to CSV
        summary = allometry_df.groupby('tier').agg({
            'r2': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'],
            'median_slope': ['mean', 'std'],
            'n_samples': ['mean', 'sum', 'min', 'max']
        }).round(3)
        
        summary.to_csv(output_dir / 'allometry_summary_by_tier.csv')
        logger.info(f"Saved summary to {output_dir / 'allometry_summary_by_tier.csv'}")

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
            valid_data = validate_ratio_data(data)
            if not valid_data:
                logger.info(f"Insufficient samples for BGB ratios {forest_type}")
                return None

            # Remove outliers if configured
            outlier_config = config.get('outlier_removal', {})
            if outlier_config.get('enabled', True):
                clean_data = remove_ratio_outliers(data)
            else:
                clean_data = data

            if len(clean_data) < min_samples:
                logger.info(f"Insufficient samples after outlier removal for BGB ratios {forest_type}: {len(clean_data)}")
                return None
            
            ratios = clean_data['BGB_Ratio']
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
                **ratio_percentiles
            )
            
            logger.debug(f"Successfully calculated BGB ratios for {forest_type}: "
                        f"mean={result.mean:.3f}")
            
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
        training_df = process_hierarchy_levels(training_df, forest_types_df)
        print(training_df)
        
        for tier_name, tier_level in HIERARCHY_LEVELS.items():
            logger.info(f"Processing tier {tier_level['tier']}: {tier_name}")

            if tier_name == 'General':
                # Fit height-AGB allometry                                                                                                                   
                try:
                    allometry_result = self._fit_height_agb_allometry(training_df, 'General', tier_level['tier'], config)
                    if allometry_result:
                        allometry_results.append(allometry_result)
                        logger.info(f"✓ Fitted allometry for General")
                except Exception as e:
                    logger.warning(f"Failed to fit allometry for General: {str(e)}")

                # Calculate BGB ratios                                                                                                                       
                try:
                    ratio_result = self._calculate_bgb_ratios(training_df, 'General', tier_level['tier'], config)
                    if ratio_result:
                        ratio_results.append(ratio_result)
                        logger.info(f"✓ Calculated BGB ratios for General")
                except Exception as e:
                    logger.warning(f"Failed to calculate BGB ratios for General: {str(e)}")

            elif tier_name not in forest_types_df.columns:
                logger.warning(f"Tier {tier_name} not found in forest types data")
                continue

            else:
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
                        allometry_result = self._fit_height_agb_allometry(type_data, forest_type, tier_level['tier'], config)
                        if allometry_result:
                            allometry_results.append(allometry_result)
                            logger.info(f"✓ Fitted allometry for {forest_type}")
                    except Exception as e:
                        logger.warning(f"Failed to fit allometry for {forest_type}: {str(e)}")
                        
                    # Calculate BGB ratios
                    try:
                        ratio_result = self._calculate_bgb_ratios(type_data, forest_type, tier_level['tier'], config)
                        if ratio_result:
                            ratio_results.append(ratio_result)
                            logger.info(f"✓ Calculated BGB ratios for {forest_type}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate BGB ratios for {forest_type}: {str(e)}")
                            

        # Convert results to DataFrames
        if not allometry_results:
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
        
        self.allometry_results = allometry_results
        self.ratio_results = ratio_results

        allometry_df = pd.DataFrame([vars(r) for r in allometry_results])
        ratio_df = pd.DataFrame([vars(r) for r in ratio_results])   

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

        # Save validation plots
        plots_dir = FITTED_PARAMETERS_FILE.parent / 'validation_plots'

        if hasattr(self, 'allometry_results') and self.allometry_results:
            logger.info("Saving validation plots...")
            output_dir = FITTED_PARAMETERS_FILE.parent
            
            for allometry_result in self.allometry_results:
                try:
                    self._save_allometry_validation_plot(
                        forest_type=allometry_result.forest_type,
                        allometry_result=allometry_result,
                        output_dir=output_dir
                    )
                except Exception as e:
                    logger.warning(f"Failed to save plot for {allometry_result.forest_type}: {e}")

        # Save and log metrics
        self._save_and_log_metrics(allometry_df, FITTED_PARAMETERS_FILE.parent) 

        return output_files
