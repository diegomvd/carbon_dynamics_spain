"""
Integrated allometry fitting pipeline for forest biomass estimation.

This module provides core fitting algorithms for height-biomass allometric relationships
and BGB-AGB ratios across hierarchical forest classifications. Processes training datasets
and fits both allometries and ratios hierarchically.

Features:
- Height-AGB allometry fitting using quantile regression  
- BGB-AGB ratio fitting using hierarchical percentile calculation
- Hierarchical forest type processing (General → Clade → Family → Genus → ForestType)
- Compatible outputs for existing biomass estimation pipeline

Ported and adapted from original fit_allometries.py to work with harmonized
path structure and current biomass_model component architecture.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import QuantileRegressor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Shared utilities
from shared_utils import get_logger
from shared_utils.central_data_paths_constants import *

# Component imports
from .allometry_utils import (
    AllometryResults, BGBRatioResults, HIERARCHY_LEVELS,
    validate_height_biomass_data, validate_ratio_data, prepare_ratio_data,
    remove_outliers, remove_ratio_outliers, process_hierarchy_levels,
    create_training_dataset, create_output_directory
)


def _fit_quantile_regressor(X: np.ndarray, y: np.ndarray, quantile: float, alpha: float) -> QuantileRegressor:
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


def _fit_power_law_quantile(height: np.ndarray, biomass: np.ndarray, quantile: float, alpha: float) -> tuple:
    """
    Fit power-law function for given quantile.
    
    Args:
        height (np.ndarray): Height measurements
        biomass (np.ndarray): Biomass measurements  
        quantile (float): Quantile level to fit
        alpha (float): Regularization parameter
        
    Returns:
        Tuple containing ((slope, intercept, r2), rmse)
    """
    # Transform to log space for power law: log(y) = log(a) + b*log(x)
    log_height = np.log(height)
    log_biomass = np.log(biomass)

    # Fit quantile regression
    model = _fit_quantile_regressor(log_height, log_biomass, quantile, alpha)
    intercept = model.intercept_
    slope = model.coef_[0]
    r2 = model.score(log_height.reshape(-1, 1), log_biomass)

    # Calculate RMSE in natural scale
    log_predictions = model.predict(log_height.reshape(-1, 1))
    predictions = np.exp(log_predictions)
    rmse = np.sqrt(np.mean((biomass - predictions)**2))
    
    return (slope, intercept, r2), rmse


def fit_height_agb_allometry(df: pd.DataFrame, height_col: str, biomass_col: str, 
                           forest_type: str, tier: int, config: dict) -> Optional[AllometryResults]:
    """
    Fit allometric relationship between height and above-ground biomass.
    
    Args:
        df (pd.DataFrame): Data with height and biomass measurements
        height_col (str): Name of height column
        biomass_col (str): Name of biomass column
        forest_type (str): Forest type identifier
        tier (int): Hierarchical tier level
        config (dict): Configuration dictionary with fitting parameters
        
    Returns:
        AllometryResults: Fitted allometry parameters or None if failed
    """
    logger = get_logger('biomass_estimation.allometry_fitting')
    height_agb_config = config['height_agb']
    quantiles = tuple(height_agb_config['quantiles'])
    min_samples = height_agb_config['min_samples']
    alpha = height_agb_config['alpha']
    max_height = height_agb_config['max_height']
    min_height = height_agb_config['min_height']
    
    if not validate_height_biomass_data(
        df, height_col, biomass_col, min_samples, max_height, min_height
    ):
        return None
        
    height = df[height_col].to_numpy()
    biomass = df[biomass_col].to_numpy()

    try:
        # Fit power law for median (0.5 quantile)
        median_params, rmse = _fit_power_law_quantile(height, biomass, 0.5, alpha)
        
        # Fit for confidence interval bounds
        lower_params, _ = _fit_power_law_quantile(height, biomass, quantiles[0], alpha)
        upper_params, _ = _fit_power_law_quantile(height, biomass, quantiles[1], alpha)

        return AllometryResults(
            forest_type=forest_type,
            tier=tier,
            n_samples=len(df),
            function_type='power',
            median_intercept=median_params[1],
            median_slope=median_params[0],
            low_bound_intercept=lower_params[1],
            low_bound_slope=lower_params[0],
            upper_bound_intercept=upper_params[1],
            upper_bound_slope=upper_params[0],
            r2=median_params[2],
            rmse=rmse
        )
        
    except Exception as e:
        logger.error(f"Failed to fit allometry for {forest_type}: {str(e)}")
        return None


def calculate_bgb_ratios(df: pd.DataFrame, forest_type: str, tier: int, config: dict) -> Optional[BGBRatioResults]:
    """
    Calculate BGB ratio statistics for a forest type using existing BGB_Ratio column.
    
    Args:
        df (pd.DataFrame): Data with existing BGB_Ratio column
        forest_type (str): Forest type identifier
        tier (int): Hierarchical tier level
        config (dict): Configuration dictionary with fitting parameters
        
    Returns:
        BGBRatioResults: Ratio statistics or None if failed
    """
    logger = get_logger('biomass_estimation.allometry_fitting')
    bgb_config = config['bgb_agb']
    min_samples = bgb_config['min_samples']
    percentiles = bgb_config['percentiles']
    
    # Validate existing ratio data
    if not validate_ratio_data(df, 'BGB_Ratio', min_samples):
        return None
    
    try:
        # Filter to valid existing ratios
        df_clean = prepare_ratio_data(df, 'BGB_Ratio')
        
        if len(df_clean) < min_samples:
            logger.warning(f"Insufficient clean samples for {forest_type}")
            return None
        
        # Get ratio values
        ratios = df_clean['BGB_Ratio'].values
        
        # Remove outliers
        ratios_clean = remove_ratio_outliers(ratios, contamination=0.12)
        
        if len(ratios_clean) < min_samples:
            logger.warning(f"Insufficient samples after outlier removal for {forest_type}")
            return None
        
        # Calculate percentiles
        mean_val = np.mean(ratios_clean)
        q05_val = np.percentile(ratios_clean, percentiles[0])
        q95_val = np.percentile(ratios_clean, percentiles[2])
        
        return BGBRatioResults(
            forest_type=forest_type,
            tier=tier,
            n_samples=len(ratios_clean),
            mean=mean_val,
            q05=q05_val,
            q95=q95_val
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate ratios for {forest_type}: {str(e)}")
        return None


def process_hierarchical_allometries(training_df: pd.DataFrame, forest_types_df: pd.DataFrame, 
                                    config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process both height-AGB allometries and BGB ratios hierarchically.
    
    Args:
        training_df (pd.DataFrame): Training dataset with height, AGB, BGB, forest types
        forest_types_df (pd.DataFrame): Forest types hierarchy table
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (allometry_results_df, bgb_ratio_results_df)
    """
    logger = get_logger('biomass_estimation.allometry_fitting')
    logger.info("Starting hierarchical allometry and ratio processing...")
    
    # Add hierarchy columns to training data
    training_df = process_hierarchy_levels(training_df, forest_types_df)
    
    # Initialize results lists
    allometry_results = []
    ratio_results = []
    
    # Process each hierarchy level
    for level_name, level_config in HIERARCHY_LEVELS.items():
        tier = level_config['tier']
        logger.info(f"Processing {level_name} (tier {tier})...")
        
        if level_name == 'General':
            # Process general allometry using all data
            df_subset = training_df.copy()
            
            # Remove outliers if enabled
            if config['outlier_removal']['enabled']:
                df_subset = remove_outliers(
                    df_subset, 
                    'Height', 
                    'AGB',
                    config['outlier_removal']['contamination']
                )
            
            # Fit height-AGB allometry
            allometry_result = fit_height_agb_allometry(
                df_subset, 'Height', 'AGB', 'General', tier, config
            )
            if allometry_result:
                allometry_results.append(allometry_result)
                logger.info(f"  ✓ General allometry fitted (n={allometry_result.n_samples}, R²={allometry_result.r2:.3f})")
            
            # Calculate BGB ratios (only for data with valid existing BGB_Ratio)
            if validate_ratio_data(df_subset, 'BGB_Ratio', config['bgb_agb']['min_samples']):
                ratio_result = calculate_bgb_ratios(df_subset, 'General', tier, config)
                if ratio_result:
                    ratio_results.append(ratio_result)
                    logger.info(f"  ✓ General BGB ratios calculated (n={ratio_result.n_samples})")
            else:
                logger.info(f"  ✗ Insufficient valid BGB ratios for General")
            continue
        
        # Process specific forest type groups
        column = level_config['column']
        for group in training_df[column].dropna().unique():
            logger.info(f"  Processing {level_name}: {group}")
            
            # Create subset for this group
            mask = training_df[column] == group
            df_subset = training_df[mask].copy()
            
            if len(df_subset) < config['height_agb']['min_samples']:
                logger.warning(f"    Insufficient samples ({len(df_subset)}) for {group}")
                continue
            
            # Remove outliers if enabled for height-AGB fitting
            df_allometry = df_subset.copy()
            if config['outlier_removal']['enabled']:
                try:
                    df_allometry = remove_outliers(
                        df_allometry, 
                        'Height', 
                        'AGB',
                        config['outlier_removal']['contamination']
                    )
                except Exception as e:
                    logger.warning(f"    Could not remove outliers for {group}: {str(e)}")
            
            # Fit height-AGB allometry
            allometry_result = fit_height_agb_allometry(
                df_allometry, 'Height', 'AGB', group, tier, config
            )
            
            if allometry_result:
                allometry_results.append(allometry_result)
                logger.info(f"    ✓ {group} allometry fitted (n={allometry_result.n_samples}, R²={allometry_result.r2:.3f})")
            else:
                logger.warning(f"    ✗ Failed to fit allometry for {group}")
            
            # Calculate BGB ratios (only for data with valid existing BGB_Ratio)
            if validate_ratio_data(df_subset, 'BGB_Ratio', config['bgb_agb']['min_samples']):
                ratio_result = calculate_bgb_ratios(df_subset, group, tier, config)
                if ratio_result:
                    ratio_results.append(ratio_result)
                    logger.info(f"    ✓ {group} BGB ratios calculated (n={ratio_result.n_samples})")
            else:
                logger.info(f"    ✗ Insufficient valid BGB ratios for {group}")
    
    # Convert results to DataFrames and verify independence
    if not allometry_results:
        logger.error("No valid allometry results obtained!")
        allometry_df = pd.DataFrame()
    else:
        allometry_df = pd.DataFrame([vars(r) for r in allometry_results])
        
        # Apply quality filters
        initial_count = len(allometry_df)
        quality_filters = config['quality_filters']
        allometry_df = allometry_df[
            (allometry_df['r2'] > quality_filters['min_r2']) & 
            (allometry_df['median_slope'] > quality_filters['min_slope'])
        ]
        logger.info(f"Applied quality filters to allometries: {len(allometry_df)}/{initial_count} retained")
    
    if not ratio_results:
        logger.error("No valid BGB ratio results obtained!")
        ratio_df = pd.DataFrame()
    else:
        ratio_df = pd.DataFrame([vars(r) for r in ratio_results])
        logger.info(f"Successfully calculated {len(ratio_df)} BGB ratio relationships")
    
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


def run_allometry_fitting_pipeline(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main allometry fitting pipeline execution.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (allometry_df, ratio_df) - Fitted allometry and BGB ratio results
        
    Raises:
        ValueError: If no valid training data found or fitting fails
        FileNotFoundError: If required input files not found
    """
    logger = get_logger('biomass_estimation.allometry_fitting')
    logger.info("Starting integrated allometry fitting pipeline...")
    
    try:
        # Load forest types hierarchy
        forest_types_file = FOREST_TYPES_TIERS_FILE
        if not forest_types_file.exists():
            raise FileNotFoundError(f"Forest types hierarchy file not found: {forest_types_file}")
        
        forest_types_df = pd.read_csv(forest_types_file)
        logger.info(f"Loaded forest types hierarchy with {len(forest_types_df)} entries")
        
        # Create training dataset by sampling height maps at NFI locations
        training_df = create_training_dataset(config)
        logger.info(f"Created training dataset with {len(training_df)} samples")
        
        # Process hierarchical allometries and ratios
        allometry_df, ratio_df = process_hierarchical_allometries(
            training_df, forest_types_df, config
        )
        
        # Validation
        if len(allometry_df) == 0 and len(ratio_df) == 0:
            raise ValueError("No valid allometries or BGB ratios could be fitted!")
        
        logger.info("Allometry fitting pipeline completed successfully")
        return allometry_df, ratio_df
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise


def save_allometry_results(allometry_df: pd.DataFrame, ratio_df: pd.DataFrame) -> Dict[str, Path]:
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
    
    # Save fitting summary
    summary_path = ALLOMETRIES_DIR / "fitting_summary.csv"
    summary_data = {
        'metric': ['n_allometries', 'n_bgb_ratios', 'mean_r2', 'mean_rmse', 'mean_bgb_ratio'],
        'value': [
            len(allometry_df),
            len(ratio_df), 
            allometry_df['r2'].mean() if len(allometry_df) > 0 else 0,
            allometry_df['rmse'].mean() if len(allometry_df) > 0 else 0,
            ratio_df['mean'].mean() if len(ratio_df) > 0 else 0
        ]
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    output_files['fitting_summary'] = summary_path
    logger.info(f"Fitting summary saved to {summary_path}")
    
    return output_files