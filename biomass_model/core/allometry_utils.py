"""
Utility functions for biomass allometry fitting and processing.

This module provides essential utilities for data validation, outlier removal,
hierarchical forest type processing, and spatial sampling for the integrated
biomass estimation and allometry fitting pipeline.

Ported and adapted from original fitting_utils.py to work with harmonized
path structure and current biomass_model component architecture.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
import logging
import yaml
import geopandas as gpd
import rasterio
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple, List
from sklearn.covariance import EllipticEnvelope
import warnings

# Shared utilities
from shared_utils import get_logger, setup_logging, load_config
from shared_utils.central_data_paths_constants import *

@dataclass
class AllometryResults:
    """Container for allometry regression results."""
    forest_type: str
    tier: int
    n_samples: int
    function_type: Literal['power']
    median_intercept: float
    median_slope: float
    low_bound_intercept: float
    low_bound_slope: float
    upper_bound_intercept: float
    upper_bound_slope: float
    r2: float
    rmse: float


@dataclass
class BGBRatioResults:
    """Container for BGB ratio statistical results."""
    forest_type: str
    tier: int
    n_samples: int
    mean: float
    q05: float
    q95: float


def validate_height_biomass_data(df: pd.DataFrame, height_col: str, biomass_col: str, 
                                min_samples: int, max_height: float, min_height: float) -> bool:
    """
    Validate input data for height-biomass allometry fitting.
    
    Args:
        df (pd.DataFrame): Input dataframe
        height_col (str): Name of height column
        biomass_col (str): Name of biomass column
        min_samples (int): Minimum required samples
        max_height (float): Maximum allowed height
        min_height (float): Minimum allowed height
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if len(df) < min_samples:
        return False
        
    # Check for non-positive values
    if df[height_col].min() <= 0 or df[biomass_col].min() <= 0:
        warnings.warn("Found non-positive values in data")
        return False

    # Check height constraints
    valid_heights = (df[height_col] >= min_height) & (df[height_col] <= max_height)
    if not valid_heights.all():
        n_filtered = (~valid_heights).sum()
        warnings.warn(f"Would remove {n_filtered} points with invalid heights")
        return len(df[valid_heights]) >= min_samples
    
    return True


def validate_ratio_data(df: pd.DataFrame, ratio_col: str = 'BGB_Ratio', min_samples: int = 10) -> bool:
    """
    Validate input data for BGB ratio calculation using existing ratio column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        ratio_col (str): Name of existing ratio column (default: 'BGB_Ratio')
        min_samples (int): Minimum required samples
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if len(df) < min_samples:
        return False
    
    # Check if ratio column exists
    if ratio_col not in df.columns:
        warnings.warn(f"Column {ratio_col} not found in dataframe")
        return False
    
    # Check for valid ratio values (> 0, not NaN, finite)
    valid_ratios = (
        df[ratio_col].notna() & 
        np.isfinite(df[ratio_col]) &
        (df[ratio_col] > 0)
    )
    
    n_valid = valid_ratios.sum()
    
    if n_valid < min_samples:
        warnings.warn(f"Insufficient valid ratios ({n_valid}) for calculation")
        return False
    
    return True


def prepare_ratio_data(df: pd.DataFrame, ratio_col: str = 'BGB_Ratio') -> pd.DataFrame:
    """
    Prepare BGB ratio data by filtering valid existing ratio values.
    
    Args:
        df (pd.DataFrame): Input dataframe with existing BGB_Ratio column
        ratio_col (str): Name of ratio column (default: 'BGB_Ratio')
        
    Returns:
        pd.DataFrame: Dataframe with valid ratios only
    """
    # Filter to valid ratio values (> 0, not NaN, finite)
    valid_mask = (
        df[ratio_col].notna() & 
        np.isfinite(df[ratio_col]) &
        (df[ratio_col] > 0)
    )
    
    df_valid = df[valid_mask].copy()
    
    return df_valid


def remove_outliers(df: pd.DataFrame, height_col: str, biomass_col: str, 
                   contamination: float) -> pd.DataFrame:
    """
    Remove outliers using Elliptic Envelope method for height-biomass data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        height_col (str): Name of height column
        biomass_col (str): Name of biomass column
        contamination (float): Proportion of outliers expected
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    initial_count = len(df)
    
    # Prepare data for outlier detection (log-transformed)
    X = df[[height_col, biomass_col]].copy()
    X.loc[:, height_col] = X[height_col].apply(lambda x: np.log(x))
    X.loc[:, biomass_col] = X[biomass_col].apply(lambda x: np.log(x))
    
    # Fit elliptic envelope and predict outliers
    envelope = EllipticEnvelope(contamination=contamination)
    labels = envelope.fit_predict(X)
    
    # Filter out outliers (label == -1 indicates outlier)
    df_clean = df[labels == 1].copy()
    
    n_removed = initial_count - len(df_clean)
    logger.debug(f"Removed {n_removed}/{initial_count} outliers ({n_removed/initial_count*100:.1f}%)")
    
    return df_clean


def remove_ratio_outliers(ratios: np.ndarray, contamination: float) -> np.ndarray:
    """
    Remove outliers from BGB ratio data using IQR method.
    
    Args:
        ratios (np.ndarray): BGB ratio values
        contamination (float): Expected proportion of outliers (not used for IQR)
        
    Returns:
        np.ndarray: Ratios with outliers removed
    """
    Q1 = np.percentile(ratios, 25)
    Q3 = np.percentile(ratios, 75)
    IQR = Q3 - Q1
    multiplier = 1.5  # Standard IQR multiplier
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Filter outliers
    outlier_mask = (ratios >= lower_bound) & (ratios <= upper_bound)
    return ratios[outlier_mask]


def get_tier_class(forest_types_df: pd.DataFrame, mfe_class: str, tier: str) -> str:
    """
    Get the tier classification for a given MFE forest type.
    
    Args:
        forest_types_df (pd.DataFrame): Forest types reference table
        mfe_class (str): MFE forest type class
        tier (str): Target tier level (Clade, Family, Genus)
        
    Returns:
        str: Tier classification value
    """
    try:
        result = forest_types_df[
            forest_types_df['ForestTypeMFE'] == mfe_class
        ].reset_index()
        
        if len(result) == 0:
            return np.nan
            
        return result.at[0, tier]
    except (KeyError, IndexError):
        return np.nan


def add_tier_column(df: pd.DataFrame, forest_types_df: pd.DataFrame, 
                   tier: str, forest_type_col: str = 'ForestType') -> pd.DataFrame:
    """
    Add hierarchical tier column to dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with forest type information
        forest_types_df (pd.DataFrame): Forest types reference table
        tier (str): Tier level to add (Clade, Family, Genus)
        forest_type_col (str): Name of forest type column in df
        
    Returns:
        pd.DataFrame: Dataframe with added tier column
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    logger.debug(f"Adding {tier} tier column...")
    
    df = df.copy()
    df[tier] = df[forest_type_col].apply(
        lambda forest_type: get_tier_class(forest_types_df, forest_type, tier)
    )
    
    return df


def process_hierarchy_levels(df: pd.DataFrame, forest_types_df: pd.DataFrame,
                           forest_type_col: str = 'ForestType') -> pd.DataFrame:
    """
    Process and add all hierarchical tier columns to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with forest type information
        forest_types_df (pd.DataFrame): Forest types reference table
        forest_type_col (str): Name of forest type column in df
        
    Returns:
        pd.DataFrame: Dataframe with all hierarchy columns added
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    logger.info("Adding hierarchical tier columns...")
    
    # Add tier columns for hierarchical levels
    tiers_to_add = ['Clade', 'Family', 'Genus']
    
    for tier in tiers_to_add:
        df = add_tier_column(df, forest_types_df, tier, forest_type_col)
    
    return df


def sample_height_at_points(height_maps_dir: Path, points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Sample canopy height values at NFI plot locations from tiled height maps.
    
    Updated to handle tiled height maps efficiently by iterating through tiles
    and sampling only NFI points within each tile's bounds.
    
    Args:
        height_maps_dir (Path): Directory containing height map tiles
        points_gdf (gpd.GeoDataFrame): GeoDataFrame with plot locations
        
    Returns:
        np.ndarray: Sampled height values (NaN for points outside raster coverage)
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    
    # Initialize height values array
    height_values = np.full(len(points_gdf), np.nan)
    
    # Find all height map tiles (any .tif files)
    height_files = list(height_maps_dir.glob('*.tif'))
    
    if not height_files:
        logger.error(f"No height map files found in {height_maps_dir}")
        return height_values
    
    logger.info(f"Processing {len(height_files)} height map tiles...")
    
    # Track how many points have been sampled
    points_sampled = 0
    
    # For each height map tile
    for height_file in height_files:
        try:
            with rasterio.open(height_file) as src:
                # Get tile bounds
                bounds = src.bounds
                
                # Find points within this tile's bounds using spatial indexing
                points_in_tile_mask = (
                    (points_gdf.geometry.x >= bounds.left) &
                    (points_gdf.geometry.x <= bounds.right) &
                    (points_gdf.geometry.y >= bounds.bottom) &
                    (points_gdf.geometry.y <= bounds.top)
                )
                
                points_in_tile = points_gdf[points_in_tile_mask]
                
                if len(points_in_tile) > 0:
                    # Reproject points to raster CRS if needed
                    if points_in_tile.crs != src.crs:
                        points_in_tile = points_in_tile.to_crs(src.crs)
                    
                    # Extract coordinates
                    coords = [(point.x, point.y) for point in points_in_tile.geometry]
                    
                    # Sample height values
                    sampled_values = [val[0] for val in src.sample(coords)]
                    
                    # Convert nodata to NaN
                    sampled_values = np.array(sampled_values, dtype=float)
                    if src.nodata is not None:
                        sampled_values[sampled_values == src.nodata] = np.nan
                    
                    # Update height values array
                    tile_indices = points_gdf.index[points_in_tile_mask]
                    height_values[tile_indices] = sampled_values
                    
                    points_sampled += len(points_in_tile)
                    logger.debug(f"Sampled {len(points_in_tile)} points from {height_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing height tile {height_file}: {e}")
            continue
    
    n_valid = np.sum(~np.isnan(height_values))
    logger.info(f"Height sampling complete: {n_valid}/{len(points_gdf)} points with valid heights")
    
    return height_values


def create_training_dataset(config: dict) -> pd.DataFrame:
    """
    Create training dataset by combining NFI data with sampled height values.
    
    Updated to use CentralDataPaths for harmonized path management and handle
    both 10m and 100m height maps as specified in config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Combined training dataset with height, AGB, BGB, and forest type
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    logger.info("Creating training dataset...")
    
    # Get paths using centralized path management
    nfi_processed_dir = FOREST_INVENTORY_PROCESSED_DIR
    
    # Determine which height maps to use (10m for fitting, 100m for estimation)
    use_10m = config['fitting']['height_res'] == 10
    if use_10m:
        height_maps_dir = HEIGHT_MAPS_10M_DIR
        logger.info("Using 10m height maps for allometry fitting")
    else:
        height_maps_dir = HEIGHT_MAPS_100M_DIR
        logger.info("Using 100m height maps for allometry fitting")
    
    target_years = config['processing']['target_years']
    
    all_data = []
    
    for year in target_years:
        logger.info(f"Processing year {year}...")
        
        # Look for NFI shapefile for this year
        nfi_pattern = f'nfi4_{year}_biomass.shp'
        nfi_path = nfi_processed_dir / nfi_pattern
        
        if not nfi_path.exists():
            logger.warning(f"NFI file not found: {nfi_path}")
            continue
        
        # Check if height maps directory exists for this year
        year_height_dir = height_maps_dir / str(year)
        if not year_height_dir.exists():
            logger.warning(f"Height maps directory not found: {year_height_dir}")
            continue
        
        # Load NFI data
        try:
            nfi_gdf = gpd.read_file(nfi_path)
            logger.info(f"Loaded {len(nfi_gdf)} NFI plots for {year}")
        except Exception as e:
            logger.error(f"Failed to load NFI data for {year}: {e}")
            continue
        
        # Sample height values from tiled maps
        height_values = sample_height_at_points(year_height_dir, nfi_gdf)
        
        # Add height to dataframe
        nfi_gdf['Height'] = height_values
        
        # Filter valid data (height > 0, AGB > 0)
        valid_mask = (
            ~np.isnan(nfi_gdf['Height']) & 
            (nfi_gdf['Height'] > 0) &
            (nfi_gdf['AGB'] > 0)
        )
        
        nfi_gdf_valid = nfi_gdf[valid_mask].copy()
        logger.info(f"Valid samples for {year}: {len(nfi_gdf_valid)}")
        
        if len(nfi_gdf_valid) > 0:
            nfi_gdf_valid['Year'] = year
            all_data.append(nfi_gdf_valid)
    
    if not all_data:
        raise ValueError("No valid training data found!")
    
    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total training samples: {len(combined_df)}")
    
    return combined_df


def create_output_directory(output_path: Path) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path (Path): Path to output file or directory
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)


# Tier name mapping for compatibility with existing pipeline
TIER_NAMES = {
    0: 'Dummy',
    1: 'Clade', 
    2: 'Family',
    3: 'Genus',
    4: 'ForestTypeMFE'
}


# Forest type hierarchy levels (processed in order)
HIERARCHY_LEVELS = {
    'General': {'tier': 0, 'column': None},
    'Clade': {'tier': 1, 'column': 'Clade'},
    'Family': {'tier': 2, 'column': 'Family'},
    'Genus': {'tier': 3, 'column': 'Genus'},
    'ForestType': {'tier': 4, 'column': 'ForestType'}
}