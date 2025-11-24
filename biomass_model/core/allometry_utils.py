"""
Utility functions for biomass allometry fitting and processing.

This module provides essential utilities for data validation, outlier removal,
hierarchical forest type processing, and spatial sampling for the integrated
biomass estimation and allometry fitting pipeline.

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
from scipy.stats import theilslopes

from rasterio.mask import mask
from rasterio.mask import mask as rio_mask

from shapely.geometry import Point


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


def load_forest_type_code_mapping() -> pd.DataFrame:
    """
    Load forest type name → code mapping (cached).
    
    Returns DataFrame with columns: code, name
    """
    cache_file = FOREST_TYPE_MFE_CODE_TO_NAME_FILE  
    return pd.read_csv(cache_file)
    
def get_forest_type_code(forest_type_name: str) -> Optional[int]:
    """Map forest type name to code."""
    mapping = load_forest_type_code_mapping()
    match = mapping[mapping['name'] == forest_type_name]
    
    if len(match) > 0:
        return int(match.iloc[0]['code'])
    return None

def parse_height_tile_metadata(filepath: Path) -> Optional[Dict]:
    """
    Extract year and forest type code from filename.
    
    Pattern: canopy_height_{year}_{coords}_code{code}.tif
    Example: canopy_height_2017_N36.0_W3.0_code0.tif
    """
    stem = filepath.stem
    
    # Extract year: canopy_height_2017_...
    year_match = re.search(r'canopy_height_(\d{4})_', stem)
    if not year_match:
        return None
    
    # Extract code: ..._code12.tif
    code_match = re.search(r'_code(\d+)', stem)
    if not code_match:
        return None
    
    return {
        'year': int(year_match.group(1)),
        'forest_type_code': int(code_match.group(1))
    }

def extract_heights_at_plot_robust(
    height_maps_dir: Path,
    masks_dir: Path,
    plot_geom: Point,
    plot_year: int,
    plot_forest_type_code: int,
    plot_radius: float = 25.0,
    min_pixels: int = 5
) -> Optional[Dict]:
    """
    Extract heights within plot buffer, applying forest type mask.
    
    Args:
        height_maps_dir: Directory with height tiles (pattern: *_interpolated.tif)
        masks_dir: Directory with forest type masks (pattern: *_code{N}.tif)
        plot_geom: Plot center point
        plot_year: Year of measurement
        plot_forest_type_code: Forest type code to filter
        plot_radius: Extraction radius (25m = 50m diameter)
        min_pixels: Minimum valid pixels required
        
    Returns:
        dict with height statistics or None
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    
    buffer = plot_geom.buffer(plot_radius)
    
    # Find all mask tiles for this year + forest type
    mask_pattern = f"canopy_height_{plot_year}_*_code{plot_forest_type_code}.tif"
    mask_tiles = list(masks_dir.glob(mask_pattern))
    
    if not mask_tiles:
        return None
    
    heights = []
    
    for mask_path in mask_tiles:
        try:
            # Extract coordinates from mask filename
            # Example: canopy_height_2018_N44.0_W2.0_code12.tif
            stem = mask_path.stem
            coord_match = re.search(r'canopy_height_(\d{4})_(N[\d.]+_W[\d.]+)_code\d+', stem)
            
            if not coord_match:
                logger.info(f"Could not parse coordinates from: {mask_path.name}")
                continue
            
            year_str = coord_match.group(1)
            coords = coord_match.group(2)  # e.g., "N44.0_W2.0"
            
            # Build corresponding height tile path
            height_filename = f"canopy_height_{year_str}_{coords}_interpolated.tif"
            height_path = height_maps_dir / height_filename
            
            if not height_path.exists():
                logger.info(f"Height tile not found: {height_filename}")
                continue
            
            # Open both height and mask tiles
            with rasterio.open(height_path) as height_src, rasterio.open(mask_path) as mask_src:
                
                # Quick bounds check
                if not (buffer.bounds[0] <= height_src.bounds[2] and 
                       buffer.bounds[2] >= height_src.bounds[0] and
                       buffer.bounds[1] <= height_src.bounds[3] and 
                       buffer.bounds[3] >= height_src.bounds[1]):
                    continue
                
                # Extract height within buffer
                height_img, height_transform = mask(height_src, [buffer], crop=True)
                height_data = height_img[0].astype(float)
                
                # Handle height nodata
                if height_src.nodata is not None:
                    height_data[height_data == height_src.nodata] = np.nan
                
                # Extract mask within same buffer
                mask_img, _ = mask(mask_src, [buffer], crop=True)
                mask_data = mask_img[0]
                
                # Apply forest type mask to heights
                height_masked = height_data.copy()
                height_masked[mask_data == 0] = np.nan  # Zero out non-forest-type pixels
                
                # Get valid heights
                valid = height_masked[(~np.isnan(height_masked)) & (height_masked > 0)]
                
                if len(valid) > 0:
                    heights.extend(valid)
                    
        except Exception as e:
            logger.info(f"Error processing {mask_path.name}: {e}")
            continue
    
    if len(heights) < min_pixels:
        return None
    
    heights = np.array(heights)
    
    return {
        'n_pixels': len(heights),
        'H_mean': np.mean(heights),
        'H_p75': np.percentile(heights, 75),
        'H_p90': np.percentile(heights, 90),
        'H_std': np.std(heights)
    }

def validate_height_biomass_data(df: pd.DataFrame, height_col: str = 'Height', 
                                 min_samples: int=20, max_height: float=60.0, min_height: float=0.0) -> bool:
    """
    Validate input data for height-biomass allometry fitting.
    
    Args:
        df (pd.DataFrame): Input dataframe
        height_col (str): Name of height column
        min_samples (int): Minimum required samples
        max_height (float): Maximum allowed height
        min_height (float): Minimum allowed height
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if len(df) < min_samples:
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


def remove_outliers(df: pd.DataFrame, height_col: str = 'Height', biomass_col: str = 'AGB', 
                    contamination: float = 0.12) -> pd.DataFrame:
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


def remove_ratio_outliers(ratios: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from BGB ratio data using IQR method.
    
    Args:
        ratios (pd.DataFrame): Dataframe with ratio values 
        
    Returns:
        pd.DataFrame: Processed dataframe without outliers
    """
    ratios_validated = prepare_ratio_data(ratios)
    ratios = np.array(ratios_validated['BGB_Ratio'])

    Q1 = np.percentile(ratios, 25)
    Q3 = np.percentile(ratios, 75)
    IQR = Q3 - Q1
    multiplier = 1.5  # Standard IQR multiplier
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Filter outliers
    ratios_clean = ratios_validated[(ratios_validated['BGB_Ratio'] >= lower_bound) & (ratios_validated['BGB_Ratio'] <= upper_bound)]
   
    return ratios_clean


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
    tiers_to_add = ['Clade', 'Family', 'Genus', 'ForestTypeMFE']
    
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


def create_training_dataset(config: dict, height_metric: str = 'mean') -> pd.DataFrame:
    """
    Create training dataset with robust plot-aggregated heights.
    
    Args:
        height_metric: 'mean', 'p75', or 'p90'
    """
    logger = get_logger('biomass_estimation.allometry_utils')
    logger.info(f"Creating training dataset using H_{height_metric}...")
    
    nfi_processed_dir = FOREST_INVENTORY_PROCESSED_DIR
    height_maps_dir = HEIGHT_MAPS_10M_DIR  # All tiles in single dir now
    masks_dir = FOREST_TYPE_MASKS_DIR 

    target_years = config['processing']['target_years']
    plot_radius = config['fitting'].get('plot_radius', 25.0)
    min_pixels = config['fitting'].get('min_pixels_per_plot', 5)
    
    all_data = []
    
    for year in target_years:
        logger.info(f"Processing year {year}...")
        
        nfi_path = nfi_processed_dir / 'per_year' / f'nfi4_{year}_biomass.shp'
        if not nfi_path.exists():
            logger.warning(f"NFI file not found: {nfi_path}")
            continue
        
        nfi_gdf = gpd.read_file(nfi_path)
        logger.info(f"  Loaded {len(nfi_gdf)} plots")
        
        for idx, plot in nfi_gdf.iterrows():
            if idx % 100 == 0:
                logger.debug(f"    Plot {idx}/{len(nfi_gdf)}")
            
            # Map forest type name → code
            forest_type_name = plot['ForestType']
            forest_type_code = get_forest_type_code(forest_type_name)
            
            if forest_type_code is None:
                logger.debug(f"Could not map forest type: {forest_type_name}")
                continue
            
            # Extract heights
            height_stats = extract_heights_at_plot_robust(
                height_maps_dir,
                masks_dir,
                plot.geometry,
                year,
                forest_type_code,
                plot_radius,
                min_pixels
            )
            
            if height_stats is None:
                continue
            
            all_data.append({
                'plot_id': plot.get('ID', idx),
                'Year': year,
                'ForestType': forest_type_name,
                'Height': height_stats[f'H_{height_metric}'],
                'AGB': plot['AGB'],
                'BGB': plot.get('BGB', np.nan),
                'BGB_Ratio': plot['BGB'] / plot['AGB'] if plot.get('BGB') and plot['AGB'] > 0 else np.nan,
                'n_pixels': height_stats['n_pixels']
            })
    
    if not all_data:
        raise ValueError("No valid training data!")
    
    df = pd.DataFrame(all_data)
    logger.info(f"Created dataset: {len(df)} samples, {df['n_pixels'].mean():.1f} pixels/plot avg")
    
    return df


def fit_theil_sen_allometry(heights: np.ndarray, agbd: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power-law using Theil-Sen (robust to outliers).
    
    Fits: AGBD = a * Height^b  (via log-log linear regression)
    
    Returns:
        (log_intercept, slope, rmse)
    """
    
    log_h = np.log(heights)
    log_agb = np.log(agbd)
    
    result = theilslopes(log_agb, log_h)
    slope = result.slope
    log_intercept = result.intercept
    
    # RMSE on original scale
    pred = np.exp(log_intercept) * heights ** slope
    rmse = np.sqrt(np.mean((agbd - pred) ** 2))
    
    return log_intercept, slope, rmse

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
    'ForestTypeMFE': {'tier': 4, 'column': 'ForestTypeMFE'}
}
