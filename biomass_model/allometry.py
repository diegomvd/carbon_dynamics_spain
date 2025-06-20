"""
Forest type classification and allometric parameter management.

This module provides functions for managing forest type hierarchies and retrieving
allometric parameters for biomass estimation. Implements a hierarchical fallback
system that traverses from specific forest types to general categories when
specific parameters are unavailable.

The hierarchy structure follows: ForestTypeMFE -> Genus -> Family -> Clade -> General
Each tier provides increasingly general allometric relationships as fallbacks.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
from pathlib import Path

from logging_utils import logger
from config import get_config


def build_forest_type_mapping(mfe_dir, cache_path='forest_type_cache.csv', use_cache=True):
    """
    Build a mapping between forest type codes and names with optional caching.
    
    Extracts forest type codes and names from MFE (Mapa Forestal de España) shapefiles
    and creates a dictionary mapping for efficient lookups during processing.
    
    Args:
        mfe_dir (str): Directory containing MFE shapefile data
        cache_path (str): Path to cache file for storing/loading mappings
        use_cache (bool): Whether to use cached mapping if available
        
    Returns:
        dict: Dictionary mapping forest type codes (str) to names (str)
        
    Raises:
        FileNotFoundError: If MFE directory doesn't exist
        ValueError: If no valid MFE files found or required columns missing
    """
    # Check for existing cache to avoid reprocessing
    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading forest type mapping from cache: {cache_path}")
        try:
            mapping_df = pd.read_csv(cache_path)
            # Ensure codes are strings for consistent comparison
            mapping_df['code'] = mapping_df['code'].astype(str)
            return dict(zip(mapping_df['code'], mapping_df['name']))
        except Exception as e:
            logger.warning(f"Failed to load cache, rebuilding mapping: {e}")
    
    logger.info("Building forest type code-to-name mapping...")
    
    # Validate MFE directory exists
    if not os.path.exists(mfe_dir):
        raise FileNotFoundError(f"MFE directory not found: {mfe_dir}")
    
    # Dictionary to store code-to-name mappings
    code_to_name = {}

    # Find all MFE shapefile files in directory
    mfe_files = list(Path(mfe_dir).glob('dissolved_by_form_MFE*.shp'))
    
    if not mfe_files:
        raise ValueError(f"No MFE shapefiles found in directory: {mfe_dir}")

    # Process each MFE file to extract mappings
    for mfe_file in mfe_files:
        try:
            # Load MFE spatial data
            mfe_data = gpd.read_file(mfe_file)

            # Extract code and name pairs from required columns
            if 'FORARB' in mfe_data.columns and 'FormArbol' in mfe_data.columns:
                # Update mapping with unique code-name pairs
                for _, row in mfe_data[['FORARB', 'FormArbol']].drop_duplicates().iterrows():
                    code_to_name[str(row['FORARB'])] = row['FormArbol']
            else:
                logger.warning(f"Required columns (FORARB, FormArbol) missing in {mfe_file}")

        except Exception as e:
            logger.error(f"Failed to extract mapping from {mfe_file}: {str(e)}")

    if not code_to_name:
        raise ValueError("No forest type mappings could be extracted from MFE files")

    logger.info(f"Created mapping for {len(code_to_name)} forest types")

    # Save cache for future use to avoid reprocessing
    try:
        mapping_df = pd.DataFrame({
            'code': list(code_to_name.keys()),
            'name': list(code_to_name.values())
        })
        mapping_df.to_csv(cache_path, index=False)
        logger.info(f"Saved forest type mapping to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
    
    return code_to_name


def update_tiers(tier, tier_names):
    """
    Update tier level when moving up the forest type hierarchy.
    
    Args:
        tier (int): Current tier level
        tier_names (dict): Dictionary mapping tier numbers to names
        
    Returns:
        tuple: (new_tier, old_tier_name, new_tier_name)
    """
    old_tier_name = tier_names[tier]
    new_tier = tier - 1
    new_tier_name = tier_names[new_tier]
    return new_tier, old_tier_name, new_tier_name


def update_forest_type(forest_types, forest_type, old_tier_name, new_tier_name):
    """
    Get parent forest type when moving up the hierarchy.
    
    Traverses the forest type hierarchy to find the parent category
    for the current forest type at a higher (more general) tier level.
    
    Args:
        forest_types (pd.DataFrame): DataFrame with forest type hierarchy
        forest_type (str): Current forest type name
        old_tier_name (str): Name of current tier level
        new_tier_name (str): Name of parent tier level
        
    Returns:
        str: Parent forest type name, or None if not found
    """
    try:
        # Look up parent forest type in hierarchy table
        new_forest_type = forest_types.reset_index().set_index(old_tier_name).loc[forest_type][new_tier_name]
        
        # Handle case where multiple matches exist (take first)
        if not isinstance(new_forest_type, str):
            new_forest_type = forest_types.reset_index().set_index(old_tier_name).loc[forest_type].iloc[0][new_tier_name]
        
        return new_forest_type
        
    except Exception as e:
        logger.error(f"Error updating forest type {forest_type}: {str(e)}")
        return None


def get_allometry_parameters(forest_type_name, forest_types, agb_allometries, bgb_allometries, tier_names):
    """
    Get allometry parameters for a forest type, traversing hierarchy if needed.
    
    Implements a hierarchical fallback system that starts with the most specific
    forest type and progressively moves to more general categories until
    parameters are found. Ensures all forest types have valid allometric
    relationships for biomass estimation.
    
    Args:
        forest_type_name (str): Specific forest type name to get parameters for
        forest_types (pd.DataFrame): DataFrame with forest type hierarchy structure
        agb_allometries (pd.DataFrame): Above-ground biomass allometry coefficients
        bgb_allometries (pd.DataFrame): Below-ground biomass ratio coefficients  
        tier_names (dict): Dictionary mapping tier numbers to tier names
        
    Returns:
        tuple: (agb_params, bgb_params) where:
            - agb_params: dict with 'median', 'p15', 'p85' allometry parameters
            - bgb_params: dict with 'mean', 'p5', 'p95' ratio parameters
    """
    # Start at the most specific tier (ForestTypeMFE)
    tier = 4  
    current_forest_type = forest_type_name
    
    # Handle non-forest areas - skip hierarchy traversal
    if forest_type_name == 'No arbolado':
        tier = -1  # Skip allometry search and go to general fallback
    
    # Initialize parameter dictionaries
    agb_params = None
    bgb_params = None

    # Traverse hierarchy from specific to general until parameters are found
    while tier >= 0:
        try:
            # Attempt to find AGB allometry parameters at current tier
            if agb_params is None:
                try:
                    agb_subset = agb_allometries[agb_allometries.tier == tier]
                    agb_row = agb_subset.loc[current_forest_type]
                    
                    # Extract function type for allometric relationship
                    function_type = agb_row.function_type
                    
                    # Build parameter dictionary for different percentiles
                    agb_params = {
                        'median': (
                            np.exp(agb_row.median_intercept),    # Transform log-intercept
                            agb_row.median_slope,
                            function_type
                        ),
                        'p15': (
                            np.exp(agb_row.low_bound_intercept), # Transform log-intercept
                            agb_row.low_bound_slope,
                            function_type
                        ),
                        'p85': (
                            np.exp(agb_row.upper_bound_intercept), # Transform log-intercept
                            agb_row.upper_bound_slope,
                            function_type
                        )
                    }
                except:
                    pass  # Continue to next tier if parameters not found
            
            # Attempt to find BGB ratio parameters at current tier
            if bgb_params is None:
                try:
                    bgb_subset = bgb_allometries[bgb_allometries.tier == tier]
                    bgb_row = bgb_subset.loc[current_forest_type]
                    
                    # Build BGB ratio parameter dictionary
                    bgb_params = {
                        'mean': bgb_row['mean'],
                        'p5': bgb_row.q05,
                        'p95': bgb_row.q95
                    }
                except:
                    pass  # Continue to next tier if parameters not found
            
            # Exit loop if both parameter sets found
            if agb_params is not None and bgb_params is not None:
                break

            # Move up hierarchy to more general tier
            tier, old_tier_name, new_tier_name = update_tiers(tier, tier_names)
            current_forest_type = update_forest_type(forest_types, current_forest_type, old_tier_name, new_tier_name)
        
            # Fallback to General category if hierarchy traversal fails
            if current_forest_type is None:
                logger.error(f'Could not get parent forest type, returning general allometry.')
                current_forest_type = 'General'
                tier = 0  # Use dummy tier for General category
                
        except Exception as e:
            logger.error(f"Error getting parameters for {forest_type_name}: {str(e)}")
            # Force fallback to General category
            current_forest_type = 'General'
            tier = 0  # Use dummy tier for General category
    
    # Final fallback to General allometry if AGB parameters still missing
    if agb_params is None:
        logger.info(f'{forest_type_name}: Could not find any parameters for H-AGB allometry. Falling back to general allometry.')
        try:
            agb_row = agb_allometries[agb_allometries.tier == 0].loc['General']
            function_type = agb_row.function_type
            agb_params = {
                'median': (
                    np.exp(agb_row.median_intercept),
                    agb_row.median_slope,
                    function_type
                ),
                'p15': (
                    np.exp(agb_row.low_bound_intercept),
                    agb_row.low_bound_slope,
                    function_type
                ),
                'p85': (
                    np.exp(agb_row.upper_bound_intercept),
                    agb_row.upper_bound_slope,
                    function_type
                )
            }
        except:
            # Absolute last resort - zero parameters
            agb_params = {
                'median': (0.0, 0.0, 'power'),
                'p15': (0.0, 0.0, 'power'),
                'p85': (0.0, 0.0, 'power')
            }
    
    # Final fallback to General BGB coefficients if still missing
    if bgb_params is None:
        logger.info(f'{forest_type_name}: Could not find any parameter for BGB coefficient. Falling back to general coefficients.')
        try:
            bgb_row = bgb_allometries[bgb_allometries.tier == 0].loc['General']
            bgb_params = {
                'mean': bgb_row['mean'],
                'p5': bgb_row.q05,
                'p95': bgb_row.q95
            }
        except:
            # Absolute last resort - zero parameters
            bgb_params = {
                'mean': 0.0,
                'p5': 0.0,
                'p95': 0.0
            }

    return agb_params, bgb_params