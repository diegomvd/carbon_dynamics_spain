"""
Allometric Relationship Management

This module manages allometric relationships for biomass estimation,
including hierarchical forest type processing and parameter retrieval.
Updated to use CentralDataPaths instead of config file paths.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import logging

# Shared utilities
from shared_utils import get_logger, validate_file_exists
from shared_utils.central_data_paths_constants import *


class AllometryManager:
    """
    Manager for allometric relationships and forest type hierarchies.
    
    Handles loading and retrieval of allometric parameters for different
    forest types with hierarchical fallback system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the allometry manager.
        
        Args:
            config: Configuration dictionary (processing parameters only)
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.allometry')
        
        # Data containers
        self.allometry_data = None
        self.forest_types_data = None
        self.bgb_ratios_data = None
        self.hierarchy_mapping = None
        
        # Hierarchy tier names mapping
        self.tier_names = {
            0: 'Dummy',
            1: 'Clade', 
            2: 'Family',
            3: 'Genus',
            4: 'ForestTypeMFE'
        }
        
        # Load all allometric data
        self._load_allometric_data()
        
        self.logger.info("AllometryManager initialized successfully")
    
    def _load_allometric_data(self) -> None:
        """Load all allometric data files using CentralDataPaths."""
        self.logger.info("Loading allometric data files...")
        
        # Load allometry relationships - UPDATED: Use CentralDataPaths
        allometry_file = FITTED_PARAMETERS_FILE
        if allometry_file.exists():
            self.allometry_data = pd.read_csv(allometry_file)
            # Set forest_type as index for efficient lookup
            if 'forest_type' in self.allometry_data.columns:
                self.allometry_data = self.allometry_data.set_index('forest_type')
            self.logger.info(f"Loaded {len(self.allometry_data)} allometric relationships")
        else:
            self.logger.warning(f"Allometry file not found: {allometry_file}")
            self.allometry_data = pd.DataFrame()
        
        # Load forest type hierarchies - UPDATED: Use CentralDataPaths
        forest_types_file = FOREST_TYPES_TIERS_FILE
        if forest_types_file.exists():
            self.forest_types_data = pd.read_csv(forest_types_file)
            self.logger.info(f"Loaded {len(self.forest_types_data)} forest type mappings")
        else:
            self.logger.warning(f"Forest types file not found: {forest_types_file}")
            self.forest_types_data = pd.DataFrame()
        
        # Load BGB ratios - UPDATED: Use CentralDataPaths
        bgb_file = BGB_RATIOS_FILE
        if bgb_file.exists():
            self.bgb_ratios_data = pd.read_csv(bgb_file)
            # Set forest_type as index for efficient lookup
            if 'forest_type' in self.bgb_ratios_data.columns:
                self.bgb_ratios_data = self.bgb_ratios_data.set_index('forest_type')
            self.logger.info(f"Loaded {len(self.bgb_ratios_data)} BGB ratio coefficients")
        else:
            self.logger.warning(f"BGB ratios file not found: {bgb_file}")
            self.bgb_ratios_data = pd.DataFrame()
        
        # Build hierarchy mapping
        self._build_hierarchy_mapping()
    
    def _build_hierarchy_mapping(self) -> None:
        """Build forest type hierarchy mapping for fallback system."""
        self.logger.info("Building forest type hierarchy mapping...")
        
        # Create hierarchical mapping from forest types data
        if self.forest_types_data is not None and not self.forest_types_data.empty:
            try:
                # Create mapping for each tier level
                self.hierarchy_mapping = {}
                for tier_name in ['Clade', 'Family', 'Genus', 'ForestTypeMFE']:
                    if tier_name in self.forest_types_data.columns:
                        tier_data = self.forest_types_data.dropna(subset=[tier_name])
                        self.hierarchy_mapping[tier_name] = tier_data.groupby(tier_name).first().to_dict('index')
                
                self.logger.info("Hierarchy mapping built successfully")
            except Exception as e:
                self.logger.error(f"Error building hierarchy mapping: {str(e)}")
                self.hierarchy_mapping = {}
        else:
            self.hierarchy_mapping = {}
    
    def update_tiers(self, tier: int) -> Tuple[int, str, str]:
        """
        Update tier level when moving up the forest type hierarchy.
        
        Args:
            tier (int): Current tier level
            
        Returns:
            tuple: (new_tier, old_tier_name, new_tier_name)
        """
        old_tier_name = self.tier_names[tier]
        new_tier = tier - 1
        new_tier_name = self.tier_names[new_tier]
        return new_tier, old_tier_name, new_tier_name
    
    def update_forest_type(self, forest_type: str, old_tier_name: str, new_tier_name: str) -> Optional[str]:
        """
        Get parent forest type when moving up the hierarchy.
        
        Traverses the forest type hierarchy to find the parent category
        for the current forest type at a higher (more general) tier level.
        
        Args:
            forest_type (str): Current forest type name
            old_tier_name (str): Name of current tier level
            new_tier_name (str): Name of parent tier level
            
        Returns:
            str: Parent forest type name, or None if not found
        """
        try:
            # Look up parent forest type in hierarchy table
            forest_match = self.forest_types_data[
                self.forest_types_data[old_tier_name] == forest_type
            ]
            
            if forest_match.empty:
                return None
            
            # Get parent tier value
            new_forest_type = forest_match.iloc[0][new_tier_name]
            
            return new_forest_type if pd.notna(new_forest_type) else None
            
        except Exception as e:
            self.logger.error(f"Error updating forest type {forest_type}: {str(e)}")
            return None
    
    def get_allometry_parameters(self, forest_type_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Get allometry parameters for a forest type, traversing hierarchy if needed.
        
        Implements a hierarchical fallback system that starts with the most specific
        forest type and progressively moves to more general categories until
        parameters are found. Ensures all forest types have valid allometric
        relationships for biomass estimation.
        
        Args:
            forest_type_name (str): Specific forest type name to get parameters for
            
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
                        agb_subset = self.allometry_data[self.allometry_data['tier'] == tier]
                        if current_forest_type in agb_subset.index:
                            agb_row = agb_subset.loc[current_forest_type]
                            
                            # Extract function type for allometric relationship
                            function_type = agb_row.get('function_type', 'power')
                            
                            # Build parameter dictionary for different percentiles
                            agb_params = {
                                'median': (
                                    np.exp(agb_row['median_intercept']),    # Transform log-intercept
                                    agb_row['median_slope'],
                                    function_type
                                ),
                                'p15': (
                                    np.exp(agb_row['low_bound_intercept']), # Transform log-intercept
                                    agb_row['low_bound_slope'],
                                    function_type
                                ),
                                'p85': (
                                    np.exp(agb_row['upper_bound_intercept']), # Transform log-intercept
                                    agb_row['upper_bound_slope'],
                                    function_type
                                )
                            }
                    except (KeyError, IndexError):
                        pass  # Continue to next tier if parameters not found
                
                # Attempt to find BGB ratio parameters at current tier
                if bgb_params is None:
                    try:
                        bgb_subset = self.bgb_ratios_data[self.bgb_ratios_data['tier'] == tier]
                        if current_forest_type in bgb_subset.index:
                            bgb_row = bgb_subset.loc[current_forest_type]
                            
                            # Build BGB ratio parameter dictionary
                            bgb_params = {
                                'mean': bgb_row['mean'],
                                'p5': bgb_row['q05'],
                                'p95': bgb_row['q95']
                            }
                    except (KeyError, IndexError):
                        pass  # Continue to next tier if parameters not found
                
                # Exit loop if both parameter sets found
                if agb_params is not None and bgb_params is not None:
                    break

                # Move up hierarchy to more general tier
                tier, old_tier_name, new_tier_name = self.update_tiers(tier)
                current_forest_type = self.update_forest_type(current_forest_type, old_tier_name, new_tier_name)
            
                # Fallback to General category if hierarchy traversal fails
                if current_forest_type is None:
                    self.logger.error(f'Could not get parent forest type, returning general allometry.')
                    current_forest_type = 'General'
                    tier = 0  # Use dummy tier for General category
                    
            except Exception as e:
                self.logger.error(f"Error getting parameters for {forest_type_name}: {str(e)}")
                # Force fallback to General category
                current_forest_type = 'General'
                tier = 0  # Use dummy tier for General category
        
        # Final fallback to General allometry if AGB parameters still missing
        if agb_params is None:
            self.logger.info(f'{forest_type_name}: Could not find any parameters for H-AGB allometry. Falling back to general allometry.')
            try:
                agb_general = self.allometry_data[self.allometry_data['tier'] == 0]
                if 'General' in agb_general.index:
                    agb_row = agb_general.loc['General']
                    function_type = agb_row.get('function_type', 'power')
                    agb_params = {
                        'median': (
                            np.exp(agb_row['median_intercept']),
                            agb_row['median_slope'],
                            function_type
                        ),
                        'p15': (
                            np.exp(agb_row['low_bound_intercept']),
                            agb_row['low_bound_slope'],
                            function_type
                        ),
                        'p85': (
                            np.exp(agb_row['upper_bound_intercept']),
                            agb_row['upper_bound_slope'],
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
            self.logger.info(f'{forest_type_name}: Could not find any parameter for BGB coefficient. Falling back to general coefficients.')
            try:
                bgb_general = self.bgb_ratios_data[self.bgb_ratios_data['tier'] == 0]
                if 'General' in bgb_general.index:
                    bgb_row = bgb_general.loc['General']
                    bgb_params = {
                        'mean': bgb_row['mean'],
                        'p5': bgb_row['q05'],
                        'p95': bgb_row['q95']
                    }
            except:
                # Absolute last resort - zero parameters
                bgb_params = {
                    'mean': 0.0,
                    'p5': 0.0,
                    'p95': 0.0
                }

        return agb_params, bgb_params
    
    def get_available_forest_types(self) -> List[str]:
        """
        Get list of available forest types with allometric parameters.
        
        Returns:
            List of forest type names
        """
        forest_types = set()
        
        # From allometry data
        if hasattr(self.allometry_data, 'index'):
            forest_types.update(self.allometry_data.index.tolist())
        
        # From hierarchy data
        for tier_name in ['Clade', 'Family', 'Genus', 'ForestTypeMFE']:
            if tier_name in self.forest_types_data.columns:
                values = self.forest_types_data[tier_name].dropna().unique()
                forest_types.update(values)
        
        return sorted(list(forest_types))
    
    def validate_allometric_data(self) -> bool:
        """
        Validate loaded allometric data for consistency.
        
        Returns:
            bool: True if data is valid
        """
        self.logger.info("Validating allometric data...")
        
        issues = []
        
        # Check required columns exist in allometry data
        required_allometry_cols = ['median_intercept', 'median_slope', 'tier']
        if self.allometry_data is not None:
            missing_cols = [col for col in required_allometry_cols if col not in self.allometry_data.columns]
            if missing_cols:
                issues.append(f"Missing allometry columns: {missing_cols}")
        
        # Check for NaN values in critical columns
        if self.allometry_data is not None:
            for col in ['median_intercept', 'median_slope']:
                if col in self.allometry_data.columns:
                    nan_count = self.allometry_data[col].isna().sum()
                    if nan_count > 0:
                        issues.append(f"Found {nan_count} NaN values in critical column {col}")
        
        # Check BGB ratios
        if self.bgb_ratios_data is not None and 'mean' in self.bgb_ratios_data.columns:
            nan_count = self.bgb_ratios_data['mean'].isna().sum()
            if nan_count > 0:
                issues.append(f"Found {nan_count} NaN values in BGB ratios")
        
        # Log issues
        if issues:
            for issue in issues:
                self.logger.warning(f"Data validation issue: {issue}")
            return False
        else:
            self.logger.info("Allometric data validation passed")
            return True
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded allometric data.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_allometric_relationships': len(self.allometry_data) if self.allometry_data is not None else 0,
            'total_forest_type_mappings': len(self.forest_types_data) if self.forest_types_data is not None else 0,
            'total_bgb_ratios': len(self.bgb_ratios_data) if self.bgb_ratios_data is not None else 0,
            'available_forest_types': len(self.get_available_forest_types()),
            'hierarchy_levels': len(self.tier_names)
        }
        
        # Parameter ranges
        if self.allometry_data is not None:
            if 'median_intercept' in self.allometry_data.columns:
                summary['intercept_range'] = (
                    self.allometry_data['median_intercept'].min(),
                    self.allometry_data['median_intercept'].max()
                )
            
            if 'median_slope' in self.allometry_data.columns:
                summary['slope_range'] = (
                    self.allometry_data['median_slope'].min(), 
                    self.allometry_data['median_slope'].max()
                )
        
        return summary