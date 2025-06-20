"""
Allometric Relationship Management

This module manages allometric relationships for biomass estimation,
including hierarchical forest type processing and parameter retrieval.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import logging

# Shared utilities
from shared_utils import get_logger, validate_file_exists


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
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.allometry')
        
        # Data containers
        self.allometry_data = None
        self.forest_types_data = None
        self.bgb_ratios_data = None
        self.hierarchy_mapping = None
        
        # Load all allometric data
        self._load_allometric_data()
        
        self.logger.info("AllometryManager initialized successfully")
    
    def _load_allometric_data(self) -> None:
        """Load all allometric data files."""
        self.logger.info("Loading allometric data files...")
        
        # Load allometry relationships
        allometry_file = validate_file_exists(
            self.config['data']['allometries_file'],
            "Allometry relationships file"
        )
        self.allometry_data = pd.read_csv(allometry_file)
        self.logger.info(f"Loaded {len(self.allometry_data)} allometric relationships")
        
        # Load forest type hierarchies
        forest_types_file = validate_file_exists(
            self.config['data']['forest_types_file'],
            "Forest types hierarchy file"
        )
        self.forest_types_data = pd.read_csv(forest_types_file)
        self.logger.info(f"Loaded {len(self.forest_types_data)} forest type mappings")
        
        # Load BGB ratios
        bgb_file = validate_file_exists(
            self.config['data']['bgb_coeffs_file'],
            "BGB ratios file"
        )
        self.bgb_ratios_data = pd.read_csv(bgb_file)
        self.logger.info(f"Loaded {len(self.bgb_ratios_data)} BGB ratio coefficients")
        
        # Build hierarchy mapping
        self._build_hierarchy_mapping()
    
    def _build_hierarchy_mapping(self) -> None:
        """Build forest type hierarchy mapping for fallback system."""
        self.logger.info("Building forest type hierarchy mapping...")
        
        self.hierarchy_mapping = {}
        tier_names = self.config['forest_types']['tier_names']
        
        for _, row in self.forest_types_data.iterrows():
            forest_type = row.get('ForestTypeMFE')
            if pd.notna(forest_type):
                hierarchy = {}
                for tier in tier_names:
                    if tier in row and pd.notna(row[tier]):
                        hierarchy[tier] = row[tier]
                
                self.hierarchy_mapping[forest_type] = hierarchy
        
        self.logger.info(f"Built hierarchy mapping for {len(self.hierarchy_mapping)} forest types")
    
    def get_allometry_parameters(self, forest_type_code: str) -> Optional[Dict[str, Any]]:
        """
        Get allometric parameters for a forest type with hierarchical fallback.
        
        Args:
            forest_type_code: Forest type code
            
        Returns:
            Dict containing allometric parameters or None if not found
        """
        try:
            # Convert code to forest type name if needed
            forest_type_name = self._code_to_forest_type(forest_type_code)
            
            if not forest_type_name:
                self.logger.warning(f"Could not resolve forest type for code: {forest_type_code}")
                return None
            
            # Try hierarchical parameter retrieval
            params = self._get_parameters_with_fallback(forest_type_name)
            
            if params:
                self.logger.debug(f"Found allometric parameters for {forest_type_name} (code: {forest_type_code})")
                return params
            else:
                self.logger.warning(f"No allometric parameters found for {forest_type_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting allometry parameters for {forest_type_code}: {str(e)}")
            return None
    
    def _code_to_forest_type(self, forest_type_code: str) -> Optional[str]:
        """
        Convert forest type code to forest type name.
        
        Args:
            forest_type_code: Numeric or string forest type code
            
        Returns:
            Forest type name or None if not found
        """
        # Handle different code formats
        if isinstance(forest_type_code, str):
            # Remove 'code' prefix if present
            code = forest_type_code.replace('code', '')
            try:
                code = int(code)
            except ValueError:
                self.logger.warning(f"Could not parse forest type code: {forest_type_code}")
                return None
        else:
            code = int(forest_type_code)
        
        # Look up in forest types data
        matching_rows = self.forest_types_data[self.forest_types_data['ForestTypeCode'] == code]
        
        if not matching_rows.empty:
            return matching_rows.iloc[0]['ForestTypeMFE']
        
        self.logger.warning(f"Forest type code {code} not found in mapping")
        return None
    
    def _get_parameters_with_fallback(self, forest_type_name: str) -> Optional[Dict[str, Any]]:
        """
        Get allometric parameters with hierarchical fallback.
        
        Args:
            forest_type_name: Forest type name
            
        Returns:
            Dictionary containing parameters or None
        """
        # Get hierarchy for this forest type
        hierarchy = self.hierarchy_mapping.get(forest_type_name, {})
        
        # Define fallback order (most specific to most general)
        tier_names = self.config['forest_types']['tier_names']
        fallback_order = list(reversed(tier_names))  # ForestType -> Genus -> Family -> Clade -> General
        
        # Try each level in the hierarchy
        for tier in fallback_order:
            if tier in hierarchy:
                tier_value = hierarchy[tier]
                
                # Look for parameters at this tier level
                agb_params = self._get_agb_parameters(tier, tier_value)
                bgb_params = self._get_bgb_parameters(tier, tier_value)
                
                if agb_params is not None:
                    result = {
                        'agb_params': agb_params,
                        'bgb_params': bgb_params,  # May be None
                        'tier_used': tier,
                        'tier_value': tier_value,
                        'forest_type': forest_type_name
                    }
                    
                    self.logger.debug(f"Found parameters at {tier} level: {tier_value}")
                    return result
        
        self.logger.warning(f"No parameters found for {forest_type_name} at any hierarchy level")
        return None
    
    def _get_agb_parameters(self, tier: str, tier_value: str) -> Optional[Dict[str, float]]:
        """
        Get AGB allometric parameters for a specific tier and value.
        
        Args:
            tier: Hierarchy tier name
            tier_value: Value at that tier
            
        Returns:
            Dictionary with AGB parameters or None
        """
        # Look for exact match in allometry data
        matching_rows = self.allometry_data[self.allometry_data[tier] == tier_value]
        
        if matching_rows.empty:
            return None
        
        # Take first match (should be unique)
        row = matching_rows.iloc[0]
        
        # Extract parameters (assuming specific column names from existing data)
        try:
            params = {
                'a_mean': row.get('a_mean', row.get('a_50')),
                'b_mean': row.get('b_mean', row.get('b_50')),
                'a_std': self._calculate_parameter_std(row, 'a'),
                'b_std': self._calculate_parameter_std(row, 'b'),
                'a_15': row.get('a_15'),
                'a_85': row.get('a_85'),
                'b_15': row.get('b_15'),
                'b_85': row.get('b_85')
            }
            
            # Validate parameters
            if all(pd.notna(v) for v in [params['a_mean'], params['b_mean']]):
                return params
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting AGB parameters: {str(e)}")
            return None
    
    def _get_bgb_parameters(self, tier: str, tier_value: str) -> Optional[Dict[str, float]]:
        """
        Get BGB ratio parameters for a specific tier and value.
        
        Args:
            tier: Hierarchy tier name
            tier_value: Value at that tier
            
        Returns:
            Dictionary with BGB parameters or None
        """
        # Look for exact match in BGB ratios data
        matching_rows = self.bgb_ratios_data[self.bgb_ratios_data[tier] == tier_value]
        
        if matching_rows.empty:
            return None
        
        # Take first match
        row = matching_rows.iloc[0]
        
        try:
            params = {
                'ratio_mean': row.get('BGB_Ratio_Mean'),
                'ratio_std': self._calculate_bgb_ratio_std(row),
                'ratio_05': row.get('BGB_Ratio_05'),
                'ratio_95': row.get('BGB_Ratio_95')
            }
            
            # Validate parameters
            if pd.notna(params['ratio_mean']):
                return params
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting BGB parameters: {str(e)}")
            return None
    
    def _calculate_parameter_std(self, row: pd.Series, param_prefix: str) -> float:
        """
        Calculate standard deviation from quantiles for allometric parameters.
        
        Args:
            row: Data row containing quantile values
            param_prefix: Parameter prefix ('a' or 'b')
            
        Returns:
            Estimated standard deviation
        """
        try:
            # Get 15th and 85th percentiles
            q15 = row.get(f'{param_prefix}_15')
            q85 = row.get(f'{param_prefix}_85')
            
            if pd.notna(q15) and pd.notna(q85):
                # Approximate std from quantiles (assuming normal distribution)
                # 85th - 15th percentile ≈ 2.07 * std
                return (q85 - q15) / 2.07
            else:
                # Fallback to small default std if quantiles not available
                return 0.1
                
        except Exception:
            return 0.1
    
    def _calculate_bgb_ratio_std(self, row: pd.Series) -> float:
        """
        Calculate standard deviation for BGB ratios from quantiles.
        
        Args:
            row: Data row containing BGB ratio quantiles
            
        Returns:
            Estimated standard deviation
        """
        try:
            q05 = row.get('BGB_Ratio_05')
            q95 = row.get('BGB_Ratio_95')
            
            if pd.notna(q05) and pd.notna(q95):
                # 95th - 5th percentile ≈ 3.29 * std
                return (q95 - q05) / 3.29
            else:
                return 0.05  # Small default
                
        except Exception:
            return 0.05
    
    def get_available_forest_types(self) -> List[str]:
        """
        Get list of available forest types with allometric parameters.
        
        Returns:
            List of forest type names
        """
        forest_types = set()
        
        # From allometry data
        for tier in self.config['forest_types']['tier_names']:
            if tier in self.allometry_data.columns:
                values = self.allometry_data[tier].dropna().unique()
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
        
        # Check required columns exist
        required_allometry_cols = ['a_mean', 'b_mean', 'a_15', 'a_85', 'b_15', 'b_85']
        missing_cols = [col for col in required_allometry_cols if col not in self.allometry_data.columns]
        if missing_cols:
            issues.append(f"Missing allometry columns: {missing_cols}")
        
        # Check for NaN values in critical columns
        for col in ['a_mean', 'b_mean']:
            if col in self.allometry_data.columns:
                nan_count = self.allometry_data[col].isna().sum()
                if nan_count > 0:
                    issues.append(f"Found {nan_count} NaN values in critical column {col}")
        
        # Check BGB ratios
        if 'BGB_Ratio_Mean' in self.bgb_ratios_data.columns:
            nan_count = self.bgb_ratios_data['BGB_Ratio_Mean'].isna().sum()
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
            'total_allometric_relationships': len(self.allometry_data),
            'total_forest_type_mappings': len(self.forest_types_data),
            'total_bgb_ratios': len(self.bgb_ratios_data),
            'available_forest_types': len(self.get_available_forest_types()),
            'hierarchy_levels': len(self.config['forest_types']['tier_names'])
        }
        
        # Parameter ranges
        if 'a_mean' in self.allometry_data.columns:
            summary['a_parameter_range'] = (
                self.allometry_data['a_mean'].min(),
                self.allometry_data['a_mean'].max()
            )
        
        if 'b_mean' in self.allometry_data.columns:
            summary['b_parameter_range'] = (
                self.allometry_data['b_mean'].min(), 
                self.allometry_data['b_mean'].max()
            )
        
        return summary
