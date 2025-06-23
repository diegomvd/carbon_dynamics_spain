#!/usr/bin/env python3
"""
Centralized Data Path Management - HARMONIZED VERSION

Provides standardized data path management for all components in the
Iberian Carbon Assessment Pipeline. Updated with harmonized structure
across all recipes (0-3).

Author: Diego Bengochea
"""

from pathlib import Path
from typing import Dict, Union, Optional, List
import logging


class CentralDataPaths:
    """
    Centralized data path management for all pipeline components.
    
    Provides standardized paths for raw data, processed outputs, results,
    and models following the harmonized data/ directory structure.
    """
    
    def __init__(self, data_root: Union[str, Path] = "data"):
        """
        Initialize centralized data paths.
        
        Args:
            data_root: Root directory for all data (default: "data")
        """
        self.data_root = Path(data_root).resolve()
        
        # Create main directories
        self.raw = self.data_root / "raw"
        self.processed = self.data_root / "processed" 
        self.results = self.data_root / "results"
        self.models = self.data_root / "models"
        
        # Define harmonized data structure mapping
        self.paths = {
            # Raw data inputs - RECIPE 0 HARMONIZED
            'forest_inventory': self.raw / "forest_inventory",
            'forest_type_maps': self.raw / "forest_type_maps",      # SEPARATED from forest_inventory
            'wood_density': self.raw / "wood_density",             # NEW: dedicated directory
            'als_canopy_height': self.raw / "als_canopy_height",   # RENAMED from pnoa_lidar
            'land_cover': self.raw / "land_cover",                 # NEW: for Corine land cover data
            'climate_raw': self.raw / "climate",
            'sentinel2_raw': self.raw / "sentinel2",
            
            # Processed intermediate data - RECIPES 0,1,2 HARMONIZED
            'forest_inventory_processed': self.processed / "forest_inventory",  # SEPARATED processed
            'sentinel2_processed': self.processed / "sentinel2",                # SEPARATED processed
            'als_canopy_height_processed': self.processed / "als_canopy_height", # SEPARATED processed
            'allometries': self.processed / "allometries",                      # NEW: fitted parameters
            'bioclim': self.processed / "bioclim",                              # NEW: processed climate
            'sentinel2_mosaics': self.processed / "sentinel2_mosaics",
            'height_maps': self.processed / "height_maps",                      # RENAMED from height_predictions
            'biomass_maps': self.processed / "biomass_maps",                    # RESTRUCTURED
            'forest_type_masks': self.processed / "forest_type_masks",
            'height_range_masks': self.processed / "height_range_masks",
            
            # Analysis results - RECIPE 3 HARMONIZED
            'analysis_outputs': self.results / "analysis_outputs",
            'tables': self.results / "tables", 
            'ml_outputs': self.results / "ml_outputs",
            
            # Model files - RECIPE 1 HARMONIZED
            'pretrained_height_models': self.data_root / "pretrained_height_models",  # MOVED from models/
            'ml_models': self.models / "ml_models"
        }
        
        # Component-specific subdirectory mappings - ALL RECIPES HARMONIZED
        self.subdirs = {
            # RECIPE 0: Data preparation
            'forest_inventory': {
                'nfi4': 'nfi4',                        # CHANGED from IFN_4_SP
                'codes': 'nfi4_codes.csv',             # CHANGED from Codigos_IFN.csv  
                'types': 'Forest_Types_Tiers.csv',
                'bgb_ratios': 'BGBRatios_Tiers.csv'
            },
            'als_canopy_height': {
                'data': 'data',                        # Contains NDSM-VEGETACION-*.tif
                'tile_metadata': 'tile_metadata',      # Coverage/metadata
                'utm_zones': ['utm_29', 'utm_30', 'utm_31']  # CHANGED from Huso_XX
            },
            'wood_density': {
                'database': 'GlobalWoodDensityDatabase.xls'
            },
            
            # RECIPE 1: Height prediction
            'height_maps': {
                'tmp_raw': 'tmp/raw',                  # Raw prediction patches
                'tmp_merged_120km': 'tmp/merged_120km', # 120km merged tiles  
                'sanitized_10m': '10m',                # Sanitized 10m maps
                'country_100m': '100m'                 # Country-wide 100m maps
            },
            
            # RECIPE 2: Biomass estimation
            'biomass_maps': {
                'raw': 'raw',                          # No LC masking
                'per_forest_type': 'per_forest_type',  # LC masked per forest type
                'full_country': 'full_country'         # Merged country-wide
            },
            'allometries': {
                'fitted_parameters': 'fitted_parameters.csv',
                'bgb_ratios': 'bgb_ratios.csv', 
                'fitting_summary': 'fitting_summary.csv',
                'validation_metrics': 'validation_metrics.csv'
            },
            
            # RECIPE 3: Analysis
            'bioclim': {
                'harmonized': 'harmonized',            # Harmonized climate data
                'variables': 'variables',              # Bioclimatic variables
                'anomalies': 'anomalies'               # Climate anomalies
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_path(self, key: str, create: bool = False) -> Path:
        """
        Get standardized path by key.
        
        Args:
            key: Path key from self.paths
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path object
        """
        if key not in self.paths:
            raise KeyError(f"Unknown path key: {key}")
        
        path = self.paths[key]
        
        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {path}")
        
        return path
    
    def create_directories(self, keys: List[str]) -> None:
        """
        Create multiple directories from path keys.
        
        Args:
            keys: List of path keys to create
        """
        for key in keys:
            self.get_path(key, create=True)
    
    # RECIPE 0: Data preparation convenience methods
    def get_nfi4_database_dir(self) -> Path:
        """Get NFI4 database directory."""
        return self.get_path('forest_inventory') / self.subdirs['forest_inventory']['nfi4']
    
    def get_forest_type_maps_dir(self) -> Path:
        """Get forest type maps directory (MFE + boundaries)."""
        return self.get_path('forest_type_maps')
    
    def get_wood_density_file(self) -> Path:
        """Get wood density database file."""
        return (self.get_path('wood_density') / 
                self.subdirs['wood_density']['database'])
    
    def get_als_data_dir(self) -> Path:
        """Get ALS canopy height data directory."""
        return (self.get_path('als_canopy_height') / 
                self.subdirs['als_canopy_height']['data'])
    
    def get_als_metadata_dir(self) -> Path:
        """Get ALS tile metadata directory."""
        return (self.get_path('als_canopy_height') / 
                self.subdirs['als_canopy_height']['tile_metadata'])
    
    def get_forest_inventory_codes_file(self) -> Path:
        """Get NFI4 codes file."""
        return (self.get_path('forest_inventory') / 
                self.subdirs['forest_inventory']['codes'])
    
    def get_corine_land_cover_file(self) -> Path:
        """Get Corine land cover file."""
        return self.get_path('land_cover') / "U2018_CLC2018_V2020_20u1.tif"
    
    # RECIPE 1: Height prediction convenience methods
    def get_pretrained_height_models_dir(self) -> Path:
        """Get pretrained height models directory."""
        return self.get_path('pretrained_height_models')
    
    def get_height_maps_tmp_raw_dir(self) -> Path:
        """Get height maps raw patches directory."""
        return self.get_path('height_maps') / self.subdirs['height_maps']['tmp_raw']
    
    def get_height_maps_tmp_merged_dir(self) -> Path:
        """Get height maps 120km merged tiles directory."""
        return self.get_path('height_maps') / self.subdirs['height_maps']['tmp_merged_120km']
    
    def get_height_maps_10m_dir(self) -> Path:
        """Get height maps sanitized 10m directory.""" 
        return self.get_path('height_maps') / self.subdirs['height_maps']['sanitized_10m']
    
    def get_height_maps_100m_dir(self) -> Path:
        """Get height maps country-wide 100m directory."""
        return self.get_path('height_maps') / self.subdirs['height_maps']['country_100m']
    
    # RECIPE 2: Biomass estimation convenience methods
    def get_biomass_maps_raw_dir(self) -> Path:
        """Get biomass maps raw directory (no LC masking)."""
        return self.get_path('biomass_maps') / self.subdirs['biomass_maps']['raw']
    
    def get_biomass_maps_per_forest_type_dir(self) -> Path:
        """Get biomass maps per forest type directory (LC masked)."""
        return self.get_path('biomass_maps') / self.subdirs['biomass_maps']['per_forest_type']
    
    def get_biomass_maps_full_country_dir(self) -> Path:
        """Get biomass maps full country directory (merged)."""
        return self.get_path('biomass_maps') / self.subdirs['biomass_maps']['full_country']
    
    def get_allometries_dir(self) -> Path:
        """Get allometries output directory."""
        return self.get_path('allometries')
    
    def get_fitted_parameters_file(self) -> Path:
        """Get fitted allometric parameters file."""
        return self.get_path('allometries') / self.subdirs['allometries']['fitted_parameters']
    
    def get_bgb_ratios_file(self) -> Path:
        """Get calculated BGB ratios file."""
        return self.get_path('allometries') / self.subdirs['allometries']['bgb_ratios']
    
    # RECIPE 3: Analysis convenience methods
    def get_bioclim_harmonized_dir(self) -> Path:
        """Get harmonized climate data directory."""
        return self.get_path('bioclim') / self.subdirs['bioclim']['harmonized']
    
    def get_bioclim_variables_dir(self) -> Path:
        """Get bioclimatic variables directory."""
        return self.get_path('bioclim') / self.subdirs['bioclim']['variables']
    
    def get_bioclim_anomalies_dir(self) -> Path:
        """Get climate anomalies directory."""
        return self.get_path('bioclim') / self.subdirs['bioclim']['anomalies']
    
    # Component configuration overrides - ALL RECIPES HARMONIZED
    def get_component_config_overrides(self, component_name: str) -> Dict[str, str]:
        """
        Get configuration path overrides for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dict[str, str]: Configuration path overrides
        """
        overrides = {}
        
        if component_name == 'biomass_model':
            overrides = {
                'data.input_data_dir': str(self.get_height_maps_100m_dir()),      # 100m for biomass estimation
                'data.input_10m_dir': str(self.get_height_maps_10m_dir()),        # 10m for allometry calibration
                'data.nfi_processed_dir': str(self.get_path('forest_inventory_processed')),
                'data.forest_types_file': str(self.get_path('forest_inventory') / "Forest_Types_Tiers.csv"),
                'data.forest_type_maps_dir': str(self.get_forest_type_maps_dir()),
                'data.allometries_output_dir': str(self.get_allometries_dir()),   # Fitted parameters output
                'data.output_base_dir': str(self.get_path('biomass_maps')),
                'data.corine_land_cover': str(self.get_corine_land_cover_file())
            }
        
        elif component_name == 'canopy_height_model':
            overrides = {
                'data.data_dir': str(self.data_root),
                'data.sentinel2_dir': str(self.get_path('sentinel2_mosaics').relative_to(self.data_root)),
                'data.sentinel2_processed_dir': str(self.get_path('sentinel2_processed').relative_to(self.data_root)),
                'data.als_processed_dir': str(self.get_path('als_canopy_height_processed').relative_to(self.data_root)),
                'data.predictions_dir': str(self.get_path('height_maps').relative_to(self.data_root)),
                'data.checkpoint_dir': str(self.get_pretrained_height_models_dir().relative_to(self.data_root)),
                'data.checkpoint_path': str(self.get_pretrained_height_models_dir() / "height_model_checkpoint.pkl")
            }
        
        elif component_name == 'biomass_analysis':
            overrides = {
                'data.base_dir': str(self.data_root),
                'data.biomass_maps_dir': str(self.get_biomass_maps_full_country_dir().relative_to(self.data_root)),
                'data.country_bounds_path': str(self.get_forest_type_maps_dir() / "gadm41_ESP_0.shp"),
                'data.forest_types_file': str(self.get_path('forest_inventory') / "Forest_Types_Tiers.csv"),
                'output.base_output_dir': str(self.get_path('analysis_outputs').relative_to(self.data_root)),
                'output.tables_dir': str(self.get_path('tables').relative_to(self.data_root))
            }
        
        elif component_name == 'climate_biomass_analysis':
            overrides = {
                'data.climate_raw_dir': str(self.get_path('climate_raw')),
                'data.bioclim_harmonized_dir': str(self.get_bioclim_harmonized_dir()),
                'data.bioclim_variables_dir': str(self.get_bioclim_variables_dir()),
                'data.bioclim_anomalies_dir': str(self.get_bioclim_anomalies_dir()),
                'data.biomass_maps_dir': str(self.get_biomass_maps_full_country_dir()),
                'output.climate_analysis_dir': str(self.get_path('analysis_outputs') / 'climate_analysis'),
                'output.ml_outputs_dir': str(self.get_path('ml_outputs'))
            }
        
        elif component_name == 'forest_inventory':
            overrides = {
                'data.base_dir': str(self.get_path('forest_inventory')),
                'data.nfi4_dir': str(self.get_nfi4_database_dir()),
                'data.forest_type_maps_dir': str(self.get_forest_type_maps_dir()),
                'data.codes_file': str(self.get_forest_inventory_codes_file()),
                'data.wood_density_file': str(self.get_wood_density_file()),
                'output.base_dir': str(self.get_path('forest_inventory_processed'))
            }
        
        elif component_name == 'sentinel2_processing':
            overrides = {
                'paths.output_dir': str(self.get_path('sentinel2_mosaics')),
                'paths.processed_dir': str(self.get_path('sentinel2_processed')),
                'paths.boundary_file': str(self.get_forest_type_maps_dir() / "gadm41_ESP_0.shp")
            }
        
        return overrides
    
    def __str__(self) -> str:
        """String representation of data paths."""
        return f"CentralDataPaths(root={self.data_root})"
    
    def __repr__(self) -> str:
        """Detailed representation of data paths."""
        return f"CentralDataPaths(root='{self.data_root}', paths={len(self.paths)} defined)"


def get_central_data_paths(data_root: Union[str, Path] = "data") -> CentralDataPaths:
    """
    Convenience function to get centralized data paths instance.
    
    Args:
        data_root: Root directory for all data
        
    Returns:
        CentralDataPaths: Configured data paths instance
    """
    return CentralDataPaths(data_root)