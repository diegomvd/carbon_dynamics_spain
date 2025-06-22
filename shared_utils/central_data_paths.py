#!/usr/bin/env python3
"""
Centralized Data Path Management

Provides standardized data path management for all components in the
Iberian Carbon Assessment Pipeline. Ensures consistent data organization
and simplifies path handling across recipe scripts.

Author: Diego Bengochea
"""

from pathlib import Path
from typing import Dict, Union, Optional, List
import logging


class CentralDataPaths:
    """
    Centralized data path management for all pipeline components.
    
    Provides standardized paths for raw data, processed outputs, results,
    and models following the agreed data/ directory structure.
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
        
        # Define standard data structure mapping
        self.paths = {
            # Raw data inputs
            'forest_inventory': self.raw / "forest_inventory",
            'sentinel2_raw': self.raw / "sentinel2",
            'climate_raw': self.raw / "climate",
            'reference_data': self.raw / "reference_data",
            
            # Processed intermediate data
            'sentinel2_mosaics': self.processed / "sentinel2_mosaics",
            'height_predictions': self.processed / "height_predictions", 
            'biomass_maps': self.processed / "biomass_maps",
            'climate_variables': self.processed / "climate_variables",
            'forest_type_masks': self.processed / "forest_type_masks",
            'height_range_masks': self.processed / "height_range_masks",
            
            # Analysis results
            'figures': self.results / "figures",
            'tables': self.results / "tables", 
            'analysis_outputs': self.results / "analysis_outputs",
            'ml_outputs': self.results / "ml_outputs",
            
            # Model files
            'height_model': self.models / "height_model_checkpoint.pkl",
            'ml_models': self.models / "ml_models"
        }
        
        # Component-specific subdirectory mappings
        self.subdirs = {
            'biomass_maps': {
                'no_masking': 'biomass_no_LC_masking',
                'with_mask': 'with_annual_crop_mask', 
                'merged': 'biomass_maps_merged'
            },
            'height_predictions': {
                'patches': 'patches',
                'merged_tiles': 'merged_tiles',
                'sanitized': 'sanitized',
                'final_mosaics': 'final_mosaics',
                'merged_full_country': 'merged_full_country_interpolated'
            },
            'climate_variables': {
                'raw_outputs': 'climate_outputs',
                'harmonized': 'harmonized_climate',
                'bioclimatic': 'bioclimatic_variables',
                'anomalies': 'climate_anomalies'
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
            Path: Standardized path object
        """
        path = self.paths.get(key, self.data_root / key)
        
        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {path}")
        
        return path
    
    def get_biomass_path(self, biomass_type: str, year: int, 
                        statistic: str = 'mean', with_mask: bool = True,
                        forest_type_code: Optional[str] = None) -> Path:
        """
        Get biomass file path following naming conventions.
        
        Args:
            biomass_type: Type of biomass (TBD, AGBD, BGBD)
            year: Year of data
            statistic: Statistic type (mean, uncertainty)
            with_mask: Whether file has annual crop mask applied
            forest_type_code: Optional forest type code for non-merged files
            
        Returns:
            Path: Full path to biomass file
        """
        # Determine subdirectory
        if with_mask:
            subdir = self.subdirs['biomass_maps']['with_mask']
        else:
            subdir = self.subdirs['biomass_maps']['no_masking']
        
        base_path = self.get_path('biomass_maps') / subdir
        
        # Build filename
        if forest_type_code:
            # Forest type specific file
            filename = f"{biomass_type}_S2_{statistic}_{year}_100m_code{forest_type_code}.tif"
            # Add biomass type subdirectory
            if biomass_type == 'TBD':
                type_subdir = 'TBD_MC_100m'
            elif biomass_type == 'AGBD':
                type_subdir = 'AGBD_MC_100m'
            elif biomass_type == 'BGBD':
                type_subdir = 'BGBD_MC_100m'
            else:
                type_subdir = f"{biomass_type}_MC_100m"
            
            full_path = base_path / type_subdir / filename
        else:
            # Merged file
            merged_dir = self.get_path('biomass_maps') / self.subdirs['biomass_maps']['merged']
            filename = f"{biomass_type}_S2_{statistic}_{year}_100m_merged.tif"
            full_path = merged_dir / filename
        
        return full_path
    
    def get_height_prediction_path(self, year: int, processing_stage: str = 'final_mosaics') -> Path:
        """
        Get canopy height prediction file path.
        
        Args:
            year: Year of predictions
            processing_stage: Processing stage (patches, merged_tiles, sanitized, final_mosaics, merged_full_country)
            
        Returns:
            Path: Full path to height prediction file
        """
        base_path = self.get_path('height_predictions')
        
        if processing_stage in self.subdirs['height_predictions']:
            subdir = self.subdirs['height_predictions'][processing_stage]
            stage_path = base_path / subdir
        else:
            stage_path = base_path / processing_stage
        
        if processing_stage == 'merged_full_country':
            filename = f"canopy_height_{year}_100m.tif"
        else:
            filename = f"canopy_height_{year}_*.tif"  # Pattern for multiple files
        
        return stage_path / filename
    
    def get_climate_path(self, variable_type: str, year: Optional[int] = None, 
                        variable_name: Optional[str] = None) -> Path:
        """
        Get climate data path.
        
        Args:
            variable_type: Type of climate data (raw_outputs, harmonized, bioclimatic, anomalies)
            year: Optional year for year-specific data
            variable_name: Optional variable name (e.g., 'bio1', 'temperature')
            
        Returns:
            Path: Climate data path
        """
        base_path = self.get_path('climate_variables')
        
        if variable_type in self.subdirs['climate_variables']:
            subdir = self.subdirs['climate_variables'][variable_type]
            var_path = base_path / subdir
        else:
            var_path = base_path / variable_type
        
        # Add year-specific subdirectory if needed
        if year and variable_type == 'anomalies':
            var_path = var_path / f"anomalies_{year}"
        
        # Add specific file if variable name provided
        if variable_name:
            if variable_type == 'bioclimatic':
                filename = f"{variable_name}.tif"
            else:
                filename = f"{variable_name}_{year}.tif" if year else f"{variable_name}.tif"
            var_path = var_path / filename
        
        return var_path
    
    def get_forest_mask_path(self, forest_type_code: str, resolution: str = "100m") -> Path:
        """
        Get forest type mask path.
        
        Args:
            forest_type_code: Forest type code
            resolution: Spatial resolution (default: 100m)
            
        Returns:
            Path: Forest type mask file path
        """
        mask_dir = self.get_path('forest_type_masks') / resolution
        filename = f"forest_type_mask_{forest_type_code}_{resolution}.tif"
        return mask_dir / filename
    
    def get_sentinel2_mosaic_path(self, year: int, tile_id: Optional[str] = None, 
                                 merged: bool = False) -> Path:
        """
        Get Sentinel-2 mosaic path.
        
        Args:
            year: Year of mosaic
            tile_id: Optional tile identifier for individual tiles
            merged: Whether to get merged country-wide mosaic
            
        Returns:
            Path: Sentinel-2 mosaic file path
        """
        base_path = self.get_path('sentinel2_mosaics')
        
        if merged:
            filename = f"S2_summer_mosaic_{year}_merged.tif"
        elif tile_id:
            filename = f"S2_summer_mosaic_{year}_{tile_id}.tif"
        else:
            filename = f"S2_summer_mosaic_{year}_*.tif"  # Pattern for all tiles
        
        return base_path / filename
    
    def create_directories(self, paths: Optional[List[str]] = None) -> None:
        """
        Create directory structure.
        
        Args:
            paths: Optional list of specific paths to create. If None, creates all standard paths.
        """
        if paths is None:
            # Create all standard directories
            paths_to_create = [
                self.raw, self.processed, self.results, self.models,
                self.get_path('forest_inventory'),
                self.get_path('sentinel2_raw'),
                self.get_path('climate_raw'),
                self.get_path('sentinel2_mosaics'),
                self.get_path('height_predictions'),
                self.get_path('biomass_maps'),
                self.get_path('climate_variables'),
                self.get_path('forest_type_masks'),
                self.get_path('figures'),
                self.get_path('tables'),
                self.get_path('analysis_outputs')
            ]
        else:
            paths_to_create = [self.get_path(p) for p in paths]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {path}")
        
        self.logger.info(f"Created data directory structure in: {self.data_root}")
    
    def get_component_config_overrides(self, component_name: str) -> Dict[str, str]:
        """
        Get configuration overrides for a specific component to use centralized paths.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dict[str, str]: Configuration path overrides
        """
        overrides = {}
        
        if component_name == 'biomass_model':
            overrides = {
                'data.input_data_dir': str(self.get_path('height_predictions') / 
                                         self.subdirs['height_predictions']['merged_full_country']),
                'data.masks_dir': str(self.get_path('forest_type_masks') / "100m"),
                'data.output_base_dir': str(self.processed),
                'data.allometries_file': str(self.get_path('forest_inventory') / 
                                            "H_AGB_Allometries_Tiers_ModelCalibrated_Quantiles_15-85_OnlyPowerLaw.csv"),
                'data.forest_types_file': str(self.get_path('forest_inventory') / "Forest_Types_Tiers.csv"),
                'data.bgb_coeffs_file': str(self.get_path('forest_inventory') / "BGBRatios_Tiers.csv"),
                'data.mfe_dir': str(self.get_path('forest_inventory') / "MFESpain"),
                'data.corine_land_cover': str(self.get_path('reference_data') / "corine_land_cover" / 
                                             "U2018_CLC2018_V2020_20u1.tif")
            }
        
        elif component_name == 'canopy_height_model':
            overrides = {
                'data.data_dir': str(self.data_root),
                'data.sentinel2_dir': str(self.get_path('sentinel2_mosaics').relative_to(self.data_root)),
                'data.predictions_dir': str(self.get_path('height_predictions').relative_to(self.data_root)),
                'data.checkpoint_dir': str(self.models.relative_to(self.data_root)),
                'data.checkpoint_path': str(self.get_path('height_model'))
            }
        
        elif component_name == 'biomass_analysis':
            overrides = {
                'data.base_dir': str(self.data_root),
                'data.biomass_maps_dir': str(self.get_path('biomass_maps').relative_to(self.data_root) / 
                                            self.subdirs['biomass_maps']['with_mask']),
                'data.country_bounds_path': str(self.get_path('reference_data').relative_to(self.data_root) / 
                                               "SpainPolygon" / "gadm41_ESP_0.shp"),
                'data.forest_types_csv': str(self.get_path('forest_inventory').relative_to(self.data_root) / 
                                             "Forest_Types_Tiers.csv"),
                'data.forest_type_masks_dir': str(self.get_path('forest_type_masks').relative_to(self.data_root) / "100m"),
                'data.canopy_height_dir': str(self.get_path('height_predictions').relative_to(self.data_root) / 
                                             self.subdirs['height_predictions']['merged_full_country']),
                'output.base_output_dir': str(self.get_path('analysis_outputs').relative_to(self.data_root))
            }
        
        elif component_name == 'climate_biomass_analysis':
            overrides = {
                'data.climate_outputs': str(self.get_path('climate_variables') / 
                                           self.subdirs['climate_variables']['raw_outputs']),
                'data.harmonized_dir': str(self.get_path('climate_variables') / 
                                          self.subdirs['climate_variables']['harmonized']),
                'data.bioclim_dir': str(self.get_path('climate_variables') / 
                                       self.subdirs['climate_variables']['bioclimatic']),
                'data.anomaly_dir': str(self.get_path('climate_variables') / 
                                       self.subdirs['climate_variables']['anomalies']),
                'data.biomass_diff_dir': str(self.get_path('analysis_outputs') / 
                                            "interannual_biomass_differences_relative"),
                'data.training_dataset': str(self.get_path('ml_outputs') / "ml_training_dataset.csv"),
                'data.clustered_dataset': str(self.get_path('ml_outputs') / "ml_dataset_with_clusters.csv")
            }
        
        elif component_name == 'forest_inventory':
            overrides = {
                'data.base_dir': str(self.get_path('forest_inventory')),
                'output.base_dir': str(self.get_path('forest_inventory') / "processed")
            }
        
        elif component_name == 'sentinel2_processing':
            overrides = {
                'paths.output_dir': str(self.get_path('sentinel2_mosaics')),
                'paths.spain_polygon': str(self.get_path('reference_data') / "SpainPolygon" / "gadm41_ESP_0.shp")
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


# Convenience functions for common use cases
def get_biomass_files(data_root: Union[str, Path] = "data", 
                     years: List[int] = None, 
                     biomass_types: List[str] = None) -> Dict[str, List[Path]]:
    """
    Get all biomass files matching criteria.
    
    Args:
        data_root: Root data directory
        years: List of years to include
        biomass_types: List of biomass types to include
        
    Returns:
        Dict mapping (year, type) tuples to file paths
    """
    paths = CentralDataPaths(data_root)
    
    if years is None:
        years = list(range(2017, 2025))  # Default years
    if biomass_types is None:
        biomass_types = ['TBD', 'AGBD', 'BGBD']
    
    files = {}
    for year in years:
        for biomass_type in biomass_types:
            key = f"{year}_{biomass_type}"
            mean_path = paths.get_biomass_path(biomass_type, year, 'mean', with_mask=True)
            uncertainty_path = paths.get_biomass_path(biomass_type, year, 'uncertainty', with_mask=True)
            
            files[key] = {
                'mean': mean_path,
                'uncertainty': uncertainty_path
            }
    
    return files


def get_height_files(data_root: Union[str, Path] = "data", 
                    years: List[int] = None) -> Dict[int, Path]:
    """
    Get canopy height prediction files.
    
    Args:
        data_root: Root data directory
        years: List of years to include
        
    Returns:
        Dict mapping years to height file paths
    """
    paths = CentralDataPaths(data_root)
    
    if years is None:
        years = list(range(2017, 2025))
    
    files = {}
    for year in years:
        files[year] = paths.get_height_prediction_path(year, 'merged_full_country')
    
    return files