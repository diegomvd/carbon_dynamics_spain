"""
Biomass Estimation Pipeline

Main processing pipeline for biomass estimation using allometric relationships
and Monte Carlo uncertainty quantification. Supports multi-scale processing
with forest type specific allometries.
Updated to use CentralDataPaths instead of config file paths.

Author: Diego Bengochea
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import dask
from dask.distributed import Client

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory, find_files
from shared_utils.central_data_paths_constants import *

# Component imports
from .allometry import AllometryManager
from .monte_carlo import MonteCarloEstimator
from .io_utils import RasterManager
from .dask_utils import DaskClusterManager


class BiomassEstimationPipeline:
    """
    Main pipeline for biomass estimation with Monte Carlo uncertainty quantification.
    
    This pipeline processes canopy height rasters through allometric relationships
    to generate biomass maps (AGBD, BGBD, Total) with uncertainty estimates.
    Supports forest type specific processing and distributed computing.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the biomass estimation pipeline.
        
        Args:
            config: Configuration dictionary (processing parameters only)
        """
        # Store configuration and data paths
        self.config = load_config(config_path, component='biomass_estimation')
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_estimation',
            log_file=self.config['logging'].get('log_file')
        )
        
        self.allometry_manager = AllometryManager()
        self.monte_carlo = MonteCarloEstimator(self.config)
        self.raster_manager = RasterManager(self.config)
        self.dask_manager = DaskClusterManager(self.config)
        
        # Pipeline state
        self.client = None
        self.start_time = None
        
        self.logger.info(f"Initialized BiomassEstimationPipeline")
    
    def setup_dask_cluster(self) -> None:
        """Setup Dask distributed computing cluster."""
        self.logger.info("Setting up Dask cluster...")
        self.client = self.dask_manager.setup_cluster()
        
        if self.client:
            self.logger.info(f"Dask cluster ready: {self.client}")
            self.logger.info(f"Dashboard: {self.client.dashboard_link}")
        else:
            self.logger.warning("Failed to setup Dask cluster, using synchronous processing")
    
    def validate_inputs(self) -> bool:
        """
        Validate that all required input files and directories exist.
        Uses CentralDataPaths instead of config paths.
        
        Returns:
            bool: True if all inputs are valid
        """
        self.logger.info("Validating input data...")
        
        # Check height maps directory - UPDATED: Use CentralDataPaths
        height_maps_dir = HEIGHT_MAPS_100M_DIR
        if not height_maps_dir.exists():
            self.logger.error(f"Height maps directory not found: {height_maps_dir}")
            return False
        
        # Check for height map files
        height_files = []
        for year_dir in height_maps_dir.iterdir():
            if year_dir.is_dir():
                year_files = list(year_dir.glob("*.tif"))
                height_files.extend(year_files)
        
        if not height_files:
            self.logger.error(f"No height map files found in {height_maps_dir}")
            return False
        
        # Check forest type maps directory - UPDATED: Use CentralDataPaths
        forest_masks_dir = FOREST_TYPE_MASKS_DIR
        if not forest_masks_dir.exists():
            self.logger.error(f"Forest type maps directory not found: {forest_masks_dir}")
            return False
        
        # Check allometric data validation
        if not self.allometry_manager.validate_allometric_data():
            self.logger.error("Allometric data validation failed")
            return False
        
        self.logger.info(f"Input validation passed. Found {len(height_files)} height map files.")
        return True
    
    def setup_output_directories(self) -> Dict[str, Path]:
        """
        Create output directory structure using CentralDataPaths.
        
        Returns:
            Dict[str, Path]: Dictionary of output directories by biomass type
        """
        self.logger.info("Setting up output directories...")
        
        base_dir = BIOMASS_MAPS_RAW_DIR
        
        output_dirs = {}
        for biomass_type in self.config['output']['types']:
            # Create subdirectory for this biomass type
            if biomass_type.lower() == 'agbd':
                subdir_name = 'AGBD_MC_100m'
            elif biomass_type.lower() == 'bgbd':
                subdir_name = 'BGBD_MC_100m'
            elif biomass_type.lower() == 'total':
                subdir_name = 'TBD_MC_100m'
            else:
                subdir_name = f'{biomass_type.upper()}_MC_100m'
            
            output_path = base_dir / subdir_name
            ensure_directory(output_path)
            output_dirs[biomass_type] = output_path
            self.logger.info(f"Created output directory for {biomass_type}: {output_path}")
        
        return output_dirs
    
    def find_height_files_for_year(self, year: int) -> List[Path]:
        """
        Find height raster files for a specific year using CentralDataPaths.
        
        Args:
            year: Year to search for
            
        Returns:
            List of matching height raster files
        """
        height_maps_dir = HEIGHT_MAPS_100M_DIR
        year_dir = height_maps_dir / str(year)
        
        if not year_dir.exists():
            self.logger.warning(f"No height maps found for year {year} in {year_dir}")
            return []
        
        height_files = list(year_dir.glob("*.tif"))
        self.logger.debug(f"Found {len(height_files)} height files for year {year}")
        return sorted(height_files)
    
    def find_forest_mask_files(self) -> List[Path]:
        """
        Find forest type mask files using CentralDataPaths.
        
        Returns:
            List of forest type mask files
        """
       
        masks_dir = FOREST_TYPE_MASKS_DIR
        
        if not masks_dir.exists():
            self.logger.warning(f"Forest masks directory not found: {masks_dir}")
            return []
        
        # Look for mask files (shapefiles or rasters)
        mask_files = []
        for pattern in ["*.shp", "*.tif"]:
            mask_files.extend(list(masks_dir.glob(pattern)))
        
        self.logger.debug(f"Found {len(mask_files)} forest mask files")
        return sorted(mask_files)
    
    def process_year(self, year: int, output_dirs: Dict[str, Path]) -> bool:
        """
        Process all forest types for a specific year.
        
        Args:
            year: Year to process
            output_dirs: Output directories by biomass type
            
        Returns:
            bool: True if processing succeeded
        """
        self.logger.info(f"Processing year {year}...")
        
        try:
            # Find height files for this year
            height_files = self.find_height_files_for_year(year)
            
            if not height_files:
                self.logger.error(f"No height files found for year {year}")
                return False
            
            # Find forest mask files
            mask_files = self.find_forest_mask_files()
            
            if not mask_files:
                self.logger.error("No forest mask files found")
                return False
            
            # Process each forest type
            forest_types = self.allometry_manager.get_available_forest_types()
            successful_types = 0
            
            for forest_type in forest_types:
                try:
                    # Find appropriate mask file for this forest type
                    mask_file = self._find_mask_for_forest_type(forest_type, mask_files)
                    
                    if mask_file:
                        success = self.process_forest_type(
                            year, forest_type, height_files, mask_file, output_dirs
                        )
                        if success:
                            successful_types += 1
                    else:
                        self.logger.warning(f"No mask file found for forest type {forest_type}")
                
                except Exception as e:
                    self.logger.error(f"Error processing forest type {forest_type}: {str(e)}")
                    continue
            
            self.logger.info(f"Year {year} processing complete: {successful_types}/{len(forest_types)} forest types successful")
            return successful_types > 0
            
        except Exception as e:
            self.logger.error(f"Error processing year {year}: {str(e)}")
            return False
    
    def _find_mask_for_forest_type(self, forest_type: str, mask_files: List[Path]) -> Optional[Path]:
        """
        Find the appropriate mask file for a forest type.
        
        Args:
            forest_type: Forest type identifier
            mask_files: List of available mask files
            
        Returns:
            Path to mask file or None if not found
        """
        # Simple matching logic - could be enhanced
        for mask_file in mask_files:
            if forest_type.lower() in mask_file.name.lower():
                return mask_file
        
        # Fallback: return first mask file if no specific match
        return mask_files[0] if mask_files else None
    
    def process_forest_type(
        self,
        year: int,
        forest_type: str,
        height_files: List[Path],
        mask_file: Path,
        output_dirs: Dict[str, Path]
    ) -> bool:
        """
        Process a specific forest type for a year.
        
        Args:
            year: Year being processed
            forest_type: Forest type identifier
            height_files: List of height raster files
            mask_file: Forest type mask file
            output_dirs: Output directories
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            self.logger.info(f"Processing forest type {forest_type} for year {year}")
            
            # Get allometric parameters for this forest type
            agb_params, bgb_params = self.allometry_manager.get_allometry_parameters(forest_type)
            
            if not agb_params:
                self.logger.warning(f"No allometry parameters found for forest type {forest_type}")
                return False
            
            # Combine parameters for Monte Carlo
            allometry_params = {
                'agb_params': agb_params,
                'bgb_params': bgb_params
            }
            
            # Process each biomass type
            for biomass_type in self.config['output']['types']:
                output_dir = output_dirs[biomass_type]
                
                # Generate output filenames
                for measure in self.config['output']['measures']:
                    output_file = self.generate_output_filename(
                        output_dir, biomass_type, measure, year, forest_type
                    )
                    
                    # Skip if file already exists (unless overwrite is enabled)
                    if output_file.exists() and not self.config.get('overwrite', False):
                        self.logger.debug(f"Output file already exists, skipping: {output_file.name}")
                        continue
                    
                    # Run Monte Carlo estimation
                    success = self.monte_carlo.estimate_biomass(
                        height_files=height_files,
                        mask_file=mask_file,
                        allometry_params=allometry_params,
                        biomass_type=biomass_type,
                        measure=measure,
                        output_file=output_file
                    )
                    
                    if success:
                        self.logger.info(f"Generated: {output_file.name}")
                    else:
                        self.logger.error(f"Failed to generate: {output_file.name}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing forest type {forest_type}: {str(e)}")
            return False
    
    def generate_output_filename(
        self, 
        output_dir: Path, 
        biomass_type: str, 
        measure: str, 
        year: int, 
        forest_type: str
    ) -> Path:
        """
        Generate standardized output filename.
        
        Args:
            output_dir: Output directory
            biomass_type: Biomass type (AGBD, BGBD, TBD)
            measure: Statistical measure (mean, uncertainty)
            year: Year
            forest_type: Forest type
            
        Returns:
            Path: Complete output file path
        """
        # Clean forest type for filename
        clean_forest_type = forest_type.replace(' ', '_').replace('/', '_')
        
        # Generate filename
        filename = f"{biomass_type.upper()}_S2_{measure}_{year}_100m_{clean_forest_type}.tif"
        
        return output_dir / filename
    
    def run_full_pipeline(self, years: Optional[List[int]] = None) -> bool:
        """
        Execute the complete biomass estimation pipeline.
        
        Args:
            years: List of years to process (uses config default if None)
            
        Returns:
            bool: True if pipeline completed successfully
        """
        self.logger.info("Starting biomass estimation pipeline...")
        self.start_time = time.time()
        
        try:
            # Use default years from config if not provided
            if years is None:
                years = self.config['processing']['target_years']
            
            # Validate inputs
            if not self.validate_inputs():
                return False
            
            # Setup output directories
            output_dirs = self.setup_output_directories()
            
            # Setup Dask cluster if enabled
            if self.config.get('compute', {}).get('use_dask', False):
                self.setup_dask_cluster()
            
            # Process each year
            successful_years = 0
            for year in years:
                success = self.process_year(year, output_dirs)
                if success:
                    successful_years += 1
                else:
                    self.logger.error(f"Failed to process year {year}")
            
            # Log completion
            duration = time.time() - self.start_time
            self.logger.info(f"Pipeline completed in {duration:.2f} seconds")
            self.logger.info(f"Successfully processed {successful_years}/{len(years)} years")
            
            return successful_years > 0
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False
        
        finally:
            # Cleanup Dask cluster
            if self.client:
                self.client.close()