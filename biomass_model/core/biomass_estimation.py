"""
Biomass Estimation Pipeline

Main processing pipeline for biomass estimation using allometric relationships
and Monte Carlo uncertainty quantification. Supports multi-scale processing
with forest type specific allometries.

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

# Component imports (to be updated with existing modules)
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
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="biomass_estimation")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_estimation',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Initialize managers
        self.allometry_manager = AllometryManager(self.config)
        self.monte_carlo = MonteCarloEstimator(self.config)
        self.raster_manager = RasterManager(self.config)
        self.dask_manager = DaskClusterManager(self.config)
        
        # Pipeline state
        self.client = None
        self.start_time = None
        
        self.logger.info(f"Initialized BiomassEstimationPipeline")
        self.logger.info(f"Configuration loaded from: {self.config['_meta']['config_file']}")
    
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
        
        Returns:
            bool: True if all inputs are valid
        """
        self.logger.info("Validating input data...")
        
        required_files = [
            self.config['data']['allometries_file'],
            self.config['data']['forest_types_file'], 
            self.config['data']['bgb_coeffs_file']
        ]
        
        required_dirs = [
            self.config['data']['input_data_dir'],
            self.config['data']['masks_dir']
        ]
        
        # Check required files
        for file_path in required_files:
            if not Path(file_path).exists():
                self.logger.error(f"Required file not found: {file_path}")
                return False
        
        # Check required directories
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                self.logger.error(f"Required directory not found: {dir_path}")
                return False
        
        # Check input rasters
        input_files = find_files(
            self.config['data']['input_data_dir'],
            self.config['processing']['file_pattern']
        )
        
        if not input_files:
            self.logger.error(f"No input raster files found in {self.config['data']['input_data_dir']}")
            return False
        
        self.logger.info(f"Input validation passed. Found {len(input_files)} input rasters.")
        return True
    
    def setup_output_directories(self) -> Dict[str, Path]:
        """
        Create output directory structure.
        
        Returns:
            Dict[str, Path]: Dictionary of output directories by biomass type
        """
        self.logger.info("Setting up output directories...")
        
        base_dir = Path(self.config['data']['output_base_dir'])
        no_mask_dir = base_dir / self.config['data']['biomass_no_masking_dir']
        
        output_dirs = {}
        for biomass_type in self.config['output']['types']:
            subdir_name = self.config['data']['subdirs'][biomass_type]
            output_path = no_mask_dir / subdir_name
            ensure_directory(output_path)
            output_dirs[biomass_type] = output_path
            self.logger.info(f"Created output directory for {biomass_type}: {output_path}")
        
        return output_dirs
    
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
            # Find height rasters for this year
            height_files = self.raster_manager.find_height_rasters_for_year(year)
            
            if not height_files:
                self.logger.warning(f"No height rasters found for year {year}")
                return False
            
            # Find forest type masks for this year
            mask_files = self.raster_manager.find_mask_files_for_year(year)
            
            if not mask_files:
                self.logger.warning(f"No forest type masks found for year {year}")
                return False
            
            # Process each forest type
            success_count = 0
            total_count = len(mask_files)
            
            for mask_file in mask_files:
                forest_type_code = self.raster_manager.extract_forest_type_from_filename(mask_file)
                
                if self.process_forest_type_year(year, forest_type_code, height_files, mask_file, output_dirs):
                    success_count += 1
                
            self.logger.info(f"Year {year} processing: {success_count}/{total_count} forest types succeeded")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing year {year}: {str(e)}")
            return False
    
    def process_forest_type_year(
        self, 
        year: int, 
        forest_type_code: str, 
        height_files: List[Path],
        mask_file: Path,
        output_dirs: Dict[str, Path]
    ) -> bool:
        """
        Process biomass estimation for a specific forest type and year.
        
        Args:
            year: Year being processed
            forest_type_code: Forest type code
            height_files: List of height raster files
            mask_file: Forest type mask file
            output_dirs: Output directories
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            self.logger.info(f"Processing forest type {forest_type_code} for year {year}")
            
            # Get allometric parameters for this forest type
            allometry_params = self.allometry_manager.get_allometry_parameters(forest_type_code)
            
            if not allometry_params:
                self.logger.warning(f"No allometry parameters found for forest type {forest_type_code}")
                return False
            
            # Process each biomass type
            for biomass_type in self.config['output']['types']:
                output_dir = output_dirs[biomass_type]
                
                # Generate output filenames
                for measure in self.config['output']['measures']:
                    output_file = self.generate_output_filename(
                        output_dir, biomass_type, measure, year, forest_type_code
                    )
                    
                    # Skip if file already exists (unless overwrite is enabled)
                    if output_file.exists():
                        self.logger.info(f"Output file already exists, skipping: {output_file.name}")
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
            self.logger.error(f"Error processing forest type {forest_type_code}: {str(e)}")
            return False
    
    def generate_output_filename(
        self, 
        output_dir: Path, 
        biomass_type: str, 
        measure: str, 
        year: int, 
        forest_type_code: str
    ) -> Path:
        """
        Generate standardized output filename.
        
        Args:
            output_dir: Output directory
            biomass_type: Type of biomass (agbd, bgbd, total)
            measure: Statistical measure (mean, uncertainty)
            year: Year
            forest_type_code: Forest type code
            
        Returns:
            Path: Output file path
        """
        prefix = self.config['output']['prefix']
        filename = f"{biomass_type.upper()}_{prefix}_{measure}_{year}_100m_code{forest_type_code}.tif"
        return output_dir / filename
    
    def run_pipeline(self, years: Optional[List[int]] = None) -> bool:
        """
        Run the complete biomass estimation pipeline.
        
        Args:
            years: List of years to process. If None, uses config default.
            
        Returns:
            bool: True if pipeline completed successfully
        """
        self.start_time = time.time()
        
        # Log pipeline start
        from shared_utils.logging_utils import log_pipeline_start
        log_pipeline_start(self.logger, "Biomass Estimation", self.config)
        
        try:
            # Validate inputs
            if not self.validate_inputs():
                return False
            
            # Setup distributed computing
            self.setup_dask_cluster()
            
            # Setup output directories
            output_dirs = self.setup_output_directories()
            
            # Determine years to process
            if years is None:
                years = self.config['processing']['target_years']
            
            self.logger.info(f"Processing years: {years}")
            
            # Process each year
            successful_years = []
            for year in years:
                if self.process_year(year, output_dirs):
                    successful_years.append(year)
            
            # Pipeline completion
            success = len(successful_years) > 0
            elapsed_time = time.time() - self.start_time
            
            from shared_utils.logging_utils import log_pipeline_end
            log_pipeline_end(self.logger, "Biomass Estimation", success, elapsed_time)
            
            if success:
                self.logger.info(f"Successfully processed years: {successful_years}")
            else:
                self.logger.error("Pipeline failed - no years processed successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            return False
        
        finally:
            # Cleanup
            if self.client:
                self.client.close()
    
    def run_single_forest_type(
        self, 
        year: int, 
        forest_type_code: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Run pipeline for a single forest type and year (useful for testing/debugging).
        
        Args:
            year: Year to process
            forest_type_code: Forest type code
            output_dir: Optional custom output directory
            
        Returns:
            bool: True if processing succeeded
        """
        self.logger.info(f"Running single forest type processing: {forest_type_code}, year {year}")
        
        if not self.validate_inputs():
            return False
        
        # Setup output directories
        if output_dir:
            custom_output_dirs = {}
            for biomass_type in self.config['output']['types']:
                custom_output_dirs[biomass_type] = ensure_directory(Path(output_dir) / biomass_type)
            output_dirs = custom_output_dirs
        else:
            output_dirs = self.setup_output_directories()
        
        # Find input files
        height_files = self.raster_manager.find_height_rasters_for_year(year)
        mask_files = self.raster_manager.find_mask_files_for_year(year)
        
        # Find specific mask file
        mask_file = None
        for mf in mask_files:
            if forest_type_code in mf.stem:
                mask_file = mf
                break
        
        if not mask_file:
            self.logger.error(f"Mask file not found for forest type {forest_type_code}")
            return False
        
        return self.process_forest_type_year(year, forest_type_code, height_files, mask_file, output_dirs)
