#!/usr/bin/env python3
"""
Recipe: Biomass Estimation

Reproduces biomass estimation results from canopy height predictions.
Executes the complete biomass model pipeline:

1. Allometry fitting (always runs - produces calibrated parameters)
2. Biomass estimation (forest type specific maps)  
3. Annual cropland masking
4. Forest type merging (country-wide maps)

This recipe produces the biomass maps used in the paper analysis.

Usage:
    python 2_biomass_estimation_recipe.py [OPTIONS]

Examples:
    # Run complete biomass estimation
    python 2_biomass_estimation_recipe.py

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

# Import utilities
from shared_utils.central_data_paths_constants import *
from shared_utils.logging_utils import setup_logging

# Import component scripts
from biomass_model.scripts.run_full_pipeline import main as run_biomass_pipeline_main


class BiomassEstimationRecipe:
    """
    Recipe for biomass estimation reproduction.
    
    Provides a simple interface for reproducing biomass estimation results
    with centralized data management and error handling.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize biomass estimation recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_recipe'
        )
        
        # Track stage results
        self.stage_results = {}
        
        self.logger.info("Initialized Biomass Estimation Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for biomass estimation...")
        
        height_100m_dir = HEIGHT_MAPS_100M_DIR
        if not height_100m_dir.exists():
            self.logger.error(f"Height maps (100m) directory not found: {height_100m_dir}")
            self.logger.error("Please run '1_canopy_height_prediction_recipe.py' first")
            return False
        
        height_files = list(height_100m_dir.glob("*.tif"))
        if not height_files:
            self.logger.error(f"No height map files found in {height_100m_dir}")
            return False
        
        self.logger.info(f"Found {len(height_files)} country-wide height maps (100m)")
        
        # Check for height maps (10m for allometry calibration) NOT PROVIDED ONLINE BECAUSE OF SIZE CONSTRAINTS
        # height_10m_dir = HEIGHT_MAPS_10M_DIR
        # if not height_10m_dir.exists():
        #     self.logger.error(f"Height maps (10m) directory not found: {height_10m_dir}")
        #     self.logger.error("10m height maps needed for allometry calibration")
        #     return False
        
        height_10m_files = list(height_10m_dir.glob("*.tif"))
        if not height_10m_files:
            self.logger.error(f"No 10m height map files found in {height_10m_dir}")
            return False
        
        self.logger.info(f"Found {len(height_10m_files)} sanitized height maps (10m)")
        
        # Check for processed NFI data 
        nfi_processed_dir = FOREST_INVENTORY_PROCESSED_DIR
        if not nfi_processed_dir.exists():
            self.logger.error(f"Processed NFI directory not found: {nfi_processed_dir}")
            self.logger.error("Please run '0_data_preparation_recipe.py' first")
            return False
        
        nfi_files = list(nfi_processed_dir.glob("*.shp"))
        if not nfi_files:
            self.logger.error(f"No processed NFI shapefiles found in {nfi_processed_dir}")
            return False
        
        self.logger.info(f"Found {len(nfi_files)} processed NFI shapefiles")
        
        # Check for forest type data - UPDATED PATHS
        forest_types_file = FOREST_TYPES_TIERS_FILE
        if not forest_types_file.exists():
            self.logger.error(f"Forest types hierarchy file not found: {forest_types_file}")
            return False
        
        self.logger.info("Forest types hierarchy file found")
        
        # Check for forest type maps - UPDATED PATH
        forest_type_maps_dir = FOREST_TYPE_MAPS_DIR
        if not forest_type_maps_dir.exists():
            self.logger.error(f"Forest type maps directory not found: {forest_type_maps_dir}")
            return False
        
        mfe_files = list(forest_type_maps_dir.glob("*.shp"))
        if not mfe_files:
            self.logger.error(f"No forest type map files found in {forest_type_maps_dir}")
            return False
        
        self.logger.info(f"Found {len(mfe_files)} forest type map files")
        
        # Check for Corine land cover data (for masking)
        corine_path = CORINE_LAND_COVER_FILE
        if not corine_path.exists():
            self.logger.warning(f"Corine land cover data not found: {corine_path}")
            self.logger.warning("Annual cropland masking may not work properly")
        else:
            self.logger.info("Corine land cover data found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        directories = [
            BIOMASS_MAPS_DIR,
            BIOMASS_MAPS_RAW_DIR,
            BIOMASS_MAPS_TEMP_DIR,
            BIOMASS_MAPS_PER_FOREST_TYPE_DIR,
            BIOMASS_MAPS_FULL_COUNTRY_DIR,
            BIOMASS_MASKING_TEMP_DIR,
            ALLOMETRIES_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)  
    
    def run_biomass_estimation(self, 
                              years: Optional[List[int]] = None,
                              continue_on_error: bool = False) -> bool:
        """
        Run the biomass estimation pipeline.
        
        Args:
            years: Specific years to process
            continue_on_error: Continue if a stage fails
            
        Returns:
            bool: True if successful
        """
        stage_name = "Biomass Estimation Pipeline"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            
            # Run the pipeline
            result = run_biomass_pipeline_main()
            
            stage_time = time.time() - stage_start
            success = result == 0
            
            self.stage_results[stage_name] = {
                'success': success,
                'duration_minutes': stage_time / 60,
                'result': result
            }
            
            if success:
                self.logger.info(f"{stage_name} completed successfully in {stage_time/60:.2f} minutes")
            else:
                self.logger.error(f"{stage_name} failed after {stage_time/60:.2f} minutes")
            
            return success
        
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"{stage_name} failed with error: {str(e)}")
            return False
    

def main():
    """Main entry point for biomass estimation recipe."""
    start_time = time.time()

    # Initialize recipe
    recipe = BiomassEstimationRecipe()
    
    # Validate prerequisites
    if not recipe.validate_prerequisites():
        recipe.logger.error("Prerequisites validation failed")
        recipe.logger.error("Please ensure height maps and processed NFI data are available")
        sys.exit(1)
    
    # Create output structure
    recipe.create_output_structure()
    
    # Run biomass estimation pipeline
    result = recipe.run_biomass_estimation()
    success = result == 0
    
    if not success:
        recipe.logger.error("Biomass estimation pipeline failed")
        sys.exit(1)
    
    # Validate outputs
    if success:
        elapsed_time = time.time() - start_time
        recipe.logger.info(f"Biomass estimation recipe completed successfully in {elapsed_time/60:.2f} minutes!")
    else:
        recipe.logger.error("Biomass estimation recipe failed or output validation failed")
        sys.exit(1)




if __name__ == "__main__":
    main()