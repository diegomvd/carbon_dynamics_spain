#!/usr/bin/env python3
"""
Recipe: Data Preparation

Reproduces the data preparation pipeline including:
1. Forest inventory processing (NFI data to biomass shapefiles)
2. Sentinel-2 mosaic creation (summer mosaics from STAC catalog)
3. ALS PNOA processing (LiDAR tiles to training data, optional)

This recipe prepares the foundational datasets used throughout the analysis.

Usage:
    python 0_data_preparation_recipe.py 

Examples:
    # Run complete data preparation
    python 0_data_preparation_recipe.py
Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

# Import utilities
from shared_utils.central_data_paths_constants import *
from shared_utils.logging_utils import setup_logging

# Import component scripts
from forest_inventory.scripts.run_nfi_processing import main as run_nfi_processing_main
from sentinel2_processing.scripts.run_postprocessing import main as run_sentinel2_postprocessing_main
from als_pnoa.scripts.run_pnoa_processing import main as run_pnoa_processing_main


class DataPreparationRecipe:
    """
    Recipe for data preparation reproduction.
    
    Orchestrates forest inventory processing and Sentinel-2 mosaic creation
    with centralized data management.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize data preparation recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='data_prep_recipe'
        )
        
        # Track stage results
        self.stage_results = {}
        
        self.logger.info("Initialized Data Preparation Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for data preparation...")
        
        nfi4_dir = FOREST_INVENTORY_RAW_DIR
        if not nfi4_dir.exists():
            self.logger.error(f"NFI4 database directory not found: {nfi4_dir}")
            self.logger.error("Expected structure: data/raw/forest_inventory/nfi4/*.accdb")
            return False
        
        nfi4_files = list(nfi4_dir.glob("*.accdb"))
        if not nfi4_files:
            self.logger.error(f"No NFI4 database files found in: {nfi4_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(nfi4_files)} NFI4 database files")
        
        forest_type_maps_dir = FOREST_TYPE_MAPS_DIR
        if not forest_type_maps_dir.exists():
            self.logger.error(f"Forest type maps directory not found: {forest_type_maps_dir}")
            self.logger.error("Expected structure: data/raw/forest_type_maps/*.shp")
            return False
        
        mfe_files = list(forest_type_maps_dir.glob("*.shp"))
        if not mfe_files:
            self.logger.error(f"No forest type map files found in: {forest_type_maps_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(mfe_files)} forest type map files")
        
        spain_boundaries = SPAIN_BOUNDARIES_FILE
        if not spain_boundaries:
            self.logger.warning("No Spain boundary shapefile found in forest_type_maps")
            self.logger.warning("Processing may fail without country boundaries")
        else:
            self.logger.info(f"✅ Found Spain boundary: {spain_boundaries[0].name}")
        
        nfi4_codes_file = NFI4_SPECIES_CODE_FILE
        if not nfi4_codes_file.exists():
            self.logger.error(f"NFI4 codes file not found: {nfi4_codes_file}")
            return False
        
        self.logger.info(f"✅ NFI4 codes file found")
        
        # Check for forest types CSV
        forest_types_file = NFI4_FOREST_TYPES_CSV
        if not forest_types_file.exists():
            self.logger.error(f"Forest types file not found: {forest_types_file}")
            return False
        
        self.logger.info(f"Forest types file found")
        
        wood_density_file = WOOD_DENSITY_FILE
        if not wood_density_file.exists():
            self.logger.error(f"Wood density database not found: {wood_density_file}")
            return False
        
        self.logger.info(f"Wood density database found")
        
        als_data_dir = ALS_CANOPY_HEIGHT_RAW_DIR
        if not als_data_dir.exists():
            self.logger.warning(f"ALS canopy height data directory not found: {als_data_dir}")
            self.logger.warning("ALS PNOA processing will be skipped")
        else:
            als_files = list(als_data_dir.glob("NDSM-VEGETACION-*.tif"))
            if als_files:
                self.logger.info(f"✅ Found {len(als_files)} ALS canopy height files")
            else:
                self.logger.warning("ALS data directory exists but no NDSM files found")
        
        als_metadata_dir = ALS_METADATA_DIR
        if not als_metadata_dir.exists():
            self.logger.warning(f"ALS tile metadata directory not found: {als_metadata_dir}")
        else:
            metadata_files = list(als_metadata_dir.rglob("*.shp"))
            if metadata_files:
                self.logger.info(f"✅ Found {len(metadata_files)} ALS metadata files")
            else:
                self.logger.warning("ALS metadata directory exists but no shapefiles found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        directories = [
            FOREST_INVENTORY_PROCESSED_DIR,
            SENTINEL2_MOSAICS_DIR,
            ALS_CANOPY_HEIGHT_PROCESSED_DIR,
            SENTINEL2_PROCESSED_DIR
        ]
    
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Output directory structure created")
    
    def run_forest_inventory_processing(self) -> bool:
        """
        Run forest inventory processing.
        
        Returns:
            bool: True if successful
        """
        stage_name = "Forest Inventory Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            forest_out_dir = FOREST_INVENTORY_PROCESSED_DIR
            existing_shapefiles = list(forest_out_dir.glob("*.shp"))
            
            if existing_shapefiles:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_shapefiles)} shapefiles")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run forest inventory processing with new paths
            import sys
            old_argv = sys.argv.copy()
            
            # Run the processing
            result = run_nfi_processing_main()
            success = result == 0
            stage_time = time.time() - stage_start
            
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
            
            self.logger.error(f"❌ {stage_name} failed with error: {str(e)}")
            return False
    
    def run_sentinel2_processing(self) -> bool:
        """
        Run Sentinel-2 mosaic processing.
        
        Args:
            years: Specific years to process
            bbox: Bounding box (south, west, north, east)
            
        Returns:
            bool: True if successful
        """
        stage_name = "Sentinel-2 Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            mosaic_dir = SENTINEL2_MOSAICS_DIR
            existing_mosaics = list(mosaic_dir.glob("*.tif"))
            
            if existing_mosaics and len(existing_mosaics) >= 5:  # Expect multiple years
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_mosaics)} mosaics")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run the processing
            result = run_sentinel2_postprocessing_main()
            
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
    
    def run_als_pnoa_processing(self) -> bool:
        """
        Run ALS PNOA processing to prepare training data.
        
        Returns:
            bool: True if successful
        """
        stage_name = "ALS PNOA Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if ALS data is available - UPDATED PATH
        als_data_dir = ALS_CANOPY_HEIGHT_RAW_DIR
        if not als_data_dir.exists():
            self.logger.info(f"ℹ️  Skipping {stage_name} - no ALS data found")
            self.stage_results[stage_name] = {
                'success': True,
                'duration_minutes': 0,
                'result': 'skipped_no_data'
            }
            return True
        
        stage_start = time.time()
        
        try:
            # Check if already completed - UPDATED PATH
            als_output_dir = ALS_CANOPY_HEIGHT_PROCESSED_DIR
            existing_outputs = list(als_output_dir.glob("PNOA_*.tif"))
            
            if existing_outputs:
                self.logger.info(f"{stage_name} - Found existing outputs: {len(existing_outputs)} files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True

            # Run the processing
            result = run_pnoa_processing_main()
            
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
    """Main entry point for data preparation recipe."""
    start_time = time.time()
    
    # Initialize recipe
    recipe = DataPreparationRecipe()
    
    if not recipe.validate_prerequisites():
        recipe.logger.error("❌ Prerequisites validation failed")
        recipe.logger.error("Please ensure required input data and model checkpoints are available")
        sys.exit(1)

    # Create output structure
    recipe.create_output_structure()
    
    # Track overall success
    overall_success = True
    
  
    success = recipe.run_forest_inventory_processing()
    
    overall_success = overall_success and success


    success = recipe.run_sentinel2_processing()
    
    overall_success = overall_success and success


    success = recipe.run_als_pnoa_processing()
    
    overall_success = overall_success and success
    
    # Validate outputs
    if overall_success :
        elapsed_time = time.time() - start_time
        recipe.logger.info(f"Data preparation recipe completed successfully in {elapsed_time/60:.2f} minutes!")
    else:
        recipe.logger.error("Data preparation recipe failed at some stage.")
        sys.exit(1)
    
if __name__ == "__main__":
    main()