#!/usr/bin/env python3
"""
Recipe: Height Modeling

Reproduces canopy height predictions from Sentinel-2 mosaics using deep learning models:
1. Height prediction using pre-trained model checkpoint
2. Post-processing (merging, sanitization, final mosaics)

This recipe generates the canopy height maps used for biomass estimation.

Usage:
    python 1_canopy_height_prediction_recipe.py [OPTIONS]

Examples:
    # Run complete height modeling
    python 1_canopy_height_prediction_recipe.py
    
    # Specific years only
    python 1_canopy_height_prediction_recipe.py --years 2020 2021 2022
    
    # Use specific model checkpoint
    python 1_canopy_height_prediction_recipe.py --checkpoint /path/to/model.ckpt
    
    # Skip model training (prediction only)
    python 1_canopy_height_prediction_recipe.py --prediction-only

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
from canopy_height_model.scripts.run_prediction import main as run_prediction_main
from canopy_height_model.scripts.run_postprocessing import main as run_postprocessing_main


class HeightModelingRecipe:
    """
    Recipe for height modeling reproduction.
    
    Orchestrates canopy height prediction and post-processing with
    centralized data management.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize height modeling recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='height_recipe'
        )
        
        # Track stage results
        self.stage_results = {}
        
        self.logger.info("Initialized Height Modeling Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for height modeling...")
        
        s2_mosaic_dir = SENTINEL2_MOSAICS_DIR
        
        # Check primary mosaic directory
        if not s2_mosaic_dir.exists():
            self.logger.error(f"Sentinel-2 mosaic directory not found: {s2_mosaic_dir}")
            self.logger.error("Please run '0_data_preparation_recipe.py' first to create Sentinel-2 mosaics")
            return False
        
        mosaic_files = list(s2_mosaic_dir.glob("*.tif"))
        if not mosaic_files:
            self.logger.error(f"No Sentinel-2 mosaic files found in {s2_mosaic_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(mosaic_files)} Sentinel-2 mosaic files")
    
        
        # Check for ALS processed data (from data prep recipe) - NEW
        als_processed_dir = ALS_CANOPY_HEIGHT_PROCESSED_DIR
        if als_processed_dir.exists():
            als_files = list(als_processed_dir.glob("*.tif"))
            if als_files:
                self.logger.info(f"✅ Found {len(als_files)} processed ALS canopy height files")
            else:
                self.logger.info("ℹ️  Processed ALS directory exists but no files found")
        else:
            self.logger.info("ℹ️  No processed ALS data found (optional for height prediction)")
        
        # Check for model checkpoint - UPDATED PATH
        pretrained_models_dir = PRETRAINED_HEIGHT_MODELS_DIR
        if not pretrained_models_dir.exists():
            self.logger.error(f"Pretrained height models directory not found: {pretrained_models_dir}")
            self.logger.error("Expected structure: data/pretrained_height_models/")
            return False
        
        # Look for model checkpoint files
        checkpoint_files = list(pretrained_models_dir.glob("*.ckpt"))
        if not checkpoint_files:
            self.logger.error(f"No model checkpoint files found in {pretrained_models_dir}")
            self.logger.error("Please provide pre-trained model checkpoint (.ckpt)")
            return False
        
        self.logger.info(f"✅ Found {len(checkpoint_files)} model checkpoint files")
        
        # Validate main checkpoint file
        main_checkpoint = HEIGHT_MODEL_CHECKPOINT_FILE
        if main_checkpoint.exists():
            try:
                # Basic file size check
                if main_checkpoint.stat().st_size < 1000000:  # Less than 1MB
                    self.logger.warning("Model checkpoint seems unusually small")
                self.logger.info(f"✅ Main checkpoint found: {main_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Could not validate checkpoint file: {e}")
        else:
            self.logger.warning("Main checkpoint (height_model_checkpoint.pkl) not found, will use first available")
    
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        directories = [
            HEIGHT_MAPS_TMP_DIR,
            HEIGHT_MAPS_TMP_120KM_DIR,
            HEIGHT_MAPS_DIR,
            HEIGHT_MAPS_TMP_RAW_DIR,
            HEIGHT_MAPS_TMP_INTERPOLATION_MASKS_DIR,
            HEIGHT_MAPS_10M_DIR,
            HEIGHT_MAPS_100M_DIR
        ]
    
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def run_height_prediction(self, years: Optional[List[int]] = None,
                             checkpoint_path: Optional[str] = None) -> bool:
        """
        Run height prediction using pre-trained model.
        
        Args:
            years: Specific years to process
            checkpoint_path: Path to model checkpoint
            
        Returns:
            bool: True if successful
        """
        stage_name = "Height Prediction"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            raw_patches_dir = HEIGHT_MAPS_TMP_RAW_DIR
            existing_predictions = list(raw_patches_dir.glob("*.tif")) if raw_patches_dir.exists() else []
            
            if existing_predictions:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_predictions)} prediction files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Determine checkpoint to use - UPDATED PATH
            if checkpoint_path:
                model_checkpoint = Path(checkpoint_path)
            else:
                # Look for default checkpoint in new location
                main_checkpoint = HEIGHT_MODEL_CHECKPOINT_FILE
                if main_checkpoint.exists():
                    model_checkpoint = main_checkpoint
                else:
                    # Use first available checkpoint
                    checkpoint_files = list(pretrained_dir.glob("*.ckpt"))
                    if not checkpoint_files:
                        raise ValueError("No checkpoint files found")
                    model_checkpoint = checkpoint_files[0]
            
            self.logger.info(f"Using model checkpoint: {model_checkpoint}")
            
            try:
                # Prepare arguments for height prediction with new paths
                sys.argv = [
                    'run_prediction.py',
                    '--checkpoint', str(model_checkpoint),
                ]
                
                # Run the prediction
                result = run_prediction_main()
                
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
                
            finally:
                sys.argv = old_argv
        
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"❌ {stage_name} failed with error: {str(e)}")
            return False
    
    def run_height_postprocessing(self, years: Optional[List[int]] = None) -> bool:
        """
        Run height prediction post-processing pipeline.
        
        Args:
            years: Specific years to process
            
        Returns:
            bool: True if successful
        """
        stage_name = "Height Post-processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed - UPDATED PATHS
            country_100m_dir = HEIGHT_MAPS_100M_DIR
            existing_outputs = list(country_100m_dir.glob("*.tif")) if country_100m_dir.exists() else []
            
            if existing_outputs:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_outputs)} final maps")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
                
            # Run the post-processing
            result = run_postprocessing_main()
            
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
    """Main entry point for height modeling recipe."""
    start_time = time.time()

    # Initialize recipe
    recipe = HeightModelingRecipe() 
    
    # Validate prerequisites
    if not recipe.validate_prerequisites():
        recipe.logger.error("❌ Prerequisites validation failed")
        recipe.logger.error("Please ensure required input data and model checkpoints are available")
        sys.exit(1)
    
    # Create output structure
    recipe.create_output_structure()
    
    # Track overall success
    overall_success = True
    
    success = recipe.run_height_prediction()
    
    overall_success = overall_success and success


    success = recipe.run_height_postprocessing()
    
    overall_success = overall_success and success
    
    # Validate outputs
    if overall_success:
        elapsed_time = time.time() - start_time
        recipe.logger.info(f"Height modeling recipe completed successfully in {elapsed_time/60:.2f} minutes!")
    else:
        recipe.logger.error("Height modeling recipe failed or output validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()