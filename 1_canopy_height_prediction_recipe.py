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
from shared_utils.central_data_paths import CentralDataPaths
from shared_utils.config_utils import load_config
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
    
    def __init__(self, data_root: str = "data", log_level: str = "INFO"):
        """
        Initialize height modeling recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        # Setup centralized data paths
        self.data_paths = CentralDataPaths(data_root)
        
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
        
        # Check for Sentinel-2 mosaics - UPDATED to use both sources
        s2_mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
        s2_processed_dir = self.data_paths.get_path('sentinel2_processed')
        
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
        
        # Check processed Sentinel-2 directory (from data prep recipe)
        if s2_processed_dir.exists():
            processed_files = list(s2_processed_dir.glob("*.tif"))
            self.logger.info(f"✅ Found {len(processed_files)} processed Sentinel-2 files")
        else:
            self.logger.warning(f"Processed Sentinel-2 directory not found: {s2_processed_dir}")
        
        # Check for ALS processed data (from data prep recipe) - NEW
        als_processed_dir = self.data_paths.get_path('als_canopy_height_processed')
        if als_processed_dir.exists():
            als_files = list(als_processed_dir.glob("*.tif"))
            if als_files:
                self.logger.info(f"✅ Found {len(als_files)} processed ALS canopy height files")
            else:
                self.logger.info("ℹ️  Processed ALS directory exists but no files found")
        else:
            self.logger.info("ℹ️  No processed ALS data found (optional for height prediction)")
        
        # Check for model checkpoint - UPDATED PATH
        pretrained_models_dir = self.data_paths.get_pretrained_height_models_dir()
        if not pretrained_models_dir.exists():
            self.logger.error(f"Pretrained height models directory not found: {pretrained_models_dir}")
            self.logger.error("Expected structure: data/pretrained_height_models/")
            return False
        
        # Look for model checkpoint files
        checkpoint_files = list(pretrained_models_dir.glob("*.pkl")) + list(pretrained_models_dir.glob("*.ckpt"))
        if not checkpoint_files:
            self.logger.error(f"No model checkpoint files found in {pretrained_models_dir}")
            self.logger.error("Please provide pre-trained model checkpoint (.pkl or .ckpt)")
            return False
        
        self.logger.info(f"✅ Found {len(checkpoint_files)} model checkpoint files")
        
        # Validate main checkpoint file
        main_checkpoint = pretrained_models_dir / "height_model_checkpoint.pkl"
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
        
        # Check for processing environment
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"✅ PyTorch available, using device: {device}")
        except ImportError:
            self.logger.warning("PyTorch not available - this may affect model loading")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        # Create main height maps directories - UPDATED STRUCTURE
        self.data_paths.create_directories(['height_maps'])
        
        # Create height maps subdirectories - NEW STRUCTURE
        height_base = self.data_paths.get_path('height_maps')
        for subdir in self.data_paths.subdirs['height_maps'].values():
            (height_base / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Output directory structure created")
        self.logger.info(f"   📁 height_maps/tmp/raw/ - raw prediction patches")
        self.logger.info(f"   📁 height_maps/tmp/merged_120km/ - 120km merged tiles")
        self.logger.info(f"   📁 height_maps/10m/ - sanitized 10m maps")
        self.logger.info(f"   📁 height_maps/100m/ - country-wide 100m map")
    
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
            # Check if already completed - UPDATED PATH
            raw_patches_dir = self.data_paths.get_height_maps_tmp_raw_dir()
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
                pretrained_dir = self.data_paths.get_pretrained_height_models_dir()
                main_checkpoint = pretrained_dir / "height_model_checkpoint.pkl"
                if main_checkpoint.exists():
                    model_checkpoint = main_checkpoint
                else:
                    # Use first available checkpoint
                    checkpoint_files = list(pretrained_dir.glob("*.pkl")) + list(pretrained_dir.glob("*.ckpt"))
                    if not checkpoint_files:
                        raise ValueError("No checkpoint files found")
                    model_checkpoint = checkpoint_files[0]
            
            self.logger.info(f"Using model checkpoint: {model_checkpoint}")
            
            # Run height prediction with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for height prediction with new paths
                sys.argv = [
                    'run_prediction.py',
                    '--checkpoint', str(model_checkpoint),
                    '--sentinel2-mosaics-dir', str(self.data_paths.get_path('sentinel2_mosaics')),
                    '--sentinel2-processed-dir', str(self.data_paths.get_path('sentinel2_processed')),
                    '--als-processed-dir', str(self.data_paths.get_path('als_canopy_height_processed')),
                    '--output-dir', str(raw_patches_dir),
                    '--log-level', 'INFO'
                ]
                
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                # Run the prediction
                result = run_prediction_main()
                
                stage_time = time.time() - stage_start
                success = result if result is not None else True
                
                self.stage_results[stage_name] = {
                    'success': success,
                    'duration_minutes': stage_time / 60,
                    'result': result
                }
                
                if success:
                    self.logger.info(f"✅ {stage_name} completed successfully in {stage_time/60:.2f} minutes")
                else:
                    self.logger.error(f"❌ {stage_name} failed after {stage_time/60:.2f} minutes")
                
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
            country_100m_dir = self.data_paths.get_height_maps_100m_dir()
            existing_outputs = list(country_100m_dir.glob("*.tif")) if country_100m_dir.exists() else []
            
            if existing_outputs:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_outputs)} final maps")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run post-processing with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for post-processing with new path structure
                sys.argv = [
                    'run_postprocessing.py',
                    '--input-dir', str(self.data_paths.get_height_maps_tmp_raw_dir()),
                    '--merged-120km-dir', str(self.data_paths.get_height_maps_tmp_merged_dir()),
                    '--sanitized-10m-dir', str(self.data_paths.get_height_maps_10m_dir()),
                    '--final-100m-dir', str(self.data_paths.get_height_maps_100m_dir()),
                    '--log-level', 'INFO'
                ]
                
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                # Run the post-processing
                result = run_postprocessing_main()
                
                stage_time = time.time() - stage_start
                success = result if result is not None else True
                
                self.stage_results[stage_name] = {
                    'success': success,
                    'duration_minutes': stage_time / 60,
                    'result': result
                }
                
                if success:
                    self.logger.info(f"✅ {stage_name} completed successfully in {stage_time/60:.2f} minutes")
                else:
                    self.logger.error(f"❌ {stage_name} failed after {stage_time/60:.2f} minutes")
                
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
    
    def validate_outputs(self) -> bool:
        """
        Validate that outputs were created successfully.
        
        Returns:
            bool: True if outputs are valid
        """
        self.logger.info("Validating height modeling outputs...")
        
        # Check for final country-wide height maps - UPDATED PATH
        country_100m_dir = self.data_paths.get_height_maps_100m_dir()
        
        if not country_100m_dir.exists():
            self.logger.error(f"Final height maps directory not found: {country_100m_dir}")
            return False
        
        final_files = list(country_100m_dir.glob("*.tif"))
        if not final_files:
            self.logger.error("No final height map files found")
            return False
        
        self.logger.info(f"✅ Found {len(final_files)} final height map files")
        
        # Check intermediate outputs - UPDATED PATHS
        intermediate_checks = [
            ('raw patches', self.data_paths.get_height_maps_tmp_raw_dir()),
            ('120km merged', self.data_paths.get_height_maps_tmp_merged_dir()),
            ('10m sanitized', self.data_paths.get_height_maps_10m_dir())
        ]
        
        for stage_name, stage_dir in intermediate_checks:
            if stage_dir.exists():
                stage_files = list(stage_dir.glob("*.tif"))
                self.logger.info(f"✅ Found {len(stage_files)} files in {stage_name} stage")
            else:
                self.logger.warning(f"⚠️  {stage_name} directory not found: {stage_dir}")
        
        # Basic file validation
        for file_path in final_files[:2]:  # Check first couple files
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb < 10:  # Less than 10MB seems small for height predictions
                    self.logger.warning(f"Small height map file: {file_path} ({file_size_mb:.1f} MB)")
                else:
                    self.logger.debug(f"Height file size OK: {file_path} ({file_size_mb:.1f} MB)")
            except Exception as e:
                self.logger.warning(f"Could not validate file size for {file_path}: {e}")
        
        return True
    
    def print_summary(self) -> None:
        """Print summary of height modeling results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"HEIGHT MODELING RECIPE SUMMARY")
        self.logger.info(f"{'='*60}")
        
        # Show results for each stage
        for stage_name, results in self.stage_results.items():
            status = "✅" if results['success'] else "❌"
            duration = results['duration_minutes']
            self.logger.info(f"  {status} {stage_name}: {duration:.2f} min")
        
        # Check output directories - UPDATED PATHS
        height_base = self.data_paths.get_path('height_maps')
        
        stage_mappings = [
            ('Raw Patches', self.data_paths.get_height_maps_tmp_raw_dir()),
            ('120km Merged', self.data_paths.get_height_maps_tmp_merged_dir()),
            ('10m Sanitized', self.data_paths.get_height_maps_10m_dir()),
            ('100m Country', self.data_paths.get_height_maps_100m_dir())
        ]
        
        for stage_name, stage_dir in stage_mappings:
            if stage_dir.exists():
                stage_files = list(stage_dir.glob("*.tif"))
                self.logger.info(f"📁 {stage_name}: {len(stage_files)} files in {stage_dir}")
        
        # Show NEW data structure
        self.logger.info(f"📂 Data structure created in: {self.data_paths.data_root}")
        self.logger.info(f"   ├── pretrained_height_models/            # Model checkpoints")
        self.logger.info(f"   └── processed/")
        self.logger.info(f"       └── height_maps/                     # NEW structure")
        self.logger.info(f"           ├── tmp/")
        self.logger.info(f"           │   ├── raw/                     # Raw prediction patches")
        self.logger.info(f"           │   └── merged_120km/            # 120km merged tiles")
        self.logger.info(f"           ├── 10m/                         # Sanitized 10m maps")
        self.logger.info(f"           └── 100m/                        # Country-wide 100m maps")
        
        self.logger.info(f"\n🎯 Next steps:")
        self.logger.info(f"   1. Run '2_biomass_estimation_recipe.py' to estimate biomass from heights")
        self.logger.info(f"   2. Check height prediction quality and coverage")
        self.logger.info(f"   3. Use height_maps/100m/ files for biomass estimation")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Height Modeling (Canopy Height Prediction + Post-processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe generates canopy height predictions by executing:
1. Height prediction using pre-trained deep learning model
2. Post-processing pipeline (merging → sanitization → final mosaics)

NEW DATA STRUCTURE:
  data/pretrained_height_models/           # Model checkpoints (was data/models/)
  data/processed/height_maps/              # Output structure (was height_predictions/)
    ├── tmp/raw/                           # Raw prediction patches
    ├── tmp/merged_120km/                  # 120km merged tiles  
    ├── 10m/                               # Sanitized 10m maps
    └── 100m/                              # Country-wide 100m maps

INPUTS FROM DATA PREP RECIPE:
  data/processed/sentinel2_mosaics/        # Primary Sentinel-2 mosaics
  data/processed/sentinel2/                # Processed Sentinel-2 data
  data/processed/als_canopy_height/        # Processed ALS training data (optional)

Examples:
  %(prog)s                              # Run complete height modeling
  %(prog)s --years 2020 2021 2022       # Specific years only
  %(prog)s --checkpoint /path/to/model.ckpt  # Custom model checkpoint
  %(prog)s --prediction-only            # Skip post-processing
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements (NEW PATHS):
  - Sentinel-2 mosaics from data preparation recipe
  - Model checkpoint in data/pretrained_height_models/
  - PyTorch environment (GPU recommended for speed)
  - Processed ALS data (optional, improves accuracy)
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Root directory for data storage (default: data)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process (default: all available)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to specific model checkpoint file'
    )
    
    parser.add_argument(
        '--prediction-only',
        action='store_true',
        help='Run only prediction stage (skip post-processing)'
    )
    
    parser.add_argument(
        '--postprocessing-only',
        action='store_true',
        help='Run only post-processing (skip prediction)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution if a stage fails'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate prerequisites without running pipeline'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for height modeling recipe."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    log_level = 'ERROR' if args.quiet else args.log_level
    
    try:
        # Initialize recipe
        recipe = HeightModelingRecipe(
            data_root=args.data_root,
            log_level=log_level
        )
        
        # Validate prerequisites
        if not recipe.validate_prerequisites():
            recipe.logger.error("❌ Prerequisites validation failed")
            recipe.logger.error("Please ensure required input data and model checkpoints are available")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Track overall success
        overall_success = True
        
        # Run height prediction
        if not args.postprocessing_only:
            success = recipe.run_height_prediction(
                years=args.years,
                checkpoint_path=args.checkpoint
            )
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Height prediction failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Run post-processing
        if not args.prediction_only:
            success = recipe.run_height_postprocessing(years=args.years)
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Height post-processing failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Validate outputs
        if overall_success and recipe.validate_outputs():
            recipe.print_summary()
            elapsed_time = time.time() - start_time
            recipe.logger.info(f"🎉 Height modeling recipe completed successfully in {elapsed_time/60:.2f} minutes!")
        else:
            recipe.logger.error("❌ Height modeling recipe failed or output validation failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Recipe interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Recipe failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()