#!/usr/bin/env python3
"""
Recipe: Height Modeling

Reproduces canopy height predictions from Sentinel-2 mosaics using deep learning models:
1. Height prediction using pre-trained model checkpoint
2. Post-processing (merging, sanitization, final mosaics)

This recipe generates the canopy height maps used for biomass estimation.

Usage:
    python reproduce_height_modeling.py [OPTIONS]

Examples:
    # Run complete height modeling
    python reproduce_height_modeling.py
    
    # Specific years only
    python reproduce_height_modeling.py --years 2020 2021 2022
    
    # Use specific model checkpoint
    python reproduce_height_modeling.py --checkpoint /path/to/model.ckpt
    
    # Skip model training (prediction only)
    python reproduce_height_modeling.py --prediction-only

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
from shared_utils.data_paths import CentralDataPaths
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
        
        # Check for Sentinel-2 mosaics
        s2_mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
        if not s2_mosaic_dir.exists():
            self.logger.error(f"Sentinel-2 mosaic directory not found: {s2_mosaic_dir}")
            self.logger.error("Please run 'reproduce_data_preparation.py' first to create Sentinel-2 mosaics")
            return False
        
        # Check for at least one mosaic file
        mosaic_files = list(s2_mosaic_dir.glob("*.tif"))
        if not mosaic_files:
            self.logger.error(f"No Sentinel-2 mosaic files found in {s2_mosaic_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(mosaic_files)} Sentinel-2 mosaic files")
        
        # Check for model checkpoint
        model_checkpoint = self.data_paths.get_path('height_model')
        if not model_checkpoint.exists():
            self.logger.error(f"Height model checkpoint not found: {model_checkpoint}")
            self.logger.error("Please provide pre-trained model checkpoint in data/models/")
            return False
        
        self.logger.info(f"✅ Model checkpoint found: {model_checkpoint}")
        
        # Validate checkpoint file
        try:
            # Basic file size check
            if model_checkpoint.stat().st_size < 1000000:  # Less than 1MB
                self.logger.warning("Model checkpoint seems unusually small")
        except Exception as e:
            self.logger.warning(f"Could not validate checkpoint file: {e}")
        
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
        
        # Create main height prediction directories
        self.data_paths.create_directories(['height_predictions'])
        
        # Create height prediction subdirectories
        height_base = self.data_paths.get_path('height_predictions')
        for subdir in self.data_paths.subdirs['height_predictions'].values():
            (height_base / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Output directory structure created")
    
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
            # Check if already completed
            patches_dir = (self.data_paths.get_path('height_predictions') / 
                          self.data_paths.subdirs['height_predictions']['patches'])
            existing_predictions = list(patches_dir.glob("*.tif")) if patches_dir.exists() else []
            
            if existing_predictions:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_predictions)} prediction files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Prepare config overrides for centralized paths
            config_overrides = self.data_paths.get_component_config_overrides('canopy_height_model')
            
            # Use provided checkpoint or default
            if checkpoint_path:
                config_overrides['data.checkpoint_path'] = checkpoint_path
            
            # Run height prediction
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for height prediction
                sys.argv = ['run_prediction.py']
                
                if checkpoint_path:
                    sys.argv.extend(['--checkpoint', checkpoint_path])
                elif self.data_paths.get_path('height_model').exists():
                    sys.argv.extend(['--checkpoint', str(self.data_paths.get_path('height_model'))])
                
                if years:
                    # Note: Adjust based on actual script interface
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                # Add output directory
                output_dir = self.data_paths.get_path('height_predictions')
                sys.argv.extend(['--output-dir', str(output_dir)])
                
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
    
    def run_postprocessing(self) -> bool:
        """
        Run height prediction post-processing.
        
        Returns:
            bool: True if successful
        """
        stage_name = "Height Post-processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            final_dir = (self.data_paths.get_path('height_predictions') / 
                        self.data_paths.subdirs['height_predictions']['merged_full_country'])
            existing_finals = list(final_dir.glob("*.tif")) if final_dir.exists() else []
            
            if existing_finals:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_finals)} final files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run height post-processing
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for post-processing
                input_dir = (self.data_paths.get_path('height_predictions') / 
                           self.data_paths.subdirs['height_predictions']['patches'])
                output_dir = self.data_paths.get_path('height_predictions')
                
                sys.argv = [
                    'run_postprocessing.py',
                    '--input-dir', str(input_dir),
                    '--output-dir', str(output_dir)
                ]
                
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
        Validate that expected outputs were created.
        
        Returns:
            bool: True if outputs are valid
        """
        self.logger.info("Validating height modeling outputs...")
        
        # Check for final merged height predictions
        final_dir = (self.data_paths.get_path('height_predictions') / 
                    self.data_paths.subdirs['height_predictions']['merged_full_country'])
        
        if not final_dir.exists():
            self.logger.error(f"Final height predictions directory not found: {final_dir}")
            return False
        
        final_files = list(final_dir.glob("*.tif"))
        if not final_files:
            self.logger.error("No final height prediction files found")
            return False
        
        self.logger.info(f"✅ Found {len(final_files)} final height prediction files")
        
        # Check intermediate outputs
        stages = ['patches', 'merged_tiles', 'sanitized']
        for stage in stages:
            stage_dir = (self.data_paths.get_path('height_predictions') / 
                        self.data_paths.subdirs['height_predictions'][stage])
            if stage_dir.exists():
                stage_files = list(stage_dir.glob("*.tif"))
                self.logger.info(f"✅ Found {len(stage_files)} files in {stage} stage")
        
        # Basic file validation
        for file_path in final_files[:2]:  # Check first couple files
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb < 10:  # Less than 10MB seems small for height predictions
                    self.logger.warning(f"Small height prediction file: {file_path} ({file_size_mb:.1f} MB)")
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
        
        # Check output directories
        height_base = self.data_paths.get_path('height_predictions')
        
        for stage, subdir in self.data_paths.subdirs['height_predictions'].items():
            stage_dir = height_base / subdir
            if stage_dir.exists():
                stage_files = list(stage_dir.glob("*.tif"))
                self.logger.info(f"📁 {stage.replace('_', ' ').title()}: {len(stage_files)} files in {stage_dir}")
        
        # Show data structure
        self.logger.info(f"📂 Data structure created in: {self.data_paths.data_root}")
        self.logger.info(f"   └── processed/")
        self.logger.info(f"       └── height_predictions/")
        self.logger.info(f"           ├── patches/")
        self.logger.info(f"           ├── merged_tiles/")
        self.logger.info(f"           ├── sanitized/")
        self.logger.info(f"           ├── final_mosaics/")
        self.logger.info(f"           └── merged_full_country/")
        
        self.logger.info(f"\n🎯 Next steps:")
        self.logger.info(f"   1. Run 'reproduce_biomass_estimation.py' to estimate biomass from heights")
        self.logger.info(f"   2. Check height prediction quality and coverage")
        self.logger.info(f"   3. Use merged_full_country/ files for biomass estimation")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Height Modeling (Canopy Height Prediction + Post-processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe generates canopy height predictions by executing:
1. Height prediction using pre-trained deep learning model
2. Post-processing (merging, sanitization, final mosaics)

Examples:
  %(prog)s                              # Run complete height modeling
  %(prog)s --years 2020 2021 2022       # Specific years only
  %(prog)s --checkpoint /path/to/model.ckpt # Use specific checkpoint
  %(prog)s --prediction-only            # Skip training, prediction only
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements:
  - Sentinel-2 mosaics (run reproduce_data_preparation.py first)
  - Pre-trained model checkpoint in data/models/
  - PyTorch environment with GPU support (recommended)
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
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--prediction-only',
        action='store_true',
        help='Skip training, run prediction and post-processing only'
    )
    
    parser.add_argument(
        '--skip-postprocessing',
        action='store_true',
        help='Skip post-processing stage'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue if a processing stage fails'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate prerequisites without running processing'
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
            recipe.logger.error("Please ensure required input data is available")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Track overall success
        overall_success = True
        
        # Run height prediction (always run unless explicitly skipped)
        if args.prediction_only or True:  # Default to prediction-only for reproduction
            success = recipe.run_height_prediction(
                years=args.years,
                checkpoint_path=args.checkpoint
            )
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Height prediction failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Run post-processing
        if not args.skip_postprocessing:
            success = recipe.run_postprocessing()
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