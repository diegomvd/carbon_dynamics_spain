#!/usr/bin/env python3
"""
Biomass Estimation Pipeline Script

Main entry point for running the biomass estimation pipeline with
Monte Carlo uncertainty quantification. Updated with recipe integration
arguments for harmonized path management.

Usage:
    python run_biomass_estimation.py [OPTIONS]
    
Examples:
    # Run full pipeline with default config
    python run_biomass_estimation.py
    
    # Run with custom config
    python run_biomass_estimation.py --config custom_config.yaml
    
    # Run specific years only
    python run_biomass_estimation.py --years 2020 2021 2022
    
    # Run single forest type for testing
    python run_biomass_estimation.py --test-mode --year 2020 --forest-type 12
    
    # Recipe integration with custom paths
    python run_biomass_estimation.py --height-100m-dir ./height_maps/100m --allometries-output-dir ./allometries

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from biomass_model.core.biomass_estimation import BiomassEstimationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end, CentralDataPaths


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Biomass Estimation Pipeline with Monte Carlo Uncertainty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default settings
  %(prog)s --config custom.yaml              # Custom configuration
  %(prog)s --years 2020 2021 2022            # Specific years only
  %(prog)s --test-mode --forest-type 12      # Test single forest type
  
Recipe Integration:
  %(prog)s --height-100m-dir ./heights       # Custom height maps directory
  %(prog)s --allometries-output-dir ./out    # Custom allometries output
  %(prog)s --biomass-output-dir ./biomass    # Custom biomass output
        """
    )
    
    # Core configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Root directory for data storage (default: data)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process'
    )
    
    parser.add_argument(
        '--forest-types',
        type=str,
        nargs='+',
        help='Specific forest types to process'
    )
    
    # Test mode
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (single forest type)'
    )
    
    parser.add_argument(
        '--forest-type',
        type=str,
        help='Forest type for test mode'
    )
    
    # **NEW: Recipe integration arguments**
    parser.add_argument(
        '--height-100m-dir',
        type=str,
        help='Custom directory for 100m height maps (overrides default)'
    )
    
    parser.add_argument(
        '--height-10m-dir',
        type=str,
        help='Custom directory for 10m height maps (overrides default)'
    )
    
    parser.add_argument(
        '--allometries-input-dir',
        type=str,
        help='Custom directory for fitted allometries input (overrides default)'
    )
    
    parser.add_argument(
        '--allometries-output-dir',
        type=str,
        help='Custom directory for allometries output (overrides default)'
    )
    
    parser.add_argument(
        '--biomass-output-dir',
        type=str,
        help='Custom directory for biomass maps output (overrides default)'
    )
    
    parser.add_argument(
        '--nfi-processed-dir',
        type=str,
        help='Custom directory for processed NFI data (overrides default)'
    )
    
    parser.add_argument(
        '--forest-type-maps-dir',
        type=str,
        help='Custom directory for forest type maps (overrides default)'
    )
    
    parser.add_argument(
        '--land-cover-file',
        type=str,
        help='Custom path to Corine land cover file (overrides default)'
    )
    
    # Pipeline control
    parser.add_argument(
        '--skip-allometry-fitting',
        action='store_true',
        help='Skip allometry fitting (use existing fitted parameters)'
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['allometry_fitting', 'biomass_estimation', 'masking', 'merging'],
        help='Specific pipeline stages to run'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing if a stage fails'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    # Logging and output
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    
    parser.add_argument(
        '--output-summary',
        type=str,
        help='Path to save pipeline execution summary'
    )
    
    # Monte Carlo settings
    parser.add_argument(
        '--monte-carlo-samples',
        type=int,
        help='Number of Monte Carlo samples (overrides config)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed for Monte Carlo sampling (overrides config)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Validate config file if provided
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    # Validate data root directory
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory not found: {args.data_root}")
        return False
    
    # Validate test mode arguments
    if args.test_mode and not args.forest_type:
        print("Error: --forest-type required when using --test-mode")
        return False
    
    # Validate custom path arguments if provided
    path_args = [
        ('height-100m-dir', args.height_100m_dir),
        ('height-10m-dir', args.height_10m_dir),
        ('allometries-input-dir', args.allometries_input_dir),
        ('nfi-processed-dir', args.nfi_processed_dir),
        ('forest-type-maps-dir', args.forest_type_maps_dir)
    ]
    
    for arg_name, arg_value in path_args:
        if arg_value and not Path(arg_value).exists():
            print(f"Error: {arg_name} directory not found: {arg_value}")
            return False
    
    if args.land_cover_file and not Path(args.land_cover_file).exists():
        print(f"Error: Land cover file not found: {args.land_cover_file}")
        return False
    
    return True


class BiomassEstimationRunner:
    """
    Biomass estimation pipeline runner with recipe integration support.
    
    Handles setup of centralized paths, configuration overrides, and
    pipeline execution with comprehensive error handling.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize pipeline runner."""
        
        # Apply custom path overrides from recipe arguments
        self._apply_path_overrides(args)
        
        # Setup logging
        log_level = 'ERROR' if args.quiet else args.log_level
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_estimation'
        )
        
        # Store arguments
        self.args = args
        
        self.logger.info("BiomassEstimationRunner initialized")
    
    def _apply_path_overrides(self, args: argparse.Namespace) -> None:
        """Apply custom path arguments to override default paths."""
        overrides = {
            'height_maps_100m': args.height_100m_dir,
            'height_maps_10m': args.height_10m_dir,
            'allometries': args.allometries_output_dir or args.allometries_input_dir,
            'biomass_maps': args.biomass_output_dir,
            'forest_inventory_processed': args.nfi_processed_dir,
            'forest_type_maps': args.forest_type_maps_dir
        }
        
    
    def create_pipeline_config(self) -> dict:
        """Create pipeline configuration with argument overrides."""
        # Load base configuration (no longer needs path overrides)
        config = load_config(self.args.config, component_name="biomass_estimation")
        
        # Apply CLI argument overrides
        if self.args.monte_carlo_samples:
            config['monte_carlo']['num_samples'] = self.args.monte_carlo_samples
        
        if self.args.random_seed:
            config['monte_carlo']['random_seed'] = self.args.random_seed
        
        if self.args.years:
            config['processing']['target_years'] = self.args.years
        
        return config
    
    def run_pipeline(self) -> bool:
        """
        Execute the biomass estimation pipeline.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        try:
            self.logger.info("Starting biomass estimation pipeline...")
            start_time = time.time()
            
            # Create pipeline configuration
            config = self.create_pipeline_config()
            
            pipeline = BiomassEstimationPipeline(config)
            
            # Validate inputs
            if not pipeline.validate_inputs():
                self.logger.error("Input validation failed")
                return False
            
            # Setup output directories
            output_dirs = pipeline.setup_output_directories()
            
            # Process each year
            all_success = True
            years_to_process = self.args.years or config['processing']['target_years']
            
            for year in years_to_process:
                self.logger.info(f"Processing year {year}...")
                
                year_success = pipeline.process_year(year, output_dirs)
                if not year_success:
                    if self.args.continue_on_error:
                        self.logger.warning(f"Year {year} failed, continuing...")
                        all_success = False
                    else:
                        self.logger.error(f"Year {year} failed, stopping pipeline")
                        return False
            
            # Log completion
            duration = time.time() - start_time
            status = "completed with warnings" if not all_success else "completed successfully"
            self.logger.info(f"Biomass estimation pipeline {status} in {duration:.2f} seconds")
            
            # Save execution summary if requested
            if self.args.output_summary:
                self._save_execution_summary(self.args.output_summary, duration, all_success)
            
            return all_success
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False
    
    def _save_execution_summary(self, summary_path: str, duration: float, success: bool) -> None:
        """Save pipeline execution summary to file."""
        try:
            summary = {
                'success': success,
                'duration_seconds': duration,
                'args': vars(self.args),
                'data_paths': {
                    'height_maps_100m': str(HEIGHT_MAPS_100M_DIR),
                    'allometries': str(FITTED_PARAMETERS_FILE),
                    'biomass_output': str(BIOMASS_MAPS_DIR)
                }
            }
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Execution summary saved to {summary_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save execution summary: {e}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return False
    
    try:
        # Initialize and run pipeline
        runner = BiomassEstimationRunner(args)
        success = runner.run_pipeline()
        
        # Log completion
        if success:
            print("\n✅ Biomass estimation completed successfully")
            return True
        else:
            print("\n❌ Biomass estimation failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)