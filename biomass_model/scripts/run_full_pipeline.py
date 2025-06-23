#!/usr/bin/env python3
"""
Biomass Model Full Pipeline Orchestrator

Complete biomass estimation pipeline that orchestrates all processing steps:
1. Allometry fitting (optional)
2. Biomass estimation (forest type specific maps)
3. Annual cropland masking 
4. Forest type merging (country-wide maps)

Updated with recipe integration arguments for harmonized path management
and seamless integration with the recipe-based execution system.

Usage:
    python run_full_pipeline.py [OPTIONS]

Examples:
    # Run complete pipeline
    python run_full_pipeline.py
    
    # Skip allometry fitting (use existing)
    python run_full_pipeline.py --skip-allometry
    
    # Specific years only
    python run_full_pipeline.py --years 2020 2021 2022
    
    # Continue on errors
    python run_full_pipeline.py --continue-on-error
    
    # Recipe integration with custom paths
    python run_full_pipeline.py --allometries-output-dir ./allometries --height-100m-dir ./heights

Author: Diego Bengochea
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Shared utilities
from shared_utils import setup_logging, load_config, CentralDataPaths


class BiomassFullPipelineOrchestrator:
    """
    Complete biomass estimation pipeline orchestrator.
    
    Manages execution of all pipeline stages with progress monitoring,
    error recovery, and recipe integration support.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize pipeline orchestrator.
        
        Args:
            args: Parsed command line arguments
        """
        # Setup centralized data paths
        self.data_paths = CentralDataPaths(args.data_root)
        
        # Apply custom path overrides from recipe arguments
        self._apply_path_overrides(args)
        
        # Setup logging
        log_level = 'ERROR' if args.quiet else args.log_level
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_full_pipeline'
        )
        
        # Store arguments and settings
        self.args = args
        self.continue_on_error = args.continue_on_error
        
        # Pipeline state tracking
        self.stage_results = {}
        self.start_time = None
        
        self.logger.info("Initialized BiomassFullPipelineOrchestrator")
    
    def _apply_path_overrides(self, args: argparse.Namespace) -> None:
        """Apply custom path arguments to override default paths."""
        overrides = {
            'height_maps_100m': args.height_100m_dir,
            'height_maps_10m': args.height_10m_dir,
            'allometries': args.allometries_output_dir,
            'biomass_maps': args.biomass_output_dir,
            'forest_inventory_processed': args.nfi_processed_dir,
            'forest_type_maps': args.forest_type_maps_dir
        }
        

    
    def run_stage(self, stage_name: str, stage_script: str, stage_args: List[str] = None) -> bool:
        """
        Run a pipeline stage with error handling and timing.
        
        Args:
            stage_name: Name of the stage for logging
            stage_script: Path to the stage script
            stage_args: Additional arguments for the stage script
            
        Returns:
            bool: True if stage succeeded
        """
        self.logger.info(f"Starting stage: {stage_name}")
        stage_start_time = time.time()
        
        try:
            # Build command
            cmd = [sys.executable, stage_script]
            if stage_args:
                cmd.extend(stage_args)
            
            # Run stage
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # Calculate duration
            duration_minutes = (time.time() - stage_start_time) / 60
            
            # Check result
            success = result.returncode == 0
            
            # Store results
            self.stage_results[stage_name] = {
                'success': success,
                'duration_minutes': duration_minutes,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if success:
                self.logger.info(f"✅ {stage_name} completed in {duration_minutes:.2f} minutes")
            else:
                self.logger.error(f"❌ {stage_name} failed in {duration_minutes:.2f} minutes")
                self.logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    self.logger.error(f"Error output:\n{result.stderr}")
                
                if not self.continue_on_error:
                    raise RuntimeError(f"Stage {stage_name} failed")
            
            return success
            
        except Exception as e:
            duration_minutes = (time.time() - stage_start_time) / 60
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': duration_minutes,
                'error': str(e)
            }
            
            self.logger.error(f"❌ {stage_name} failed with exception in {duration_minutes:.2f} minutes: {str(e)}")
            
            if not self.continue_on_error:
                raise
            
            return False
    
    def build_common_args(self) -> List[str]:
        """Build common arguments to pass to all stage scripts."""
        common_args = []
        
        # Add years if specified
        if self.args.years:
            common_args.extend(['--years'] + [str(y) for y in self.args.years])
        
        # Add logging level
        if not self.args.quiet:
            common_args.extend(['--log-level', self.args.log_level])
        
        # Add error handling
        if self.args.continue_on_error:
            common_args.append('--continue-on-error')
        
        if self.args.overwrite:
            common_args.append('--overwrite')
        
        return common_args
    
    def run_allometry_fitting_stage(self) -> bool:
        """Run allometry fitting stage."""
        if self.args.skip_allometry:
            self.logger.info("⏭️ Skipping allometry fitting stage")
            return True
        
        stage_args = self.build_common_args()
        
        # Add allometry-specific arguments
        if self.args.use_100m_for_fitting:
            stage_args.append('--use-100m')
        
        return self.run_stage(
            'Allometry Fitting',
            'run_allometry_fitting.py',
            stage_args
        )
    
    def run_biomass_estimation_stage(self) -> bool:
        """Run biomass estimation stage."""
        stage_args = self.build_common_args()
        
        # Skip allometry fitting in biomass estimation if we already did it
        if not self.args.skip_allometry:
            stage_args.append('--skip-allometry-fitting')
        
        return self.run_stage(
            'Biomass Estimation',
            'run_biomass_estimation.py',
            stage_args
        )
    
    def run_masking_stage(self) -> bool:
        """Run annual cropland masking stage."""
        stage_args = self.build_common_args()
        
        # Add masking-specific arguments
        if self.args.biomass_output_dir:
            stage_args.extend(['--input-dir', self.args.biomass_output_dir])
        
        return self.run_stage(
            'Annual Cropland Masking',
            'run_masking.py', 
            stage_args
        )
    
    def run_merging_stage(self) -> bool:
        """Run forest type merging stage."""
        stage_args = self.build_common_args()
        
        return self.run_stage(
            'Forest Type Merging',
            'run_merging.py',
            stage_args
        )
    
    def run_full_pipeline(self) -> bool:
        """
        Execute the complete biomass estimation pipeline.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        self.logger.info("🚀 Starting complete biomass estimation pipeline...")
        self.start_time = time.time()
        
        # Define pipeline stages
        stages = [
            ('allometry_fitting', self.run_allometry_fitting_stage),
            ('biomass_estimation', self.run_biomass_estimation_stage), 
            ('masking', self.run_masking_stage),
            ('merging', self.run_merging_stage)
        ]
        
        # Filter stages if specific stages requested
        if self.args.stages:
            stages = [(name, func) for name, func in stages if name in self.args.stages]
        
        # Run stages
        overall_success = True
        for stage_name, stage_func in stages:
            try:
                stage_success = stage_func()
                if not stage_success:
                    overall_success = False
                    if not self.continue_on_error:
                        break
            except Exception as e:
                self.logger.error(f"Stage {stage_name} failed with exception: {str(e)}")
                overall_success = False
                if not self.continue_on_error:
                    break
        
        # Log summary
        self._log_pipeline_summary()
        
        return overall_success
    
    def _log_pipeline_summary(self) -> None:
        """Log pipeline execution summary."""
        if not self.stage_results:
            self.logger.info("No pipeline results to summarize")
            return
        
        total_time = (time.time() - self.start_time) / 60 if self.start_time else 0
        successful_stages = sum(1 for r in self.stage_results.values() if r['success'])
        total_stages = len(self.stage_results)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BIOMASS PIPELINE SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total stages: {total_stages}")
        self.logger.info(f"Successful: {successful_stages}")
        self.logger.info(f"Failed: {total_stages - successful_stages}")
        self.logger.info(f"Total time: {total_time:.2f} minutes")
        
        for stage_name, results in self.stage_results.items():
            status = "✅" if results['success'] else "❌"
            duration = results['duration_minutes']
            self.logger.info(f"  {status} {stage_name}: {duration:.2f} min")
        
        # Show output directories
        self.logger.info(f"\n📂 Output directories:")
        self.logger.info(f"   Allometries: {str(FITTED_PARAMETERS_FILE)}")
        self.logger.info(f"   Biomass maps: {str(BIOMASS_MAPS_FULL_COUNTRY_DIR)}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Biomass Model Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run complete pipeline
  %(prog)s --skip-allometry                   # Skip allometry fitting
  %(prog)s --years 2020 2021 2022             # Specific years only
  %(prog)s --continue-on-error                # Continue despite failures
  %(prog)s --stages estimation masking        # Run specific stages only
  
Recipe Integration:
  %(prog)s --allometries-output-dir ./allom   # Custom allometries output
  %(prog)s --height-100m-dir ./heights        # Custom height maps directory
  %(prog)s --biomass-output-dir ./biomass     # Custom biomass output
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
    
    # Pipeline control
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['allometry_fitting', 'biomass_estimation', 'masking', 'merging'],
        help='Specific stages to run'
    )
    
    parser.add_argument(
        '--skip-allometry',
        action='store_true',
        help='Skip allometry fitting stage (use existing allometries)'
    )
    
    parser.add_argument(
        '--use-100m-for-fitting',
        action='store_true',
        help='Use 100m height maps for allometry fitting instead of 10m'
    )
    
    # Processing parameters
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution if a stage fails'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
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
    
    # Validate custom path arguments if provided
    path_args = [
        ('height-100m-dir', args.height_100m_dir),
        ('height-10m-dir', args.height_10m_dir),
        ('nfi-processed-dir', args.nfi_processed_dir),
        ('forest-type-maps-dir', args.forest_type_maps_dir)
    ]
    
    for arg_name, arg_value in path_args:
        if arg_value and not Path(arg_value).exists():
            print(f"Error: {arg_name} directory not found: {arg_value}")
            return False
    
    return True


def main():
    """Main entry point for biomass full pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return False
    
    try:
        # Initialize and run pipeline
        orchestrator = BiomassFullPipelineOrchestrator(args)
        success = orchestrator.run_full_pipeline()
        
        # Log completion
        if success:
            print("\n🎉 Biomass model pipeline completed successfully!")
            return True
        else:
            print("\n💥 Biomass model pipeline failed!")
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