#!/usr/bin/env python3
"""
Biomass Model Full Pipeline Orchestrator

Complete biomass estimation pipeline that orchestrates all processing steps:
1. Allometry fitting (optional)
2. Biomass estimation (forest type specific maps)
3. Annual cropland masking 
4. Forest type merging (country-wide maps)

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

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from biomass_model.scripts.run_allometry_fitting import AllometryFittingPipeline
from biomass_model.scripts.run_biomass_estimation import main as run_biomass_estimation_main
from biomass_model.scripts.run_masking import main as run_masking_main  
from biomass_model.scripts.run_merging import main as run_merging_main

# Shared utilities
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end


class BiomassFullPipelineOrchestrator:
    """
    Complete biomass estimation pipeline orchestrator.
    
    Manages execution of all pipeline stages with progress monitoring
    and error recovery.
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_full_pipeline'
        )
        
        # Load configuration
        self.config = load_config(config_path, component_name="biomass_estimation")
        
        # Pipeline state tracking
        self.stage_results = {}
        self.start_time = None
        
        self.logger.info("Initialized BiomassFullPipelineOrchestrator")
    
    def run_stage(self, stage_name: str, stage_func, *args, **kwargs) -> bool:
        """
        Run a pipeline stage with error handling and timing.
        
        Args:
            stage_name: Name of the stage
            stage_func: Function to execute
            *args, **kwargs: Arguments for stage function
            
        Returns:
            bool: True if stage completed successfully
        """
        stage_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Execute stage
            result = stage_func(*args, **kwargs)
            stage_time = time.time() - stage_start
            
            # Determine success based on result type
            if isinstance(result, bool):
                success = result
            elif result is None:
                success = True  # Assume success if no return value
            else:
                success = True  # Assume success for other return types
            
            # Record results
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
            
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"❌ {stage_name} failed with error after {stage_time/60:.2f} minutes: {str(e)}")
            return False
    
    def check_stage_completion(self, stage_name: str, check_paths: List[str], 
                              check_patterns: List[str] = None) -> bool:
        """
        Check if a stage has already been completed.
        
        Args:
            stage_name: Name of the stage
            check_paths: List of paths to check for existence
            check_patterns: Optional file patterns to check within paths
            
        Returns:
            bool: True if stage appears to be completed
        """
        try:
            # Check if all paths exist
            all_exist = all(Path(path).exists() for path in check_paths)
            
            if not all_exist:
                return False
            
            # Check for specific patterns if provided
            if check_patterns:
                for path in check_paths:
                    path_obj = Path(path)
                    if path_obj.is_dir():
                        for pattern in check_patterns:
                            if not any(path_obj.glob(pattern)):
                                return False
            
            self.logger.info(f"✅ {stage_name} appears to be already completed")
            return True
            
        except Exception as e:
            self.logger.debug(f"Error checking stage completion for {stage_name}: {e}")
            return False
    
    def run_allometry_fitting(self, years: Optional[List[int]] = None, 
                            overwrite: bool = False) -> bool:
        """Run allometry fitting stage."""
        stage_name = "Allometry Fitting"
        
        # Check if allometries already exist
        allometries_file = self.config['data'].get('allometries_file')
        if allometries_file and Path(allometries_file).exists() and not overwrite:
            self.logger.info(f"✅ {stage_name} - Using existing allometries: {allometries_file}")
            self.stage_results[stage_name] = {
                'success': True,
                'duration_minutes': 0,
                'result': 'existing_allometries_used'
            }
            return True
        
        def _run():
            pipeline = AllometryFittingPipeline(config_path=None)
            return pipeline.fit_allometries(years=years)
        
        return self.run_stage(stage_name, _run)
    
    def run_biomass_estimation(self, years: Optional[List[int]] = None) -> bool:
        """Run biomass estimation stage."""
        stage_name = "Biomass Estimation"
        
        # Check if estimation already completed
        output_dir = Path(self.config['data']['output_base_dir']) / self.config['data']['biomass_no_masking_dir']
        check_paths = [str(output_dir)]
        check_patterns = ["*.tif"]
        
        if self.check_stage_completion(stage_name, check_paths, check_patterns):
            return True
        
        def _run():
            # Import here to avoid circular imports
            import sys
            from io import StringIO
            
            # Capture run_biomass_estimation_main output
            old_argv = sys.argv.copy()
            try:
                # Prepare arguments for biomass estimation
                sys.argv = ['run_biomass_estimation.py']
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                # Run biomass estimation
                result = run_biomass_estimation_main()
                return result if result is not None else True
                
            finally:
                sys.argv = old_argv
        
        return self.run_stage(stage_name, _run)
    
    def run_masking(self) -> bool:
        """Run annual cropland masking stage."""
        stage_name = "Annual Cropland Masking"
        
        # Check if masking already completed
        output_dir = Path(self.config['data']['output_base_dir']) / self.config['data']['biomass_with_mask_dir']
        check_paths = [str(output_dir)]
        check_patterns = ["*.tif"]
        
        if self.check_stage_completion(stage_name, check_paths, check_patterns):
            return True
        
        def _run():
            import sys
            
            old_argv = sys.argv.copy()
            try:
                # Prepare arguments for masking
                input_dir = Path(self.config['data']['output_base_dir']) / self.config['data']['biomass_no_masking_dir']
                output_dir = Path(self.config['data']['output_base_dir']) / self.config['data']['biomass_with_mask_dir']
                
                sys.argv = [
                    'run_masking.py',
                    '--input-dir', str(input_dir),
                    '--output-dir', str(output_dir)
                ]
                
                result = run_masking_main()
                return result if result is not None else True
                
            finally:
                sys.argv = old_argv
        
        return self.run_stage(stage_name, _run)
    
    def run_merging(self) -> bool:
        """Run forest type merging stage."""
        stage_name = "Forest Type Merging"
        
        # Check if merging already completed
        output_dir = Path(self.config['data']['output_base_dir']) / "biomass_maps_merged"
        check_paths = [str(output_dir)]
        check_patterns = ["*_merged.tif"]
        
        if self.check_stage_completion(stage_name, check_paths, check_patterns):
            return True
        
        def _run():
            import sys
            
            old_argv = sys.argv.copy()
            try:
                # Prepare arguments for merging
                input_dir = Path(self.config['data']['output_base_dir']) / self.config['data']['biomass_with_mask_dir']
                output_dir = Path(self.config['data']['output_base_dir']) / "biomass_maps_merged"
                
                sys.argv = [
                    'run_merging.py',
                    '--input-dir', str(input_dir),
                    '--output-dir', str(output_dir)
                ]
                
                result = run_merging_main()
                return result if result is not None else True
                
            finally:
                sys.argv = old_argv
        
        return self.run_stage(stage_name, _run)
    
    def run_full_pipeline(self, stages: Optional[List[str]] = None,
                         skip_allometry: bool = False,
                         continue_on_error: bool = False,
                         years: Optional[List[int]] = None,
                         overwrite_allometry: bool = False) -> Dict:
        """
        Run the complete biomass estimation pipeline.
        
        Args:
            stages: Optional list of specific stages to run
            skip_allometry: Skip allometry fitting stage
            continue_on_error: Continue if a stage fails
            years: Specific years to process
            overwrite_allometry: Overwrite existing allometries
            
        Returns:
            dict: Pipeline execution results
        """
        self.start_time = time.time()
        
        # Define all available stages
        all_stages = [
            'allometry_fitting',
            'biomass_estimation', 
            'masking',
            'merging'
        ]
        
        # Filter stages based on options
        if skip_allometry:
            all_stages = [s for s in all_stages if s != 'allometry_fitting']
        
        # Use specified stages or filtered stages
        stages_to_run = stages if stages else all_stages
        
        self.logger.info(f"Starting biomass estimation pipeline")
        self.logger.info(f"Stages to run: {', '.join(stages_to_run)}")
        self.logger.info(f"Continue on error: {continue_on_error}")
        if years:
            self.logger.info(f"Processing years: {years}")
        
        # Run each stage
        for stage in stages_to_run:
            if stage == 'allometry_fitting':
                success = self.run_allometry_fitting(years=years, overwrite=overwrite_allometry)
            elif stage == 'biomass_estimation':
                success = self.run_biomass_estimation(years=years)
            elif stage == 'masking':
                success = self.run_masking()
            elif stage == 'merging':
                success = self.run_merging()
            else:
                self.logger.warning(f"Unknown stage: {stage}")
                continue
            
            if not success and not continue_on_error:
                self.logger.error(f"Pipeline stopped due to failure in {stage}")
                break
        
        return self.stage_results
    
    def print_pipeline_summary(self):
        """Print a summary of pipeline execution."""
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Biomass Model Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run complete pipeline
  %(prog)s --skip-allometry             # Skip allometry fitting
  %(prog)s --years 2020 2021 2022       # Specific years only
  %(prog)s --continue-on-error          # Continue despite failures
  %(prog)s --stages estimation masking  # Run specific stages only
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
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
        '--overwrite-allometry',
        action='store_true',
        help='Overwrite existing allometry files'
    )
    
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


def main():
    """Main entry point for biomass full pipeline."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    log_level = 'ERROR' if args.quiet else args.log_level
    
    # Initialize pipeline orchestrator
    try:
        orchestrator = BiomassFullPipelineOrchestrator(args.config, log_level)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    try:
        # Log pipeline start
        log_pipeline_start(orchestrator.logger, "Biomass Full Pipeline", orchestrator.config)
        
        # Run pipeline
        results = orchestrator.run_full_pipeline(
            stages=args.stages,
            skip_allometry=args.skip_allometry,
            continue_on_error=args.continue_on_error,
            years=args.years,
            overwrite_allometry=args.overwrite_allometry
        )
        
        # Print summary
        orchestrator.print_pipeline_summary()
        
        # Determine exit code
        failed_stages = [k for k, v in results.items() if not v['success']]
        if failed_stages and not args.continue_on_error:
            orchestrator.logger.error("Pipeline failed due to stage failures")
            sys.exit(1)
        elif failed_stages:
            orchestrator.logger.warning("Pipeline completed with some stage failures")
        else:
            orchestrator.logger.info("Pipeline completed successfully!")
        
        # Log pipeline end
        elapsed_time = time.time() - start_time
        success = len(failed_stages) == 0
        log_pipeline_end(orchestrator.logger, "Biomass Full Pipeline", success, elapsed_time)
        
    except KeyboardInterrupt:
        orchestrator.logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        orchestrator.logger.error(f"Pipeline execution error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            orchestrator.logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()