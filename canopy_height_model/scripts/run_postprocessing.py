#!/usr/bin/env python3
"""
Canopy Height Post-Processing Script

Multi-step post-processing pipeline for canopy height predictions:
1. Merge prediction patches into 120km tiles
2. Sanitize outliers and interpolate temporal gaps  
3. Create final country-wide mosaics at 100m resolution

Usage:
    python run_postprocessing.py [OPTIONS]
    
Examples:
    # Run complete 3-step pipeline
    python run_postprocessing.py
    
    # Run individual steps
    python run_postprocessing.py --steps merge
    python run_postprocessing.py --steps sanitize
    python run_postprocessing.py --steps final_merge
    
    # Run multiple steps
    python run_postprocessing.py --steps merge,sanitize
    
    # Continue on errors
    python run_postprocessing.py --continue-on-error
    
    # Custom input/output directories
    python run_postprocessing.py --steps merge --input-dir ./patches --output-dir ./tiles

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from canopy_height_model.core.postprocessing import PostProcessingPipeline, PipelineStep
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Post-Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  merge        - Merge prediction patches into 120km tiles
  sanitize     - Remove outliers and interpolate temporal gaps
  final_merge  - Create final country-wide mosaics at 100m
  all          - Run complete pipeline (default)

Examples:
  %(prog)s                                    # Run complete pipeline
  %(prog)s --steps merge                      # Run only merge step
  %(prog)s --steps merge,sanitize             # Run specific steps
  %(prog)s --continue-on-error                # Continue even if step fails
  %(prog)s --steps merge --input-dir ./patches --output-dir ./tiles
        """
    )
    
    # Pipeline control
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run (default: all)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution even if a step fails'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Directory overrides (for individual steps)
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Override input directory for current step'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory for current step'
    )
    
    # Processing options
    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of parallel workers (default: from config)'
    )
    
    parser.add_argument(
        '--tile-size-km',
        type=int,
        help='Tile size in kilometers for merge step (default: from config)'
    )
    
    parser.add_argument(
        '--target-resolution',
        type=int,
        help='Target resolution in meters for final merge (default: from config)'
    )
    
    # Quality control options
    parser.add_argument(
        '--height-min',
        type=float,
        help='Minimum valid height for outlier detection (default: from config)'
    )
    
    parser.add_argument(
        '--height-max',
        type=float,
        help='Maximum valid height for outlier detection (default: from config)'
    )
    
    parser.add_argument(
        '--skip-outlier-detection',
        action='store_true',
        help='Skip outlier detection in sanitize step'
    )
    
    parser.add_argument(
        '--skip-temporal-interpolation',
        action='store_true',
        help='Skip temporal interpolation in sanitize step'
    )
    
    # Output options
    parser.add_argument(
        '--compression',
        choices=['none', 'lzw', 'deflate', 'jpeg'],
        help='Output compression method (default: from config)'
    )
    
    parser.add_argument(
        '--create-overviews',
        action='store_true',
        help='Create overview pyramids for final outputs'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    # Validation and debugging
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without processing'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Show detailed progress information'
    )
    
    return parser.parse_args()


def parse_steps_argument(steps_str: str) -> List[PipelineStep]:
    """Parse command line steps argument."""
    if not steps_str:
        return [PipelineStep.ALL]
    
    step_names = [s.strip().lower() for s in steps_str.split(',')]
    steps = []
    
    for step_name in step_names:
        try:
            if step_name == 'all':
                step = PipelineStep.ALL
            elif step_name == 'merge':
                step = PipelineStep.MERGE
            elif step_name == 'sanitize':
                step = PipelineStep.SANITIZE
            elif step_name in ['final_merge', 'final-merge', 'finalize']:
                step = PipelineStep.FINAL_MERGE
            else:
                raise ValueError(f"Unknown step: {step_name}")
            
            steps.append(step)
        except ValueError as e:
            print(f"Warning: {e}")
            print(f"Available steps: merge, sanitize, final_merge, all")
    
    return steps if steps else [PipelineStep.ALL]


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Check config file if provided
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    # Validate directory overrides
    if args.input_dir and not Path(args.input_dir).exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return False
    
    # Validate numeric parameters
    if args.tile_size_km and args.tile_size_km <= 0:
        print("Error: Tile size must be positive")
        return False
    
    if args.target_resolution and args.target_resolution <= 0:
        print("Error: Target resolution must be positive")
        return False
    
    if args.num_workers and args.num_workers <= 0:
        print("Error: Number of workers must be positive")
        return False
    
    if args.height_min is not None and args.height_max is not None:
        if args.height_min >= args.height_max:
            print("Error: height_min must be less than height_max")
            return False
    
    return True


def apply_argument_overrides(pipeline: PostProcessingPipeline, args: argparse.Namespace) -> None:
    """Apply command line argument overrides to pipeline configuration."""
    config = pipeline.config
    
    # Processing overrides
    if args.num_workers:
        config['post_processing']['merge']['num_workers'] = args.num_workers
        pipeline.logger.info(f"Override num_workers: {args.num_workers}")
    
    if args.tile_size_km:
        config['post_processing']['merge']['tile_size_km'] = args.tile_size_km
        pipeline.logger.info(f"Override tile_size_km: {args.tile_size_km}")
    
    if args.target_resolution:
        config['post_processing']['final_merge']['target_resolution'] = args.target_resolution
        pipeline.logger.info(f"Override target_resolution: {args.target_resolution}")
    
    # Quality control overrides
    if args.height_min is not None:
        config['post_processing']['sanitize']['height_min'] = args.height_min
        pipeline.logger.info(f"Override height_min: {args.height_min}")
    
    if args.height_max is not None:
        config['post_processing']['sanitize']['height_max'] = args.height_max
        pipeline.logger.info(f"Override height_max: {args.height_max}")
    
    if args.skip_outlier_detection:
        config['post_processing']['sanitize']['outlier_detection'] = False
        pipeline.logger.info("Outlier detection disabled")
    
    if args.skip_temporal_interpolation:
        config['post_processing']['sanitize']['temporal_interpolation'] = False
        pipeline.logger.info("Temporal interpolation disabled")
    
    # Output overrides
    if args.compression:
        config['post_processing']['merge']['compression'] = args.compression
        pipeline.logger.info(f"Override compression: {args.compression}")
    
    if args.create_overviews:
        config['post_processing']['final_merge']['create_overview_pyramids'] = True
        pipeline.logger.info("Overview pyramid creation enabled")
    
    # Directory overrides (only if single step being run)
    if args.input_dir:
        # Apply to all step configs - the specific step will use it
        for step_config in ['merge', 'sanitize', 'final_merge']:
            if step_config in config['post_processing']:
                config['post_processing'][step_config]['input_dir'] = args.input_dir
        pipeline.logger.info(f"Override input_dir: {args.input_dir}")
    
    if args.output_dir:
        # Apply to all step configs
        for step_config in ['merge', 'sanitize', 'final_merge']:
            if step_config in config['post_processing']:
                config['post_processing'][step_config]['output_dir'] = args.output_dir
        pipeline.logger.info(f"Override output_dir: {args.output_dir}")


def validate_pipeline_only(pipeline: PostProcessingPipeline, steps: List[PipelineStep]) -> bool:
    """Run pipeline validation only."""
    pipeline.logger.info("Running pipeline validation...")
    
    try:
        all_valid = True
        
        for step in steps:
            pipeline.logger.info(f"Validating step: {step.value}")
            if not pipeline.validate_step_prerequisites(step):
                all_valid = False
        
        if all_valid:
            pipeline.logger.info("✅ All pipeline validation passed")
        else:
            pipeline.logger.error("❌ Pipeline validation failed")
        
        return all_valid
        
    except Exception as e:
        pipeline.logger.error(f"Validation failed: {str(e)}")
        return False


def print_step_summary(results: dict, step_times: dict, logger) -> None:
    """Print summary of pipeline execution."""
    logger.info("\n" + "="*60)
    logger.info("POST-PROCESSING PIPELINE SUMMARY")
    logger.info("="*60)
    
    total_steps = len(results)
    successful_steps = sum(1 for success in results.values() if success)
    
    logger.info(f"Steps completed: {successful_steps}/{total_steps}")
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        elapsed = step_times.get(step_name, 0)
        
        if elapsed > 0:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"({hours:02d}:{minutes:02d}:{seconds:02d})"
        else:
            time_str = ""
        
        logger.info(f"  {step_name:12} {status:10} {time_str}")
    
    if successful_steps == total_steps:
        logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.info(f"\n⚠️  PIPELINE COMPLETED WITH {total_steps - successful_steps} ERRORS")
    
    logger.info("="*60)


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Parse steps
    steps = parse_steps_argument(args.steps)
    
    if not steps:
        print("Error: No valid steps specified")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        component_name='canopy_height_postprocessing',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Canopy Height Post-Processing Pipeline...")
        pipeline = PostProcessingPipeline(config_path=args.config)
        
        # Apply command line overrides
        apply_argument_overrides(pipeline, args)
        
        # Log pipeline start
        log_pipeline_start(logger, "Canopy Height Post-Processing", pipeline.config)
        
        logger.info(f"Pipeline steps to execute: {[step.value for step in steps]}")
        
        # Validation only mode
        if args.validate_only:
            success = validate_pipeline_only(pipeline, steps)
            if success:
                logger.info("✅ Validation successful - exiting")
                sys.exit(0)
            else:
                logger.error("❌ Validation failed")
                sys.exit(1)
        
        # Run pipeline
        results = pipeline.run_pipeline(
            steps=steps,
            continue_on_error=args.continue_on_error
        )
        
        # Print detailed summary
        print_step_summary(results, pipeline.step_times, logger)
        
        # Pipeline completion
        overall_success = all(results.values())
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Canopy Height Post-Processing", overall_success, elapsed_time)
        
        if overall_success:
            logger.info("🎉 Post-processing completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Post-processing failed!")
            failed_steps = [step for step, success in results.items() if not success]
            logger.error(f"Failed steps: {failed_steps}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Post-processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Post-processing pipeline failed with unexpected error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
