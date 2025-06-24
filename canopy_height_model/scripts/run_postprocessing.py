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
    )
    
    # Pipeline control
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run (default: all)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
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
    
    return True

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
        level='INFO',
        component_name='canopy_height_postprocessing'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Canopy Height Post-Processing Pipeline...")
        pipeline = PostProcessingPipeline(config_path=args.config)
        
        # Log pipeline start
        log_pipeline_start(logger, "Canopy Height Post-Processing", pipeline.config)
        
        logger.info(f"Pipeline steps to execute: {[step.value for step in steps]}")
        
        # Run pipeline
        results = pipeline.run_pipeline(
            steps=steps,
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
        sys.exit(1)


if __name__ == "__main__":
    main()
