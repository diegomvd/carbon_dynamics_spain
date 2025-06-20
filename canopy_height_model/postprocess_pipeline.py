"""
Complete post-processing pipeline for canopy height predictions.

This script orchestrates the complete post-processing workflow for canopy height
predictions, running all three major steps in sequence: tile merging, sanitization
with temporal interpolation, and final downsampling with country-wide merging.

The pipeline provides configurable execution modes, comprehensive logging,
and error recovery capabilities for robust large-scale processing.

Workflow:
1. Merge individual prediction patches into 120km tiles
2. Sanitize predictions (outlier removal) and interpolate temporal gaps  
3. Downsample and merge into final country-wide 100m mosaics

Author: Diego Bengochea
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

# Local imports
from config import load_config, setup_logging
from merge_predictions import PredictionMerger  
from sanitize_predictions import HeightSanitizer
from downsample_merge import FinalMerger


class PipelineStep(Enum):
    """Enumeration of available pipeline steps."""
    MERGE = "merge"
    SANITIZE = "sanitize" 
    FINAL_MERGE = "final_merge"
    ALL = "all"


class PostProcessingPipeline:
    """
    Complete post-processing pipeline for canopy height predictions.
    
    This class orchestrates the entire post-processing workflow with proper
    error handling, logging, and configurable execution modes.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the post-processing pipeline.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging']['level'])
        
        # Initialize processing components
        self.merger = None
        self.sanitizer = None
        self.final_merger = None
        
        # Track execution times
        self.step_times = {}
        
        self.logger.info("Post-processing pipeline initialized")
        
    def _log_step_start(self, step_name: str) -> float:
        """
        Log the start of a pipeline step.
        
        Args:
            step_name (str): Name of the step starting
            
        Returns:
            float: Start timestamp
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"STARTING STEP: {step_name.upper()}")
        self.logger.info(f"{'='*80}")
        return time.time()
    
    def _log_step_end(self, step_name: str, start_time: float) -> None:
        """
        Log the completion of a pipeline step.
        
        Args:
            step_name (str): Name of the completed step
            start_time (float): Start timestamp of the step
        """
        elapsed_time = time.time() - start_time
        self.step_times[step_name] = elapsed_time
        
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"COMPLETED STEP: {step_name.upper()}")
        self.logger.info(f"Execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.logger.info(f"{'='*80}\n")
    
    def run_merge_step(self) -> bool:
        """
        Run the tile merging step.
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "merge_predictions"
        start_time = self._log_step_start(step_name)
        
        try:
            if self.merger is None:
                self.merger = PredictionMerger()
            
            self.merger.run_parallel_merge()
            self._log_step_end(step_name, start_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Merge step failed: {str(e)}")
            return False
    
    def run_sanitize_step(self) -> bool:
        """
        Run the sanitization and interpolation step.
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "sanitize_and_interpolate"
        start_time = self._log_step_start(step_name)
        
        try:
            if self.sanitizer is None:
                self.sanitizer = HeightSanitizer()
            
            self.sanitizer.run_sanitization_pipeline()
            self._log_step_end(step_name, start_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Sanitization step failed: {str(e)}")
            return False
    
    def run_final_merge_step(self) -> bool:
        """
        Run the final downsampling and merging step.
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "final_merge"
        start_time = self._log_step_start(step_name)
        
        try:
            if self.final_merger is None:
                self.final_merger = FinalMerger()
            
            self.final_merger.run_final_merge_pipeline()
            self._log_step_end(step_name, start_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Final merge step failed: {str(e)}")
            return False
    
    def validate_pipeline_prerequisites(self, step: PipelineStep) -> bool:
        """
        Validate that prerequisites are met for the given step.
        
        Args:
            step (PipelineStep): Step to validate prerequisites for
            
        Returns:
            bool: True if prerequisites are met, False otherwise
        """
        if step in [PipelineStep.MERGE, PipelineStep.ALL]:
            # Check if prediction files exist
            merge_config = self.config['post_processing']['merge']
            input_dir = Path(merge_config['input_dir'])
            
            if not input_dir.exists():
                self.logger.error(f"Merge input directory does not exist: {input_dir}")
                return False
            
            # Check for prediction subdirectories
            prediction_dirs = list(input_dir.glob("*.0"))
            if not prediction_dirs:
                self.logger.error(f"No prediction directories found in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(prediction_dirs)} prediction directories for merging")
        
        if step in [PipelineStep.SANITIZE, PipelineStep.ALL]:
            # Check if merged tiles exist
            sanitize_config = self.config['post_processing']['sanitize']
            input_dir = Path(sanitize_config['input_dir'])
            
            if not input_dir.exists():
                self.logger.error(f"Sanitize input directory does not exist: {input_dir}")
                return False
            
            tiles = list(input_dir.glob("canopy_height_*.tif"))
            if not tiles:
                self.logger.error(f"No tiles found for sanitization in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(tiles)} tiles for sanitization")
        
        if step in [PipelineStep.FINAL_MERGE, PipelineStep.ALL]:
            # Check if sanitized/interpolated tiles exist
            final_config = self.config['post_processing']['final_merge']
            input_dir = Path(final_config['input_dir'])
            
            if not input_dir.exists():
                self.logger.error(f"Final merge input directory does not exist: {input_dir}")
                return False
            
            tiles = list(input_dir.glob(final_config['file_pattern']))
            if not tiles:
                self.logger.error(f"No tiles found for final merging in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(tiles)} tiles for final merging")
        
        return True
    
    def run_pipeline(
        self, 
        steps: List[PipelineStep] = None,
        continue_on_error: bool = False
    ) -> Dict[str, bool]:
        """
        Run the complete or partial post-processing pipeline.
        
        Args:
            steps (List[PipelineStep], optional): Specific steps to run. 
                If None, runs all steps.
            continue_on_error (bool): Whether to continue if a step fails
            
        Returns:
            Dict[str, bool]: Dictionary mapping step names to success status
        """
        if steps is None:
            steps = [PipelineStep.ALL]
        
        # Expand ALL step into individual steps
        if PipelineStep.ALL in steps:
            steps = [PipelineStep.MERGE, PipelineStep.SANITIZE, PipelineStep.FINAL_MERGE]
        
        pipeline_start_time = time.time()
        results = {}
        
        self.logger.info(f"Starting post-processing pipeline with steps: {[s.value for s in steps]}")
        
        # Validate prerequisites
        for step in steps:
            if not self.validate_pipeline_prerequisites(step):
                self.logger.error(f"Prerequisites not met for step: {step.value}")
                return {step.value: False for step in steps}
        
        # Execute steps in order
        step_functions = {
            PipelineStep.MERGE: self.run_merge_step,
            PipelineStep.SANITIZE: self.run_sanitize_step,
            PipelineStep.FINAL_MERGE: self.run_final_merge_step
        }
        
        for step in steps:
            if step in step_functions:
                success = step_functions[step]()
                results[step.value] = success
                
                if not success and not continue_on_error:
                    self.logger.error(f"Pipeline stopped due to failure in step: {step.value}")
                    break
            else:
                self.logger.warning(f"Unknown step: {step.value}")
                results[step.value] = False
        
        # Print final summary
        self._print_pipeline_summary(pipeline_start_time, results)
        
        return results
    
    def _print_pipeline_summary(
        self, 
        pipeline_start_time: float, 
        results: Dict[str, bool]
    ) -> None:
        """
        Print comprehensive pipeline execution summary.
        
        Args:
            pipeline_start_time (float): Pipeline start timestamp
            results (Dict[str, bool]): Step execution results
        """
        total_time = time.time() - pipeline_start_time
        
        self.logger.info("\n" + "="*100)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*100)
        
        # Step results
        successful_steps = sum(1 for success in results.values() if success)
        total_steps = len(results)
        
        self.logger.info(f"Steps completed: {successful_steps}/{total_steps}")
        
        for step_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            step_time = self.step_times.get(step_name, 0)
            hours = int(step_time // 3600)
            minutes = int((step_time % 3600) // 60)
            seconds = int(step_time % 60)
            self.logger.info(f"  {step_name:20} {status:10} ({hours:02d}:{minutes:02d}:{seconds:02d})")
        
        # Total time
        total_hours = int(total_time // 3600)
        total_minutes = int((total_time % 3600) // 60)
        total_seconds = int(total_time % 60)
        
        self.logger.info(f"\nTotal pipeline time: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}")
        
        # Final status
        if successful_steps == total_steps:
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            self.logger.info("⚠️  PIPELINE COMPLETED WITH ERRORS")
        
        self.logger.info("="*100)


def parse_steps_argument(steps_str: str) -> List[PipelineStep]:
    """
    Parse command line steps argument.
    
    Args:
        steps_str (str): Comma-separated list of steps
        
    Returns:
        List[PipelineStep]: List of pipeline steps
    """
    if not steps_str:
        return [PipelineStep.ALL]
    
    step_names = [s.strip().lower() for s in steps_str.split(',')]
    steps = []
    
    for step_name in step_names:
        try:
            step = PipelineStep(step_name)
            steps.append(step)
        except ValueError:
            print(f"Warning: Unknown step '{step_name}'. Available steps: {[s.value for s in PipelineStep]}")
    
    return steps if steps else [PipelineStep.ALL]


def main():
    """Main entry point for the post-processing pipeline."""
    try:
        # Parse command line arguments (simple implementation)
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Canopy Height Post-Processing Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Available steps:
  merge        - Merge prediction patches into 120km tiles
  sanitize     - Remove outliers and interpolate temporal gaps
  final_merge  - Downsample and create country-wide mosaics
  all          - Run complete pipeline (default)

Examples:
  python postprocess_pipeline.py                    # Run complete pipeline
  python postprocess_pipeline.py --steps merge      # Run only merge step
  python postprocess_pipeline.py --steps merge,sanitize  # Run specific steps
            """
        )
        
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
        
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )
        
        args = parser.parse_args()
        
        # Parse steps
        steps = parse_steps_argument(args.steps)
        
        if not steps:
            print("Error: No valid steps specified")
            sys.exit(1)
        
        # Initialize and run pipeline
        pipeline = PostProcessingPipeline(config_path=args.config)
        results = pipeline.run_pipeline(
            steps=steps,
            continue_on_error=args.continue_on_error
        )
        
        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger = setup_logging('INFO')
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()