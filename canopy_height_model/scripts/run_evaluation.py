#!/usr/bin/env python3
"""
Canopy Height Model Evaluation Script

Entry point for comprehensive evaluation of trained canopy height regression models.
Uses the ModelEvaluationPipeline from the core module to generate statistical analysis,
residual plots, prediction density visualizations, and height distribution comparisons.

Usage:
    python run_evaluation.py
    
Examples:
    # Evaluate with default configuration
    python run_evaluation.py
    
    # Custom configuration and checkpoint
    python run_evaluation.py --config custom_config.yaml --checkpoint model.ckpt
    
    # Multiple checkpoints comparison
    python run_evaluation.py --checkpoint model1.ckpt model2.ckpt

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
from canopy_height_model.core.model_evaluation import ModelEvaluationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: use component config)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        help='Path(s) to model checkpoint(s) for evaluation'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    if args.checkpoint:
        for checkpoint in args.checkpoint:
            if not Path(checkpoint).exists():
                print(f"Error: Checkpoint file not found: {checkpoint}")
                return False
    
    return True


def run_single_evaluation(
    pipeline: ModelEvaluationPipeline,
    checkpoint_path: str,
    output_dir: Optional[str],
    logger
) -> bool:
    """
    Run evaluation for a single checkpoint.
    
    Args:
        pipeline: Evaluation pipeline instance
        checkpoint_path: Path to checkpoint file
        output_dir: Custom output directory
        logger: Logger instance
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        
        # Run evaluation
        results = pipeline.evaluate_model(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir
        )
        
        # Log key metrics if available
        if results and 'test_loss' in results:
            logger.info(f"Evaluation completed. Test loss: {results['test_loss']:.4f}")
        else:
            logger.info("Evaluation completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to evaluate checkpoint {checkpoint_path}: {e}")
        return False


def run_multiple_evaluations(
    pipeline: ModelEvaluationPipeline,
    checkpoint_paths: List[str],
    base_output_dir: Optional[str],
    logger
) -> bool:
    """
    Run evaluation for multiple checkpoints.
    
    Args:
        pipeline: Evaluation pipeline instance  
        checkpoint_paths: List of checkpoint file paths
        base_output_dir: Base output directory
        logger: Logger instance
        
    Returns:
        bool: True if all evaluations successful
    """
    success_count = 0
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}")
        
        # Create checkpoint-specific output directory
        if base_output_dir:
            checkpoint_name = Path(checkpoint_path).stem
            checkpoint_output_dir = str(Path(base_output_dir) / checkpoint_name)
        else:
            checkpoint_output_dir = None
        
        # Run evaluation
        if run_single_evaluation(pipeline, checkpoint_path, checkpoint_output_dir, logger):
            success_count += 1
    
    logger.info(f"Completed {success_count}/{len(checkpoint_paths)} evaluations successfully")
    return success_count == len(checkpoint_paths)


def main():
    """Main entry point for the evaluation pipeline."""
    start_time = time.time()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args):
            sys.exit(1)
        
        # Setup logging
        logger = setup_logging('INFO')
        log_pipeline_start(logger, "Canopy Height Model Evaluation Pipeline")
        
        # Initialize evaluation pipeline
        try:
            pipeline = ModelEvaluationPipeline(config_path=args.config)
        except Exception as e:
            logger.error(f"Failed to initialize evaluation pipeline: {e}")
            sys.exit(1)
        
        # Determine checkpoints to evaluate
        checkpoints = args.checkpoint
        if not checkpoints:
            logger.error("No checkpoint specified. Use --checkpoint to specify checkpoint file(s)")
            sys.exit(1)
        
        # Run evaluation(s)
        if len(checkpoints) == 1:
            success = run_single_evaluation(
                pipeline, 
                checkpoints[0], 
                args.output_dir, 
                logger
            )
        else:
            success = run_multiple_evaluations(
                pipeline, 
                checkpoints, 
                args.output_dir, 
                logger
            )
        
        if success:
            log_pipeline_end(logger, "Canopy Height Model Evaluation Pipeline", start_time)
            logger.info("Evaluation pipeline completed successfully!")
        else:
            logger.error("Evaluation pipeline completed with errors")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Evaluation pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Evaluation pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
