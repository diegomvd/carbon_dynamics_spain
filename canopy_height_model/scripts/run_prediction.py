#!/usr/bin/env python3
"""
Canopy Height Model Prediction Script

Large-scale prediction script for canopy height estimation using trained models.
Supports both single raster and batch directory processing.

Usage:
    python run_prediction.py [OPTIONS]
    
Examples:
    
    # Process entire directory
    python run_prediction.py --checkpoint model.ckpt --input-dir ./sentinel2/ --output-dir ./predictions/
    
    # Batch processing with pattern
    python run_prediction.py --checkpoint model.ckpt --input-dir ./data/ --output-dir ./pred/ --pattern "S2_*_mosaic_*.tif"

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from canopy_height_model.core.model_prediction import ModelPredictionPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import * 


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Large-Scale Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.ckpt file)'
    )
    
    # Processing options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (uses component default if not specified)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif',
        help='File pattern for directory processing (default: *.tif)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Check checkpoint file
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return False
    
    # Check config file if provided
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    return True

# 
def run_directory_prediction(pipeline: ModelPredictionPipeline, args: argparse.Namespace) -> bool:
    """Run prediction for all files in a directory."""
    
    # Run prediction
    return pipeline.predict_directory(
        input_dir=SENTINEL2_MOSAICS_DIR,
        output_dir=HEIGHT_MAPS_TMP_RAW_DIR,
        file_pattern=args.pattern
    )


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(
        level='INFO',
        component_name='canopy_height_prediction'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Canopy Height Prediction Pipeline...")
        pipeline = ModelPredictionPipeline(config_path=args.config)
        
        # Log pipeline start
        log_pipeline_start(logger, "Canopy Height Prediction", pipeline.config)
        
        # Load model
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        if not pipeline.load_model(args.checkpoint):
            logger.error("Failed to load model")
            sys.exit(1)
               
        success = run_directory_prediction(pipeline, args)
        
        # Get prediction summary
        summary = pipeline.get_prediction_summary()
        logger.info("Prediction Summary:")
        for section, details in summary.items():
            if isinstance(details, dict):
                logger.info(f"  {section}:")
                for key, value in details.items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.info(f"  {section}: {details}")
        
        # Pipeline completion
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Canopy Height Prediction", success, elapsed_time)
        
        if success:
            logger.info("🎉 Prediction completed successfully!")
            if args.output:
                logger.info(f"Output saved to: {args.output}")
            elif args.output_dir:
                logger.info(f"Outputs saved to: {args.output_dir}")
            sys.exit(0)
        else:
            logger.error("💥 Prediction failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed with unexpected error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
