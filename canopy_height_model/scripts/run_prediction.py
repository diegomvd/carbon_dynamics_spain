#!/usr/bin/env python3
"""
Canopy Height Model Prediction Script

Large-scale prediction script for canopy height estimation using trained models.
Supports both single raster and batch directory processing.

Usage:
    python run_prediction.py [OPTIONS]
    
Examples:
    # Predict single raster
    python run_prediction.py --checkpoint model.ckpt --input data.tif --output pred.tif
    
    # Process entire directory
    python run_prediction.py --checkpoint model.ckpt --input-dir ./sentinel2/ --output-dir ./predictions/
    
    # Custom tile settings
    python run_prediction.py --checkpoint model.ckpt --input data.tif --output pred.tif --tile-size 1024 --overlap 128
    
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Large-Scale Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checkpoint model.ckpt --input data.tif --output pred.tif
  %(prog)s --checkpoint model.ckpt --input-dir ./sentinel2/ --output-dir ./predictions/
  %(prog)s --checkpoint model.ckpt --input data.tif --output pred.tif --tile-size 1024
  %(prog)s --checkpoint model.ckpt --input-dir ./data/ --output-dir ./pred/ --pattern "*.tif"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.ckpt file)'
    )
    
    # Input/output options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        help='Path to input Sentinel-2 raster file'
    )
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='Path to directory containing Sentinel-2 rasters'
    )
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output',
        type=str,
        help='Path to output prediction file (for single input)'
    )
    output_group.add_argument(
        '--output-dir',
        type=str,
        help='Path to output directory (for directory input)'
    )
    
    # Processing options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (uses component default if not specified)'
    )
    
    parser.add_argument(
        '--tile-size',
        type=int,
        help='Size of prediction tiles (default: from config)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        help='Overlap between tiles in pixels (default: from config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for prediction (default: from config)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif',
        help='File pattern for directory processing (default: *.tif)'
    )
    
    # Hardware options
    parser.add_argument(
        '--device',
        choices=['auto', 'gpu', 'cpu', 'mps'],
        help='Device to use for prediction (default: auto-detect)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dtype',
        choices=['float32', 'float64', 'int16', 'uint16'],
        help='Output data type (default: from config)'
    )
    
    parser.add_argument(
        '--compress',
        choices=['none', 'lzw', 'deflate', 'jpeg'],
        help='Output compression (default: from config)'
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
        help='Only validate inputs without running prediction'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show detailed progress information'
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
    
    # Validate input paths
    if args.input:
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            return False
        
        if not args.output:
            print("Error: --output required when using --input")
            return False
    
    if args.input_dir:
        if not Path(args.input_dir).exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            return False
        
        if not args.output_dir:
            print("Error: --output-dir required when using --input-dir")
            return False
    
    # Validate tile settings
    if args.tile_size and args.tile_size <= 0:
        print("Error: Tile size must be positive")
        return False
    
    if args.overlap and args.overlap < 0:
        print("Error: Overlap must be non-negative")
        return False
    
    if args.batch_size and args.batch_size <= 0:
        print("Error: Batch size must be positive")
        return False
    
    return True


def apply_argument_overrides(pipeline: ModelPredictionPipeline, args: argparse.Namespace) -> None:
    """Apply command line argument overrides to pipeline configuration."""
    config = pipeline.config
    
    if args.tile_size:
        config['prediction']['tile_size'] = args.tile_size
        pipeline.logger.info(f"Override tile_size: {args.tile_size}")
    
    if args.overlap:
        config['prediction']['overlap'] = args.overlap
        pipeline.logger.info(f"Override overlap: {args.overlap}")
    
    if args.batch_size:
        config['prediction']['batch_size'] = args.batch_size
        pipeline.logger.info(f"Override batch_size: {args.batch_size}")
    
    if args.output_dtype:
        config['prediction']['output_dtype'] = args.output_dtype
        pipeline.logger.info(f"Override output_dtype: {args.output_dtype}")
    
    if args.compress:
        config['prediction']['compress'] = args.compress
        pipeline.logger.info(f"Override compress: {args.compress}")


def validate_inputs_only(pipeline: ModelPredictionPipeline, args: argparse.Namespace) -> bool:
    """Run input validation only."""
    pipeline.logger.info("Running input validation...")
    
    try:
        # Load model to validate checkpoint
        if not pipeline.load_model(args.checkpoint):
            return False
        
        # Validate input files
        if args.input:
            if not pipeline.validate_input_raster(Path(args.input)):
                return False
            pipeline.logger.info(f"✅ Input raster validation passed: {args.input}")
        
        if args.input_dir:
            from shared_utils import find_files
            input_files = find_files(Path(args.input_dir), args.pattern)
            
            if not input_files:
                pipeline.logger.error(f"No files matching pattern '{args.pattern}' in {args.input_dir}")
                return False
            
            pipeline.logger.info(f"Found {len(input_files)} input files for processing")
            
            # Validate a few sample files
            sample_files = input_files[:3]  # Check first 3 files
            for sample_file in sample_files:
                if not pipeline.validate_input_raster(sample_file):
                    return False
            
            pipeline.logger.info(f"✅ Sample input validation passed ({len(sample_files)} files checked)")
        
        pipeline.logger.info("✅ All input validation passed")
        return True
        
    except Exception as e:
        pipeline.logger.error(f"Validation failed: {str(e)}")
        return False


def run_single_file_prediction(pipeline: ModelPredictionPipeline, args: argparse.Namespace) -> bool:
    """Run prediction for a single input file."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if output exists and overwrite flag
    if output_path.exists() and not args.overwrite:
        pipeline.logger.info(f"Output file exists and --overwrite not specified: {output_path}")
        return True
    
    # Run prediction
    return pipeline.predict_raster(input_path, output_path)


def run_directory_prediction(pipeline: ModelPredictionPipeline, args: argparse.Namespace) -> bool:
    """Run prediction for all files in a directory."""
    # Check for existing outputs if not overwriting
    if not args.overwrite:
        pipeline.logger.info("Checking for existing outputs (use --overwrite to force regeneration)")
    
    # Run prediction
    return pipeline.predict_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
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
        level=args.log_level,
        component_name='canopy_height_prediction',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Canopy Height Prediction Pipeline...")
        pipeline = ModelPredictionPipeline(config_path=args.config)
        
        # Apply command line overrides
        apply_argument_overrides(pipeline, args)
        
        # Log pipeline start
        log_pipeline_start(logger, "Canopy Height Prediction", pipeline.config)
        
        # Load model
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        if not pipeline.load_model(args.checkpoint):
            logger.error("Failed to load model")
            sys.exit(1)
        
        # Validation only mode
        if args.validate_only:
            success = validate_inputs_only(pipeline, args)
            if success:
                logger.info("✅ Validation successful - exiting")
                sys.exit(0)
            else:
                logger.error("❌ Validation failed")
                sys.exit(1)
        
        # Run prediction based on input type
        if args.input:
            logger.info(f"Processing single file: {args.input}")
            success = run_single_file_prediction(pipeline, args)
        else:
            logger.info(f"Processing directory: {args.input_dir}")
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
