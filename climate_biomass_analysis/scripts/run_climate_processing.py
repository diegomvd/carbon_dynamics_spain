#!/usr/bin/env python3
"""
Climate data processing script.

Command-line interface for processing climate GRIB files to GeoTIFF format
with proper georeferencing, coordinate transformations, and clipping to Spain.

Usage:
    python run_climate_processing.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.climate_raster_processing import ClimateProcessor
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process climate GRIB files to GeoTIFF format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing GRIB files'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Output directory for GeoTIFF files'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.grib',
        help='File pattern to match in input directory'
    )
    
    # Processing options
    parser.add_argument(
        '--reference-grid',
        type=str,
        help='Reference raster file for exact grid alignment'
    )
    
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Skip clipping to Spain boundary'
    )
    
    parser.add_argument(
        '--target-crs',
        type=str,
        help='Target coordinate reference system (overrides config)'
    )
    
    # Processing control
    parser.add_argument(
        '--validate-outputs',
        action='store_true',
        help='Validate output files after processing'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for climate processing script."""
    args = parse_arguments()
    
    # Setup logging
    if args.quiet:
        log_level = 'ERROR'
    else:
        log_level = args.log_level
    
    logger = setup_logging(level=log_level, component_name='climate_processing_script')
    
    try:
        # Initialize processor
        logger.info("Initializing climate processor...")
        processor = ClimateProcessor(config_path=args.config)
        
        # Override config with command line arguments if provided
        if args.target_crs:
            processor.target_crs = args.target_crs
            logger.info(f"Using command line target CRS: {args.target_crs}")
        
        # Determine input and output directories
        if args.input_dir and args.output_dir:
            # Process directory mode
            logger.info(f"Processing directory: {args.input_dir}")
            logger.info(f"Output directory: {args.output_dir}")
            
            results = processor.process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                reference_grid=args.reference_grid,
                clip_to_spain=not args.no_clip
            )
            
            # Report results
            successful = sum(results.values())
            total = len(results)
            logger.info(f"Processing completed: {successful}/{total} files successful")
            
            if successful < total:
                failed_files = [f for f, success in results.items() if not success]
                logger.warning(f"Failed files: {failed_files}")
        
        else:
            # Use configuration file paths
            logger.info("Using configuration file for input/output paths")
            logger.info("Note: Directory processing requires --input-dir and --output-dir arguments")
            logger.info("For configuration-based processing, implement additional logic here")
        
        # Validate outputs if requested
        if args.validate_outputs and args.output_dir:
            logger.info("Validating output files...")
            validation_result = processor.validate_outputs(args.output_dir)
            
            if validation_result:
                logger.info("✅ All output files validated successfully")
            else:
                logger.error("❌ Output validation failed")
                sys.exit(1)
        
        logger.info("Climate processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Climate processing failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()