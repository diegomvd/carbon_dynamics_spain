#!/usr/bin/env python3
"""
Climate data processing script.

Command-line interface for processing climate GRIB files to GeoTIFF format
with proper georeferencing, coordinate transformations, and clipping to Spain.

Usage:
    python run_climate_processing.py

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
from shared_utils.central_data_paths_constants import *


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
    
    return parser.parse_args()


def main():
    """Main entry point for climate processing script."""
    args = parse_arguments()

    log_level = 'INFO'
    logger = setup_logging(level=log_level, component_name='climate_processing_script')
    
    try:
        # Initialize processor
        logger.info("Initializing climate processor...")
        processor = ClimateProcessor(config_path=args.config)
        
        # Process directory mode
        logger.info(f"Processing directory: {str(CLIMATE_RAW_DIR)}")
        logger.info(f"Output directory: {str(CLIMATE_HARMONIZED_DIR)}")
        
        results = processor.process_directory(
            input_dir=str(CLIMATE_RAW_DIR),
            output_dir=str(CLIMATE_HARMONIZED_DIR)
        )
        
        # Report results
        successful = sum(results.values())
        total = len(results)
        logger.info(f"Processing completed: {successful}/{total} files successful")
        
        if successful < total:
            failed_files = [f for f, success in results.items() if not success]
            logger.warning(f"Failed files: {failed_files}")
    
  
            logger.info("Validating output files...")
            validation_result = processor.validate_outputs(str(CLIMATE_HARMONIZED_DIR))
            
            if validation_result:
                logger.info("✅ All output files validated successfully")
            else:
                logger.error("❌ Output validation failed")
                sys.exit(1)
        
        logger.info("Climate processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Climate processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()