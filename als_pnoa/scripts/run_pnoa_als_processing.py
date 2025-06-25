#!/usr/bin/env python3
"""
PNOA LiDAR Processing Script

Processes PNOA LiDAR tiles by selecting those that intersect with Sentinel-2 tiles
and standardizes them for training data preparation.

Usage:
    python run_pnoa_processing.py [OPTIONS]

Examples:
    # Basic processing
    python run_pnoa_processing.py
    
    # Custom configuration
    python run_pnoa_processing.py --config custom_config.yaml

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from als_pnoa.core.pnoa_processor import PNOAProcessingPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Selects relevant PNOA tiles and reprojects them to UTM30",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    return parser.parse_args()

def main():
    """Main entry point for PNOA processing script."""

    # Parse arguments
    args = parse_arguments()
    pnoa_tiles = PNOAProcessingPipeline()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Set logging level
    log_level = 'INFO'
    
    # Setup logging
    logger = setup_logging(
        level=log_level,
        component_name='pnoa_processing'
    )
    
    try:
        config = load_config(args.config, component_name="als_pnoa")

        # Initialize processor
        processor = PNOAProcessor(config)
        
        logger.info("Starting PNOA LiDAR processing pipeline...")
        
        # Validate inputs
        if not processor.validate_inputs():
            logger.error("Input validation failed")
            sys.exit(1)
        
        # Process tiles
        logger.info("Processing PNOA tiles...")
        results = processor.process_all_tiles()
        
        # Print results
        execution_time = time.time() - start_time
        
        if not args.quiet:
            print_results_summary(results, execution_time)
        
        # Determine success
        if results['errors'] == 0:
            logger.info("🎉 PNOA processing completed successfully!")
        else:
            error_rate = (results['errors'] / results['total_selected']) * 100
            if error_rate > 10:  # More than 10% errors
                logger.error(f"❌ High error rate: {error_rate:.1f}%")
                sys.exit(1)
            else:
                logger.warning(f"⚠️  Processing completed with some errors: {error_rate:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Processing failed with error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()