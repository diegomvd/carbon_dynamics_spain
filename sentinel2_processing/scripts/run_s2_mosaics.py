#!/usr/bin/env python3
"""
Sentinel-2 Mosaic Processing Script

Command-line interface for the main Sentinel-2 mosaic processing pipeline.
Creates summer mosaics over Spain using distributed computing with STAC catalog
integration and optimized memory management.

Usage Examples:
    # Run with default configuration
    python scripts/run_mosaic_processing.py
    
    # Run with custom configuration
    python scripts/run_mosaic_processing.py --config custom_config.yaml

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.mosaic_processing import MosaicProcessingPipeline


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Mosaic Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    if args.config and not Path(args.config).exists():
        print(f"Error: Configuration file does not exist: {args.config}")
        return False
    
    return True

def print_processing_summary(pipeline: MosaicProcessingPipeline) -> None:
    """
    Print processing summary information.
    
    Args:
        pipeline: Initialized pipeline instance
    """
    summary = pipeline.get_processing_summary()
    
    print("\n" + "="*60)
    print("SENTINEL-2 MOSAIC PROCESSING SUMMARY")
    print("="*60)
    print(f"Processing Years:     {summary['config_summary']['years']}")
    print(f"Scenes per Mosaic:    {summary['config_summary']['n_scenes']}")
    print(f"Tile Size:            {summary['config_summary']['tile_size']}")
    print(f"Total Combinations:   {summary['total_combinations']}")
    print(f"Dask Workers:         {summary['config_summary']['n_workers']}")
    print("="*60)


def main() -> int:
    """
    Main entry point for mosaic processing script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Setup logging
        setup_logging(level='INFO', log_file='sentinel2_mosaicing')
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        
        # Initialize pipeline
        pipeline = MosaicProcessingPipeline()
        pipeline.config = config
        
        # Validate configuration
        if not pipeline.validate_configuration():
            logger.error("Configuration validation failed")
            return 1
        
        # Run processing
        logger.info("Starting Sentinel-2 mosaic processing...")
        start_time = time.time()
        
        results = pipeline.run_processing()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Total Duration: {duration/60:.1f} minutes")
        logger.info(f"Processed: {results['processed_count']}")
        logger.info(f"Skipped: {results['skipped_count']}")
        logger.info(f"Errors: {results['error_count']}")
        logger.info(f"Success Rate: {results['success_rate']:.1f}%")
        logger.info("="*60)
        
        return 0 if results['error_count'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())