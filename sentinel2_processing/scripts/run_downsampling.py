#!/usr/bin/env python3
"""
Sentinel-2 Downsampling and Merging Script

Command-line interface for spatial downsampling and merging operations.
Converts raw mosaic tiles into analysis-ready products through sequential
downsampling and yearly merging operations.

Usage Examples:
    # Run complete workflow
    python scripts/run_downsampling.py
    
    # Use custom scale factor
    python scripts/run_downsampling.py --scale-factor 5
    
    # Only run downsampling step
    python scripts/run_downsampling.py --downsample-only
    
    # Only run merging step
    python scripts/run_downsampling.py --merge-only
    
    # Use custom configuration
    python scripts/run_downsampling.py --config custom.yaml
    
    # Override directories
    python scripts/run_downsampling.py --input-dir /custom/input --output-dir /custom/output

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.postprocessing import DownsamplingMergingProcessor
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Downsampling and Merging Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
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

def main() -> int:
    """
    Main entry point for downsampling script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Setup logging
        setup_logging(level='INFO', log_file='sentinel2_downsampling_and_merging')
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        
        # Initialize processor
        processor = DownsamplingMergingProcessor(config)

        logger.info("Running complete downsampling and merging workflow...")
        results = processor.run_complete_workflow()
        
        if results['success']:
            logger.info("Complete workflow successful!")
            logger.info(f"Downsampling: {results['downsampling']['successful']} successful, {results['downsampling']['failed']} failed")
            logger.info(f"Merging: {results['merging']['successful']} successful, {results['merging']['failed']} failed")
        else:
            logger.error(f"Workflow failed: {results.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Downsampling pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())