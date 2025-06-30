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


def main() -> int:
    """
    Main entry point for mosaic processing script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    s2_mosaicing = MosaicProcessingPipeline(args.config)
    success = s2_mosaicing.run_full_pipeline()
    return success 

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)