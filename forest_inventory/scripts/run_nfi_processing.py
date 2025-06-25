#!/usr/bin/env python3
"""
NFI Processing Script

Command-line interface for Spanish National Forest Inventory (NFI) biomass processing.
Provides configurable access to the main processing pipeline with validation and logging.

Usage Examples:
    # Run with default configuration
    python scripts/run_nfi_processing.py
    
    # Run with custom configuration
    python scripts/run_nfi_processing.py --config custom_config.yaml

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


# Component imports
from forest_inventory.core.nfi_processing import NFIProcessingPipeline
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Spanish National Forest Inventory (NFI) Biomass Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: uses component default)'
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for NFI processing script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    nfi_processing = NFIProcessingPipeline(args.config)
    success = nfi_processing.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
