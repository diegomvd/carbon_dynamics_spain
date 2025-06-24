#!/usr/bin/env python3
"""
Forest Type Merging Script

Script for merging forest type specific biomass maps into country-wide maps.
Updated with recipe integration arguments for harmonized path management.

Usage:
    python run_merging.py [OPTIONS]
    
Examples:
    # Run with default directories
    python run_merging.py

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_model.core.forest_type_merging import ForestTypeMergingPipeline

# Shared utilities
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Forest Type Merging for Country-wide Biomass Maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Core configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    runner = Fo

    pipeline = ForestTypeMergingPipeline(args.config)
    success = runner.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)