#!/usr/bin/env python3
"""
Annual Cropland Masking Script

Script for masking biomass maps to exclude annual cropland areas using
Corine Land Cover data. Updated with recipe integration arguments.

Usage:
    python run_masking.py [OPTIONS]
    
Examples:
    # Run with default directories
    python run_masking.py

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from biomass_model.core.land_use_masking import LandUseMaskingPipeline

# Shared utilities
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Annual Cropland Masking for Biomass Maps",
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

    pipeline = LandUseMaskingPipeline(args.config)
    success = pipeline.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
