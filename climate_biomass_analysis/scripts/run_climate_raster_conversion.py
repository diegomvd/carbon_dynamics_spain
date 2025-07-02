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

from climate_biomass_analysis.core.climate_raster_conversion import ClimateRasterConversionPipeline
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
    climate_processing = ClimateRasterConversionPipeline(args.config)
    success = climate_processing.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)