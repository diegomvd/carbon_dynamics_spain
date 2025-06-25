#!/usr/bin/env python3
"""
Spatial analysis script.

Command-line interface for performing spatial autocorrelation analysis and
creating spatial clusters for cross-validation in machine learning.

Usage:
    python run_spatial_analysis.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.spatial_analysis import SpatialAnalyzer
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform spatial autocorrelation analysis and clustering",
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
    """Main entry point for spatial analysis script."""
    args = parse_arguments()
    
    spatial_analysis = SpatialAnalysisPipeline(args.config)
    success = spatial_analysis.run_full_pipeline()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)