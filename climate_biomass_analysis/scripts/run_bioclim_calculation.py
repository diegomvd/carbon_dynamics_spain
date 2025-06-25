#!/usr/bin/env python3
"""
Bioclimatic variables calculation script.

Command-line interface for calculating bioclimatic variables (bio1-bio19) from
monthly temperature and precipitation data, and computing climate anomalies.

Usage:
    python run_bioclim_calculation.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import glob
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculationPipeline
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate bioclimatic variables and anomalies",
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
    """Main entry point for bioclimatic calculation script."""
    args = parse_arguments()
    
    climate_processing = BioclimCalculationPipeline(args.config)
    success = climate_processing.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
