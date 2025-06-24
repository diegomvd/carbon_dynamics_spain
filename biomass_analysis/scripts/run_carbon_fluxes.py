#!/usr/bin/env python3
"""
Carbon flux analysis script.

Command-line interface for calculating interannual carbon fluxes from Monte Carlo
biomass samples. This is a thin wrapper around the core CarbonFluxPipeline class.

Usage:
    python run_carbon_fluxes.py 

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.carbon_flux_analysis import CarbonFluxPipeline
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate interannual carbon fluxes from Monte Carlo biomass samples",
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
    """Main entry point for carbon flux analysis script."""
    args = parse_arguments()
    
    fluxes = CarbonFluxPipeline(args.config)
    success = fluxes.run_full_pipeline()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
