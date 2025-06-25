#!/usr/bin/env python3
"""
Biomass-climate data integration script.

Command-line interface for integrating biomass change data with climate anomalies
to create machine learning training datasets with proper spatial alignment.

Usage:
    python run_biomass_integration.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.biomass_integration import BiomassIntegrationPipeline
from shared_utils import setup_logging, ensure_directory
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integrate biomass changes with climate anomalies",
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
    """Main entry point for biomass integration script."""
    args = parse_arguments()
    
    climate_biomass_integration = BiomassIntegrationPipeline()
    success = climate_biomass_integration.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)