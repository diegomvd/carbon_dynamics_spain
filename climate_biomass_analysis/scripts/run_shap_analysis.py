#!/usr/bin/env python3
"""
SHAP analysis script.

Command-line interface for running comprehensive SHAP analysis on climate-biomass
optimization results, including feature importance, PDP analysis, and interactions.

Usage:
    python run_shap_analysis.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.shap_analysis import ShapAnalysisPipeline
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive SHAP analysis on climate-biomass models",
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
    """Main entry point for SHAP analysis script."""
    args = parse_arguments()

    shap_analysis = ShapAnalysisPipeline(args.config)
    success = shap_analysis.run_full_pipeline()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)