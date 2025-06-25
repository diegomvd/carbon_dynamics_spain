#!/usr/bin/env python3
"""
Machine learning optimization script.

Command-line interface for running Bayesian optimization to select optimal
climate predictors and hyperparameters for biomass change prediction.

Usage:
    python run_optimization.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline
from shared_utils import setup_logging, ensure_directory
from shared_utils.central_data_paths import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for biomass-climate modeling",
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
    """Main entry point for optimization script."""
    args = parse_arguments()

    optimization = OptimizationPipeline(args.config)
    success = optimization.run_optimization_pipeline()
    return success 

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)