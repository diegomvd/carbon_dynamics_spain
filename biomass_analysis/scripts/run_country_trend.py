#!/usr/bin/env python3
"""
Country-level biomass time series analysis script.

Command-line interface for country-level biomass analysis with Monte Carlo 
uncertainty quantification. This is a thin wrapper around the core 
MonteCarloAnalyzer class.

Usage:
    python run_country_trend.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.monte_carlo_analysis import MonteCarloAggregationPipeline
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Country-level biomass time series analysis with Monte Carlo uncertainty quantification",
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
    """Main entry point for country trend analysis script."""
    args = parse_arguments()
    biomass_country_level = MonteCarloAggregationPipeline(args.config)
    success = biomass_country_level.run_full_pipeline()

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
