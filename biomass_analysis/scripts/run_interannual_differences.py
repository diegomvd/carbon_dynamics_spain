#!/usr/bin/env python3
"""
Interannual biomass difference mapping script.

Command-line interface for creating interannual biomass difference maps between
consecutive years. This is a thin wrapper around the core InterannualAnalyzer class.

Usage:
    python run_interannual_differences.py 

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.interannual_analysis import InterannualChangePipeline
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create interannual biomass difference maps for consecutive years",
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
    """Main entry point for interannual differences script."""
    args = parse_arguments()

    biomass_change = InterannualChangePipeline(args.config)
    success = biomass_change.run_difference_mapping()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

