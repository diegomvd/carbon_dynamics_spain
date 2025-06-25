#!/usr/bin/env python3
"""
Biomass transition distribution analysis script.

Command-line interface for analyzing biomass transition distributions between
consecutive years. This is a thin wrapper around the core InterannualAnalyzer class.

Usage:
    python run_transition_analysis.py [OPTIONS]

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
        description="Analyze biomass transition distributions between consecutive years",
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
    """Main entry point for transition analysis script."""
    args = parse_arguments()
    
    # Initialize analyzer
    biomass_change = InterannualChangePipeline(args.config)
    success = biomass_change.run_transition_analysis(True)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

