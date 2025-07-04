#!/usr/bin/env python3
"""
Biomass Estimation Pipeline Script

Main entry point for running the biomass estimation pipeline with
Monte Carlo uncertainty quantification. Updated with recipe integration
arguments for harmonized path management.

Usage:
    python run_biomass_estimation.py [OPTIONS]
    
Examples:
    # Run full pipeline with default config
    python run_biomass_estimation.py
    
    # Run with custom config
    python run_biomass_estimation.py --config custom_config.yaml

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from biomass_model.core.biomass_estimation import BiomassEstimationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Biomass Estimation Pipeline with Monte Carlo Uncertainty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Core configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    pipeline = BiomassEstimationPipeline(config_path=args.config)
    success = pipeline.run_full_pipeline()

    return success
        
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)