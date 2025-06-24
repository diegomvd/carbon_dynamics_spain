#!/usr/bin/env python3
"""
Allometry Fitting Pipeline Script

Entry point for fitting allometric relationships from NFI data and 
canopy height predictions. Uses complete implementation with quantile
regression and hierarchical forest type processing.

Features:
- Dynamic training dataset creation from NFI data + height maps
- Height-AGB allometry fitting using quantile regression  
- BGB-AGB ratio fitting using hierarchical percentile calculation
- Hierarchical forest type processing (General → Clade → Family → Genus → ForestType)
- Compatible outputs for existing biomass estimation pipeline
- Integration with harmonized data paths

Usage:
    python run_allometry_fitting.py 
    
Examples:
    # Fit allometries with default config
    python run_allometry_fitting.py
    
    # Custom configuration
    python run_allometry_fitting.py --config fitting_config.yaml

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Shared utilities
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import *

# Component imports
from biomass_model.core.allometry_fitting import AllometryFittingPipeline
from biomass_model.core.allometry import AllometryManager


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Allometric Relationship Fitting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    
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
    # Initialize and run pipeline
    pipeline = AllometryFittingPipeline(args.config)
    success = pipeline.run_full_pipeline()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)