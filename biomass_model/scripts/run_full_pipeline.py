#!/usr/bin/env python3
"""
Biomass Model Full Pipeline Orchestrator

Complete biomass estimation pipeline that orchestrates all processing steps:
1. Allometry fitting (optional)
2. Biomass estimation (forest type specific maps)
3. Annual cropland masking 
4. Forest type merging (country-wide maps)

Usage:
    python run_full_pipeline.py 

Examples:
    # Run complete pipeline
    python run_full_pipeline.py

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_model.core.biomass_estimation import BiomassEstimationPipeline
from biomass_model.core.allometry_fitting import AllometryFittingPipeline
from biomass_model.core.land_use_masking import LandUseMaskingPipeline
from biomass_model.core.forest_type_merging import ForestTypeMergingPipeline

# Shared utilities
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end


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

    allometry = AllometryFittingPipeline(args.config)
    success_allom = allometry.run_full_pipeline()

    biomass = BiomassEstimationPipeline(args.config)
    success_biomass = biomass.run_full_pipeline()

    land_use_mask = LandUseMaskingPipeline(args.config)
    succes_lum = land_use_mask.run_full_pipeline()

    merge = ForestTypeMergingPipeline(args.config)
    succes_merge = merge.run_full_pipeline()

    return all([success_allom,success_biomass,succes_lum,succes_merge])
        
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

