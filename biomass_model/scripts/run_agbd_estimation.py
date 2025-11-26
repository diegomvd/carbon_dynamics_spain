#!/usr/bin/env python3
"""
AGBD Estimation Pipeline Script

Entry point for running the AGBD-only estimation pipeline using direct
quantile allometries (no Monte Carlo sampling). Produces mean and uncertainty
bounds (70% CI) for above-ground biomass density.

This pipeline is optimized for generating AGBD maps that will be used in
post-processing to distinguish stress-induced apparent loss from structural loss.

Usage:
    python run_agbd_estimation.py [OPTIONS]
    
Examples:
    # Run full AGBD pipeline with default config
    python run_agbd_estimation.py
    
    # Run with custom config
    python run_agbd_estimation.py --config custom_config.yaml

Author: Diego Bengochea
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from biomass_model.core.agbd_estimation import AGBDEstimationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AGBD Estimation Pipeline with Direct Quantile Allometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
  Produces 3 rasters per forest type per tile:
    - AGBD_mean:  Mean estimate (from median allometry)
    - AGBD_lower: Lower bound (from 15th percentile, 70% CI)
    - AGBD_upper: Upper bound (from 85th percentile, 70% CI)
    
  Directory structure:
    AGBD_mean/2020/AGBD_mean_2020_100m_tile_XXX_code_YY.tif
    AGBD_lower/2020/AGBD_lower_2020_100m_tile_XXX_code_YY.tif
    AGBD_upper/2020/AGBD_upper_2020_100m_tile_XXX_code_YY.tif

Notes:
  - This pipeline is PARALLEL to run_biomass_estimation.py
  - Does not produce BGBD or total biomass outputs
  - No Monte Carlo sampling (faster than full pipeline)
  - Uses 70% confidence intervals (p15-p85) instead of 95% CI
"""
    )
    
    # Core configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file',
        default=None
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize pipeline
    try:
        pipeline = AGBDEstimationPipeline(config_path=args.config)
    except Exception as e:
        print(f"ERROR: Failed to initialize AGBD pipeline: {e}")
        return False
    
    # Log pipeline start
    
    # Run pipeline
    try:
        success = pipeline.run_full_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

        
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)