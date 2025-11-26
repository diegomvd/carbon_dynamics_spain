#!/usr/bin/env python3
"""
BGBD Estimation Pipeline Script

Entry point for running the BGBD and Total biomass estimation pipeline.
Uses analytical error propagation with Dask for efficient processing.

Usage:
    python run_bgbd_estimation.py [OPTIONS]
    
Examples:
    # Run full BGBD pipeline with default config
    python run_bgbd_estimation.py
    
    # Run with custom config
    python run_bgbd_estimation.py --config custom_config.yaml

Author: Diego Bengochea
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from biomass_model.core.bgbd_estimation import BGBDEstimationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BGBD and Total Biomass Estimation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
  Produces 4 rasters per tile:
    - BGBD_mean:  BGBD mean estimate
    - BGBD_uncertainty: BGBD uncertainty (80% CI half-width)
    - Total_mean: Total biomass mean estimate
    - Total_uncertainty: Total biomass uncertainty (80% CI half-width)
    
  Directory structure:
    BGBD_mean/2020/BGBD_mean_S2_2020_N37.0_W3.0_interpolated.tif
    BGBD_uncertainty/2020/BGBD_uncertainty_S2_2020_N37.0_W3.0_interpolated.tif
    Total_mean/2020/Total_mean_S2_2020_N37.0_W3.0_interpolated.tif
    Total_uncertainty/2020/Total_uncertainty_S2_2020_N37.0_W3.0_interpolated.tif

Notes:
  - Requires AGBD maps to be already generated
  - Uses analytical error propagation (no Monte Carlo)
  - Processes with Dask for memory efficiency
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
    args = parse_arguments()
    
    # Initialize logging
    logger = setup_logging(level='INFO', component_name='bgbd_estimation')
    
    # Log pipeline start
    log_pipeline_start(logger, "BGBD Estimation Pipeline")
    
    try:
        # Initialize pipeline
        pipeline = BGBDEstimationPipeline(config_path=args.config)
        
        # Run pipeline
        success = pipeline.run_full_pipeline()
        
        # Log completion
        log_pipeline_end(logger, success)
        
        return success
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)