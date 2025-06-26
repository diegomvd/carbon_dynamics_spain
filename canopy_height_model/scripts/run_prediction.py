#!/usr/bin/env python3
"""
Canopy Height Model Prediction Script

Large-scale prediction script for canopy height estimation using trained models.
Supports both single raster and batch directory processing.

Usage:
    python run_prediction.py [OPTIONS]
    
Examples:
    
    # Process entire directory
    python run_prediction.py --checkpoint model.ckpt
    
    # Batch processing with pattern
    python run_prediction.py --checkpoint model.ckpt --pattern "S2_*_mosaic_*.tif"

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from canopy_height_model.core.model_prediction import ModelPredictionPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import * 


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Large-Scale Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.ckpt file)'
    )
    
    # Processing options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (uses component default if not specified)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif',
        help='File pattern for directory processing (default: *.tif)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    model_prediction = ModelPredictionPipeline(args.config, args.checkpoint, args.pattern)
    success = model_prediction.run_full_pipeline(SENTINEL2_MOSAICS_DIR,HEIGHT_MAPS_TMP_RAW_DIR)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)