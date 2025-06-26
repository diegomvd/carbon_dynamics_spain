#!/usr/bin/env python3
"""
Canopy Height Post-Processing Script

Multi-step post-processing pipeline for canopy height predictions:
1. Merge prediction patches into 120km tiles
2. Sanitize outliers and interpolate temporal gaps  
3. Create final country-wide mosaics at 100m resolution

Usage:
    python run_postprocessing.py [OPTIONS]
    
Examples:
    # Run complete 3-step pipeline
    python run_postprocessing.py
    
    # Run individual steps
    python run_postprocessing.py --steps merge
    python run_postprocessing.py --steps sanitize
    python run_postprocessing.py --steps final_merge
    
    # Run multiple steps
    python run_postprocessing.py --steps merge,sanitize

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from canopy_height_model.core.postprocessing import PostProcessingPipeline, PipelineStep
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Post-Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Pipeline control
    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated list of steps to run (default: all)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    postprocessing = PostProcessingPipeline(args.config,args.steps)
    success = postprocessing.run_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)