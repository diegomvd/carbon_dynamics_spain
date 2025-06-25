#!/usr/bin/env python3
"""
PNOA LiDAR Processing Script

Processes PNOA LiDAR tiles by selecting those that intersect with Sentinel-2 tiles
and standardizes them for training data preparation.

Usage:
    python run_pnoa_processing.py [OPTIONS]

Examples:
    # Basic processing
    python run_pnoa_processing.py
    
    # Custom configuration
    python run_pnoa_processing.py --config custom_config.yaml

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from als_pnoa.core.pnoa_processor import PNOAProcessingPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Selects relevant PNOA tiles and reprojects them to UTM30",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    return parser.parse_args()

def main():
    """Main entry point for PNOA processing script."""

    # Parse arguments
    args = parse_arguments()
    pnoa_tiles = PNOAProcessingPipeline()
    success = pnoa_tiles.run_full_pipeline()
    return success 

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
