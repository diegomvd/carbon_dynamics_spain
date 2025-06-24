#!/usr/bin/env python3
"""
Robustness Assessment Script

Command-line interface for evaluating mosaic robustness with varying numbers
of input scenes to optimize scene selection parameters. Integrates with STAC
catalog for comprehensive scene testing and provides recommendations for
optimal processing configurations.

Usage Examples:
    # Use example area (San Francisco Bay)
    python scripts/run_robustness_assessment.py

Author: Diego Bengochea
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.postprocessing import RobustnessAssessor


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Robustness Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    if args.config and not Path(args.config).exists():
        print(f"Error: Configuration file does not exist: {args.config}")
        return False
    return True

def parse_dates(args: argparse.Namespace) -> tuple:
    """
    Parse start and end dates from arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    # Default to 2022
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)

    return start_date, end_date


def print_brief_summary(results: dict) -> None:
    """
    Print brief assessment summary.
    
    Args:
        results: Assessment results dictionary
    """
    if results['success']:
        print(f"Assessment: SUCCESS")
        print(f"Optimal scenes: {results['optimal_scenes']}")
        print(f"Total scenes analyzed: {results['total_scenes_loaded']}")
        print(f"Area: {results['bbox']}")
        print(f"Period: {results['time_period']}")
    else:
        print(f"Assessment: FAILED - {results.get('error', 'Unknown error')}")


def save_detailed_results(results: dict) -> None:
    """
    Save detailed results to file.
    
    Args:
        results: Assessment results dictionary
        filename: Output filename
    """
    import json
    filename = SENTINEL2_PROCESSED_DIR / "median_robustness_analysis.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def main() -> int:
    """
    Main entry point for robustness assessment script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Setup logging
        setup_logging(level='INFO', log_file='sentinel2_median_robustness')
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        
        # Initialize assessor
        assessor = RobustnessAssessor(config)
        
        # Set up assessment parameters
        bbox = None
        start_date, end_date = parse_dates(args)
        
        logger.info(f"Assessment period: {start_date.date()} to {end_date.date()}")
        
        # Run assessment
        logger.info("Starting robustness assessment...")
        results = assessor.run_robustness_assessment(bbox, start_date, end_date)
        
        if results['success']:
            logger.info(f"Assessment completed successfully!")
            logger.info(f"Optimal scenes recommendation: {results['optimal_scenes']}")
            logger.info(f"Total scenes analyzed: {results['total_scenes_loaded']}")
            logger.info(f"Assessment area: {results['bbox']}")
            logger.info(f"Time period: {results['time_period']}")
        
            save_detailed_results(results)
            logger.info(f"Detailed results saved to")
        
            return 0
            
        else:
            logger.error(f"Assessment failed: {results.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        logger.error(f"Robustness assessment failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())