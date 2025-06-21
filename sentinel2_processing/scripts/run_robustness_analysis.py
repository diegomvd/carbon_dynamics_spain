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
    
    # Use custom bounding box
    python scripts/run_robustness_assessment.py --bbox -10 35 5 45
    
    # Assess specific year
    python scripts/run_robustness_assessment.py --year 2023
    
    # Custom scene range testing
    python scripts/run_robustness_assessment.py --min-scenes 3 --max-scenes 20
    
    # Use custom configuration
    python scripts/run_robustness_assessment.py --config custom.yaml
    
    # Spain-specific assessment
    python scripts/run_robustness_assessment.py --bbox -9.5 36.0 3.3 43.8 --year 2022

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
        epilog="""
Examples:
  %(prog)s                                    # Use example area (San Francisco)
  %(prog)s --bbox -10 35 5 45                # Use custom bounding box
  %(prog)s --year 2023                       # Assess specific year
  %(prog)s --min-scenes 3 --max-scenes 20    # Custom scene range
  %(prog)s --config custom.yaml              # Use custom configuration
  %(prog)s --bbox -9.5 36.0 3.3 43.8 --year 2022  # Spain assessment

Bounding box format: WEST SOUTH EAST NORTH (longitude latitude)

For more information, see the component documentation.
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Spatial options
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help='Bounding box coordinates (longitude latitude)'
    )
    
    # Temporal options
    parser.add_argument(
        '--year',
        type=int,
        help='Year to assess (default: 2022)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD format)'
    )
    
    # Assessment parameters
    parser.add_argument(
        '--min-scenes',
        type=int,
        help='Minimum number of scenes to test (overrides config)'
    )
    
    parser.add_argument(
        '--max-scenes',
        type=int,
        help='Maximum number of scenes to test (overrides config)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        help='Step size for scene number testing (overrides config)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        help='Number of random samples per scene count (overrides config)'
    )
    
    # Output options
    parser.add_argument(
        '--save-results',
        type=str,
        help='File to save detailed results (default: no file saved)'
    )
    
    parser.add_argument(
        '--brief',
        action='store_true',
        help='Show only brief summary (suppress detailed output)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: console only)'
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
    
    if args.bbox and len(args.bbox) != 4:
        print("Error: Bounding box must have exactly 4 coordinates")
        return False
    
    if args.bbox:
        west, south, east, north = args.bbox
        if west >= east:
            print("Error: West longitude must be less than east longitude")
            return False
        if south >= north:
            print("Error: South latitude must be less than north latitude")
            return False
    
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print("Error: Start date must be in YYYY-MM-DD format")
            return False
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print("Error: End date must be in YYYY-MM-DD format")
            return False
    
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start >= end:
            print("Error: Start date must be before end date")
            return False
    
    if args.min_scenes and args.max_scenes and args.min_scenes >= args.max_scenes:
        print("Error: Minimum scenes must be less than maximum scenes")
        return False
    
    if args.save_results and not Path(args.save_results).parent.exists():
        print(f"Error: Results file directory does not exist: {Path(args.save_results).parent}")
        return False
    
    if args.log_file and not Path(args.log_file).parent.exists():
        print(f"Error: Log file directory does not exist: {Path(args.log_file).parent}")
        return False
    
    return True


def apply_config_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Apply command-line argument overrides to configuration.
    
    Args:
        config: Configuration dictionary
        args: Parsed arguments
        
    Returns:
        dict: Updated configuration
    """
    if args.min_scenes:
        config.setdefault('robustness', {})['min_scenes'] = args.min_scenes
    
    if args.max_scenes:
        config.setdefault('robustness', {})['max_scenes'] = args.max_scenes
    
    if args.step:
        config.setdefault('robustness', {})['step'] = args.step
    
    if args.n_samples:
        config.setdefault('robustness', {})['n_samples'] = args.n_samples
    
    return config


def parse_dates(args: argparse.Namespace) -> tuple:
    """
    Parse start and end dates from arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    elif args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
    else:
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


def save_detailed_results(results: dict, filename: str) -> None:
    """
    Save detailed results to file.
    
    Args:
        results: Assessment results dictionary
        filename: Output filename
    """
    import json
    
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
        setup_logging(level=args.log_level, log_file=args.log_file)
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        config = apply_config_overrides(config, args)
        
        # Initialize assessor
        assessor = RobustnessAssessor(config)
        
        # Set up assessment parameters
        bbox = args.bbox if args.bbox else None
        start_date, end_date = parse_dates(args)
        
        if bbox:
            logger.info(f"Using custom bounding box: {bbox}")
        else:
            logger.info("Using example area (San Francisco Bay Area)")
        
        logger.info(f"Assessment period: {start_date.date()} to {end_date.date()}")
        
        # Run assessment
        logger.info("Starting robustness assessment...")
        results = assessor.run_robustness_assessment(bbox, start_date, end_date)
        
        if results['success']:
            if args.brief:
                print_brief_summary(results)
            else:
                logger.info(f"Assessment completed successfully!")
                logger.info(f"Optimal scenes recommendation: {results['optimal_scenes']}")
                logger.info(f"Total scenes analyzed: {results['total_scenes_loaded']}")
                logger.info(f"Assessment area: {results['bbox']}")
                logger.info(f"Time period: {results['time_period']}")
            
            # Save detailed results if requested
            if args.save_results:
                save_detailed_results(results, args.save_results)
                logger.info(f"Detailed results saved to: {args.save_results}")
            
            return 0
            
        else:
            if args.brief:
                print_brief_summary(results)
            else:
                logger.error(f"Assessment failed: {results.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        logger.error(f"Robustness assessment failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())