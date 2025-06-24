#!/usr/bin/env python3
"""
Missing Tiles Analysis Script

Command-line interface for detecting and analyzing missing tile-year combinations.
Provides comprehensive quality assurance through gap detection, completeness
statistics, and detailed reporting for reprocessing needs.

Usage Examples:
    # Analyze default output directory
    python scripts/run_missing_analysis.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.postprocessing import MissingTilesAnalyzer
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Missing Tiles Analysis",
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

def print_summary_statistics(results: dict) -> None:
    """
    Print summary statistics from analysis results.
    
    Args:
        results: Analysis results dictionary
    """
    if not results['success']:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return
    
    stats = results['statistics']
    
    print("\n" + "="*60)
    print("MISSING TILES ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Locations:      {stats['total_locations']}")
    print(f"Years Found:          {', '.join(results['all_years'])}")
    print(f"Expected Files:       {stats['total_expected']}")
    print(f"Existing Files:       {stats['total_existing']}")
    print(f"Missing Files:        {stats['total_missing']}")
    print(f"Completeness Rate:    {stats['completeness_rate']:.1f}%")
    print("="*60)


def main() -> int:
    """
    Main entry point for missing analysis script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Setup logging
        setup_logging(level='INFO', log_file='sentinel2_missing_files')
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        
        # Initialize analyzer
        analyzer = MissingTilesAnalyzer(config)
        
        # Run analysis
        logger.info("Starting missing tiles analysis...")
        results = analyzer.run_missing_analysis()
        
        if results['success']:

            # Brief summary only
            stats = results['statistics']
            print(f"Completeness: {stats['completeness_rate']:.1f}% ({stats['total_existing']}/{stats['total_expected']} files)")
            
            # Save missing paths if requested
            save_file = SENTINEL2_PROCESSED_DIR / "missing_file_paths.txt"
            if results['missing_file_paths']:
                analyzer.save_missing_file_paths(results['missing_file_paths'], save_file)
                logger.info(f"Missing file paths saved to: {save_file}")
            
            logger.info("Missing tiles analysis completed successfully!")
            
            # Return appropriate exit code based on completeness
            completeness = results['statistics']['completeness_rate']
            if completeness < 95.0:  # Warning threshold
                logger.warning(f"Low completeness rate: {completeness:.1f}%")
                return 1 if completeness < 80.0 else 0  # Error if very low
            
            return 0
            
        else:
            logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        logger.error(f"Missing analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())