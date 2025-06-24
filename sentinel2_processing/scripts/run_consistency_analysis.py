#!/usr/bin/env python3
"""
Interannual Consistency Analysis Script

Command-line interface for analyzing spectral consistency across years
to detect processing artifacts and temporal stability issues. Performs
statistical tests and generates comprehensive visualizations and reports.

Usage Examples:
    # Analyze merged mosaics in default directory
    python scripts/run_consistency_analysis.py
    
    # Use custom input directory
    python scripts/run_consistency_analysis.py --input-dir /custom/merged
    
    # Use smaller sample size for KS tests
    python scripts/run_consistency_analysis.py --sample-size 5000
    
    # Save results to custom directory
    python scripts/run_consistency_analysis.py --output-dir /custom/results
    
    # Use custom configuration
    python scripts/run_consistency_analysis.py --config custom.yaml
    
    # Quick analysis with minimal output
    python scripts/run_consistency_analysis.py --summary-only

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.postprocessing import InterannualConsistencyAnalyzer


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Interannual Consistency Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
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


def filter_analysis_files(input_dir: str, args: argparse.Namespace) -> list:
    """
    Filter analysis files based on year criteria.
    
    Args:
        input_dir: Input directory path
        args: Parsed arguments
        
    Returns:
        list: Filtered list of files to analyze
    """
    import os
    import re
    
    # Get all merged files
    pattern = re.compile(r'S2_summer_mosaic_(\d{4})_merged\.tif$', re.IGNORECASE)
    all_files = []
    
    for file in os.listdir(input_dir):
        match = pattern.match(file)
        if match:
            year = int(match.group(1))
            file_path = os.path.join(input_dir, file)
            all_files.append((year, file_path))
    
    # Apply year filtering
    if args.years:
        all_files = [(year, path) for year, path in all_files if year in args.years]
    
    if args.exclude_years:
        all_files = [(year, path) for year, path in all_files if year not in args.exclude_years]
    
    return [path for year, path in sorted(all_files)]


def print_analysis_summary(results: dict) -> None:
    """
    Print analysis summary information.
    
    Args:
        results: Analysis results dictionary
    """
    if results['success']:
        print("\n" + "="*60)
        print("INTERANNUAL CONSISTENCY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Years Analyzed:       {', '.join(results['years_analyzed'])}")
        print(f"Bands Analyzed:       {len(results['bands_analyzed'])}")
        print(f"Statistical Tests:    {results['n_statistical_tests']}")
        print(f"Results Directory:    {results['output_directory']}")
        print("="*60)
    else:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")


def main() -> int:
    """
    Main entry point for consistency analysis script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Setup logging
        setup_logging(level='INFO', log_file='sentinel2_interannual_consistency')
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        
        # Initialize analyzer
        analyzer = InterannualConsistencyAnalyzer(config)
        
        # Determine input directory
        input_dir = SENTINEL2_MERGED_DIR
        
        # Filter files if year criteria specified
        if args.years or args.exclude_years:
            filtered_files = filter_analysis_files(input_dir, args)
            if not filtered_files:
                logger.error("No files found matching year criteria")
                return 1
            logger.info(f"Analyzing {len(filtered_files)} files based on year criteria")
        
        # Run analysis
        logger.info("Starting interannual consistency analysis...")
        
        if args.summary_only:
            logger.info("Running summary-only analysis...")
        
        if args.no_plots:
            logger.info("Skipping plot generation...")
        
        results = analyzer.run_consistency_analysis()
        
        if results['success']:
            if args.summary_only:
                print_analysis_summary(results)
            else:
                logger.info(f"Analysis completed successfully!")
                logger.info(f"Years analyzed: {', '.join(results['years_analyzed'])}")
                logger.info(f"Bands analyzed: {len(results['bands_analyzed'])}")
                logger.info(f"Statistical tests performed: {results['n_statistical_tests']}")
                logger.info(f"Results saved to: {results['output_directory']}")
            
            return 0
            
        else:
            logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        logger.error(f"Consistency analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())