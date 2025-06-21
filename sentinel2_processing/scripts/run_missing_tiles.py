#!/usr/bin/env python3
"""
Missing Tiles Analysis Script

Command-line interface for detecting and analyzing missing tile-year combinations.
Provides comprehensive quality assurance through gap detection, completeness
statistics, and detailed reporting for reprocessing needs.

Usage Examples:
    # Analyze default output directory
    python scripts/run_missing_analysis.py
    
    # Analyze custom directory
    python scripts/run_missing_analysis.py --directory /custom/output
    
    # Save missing paths to file
    python scripts/run_missing_analysis.py --save-paths missing_tiles.txt
    
    # Use custom configuration
    python scripts/run_missing_analysis.py --config custom.yaml
    
    # Generate detailed report
    python scripts/run_missing_analysis.py --detailed-report

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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Missing Tiles Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Analyze default output directory
  %(prog)s --directory /custom/output         # Analyze custom directory
  %(prog)s --save-paths missing_tiles.txt     # Save missing paths to file
  %(prog)s --config custom.yaml              # Use custom configuration
  %(prog)s --detailed-report                 # Generate detailed report

For more information, see the component documentation.
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Directory options
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory to analyze for missing files (overrides config)'
    )
    
    # Output options
    parser.add_argument(
        '--save-paths',
        type=str,
        help='File to save missing file paths (default: missing_file_paths.txt)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory for analysis reports (default: same as analyzed directory)'
    )
    
    # Analysis options
    parser.add_argument(
        '--detailed-report',
        action='store_true',
        help='Generate detailed analysis report'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary statistics'
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
    
    if args.directory and not Path(args.directory).exists():
        print(f"Error: Analysis directory does not exist: {args.directory}")
        return False
    
    if args.save_paths and not Path(args.save_paths).parent.exists():
        print(f"Error: Output file directory does not exist: {Path(args.save_paths).parent}")
        return False
    
    if args.output_dir and not Path(args.output_dir).exists():
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            print(f"Error: Cannot create output directory: {args.output_dir}")
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
    if args.directory:
        config.setdefault('paths', {})['output_dir'] = args.directory
    
    return config


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
        setup_logging(level=args.log_level, log_file=args.log_file)
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        config = apply_config_overrides(config, args)
        
        # Initialize analyzer
        analyzer = MissingTilesAnalyzer(config)
        
        # Run analysis
        logger.info("Starting missing tiles analysis...")
        results = analyzer.run_missing_analysis()
        
        if results['success']:
            # Print summary statistics
            if not args.summary_only:
                print_summary_statistics(results)
            else:
                # Brief summary only
                stats = results['statistics']
                print(f"Completeness: {stats['completeness_rate']:.1f}% ({stats['total_existing']}/{stats['total_expected']} files)")
            
            # Save missing paths if requested
            save_file = args.save_paths or "missing_file_paths.txt"
            if results['missing_file_paths'] and not args.summary_only:
                analyzer.save_missing_file_paths(results['missing_file_paths'], save_file)
                logger.info(f"Missing file paths saved to: {save_file}")
            
            # Generate detailed report if requested
            if args.detailed_report:
                logger.info("Detailed analysis report generated in log output above")
            
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