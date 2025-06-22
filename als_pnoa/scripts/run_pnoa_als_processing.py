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
    
    # Specific years only
    python run_pnoa_processing.py --years 2020 2021
    
    # Validate inputs only
    python run_pnoa_processing.py --validate-only

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
from als_pnoa.core.pnoa_processor import PNOAProcessor
from shared_utils import setup_logging, load_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PNOA LiDAR Processing for Training Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Process all years with default config
  %(prog)s --years 2020 2021 2022       # Process specific years only
  %(prog)s --config custom.yaml         # Use custom configuration
  %(prog)s --validate-only              # Validate inputs without processing
  %(prog)s --summary                    # Show processing summary
  %(prog)s --output-dir ./processed     # Custom output directory

This script processes PNOA LiDAR tiles by:
1. Finding tiles that intersect with Sentinel-2 mosaics
2. Filtering by target years
3. Standardizing naming and format for ML training
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process (default: all configured years)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory for processed tiles'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate input data without processing'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show processing summary without executing'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    if args.output_dir:
        output_path = Path(args.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory {args.output_dir}: {e}")
            return False
    
    if args.years:
        for year in args.years:
            if year < 2010 or year > 2030:
                print(f"Warning: Year {year} seems outside reasonable range")
    
    return True


def print_processing_summary(processor: PNOAProcessor) -> None:
    """Print a summary of what will be processed."""
    print(f"\n{'='*60}")
    print(f"PNOA PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    summary = processor.get_processing_summary()
    
    print(f"Target years: {summary['target_years']}")
    print(f"Coverage areas: {summary['coverage_areas']}")
    print(f"Total PNOA files available: {summary['total_pnoa_files']}")
    
    print(f"\nSentinel-2 tiles by year:")
    for year, count in summary['sentinel_tiles_by_year'].items():
        print(f"  {year}: {count} tiles")
    
    print(f"\nConfiguration:")
    print(f"  Sentinel-2 path: {processor.sentinel_path}")
    print(f"  PNOA data path: {processor.pnoa_data_dir}")
    print(f"  Output path: {processor.target_output_dir}")
    print(f"  Target CRS: {processor.target_crs}")


def print_results_summary(results: dict, execution_time: float) -> None:
    """Print summary of processing results."""
    print(f"\n{'='*60}")
    print(f"PNOA PROCESSING RESULTS")
    print(f"{'='*60}")
    
    print(f"Execution time: {execution_time/60:.2f} minutes")
    print(f"Total tiles selected: {results['total_selected']}")
    print(f"Successfully processed: {results['successfully_processed']}")
    print(f"Errors: {results['errors']}")
    
    if results['errors'] > 0:
        error_rate = (results['errors'] / results['total_selected']) * 100
        print(f"Error rate: {error_rate:.1f}%")
    
    print(f"\nOutput directory: {results['output_directory']}")
    
    # Show some example output files
    output_dir = Path(results['output_directory'])
    if output_dir.exists():
        output_files = list(output_dir.glob("PNOA_*.tif"))
        print(f"Output files created: {len(output_files)}")
        
        if output_files:
            print(f"\nExample output files:")
            for i, file in enumerate(output_files[:3]):
                print(f"  {file.name}")
            if len(output_files) > 3:
                print(f"  ... and {len(output_files) - 3} more")


def main():
    """Main entry point for PNOA processing script."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Set logging level
    log_level = 'ERROR' if args.quiet else args.log_level
    
    # Setup logging
    logger = setup_logging(
        level=log_level,
        component_name='pnoa_processing'
    )
    
    try:
        # Load configuration
        config = load_config(args.config, component_name="als_pnoa")
        
        # Apply command line overrides
        if args.years:
            config['processing']['target_years'] = args.years
        
        if args.output_dir:
            config['paths']['output_dir'] = args.output_dir
        
        # Initialize processor
        processor = PNOAProcessor(config)
        
        logger.info("Starting PNOA LiDAR processing pipeline...")
        
        # Validate inputs
        if not processor.validate_inputs():
            logger.error("Input validation failed")
            sys.exit(1)
        
        # Show summary if requested
        if args.summary:
            print_processing_summary(processor)
            return
        
        # Exit if validation only
        if args.validate_only:
            logger.info("✅ Input validation passed - exiting")
            return
        
        # Process tiles
        logger.info("Processing PNOA tiles...")
        results = processor.process_all_tiles()
        
        # Print results
        execution_time = time.time() - start_time
        
        if not args.quiet:
            print_results_summary(results, execution_time)
        
        # Determine success
        if results['errors'] == 0:
            logger.info("🎉 PNOA processing completed successfully!")
        else:
            error_rate = (results['errors'] / results['total_selected']) * 100
            if error_rate > 10:  # More than 10% errors
                logger.error(f"❌ High error rate: {error_rate:.1f}%")
                sys.exit(1)
            else:
                logger.warning(f"⚠️  Processing completed with some errors: {error_rate:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Processing failed with error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()