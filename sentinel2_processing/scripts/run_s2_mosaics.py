#!/usr/bin/env python3
"""
Sentinel-2 Mosaic Processing Script

Command-line interface for the main Sentinel-2 mosaic processing pipeline.
Creates summer mosaics over Spain using distributed computing with STAC catalog
integration and optimized memory management.

Usage Examples:
    # Run with default configuration
    python scripts/run_mosaic_processing.py
    
    # Run with custom configuration
    python scripts/run_mosaic_processing.py --config custom_config.yaml
    
    # Process specific years only
    python scripts/run_mosaic_processing.py --years 2021 2022
    
    # Use different tile size
    python scripts/run_mosaic_processing.py --tile-size 6144
    
    # Enable debug logging and custom output
    python scripts/run_mosaic_processing.py --log-level DEBUG --output-dir /custom/output
    
    # Validate configuration only
    python scripts/run_mosaic_processing.py --validate-only
    
    # Show processing summary
    python scripts/run_mosaic_processing.py --summary

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.mosaic_processing import MosaicProcessingPipeline


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Mosaic Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default config
  %(prog)s --config custom.yaml              # Use custom configuration
  %(prog)s --years 2021 2022                 # Process specific years
  %(prog)s --tile-size 6144                  # Use different tile size
  %(prog)s --log-level DEBUG                 # Enable debug logging
  %(prog)s --validate-only                   # Validate inputs only
  %(prog)s --summary                         # Show processing summary

For more information, see the component documentation.
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Years to process (overrides config)'
    )
    
    parser.add_argument(
        '--tile-size',
        type=int,
        help='Processing tile size (overrides config)'
    )
    
    parser.add_argument(
        '--n-scenes',
        type=int,
        help='Number of best scenes to select (overrides config)'
    )
    
    # Directory options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--spain-polygon',
        type=str,
        help='Path to Spain polygon shapefile (overrides config)'
    )
    
    # Compute options
    parser.add_argument(
        '--n-workers',
        type=int,
        help='Number of dask workers (overrides config)'
    )
    
    parser.add_argument(
        '--memory-per-worker',
        type=str,
        help='Memory limit per worker (overrides config)'
    )
    
    # Execution options
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and inputs without processing'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show processing summary and exit'
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
    
    if args.output_dir and not Path(args.output_dir).parent.exists():
        print(f"Error: Output directory parent does not exist: {args.output_dir}")
        return False
    
    if args.spain_polygon and not Path(args.spain_polygon).exists():
        print(f"Error: Spain polygon file does not exist: {args.spain_polygon}")
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
    if args.years:
        config.setdefault('processing', {})['years'] = args.years
    
    if args.tile_size:
        config.setdefault('processing', {})['tile_size'] = args.tile_size
    
    if args.n_scenes:
        config.setdefault('processing', {})['n_scenes'] = args.n_scenes
    
    if args.output_dir:
        config.setdefault('paths', {})['output_dir'] = args.output_dir
    
    if args.spain_polygon:
        config.setdefault('paths', {})['spain_polygon'] = args.spain_polygon
    
    if args.n_workers:
        config.setdefault('compute', {})['n_workers'] = args.n_workers
    
    if args.memory_per_worker:
        config.setdefault('compute', {})['memory_per_worker'] = args.memory_per_worker
    
    return config


def print_processing_summary(pipeline: MosaicProcessingPipeline) -> None:
    """
    Print processing summary information.
    
    Args:
        pipeline: Initialized pipeline instance
    """
    summary = pipeline.get_processing_summary()
    
    print("\n" + "="*60)
    print("SENTINEL-2 MOSAIC PROCESSING SUMMARY")
    print("="*60)
    print(f"Processing Years:     {summary['config_summary']['years']}")
    print(f"Scenes per Mosaic:    {summary['config_summary']['n_scenes']}")
    print(f"Tile Size:            {summary['config_summary']['tile_size']}")
    print(f"Total Combinations:   {summary['total_combinations']}")
    print(f"Dask Workers:         {summary['config_summary']['n_workers']}")
    print("="*60)


def main() -> int:
    """
    Main entry point for mosaic processing script.
    
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
        
        # Initialize pipeline
        pipeline = MosaicProcessingPipeline()
        pipeline.config = config
        
        # Show summary if requested
        if args.summary:
            pipeline.create_processing_plan()
            print_processing_summary(pipeline)
            return 0
        
        # Validate configuration
        if not pipeline.validate_configuration():
            logger.error("Configuration validation failed")
            return 1
        
        # Validate only if requested
        if args.validate_only:
            logger.info("Configuration validation successful")
            return 0
        
        # Run processing
        logger.info("Starting Sentinel-2 mosaic processing...")
        start_time = time.time()
        
        results = pipeline.run_processing()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Total Duration: {duration/60:.1f} minutes")
        logger.info(f"Processed: {results['processed_count']}")
        logger.info(f"Skipped: {results['skipped_count']}")
        logger.info(f"Errors: {results['error_count']}")
        logger.info(f"Success Rate: {results['success_rate']:.1f}%")
        logger.info("="*60)
        
        return 0 if results['error_count'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())