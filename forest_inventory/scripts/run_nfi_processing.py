#!/usr/bin/env python3
"""
NFI Processing Script

Command-line interface for Spanish National Forest Inventory (NFI) biomass processing.
Provides configurable access to the main processing pipeline with validation and logging.

Usage Examples:
    # Run with default configuration
    python scripts/run_nfi_processing.py
    
    # Run with custom configuration
    python scripts/run_nfi_processing.py --config custom_config.yaml
    
    # Run with specific data and output directories
    python scripts/run_nfi_processing.py --data-dir /path/to/data --output-dir /path/to/output
    
    # Enable debug logging
    python scripts/run_nfi_processing.py --log-level DEBUG
    
    # Validate inputs only (no processing)
    python scripts/run_nfi_processing.py --validate-only
    
    # Show processing summary
    python scripts/run_nfi_processing.py --summary

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
from forest_inventory.core.nfi_processing import NFIProcessingPipeline


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Spanish National Forest Inventory (NFI) Biomass Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default config
  %(prog)s --config custom.yaml              # Use custom configuration
  %(prog)s --data-dir /data --output-dir /out # Override directories
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
        default=None,
        help='Path to configuration file (default: uses component default)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: console only)'
    )
    
    # Processing options
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate inputs only, do not run processing'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show processing summary and exit'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force processing even if validation warnings exist'
    )
    
    # UTM zone filtering (advanced option)
    parser.add_argument(
        '--utm-zones',
        nargs='+',
        type=int,
        choices=[29, 30, 31],
        default=None,
        help='Process only specific UTM zones (default: all valid zones)'
    )
    
    return parser.parse_args()


def override_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Override configuration values from command-line arguments.
    
    Args:
        config: Configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        dict: Updated configuration
    """    
    # Override logging settings
    config['logging']['level'] = args.log_level
    if args.log_file:
        config['logging']['log_file'] = str(Path(args.log_file).resolve())
    
    # Override UTM zones if specified
    if args.utm_zones:
        config['processing']['valid_utm_zones'] = args.utm_zones
    
    return config


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    # Check that directories exist if specified
    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"Error: Data directory does not exist: {data_path}")
            return False
        if not data_path.is_dir():
            print(f"Error: Data path is not a directory: {data_path}")
            return False
    
    # Check that config file exists if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file does not exist: {config_path}")
            return False
    
    # Check that log file directory exists if specified
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.parent.exists():
            print(f"Error: Log file directory does not exist: {log_path.parent}")
            return False
    
    return True


def print_processing_summary(pipeline: NFIProcessingPipeline) -> None:
    """
    Print processing summary information.
    
    Args:
        pipeline: Initialized pipeline instance
    """
    summary = pipeline.get_processing_summary()
    
    print("\n" + "="*60)
    print("NFI PROCESSING PIPELINE SUMMARY")
    print("="*60)
    print(f"Data Directory:    {summary['data_directory']}")
    print(f"Output Directory:  {summary['output_directory']}")
    print(f"Target CRS:        {summary['target_crs']}")
    print(f"UTM Zones:         {summary['valid_utm_zones']}")
    print(f"Temp Directory:    {summary['temp_directory']}")
    print(f"Component Version: {summary['component_version']}")
    print("="*60)


def main() -> int:
    """
    Main entry point for NFI processing script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Initialize logging
        logger = setup_logging(
            level=args.log_level,
            component_name='forest_inventory',
            log_file=args.log_file
        )
        
        logger.info("Starting NFI processing script")
        logger.info(f"Command-line arguments: {vars(args)}")
        
        # Load configuration
        config = load_config(args.config, component_name="forest_inventory")
        
        # Override configuration from command-line arguments
        config = override_config_from_args(config, args)
        
        # Initialize pipeline
        logger.info("Initializing NFI processing pipeline")
        pipeline = NFIProcessingPipeline()
        pipeline.config = config  # Use the overridden config
        
        # Update pipeline attributes from config
        pipeline.data_dir = NFI4_DATABASE_DIR
        pipeline.output_dir = FOREST_INVENTORY_PROCESSED_DIR
        pipeline.temp_dir = FOREST_INVENTORY_PROCESSED_DIR / "tmp"
        pipeline.target_crs = config['output']['target_crs']
        
        # Show summary if requested
        if args.summary:
            print_processing_summary(pipeline)
            return 0
        
        # Validate inputs
        logger.info("Validating input files and directories")
        validation_result = pipeline.validate_inputs()
        
        if not validation_result:
            logger.error("Input validation failed")
            if not args.force:
                logger.error("Use --force to proceed despite validation errors")
                return 1
            else:
                logger.warning("Continuing despite validation errors (--force specified)")
        
        # Exit if validation-only mode
        if args.validate_only:
            if validation_result:
                logger.info("Input validation completed successfully")
                print("✅ All inputs validated successfully")
            else:
                logger.error("Input validation failed")
                print("❌ Input validation failed")
            return 0 if validation_result else 1
        
        # Run the processing pipeline
        start_time = time.time()
        success = pipeline.run_full_pipeline()
        elapsed_time = time.time() - start_time
        
        if success:
            logger.info(f"NFI processing completed successfully in {elapsed_time:.2f} seconds")
            print(f"\n✅ Processing completed successfully!")
            print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
            print(f"📁 Results saved to: {pipeline.output_dir}")
            return 0
        else:
            logger.error("NFI processing failed")
            print("\n❌ Processing failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\n⚠️  Processing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
