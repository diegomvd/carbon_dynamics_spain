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
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Spanish National Forest Inventory (NFI) Biomass Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: uses component default)'
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
    
    # Check that config file exists if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file does not exist: {config_path}")
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
            level='INFO',
            component_name='forest_inventory',
        )
        
        logger.info("Starting NFI processing script")
        
        # Load configuration
        config = load_config(args.config, component_name="forest_inventory")
        
        # Initialize pipeline
        logger.info("Initializing NFI processing pipeline")
        pipeline = NFIProcessingPipeline()
        pipeline.config = config  
        
        # Update pipeline attributes from config
        pipeline.data_dir = NFI4_DATABASE_DIR
        pipeline.output_dir = FOREST_INVENTORY_PROCESSED_DIR
        pipeline.temp_dir = FOREST_INVENTORY_PROCESSED_DIR / "tmp"
        pipeline.target_crs = config['output']['target_crs']
        
        
        # Validate inputs
        logger.info("Validating input files and directories")
        validation_result = pipeline.validate_inputs()
        
        if not validation_result:
            logger.error("Input validation failed")
            return 1
        
        # Run the processing pipeline
        start_time = time.time()
        success = pipeline.run_full_pipeline()
        elapsed_time = time.time() - start_time
        
        if success:
            logger.info(f"NFI processing completed successfully in {elapsed_time:.2f} seconds")
            print(f"\nProcessing completed successfully!")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"Results saved to: {pipeline.output_dir}")
            return 0
        else:
            logger.error("NFI processing failed")
            print("\nProcessing failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\nProcessing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
