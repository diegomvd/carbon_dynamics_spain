#!/usr/bin/env python3
"""
Sentinel-2 Downsampling and Merging Script

Command-line interface for spatial downsampling and merging operations.
Converts raw mosaic tiles into analysis-ready products through sequential
downsampling and yearly merging operations.

Usage Examples:
    # Run complete workflow
    python scripts/run_downsampling.py
    
    # Use custom scale factor
    python scripts/run_downsampling.py --scale-factor 5
    
    # Only run downsampling step
    python scripts/run_downsampling.py --downsample-only
    
    # Only run merging step
    python scripts/run_downsampling.py --merge-only
    
    # Use custom configuration
    python scripts/run_downsampling.py --config custom.yaml
    
    # Override directories
    python scripts/run_downsampling.py --input-dir /custom/input --output-dir /custom/output

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.postprocessing import DownsamplingMergingProcessor


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Downsampling and Merging Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run complete workflow
  %(prog)s --scale-factor 5                  # Use 5x downsampling
  %(prog)s --downsample-only                 # Only downsample
  %(prog)s --merge-only                      # Only merge
  %(prog)s --config custom.yaml              # Custom configuration
  %(prog)s --input-dir /input --output-dir /out # Override directories

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
        '--input-dir',
        type=str,
        help='Input directory containing raw mosaics (overrides config)'
    )
    
    parser.add_argument(
        '--downsampled-dir',
        type=str,
        help='Output directory for downsampled files (overrides config)'
    )
    
    parser.add_argument(
        '--merged-dir',
        type=str,
        help='Output directory for merged files (overrides config)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--scale-factor',
        type=int,
        help='Downsampling scale factor (overrides config)'
    )
    
    # Execution options
    parser.add_argument(
        '--downsample-only',
        action='store_true',
        help='Only run downsampling step'
    )
    
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only run merging step'
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
    
    if args.input_dir and not Path(args.input_dir).exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return False
    
    if args.downsample_only and args.merge_only:
        print("Error: Cannot specify both --downsample-only and --merge-only")
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
    if args.input_dir:
        config.setdefault('paths', {})['output_dir'] = args.input_dir
    
    if args.downsampled_dir:
        config.setdefault('paths', {})['downsampled_dir'] = args.downsampled_dir
    
    if args.merged_dir:
        config.setdefault('paths', {})['merged_dir'] = args.merged_dir
    
    if args.scale_factor:
        config.setdefault('postprocessing', {}).setdefault('downsample', {})['scale_factor'] = args.scale_factor
    
    return config


def main() -> int:
    """
    Main entry point for downsampling script.
    
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
        
        # Initialize processor
        processor = DownsamplingMergingProcessor(config)
        
        if args.downsample_only:
            logger.info("Running downsampling workflow only...")
            stats = processor.downsample_rasters(
                config['paths']['output_dir'],
                config['paths']['downsampled_dir']
            )
            logger.info(f"Downsampling completed: {stats['successful']} successful, {stats['failed']} failed")
            
        elif args.merge_only:
            logger.info("Running merging workflow only...")
            stats = processor.merge_rasters_by_year(
                config['paths']['downsampled_dir'],
                config['paths']['merged_dir']
            )
            logger.info(f"Merging completed: {stats['successful']} successful, {stats['failed']} failed")
            
        else:
            logger.info("Running complete downsampling and merging workflow...")
            results = processor.run_complete_workflow()
            
            if results['success']:
                logger.info("Complete workflow successful!")
                logger.info(f"Downsampling: {results['downsampling']['successful']} successful, {results['downsampling']['failed']} failed")
                logger.info(f"Merging: {results['merging']['successful']} successful, {results['merging']['failed']} failed")
            else:
                logger.error(f"Workflow failed: {results.get('error', 'Unknown error')}")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Downsampling pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())