#!/usr/bin/env python3
"""
Biomass Estimation Pipeline Script

Main entry point for running the biomass estimation pipeline with
Monte Carlo uncertainty quantification.

Usage:
    python run_biomass_estimation.py [OPTIONS]
    
Examples:
    # Run full pipeline with default config
    python run_biomass_estimation.py
    
    # Run with custom config
    python run_biomass_estimation.py --config custom_config.yaml
    
    # Run specific years only
    python run_biomass_estimation.py --years 2020 2021 2022
    
    # Run single forest type for testing
    python run_biomass_estimation.py --test-mode --year 2020 --forest-type 12

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from biomass_model.core.biomass_estimation import BiomassEstimationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Biomass Estimation Pipeline with Monte Carlo Uncertainty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full pipeline
  %(prog)s --config custom_config.yaml       # Custom configuration  
  %(prog)s --years 2020 2021                 # Specific years only
  %(prog)s --test-mode --year 2020 --forest-type 12  # Test single forest type
  %(prog)s --validate-only                   # Validate inputs only
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: use component config)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process (default: all years in config)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (single forest type)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='Year for test mode'
    )
    
    parser.add_argument(
        '--forest-type',
        type=str,
        help='Forest type code for test mode'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without processing'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-dask',
        action='store_true',
        help='Disable Dask distributed processing'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    if args.test_mode:
        if not args.year or not args.forest_type:
            print("Error: --test-mode requires --year and --forest-type")
            return False
    
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    return True


def run_full_pipeline(pipeline: BiomassEstimationPipeline, years: Optional[List[int]]) -> bool:
    """
    Run the complete biomass estimation pipeline.
    
    Args:
        pipeline: Pipeline instance
        years: Years to process
        
    Returns:
        bool: True if successful
    """
    return pipeline.run_pipeline(years=years)


def run_test_mode(
    pipeline: BiomassEstimationPipeline, 
    year: int, 
    forest_type_code: str,
    output_dir: Optional[str]
) -> bool:
    """
    Run pipeline in test mode for single forest type.
    
    Args:
        pipeline: Pipeline instance
        year: Year to process
        forest_type_code: Forest type code
        output_dir: Optional custom output directory
        
    Returns:
        bool: True if successful
    """
    return pipeline.run_single_forest_type(
        year=year,
        forest_type_code=forest_type_code,
        output_dir=output_dir
    )


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        component_name='biomass_estimation',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Biomass Estimation Pipeline...")
        pipeline = BiomassEstimationPipeline(config_path=args.config)
        
        # Log pipeline start
        log_pipeline_start(logger, "Biomass Estimation", pipeline.config)
        
        # Validation-only mode
        if args.validate_only:
            logger.info("Running input validation...")
            if pipeline.validate_inputs():
                logger.info("✅ Input validation successful")
                sys.exit(0)
            else:
                logger.error("❌ Input validation failed")
                sys.exit(1)
        
        # Disable Dask if requested
        if args.no_dask:
            logger.info("Dask disabled by user request")
            pipeline.client = None
        
        # Run pipeline
        success = False
        
        if args.test_mode:
            logger.info(f"Running test mode: year {args.year}, forest type {args.forest_type}")
            success = run_test_mode(
                pipeline=pipeline,
                year=args.year,
                forest_type_code=args.forest_type,
                output_dir=args.output_dir
            )
        else:
            logger.info("Running full biomass estimation pipeline...")
            success = run_full_pipeline(
                pipeline=pipeline,
                years=args.years
            )
        
        # Pipeline completion
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Biomass Estimation", success, elapsed_time)
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
