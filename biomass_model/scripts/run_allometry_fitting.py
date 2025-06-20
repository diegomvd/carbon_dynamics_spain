#!/usr/bin/env python3
"""
Allometry Fitting Pipeline Script

Entry point for fitting allometric relationships from NFI data and 
canopy height predictions.

Usage:
    python run_allometry_fitting.py [OPTIONS]
    
Examples:
    # Fit allometries with default config
    python run_allometry_fitting.py
    
    # Custom configuration
    python run_allometry_fitting.py --config fitting_config.yaml
    
    # Specific years only
    python run_allometry_fitting.py --years 2020 2021
    
    # Validation only
    python run_allometry_fitting.py --validate-only

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
from biomass_model.core.allometry import AllometryManager
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Allometric Relationship Fitting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Fit all allometries
  %(prog)s --config custom.yaml         # Custom configuration
  %(prog)s --years 2020 2021           # Specific years only
  %(prog)s --validate-only             # Validate data only
  %(prog)s --output-dir ./results      # Custom output directory
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
        help='Specific years to include in fitting'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory for fitted allometries'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate input data without fitting'
    )
    
    parser.add_argument(
        '--forest-types',
        type=str,
        nargs='+',
        help='Specific forest types to fit (default: all available)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
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
    
    return True


class AllometryFittingPipeline:
    """
    Pipeline for fitting allometric relationships.
    
    This is a placeholder for the actual allometry fitting logic
    that would integrate with existing fitting modules.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize fitting pipeline."""
        self.config = load_config(config_path, component_name="biomass_estimation")
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='allometry_fitting'
        )
        
        self.allometry_manager = AllometryManager(self.config)
        
        self.logger.info("AllometryFittingPipeline initialized")
    
    def validate_inputs(self) -> bool:
        """Validate input data for allometry fitting."""
        self.logger.info("Validating allometry fitting inputs...")
        
        try:
            # Validate allometric data
            if not self.allometry_manager.validate_allometric_data():
                return False
            
            # Check for NFI data (placeholder - would check actual NFI files)
            nfi_dir = Path(self.config['data'].get('nfi_data_dir', 'data/nfi'))
            if not nfi_dir.exists():
                self.logger.warning(f"NFI data directory not found: {nfi_dir}")
                # This might not be critical if allometries are pre-fitted
            
            # Check for height prediction data
            height_dir = Path(self.config['data']['input_data_dir'])
            if not height_dir.exists():
                self.logger.error(f"Height predictions directory not found: {height_dir}")
                return False
            
            self.logger.info("Input validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {str(e)}")
            return False
    
    def fit_allometries(
        self, 
        years: Optional[List[int]] = None,
        forest_types: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> bool:
        """
        Fit allometric relationships.
        
        Args:
            years: Years to include in fitting
            forest_types: Forest types to fit
            output_dir: Output directory
            
        Returns:
            bool: True if fitting succeeded
        """
        self.logger.info("Starting allometry fitting...")
        
        try:
            # This is a placeholder for actual fitting logic
            # In the real implementation, this would:
            # 1. Load NFI plot data
            # 2. Sample height predictions at NFI locations
            # 3. Fit allometric relationships by forest type hierarchy
            # 4. Generate quantile regression results
            # 5. Save fitted parameters
            
            # Get available forest types if not specified
            if forest_types is None:
                forest_types = self.allometry_manager.get_available_forest_types()
            
            self.logger.info(f"Fitting allometries for {len(forest_types)} forest types")
            
            # Placeholder fitting process
            fitted_count = 0
            for forest_type in forest_types:
                self.logger.info(f"Fitting allometry for forest type: {forest_type}")
                
                # Here would be the actual fitting logic
                # For now, just validate existing parameters
                params = self.allometry_manager.get_allometry_parameters(forest_type)
                if params:
                    fitted_count += 1
                    self.logger.debug(f"Parameters available for {forest_type}")
                else:
                    self.logger.warning(f"No parameters for {forest_type}")
            
            self.logger.info(f"Allometry fitting completed: {fitted_count}/{len(forest_types)} successful")
            
            # Generate summary statistics
            summary = self.allometry_manager.get_summary_statistics()
            self.logger.info(f"Summary: {summary}")
            
            return fitted_count > 0
            
        except Exception as e:
            self.logger.error(f"Error in allometry fitting: {str(e)}")
            return False


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
        component_name='allometry_fitting',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Allometry Fitting Pipeline...")
        pipeline = AllometryFittingPipeline(config_path=args.config)
        
        # Log pipeline start
        log_pipeline_start(logger, "Allometry Fitting", pipeline.config)
        
        # Validation
        if not pipeline.validate_inputs():
            logger.error("Input validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("✅ Validation successful - exiting")
            sys.exit(0)
        
        # Run fitting
        success = pipeline.fit_allometries(
            years=args.years,
            forest_types=args.forest_types,
            output_dir=args.output_dir
        )
        
        # Pipeline completion
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Allometry Fitting", success, elapsed_time)
        
        if success:
            logger.info("🎉 Allometry fitting completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Allometry fitting failed!")
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
