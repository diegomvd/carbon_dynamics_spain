#!/usr/bin/env python3
"""
Allometry Fitting Pipeline Script

Entry point for fitting allometric relationships from NFI data and 
canopy height predictions. Uses complete implementation with quantile
regression and hierarchical forest type processing.

Features:
- Dynamic training dataset creation from NFI data + height maps
- Height-AGB allometry fitting using quantile regression  
- BGB-AGB ratio fitting using hierarchical percentile calculation
- Hierarchical forest type processing (General → Clade → Family → Genus → ForestType)
- Compatible outputs for existing biomass estimation pipeline
- Integration with harmonized data paths

Usage:
    python run_allometry_fitting.py 
    
Examples:
    # Fit allometries with default config
    python run_allometry_fitting.py
    
    # Custom configuration
    python run_allometry_fitting.py --config fitting_config.yaml

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Shared utilities
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import *

# Component imports
from biomass_model.core.allometry_fitting import run_allometry_fitting_pipeline, save_allometry_results
from biomass_model.core.allometry import AllometryManager


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Allometric Relationship Fitting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Validate config file if provided
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    return True


def create_fitting_config(args: argparse.Namespace) -> dict:
    """
    Create fitting configuration from arguments and default values.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        dict: Configuration dictionary for allometry fitting
    """
    # Default fitting configuration
    default_config = {
        'data': {
            'target_years': args.years or [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        },
        'height_agb': {
            'quantiles': [0.15, 0.85],
            'alpha': 0.05,
            'min_samples': 10,
            'max_height': 50.0,
            'min_height': 1.0
        },
        'bgb_agb': {
            'min_samples': 25,
            'percentiles': [5, 50, 95]
        },
        'outlier_removal': {
            'enabled': True,
            'contamination': 0.12
        },
        'quality_filters': {
            'min_r2': 0.1,
            'min_slope': 0.0
        },
        'use_10m_for_fitting': not args.use_100m  # Use 10m by default, unless --use-100m specified
    }
    
    # Load custom config if provided
    if args.config:
        try:
            custom_config = load_config(args.config)
            # Merge custom config with defaults
            for key, value in custom_config.items():
                if key in default_config:
                    if isinstance(value, dict) and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                else:
                    default_config[key] = value
        except Exception as e:
            print(f"Warning: Could not load custom config {args.config}: {e}")
    
    return default_config


class AllometryFittingPipeline:
    """
    Complete allometry fitting pipeline with harmonized path integration.
    
    Orchestrates the full pipeline from NFI data loading to fitted parameter
    output using the new allometry fitting modules.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize fitting pipeline."""
        
        # Create fitting configuration
        self.config = create_fitting_config(args)
        
        # Setup logging
        self.logger = setup_logging(
            level="INFO",
            component_name='allometry_fitting'
        )
        
        self.logger.info("AllometryFittingPipeline initialized")
        self.logger.info(f"Using {'10m' if self.config['use_10m_for_fitting'] else '100m'} height maps for fitting")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites...")
        
        issues = []
        
        # Check NFI processed data directory
        nfi_dir = FOREST_INVENTORY_PROCESSED_DIR
        if not nfi_dir.exists():
            issues.append(f"NFI processed directory not found: {nfi_dir}")
        
        # Check forest types hierarchy file
        forest_types_file = FOREST_TYPES_TIERS_FILE
        if not forest_types_file.exists():
            issues.append(f"Forest types hierarchy file not found: {forest_types_file}")
        
        # Check height maps directory
        if self.config['use_10m_for_fitting']:
            height_dir = HEIGHT_MAPS_10M_DIR
            resolution = "10m"
        else:
            height_dir = HEIGHT_MAPS_100M_DIR
            resolution = "100m"
        
        if not height_dir.exists():
            issues.append(f"{resolution} height maps directory not found: {height_dir}")
        
        # Check for at least one year of data
        found_data = False
        for year in self.config['data']['target_years']:
            nfi_file = nfi_dir / f'nfi4_{year}_biomass.shp'
            height_year_dir = height_dir / str(year)
            
            if nfi_file.exists() and height_year_dir.exists():
                found_data = True
                break
        
        if not found_data:
            issues.append(f"No matching NFI and height data found for target years: {self.config['data']['target_years']}")
        
        # Log issues
        if issues:
            for issue in issues:
                self.logger.error(f"Prerequisite check failed: {issue}")
            return False
        else:
            self.logger.info("All prerequisites validated successfully")
            return True
    
    def run_fitting(self) -> bool:
        """
        Run the complete allometry fitting pipeline.
        
        Returns:
            bool: True if fitting succeeded
        """
        try:
            self.logger.info("Starting allometry fitting pipeline...")
            start_time = time.time()
            
            # Run the fitting pipeline
            allometry_df, ratio_df = run_allometry_fitting_pipeline(self.config)
            
            # Save results to harmonized output locations
            output_files = save_allometry_results(
                allometry_df, ratio_df
            )
            
            # Log summary
            duration = time.time() - start_time
            self.logger.info(f"Allometry fitting completed in {duration:.2f} seconds")
            self.logger.info(f"Output files created:")
            for file_type, file_path in output_files.items():
                self.logger.info(f"  {file_type}: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Allometry fitting failed: {str(e)}")
            if not self.continue_on_error:
                raise
            return False
    
    def run(self) -> bool:
        """
        Execute complete pipeline with validation and error handling.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        try:
            # Validate prerequisites
            if not self.validate_prerequisites():
                self.logger.error("Prerequisites validation failed")
                return False
            
            # If validation only, stop here
            if self.validate_only:
                self.logger.info("Validation completed successfully (validation-only mode)")
                return True
            
            # Run fitting
            success = self.run_fitting()
            
            if success:
                self.logger.info("Allometry fitting pipeline completed successfully")
            else:
                self.logger.error("Allometry fitting pipeline failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            return False


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return False
    
    # Setup pipeline start logging
    start_time = time.time()
    
    try:
        # Initialize and run pipeline
        pipeline = AllometryFittingPipeline(args)
        success = pipeline.run()
        
        # Log completion
        duration = time.time() - start_time
        if success:
            print(f"\n✅ Allometry fitting completed successfully in {duration:.2f} seconds")
            return True
        else:
            print(f"\n❌ Allometry fitting failed after {duration:.2f} seconds")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 Pipeline failed with error after {duration:.2f} seconds: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)