#!/usr/bin/env python3
"""
Biomass Estimation Pipeline Script

Main entry point for running the biomass estimation pipeline with
Monte Carlo uncertainty quantification. Updated with recipe integration
arguments for harmonized path management.

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
    
    # Recipe integration with custom paths
    python run_biomass_estimation.py --height-100m-dir ./height_maps/100m --allometries-output-dir ./allometries

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
    )
    
    # Core configuration
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


class BiomassEstimationRunner:
    """
    Biomass estimation pipeline runner with recipe integration support.
    
    Handles setup of centralized paths, configuration overrides, and
    pipeline execution with comprehensive error handling.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize pipeline runner."""
        
        # Setup logging
        log_level = 'INFO' 
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_estimation'
        )
        
        # Store arguments
        self.args = args
        
        self.logger.info("BiomassEstimationRunner initialized")
        
    def create_pipeline_config(self) -> dict:
        """Create pipeline configuration with argument overrides."""
        # Load base configuration (no longer needs path overrides)
        config = load_config(self.args.config, component_name="biomass_estimation")
        return config
    
    def run_pipeline(self) -> bool:
        """
        Execute the biomass estimation pipeline.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        try:
            self.logger.info("Starting biomass estimation pipeline...")
            start_time = time.time()
            
            # Create pipeline configuration
            config = self.create_pipeline_config()
            
            pipeline = BiomassEstimationPipeline(config)
            
            # Validate inputs
            if not pipeline.validate_inputs():
                self.logger.error("Input validation failed")
                return False
            
            # Setup output directories
            output_dirs = pipeline.setup_output_directories()
            
            # Process each year
            all_success = True
            years_to_process = config['processing']['target_years']
            
            for year in years_to_process:
                self.logger.info(f"Processing year {year}...")
                
                year_success = pipeline.process_year(year, output_dirs)
                if not year_success:
                    self.logger.warning(f"Year {year} failed, continuing...")
                    all_success = False
            
            # Log completion
            duration = time.time() - start_time
            status = "completed with warnings" if not all_success else "completed successfully"
            self.logger.info(f"Biomass estimation pipeline {status} in {duration:.2f} seconds")
            
            return all_success
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False
    
def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return False
    
    try:
        # Initialize and run pipeline
        runner = BiomassEstimationRunner(args)
        success = runner.run_pipeline()
        
        # Log completion
        if success:
            print("\n✅ Biomass estimation completed successfully")
            return True
        else:
            print("\n❌ Biomass estimation failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)