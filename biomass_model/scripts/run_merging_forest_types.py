#!/usr/bin/env python3
"""
Forest Type Merging Script

Script for merging forest type specific biomass maps into country-wide maps.
Updated with recipe integration arguments for harmonized path management.

Usage:
    python run_merging.py [OPTIONS]
    
Examples:
    # Run with default directories
    python run_merging.py

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
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Forest Type Merging for Country-wide Biomass Maps",
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


class ForestTypeMergingRunner:
    """
    Forest type merging runner with recipe integration support.
    
    Handles setup of centralized paths, configuration overrides, and
    merging execution with comprehensive error handling.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize merging runner."""
        
        # Setup logging
        log_level = 'INFO'
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_merging'
        )
        
        # Store arguments
        self.args = args
        
        self.logger.info("ForestTypeMergingRunner initialized")
    

    def determine_input_output_dirs(self) -> tuple:
        """Determine input and output directories from arguments and defaults."""
        input_dir = BIOMASS_MAPS_PER_FOREST_TYPE_DIR
        output_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR
        
        return input_dir, output_dir
    
    def run_merging(self) -> bool:
        """
        Execute the forest type merging process.
        
        Returns:
            bool: True if merging completed successfully
        """
        try:
            self.logger.info("Starting forest type merging...")
            start_time = time.time()
            
            # Determine input and output directories
            input_dir, output_dir = self.determine_input_output_dirs()
            
            self.logger.info(f"Input directory: {input_dir}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Validate paths
            if not input_dir.exists():
                self.logger.error(f"Input directory not found: {input_dir}")
                return False
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Import merging logic here to avoid import issues
            try:
                from biomass_model.core.forest_type_merging import ForestTypeMergingPipeline
                
                # Initialize merging pipeline
                merging_pipeline = ForestTypeMergingPipeline()
                
                # Run merging
                success = merging_pipeline.process_directory(
                    input_dir=str(input_dir),
                    output_dir=str(output_dir)
                )
                
            except ImportError:
                # Fallback implementation if core merging module not available
                self.logger.warning("Core merging module not available")
                return False
            
            # Log completion
            duration = time.time() - start_time
            if success:
                self.logger.info(f"Forest type merging completed in {duration:.2f} seconds")
            else:
                self.logger.error(f"Forest type merging failed after {duration:.2f} seconds")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Merging execution failed: {str(e)}")
            return False
    
def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return False
    
    try:
        # Initialize and run merging
        runner = ForestTypeMergingRunner(args)
        success = runner.run_merging()
        
        # Log completion
        if success:
            print("\n✅ Forest type merging completed successfully")
            return True
        else:
            print("\n❌ Forest type merging failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Merging interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Merging failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)