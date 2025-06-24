#!/usr/bin/env python3
"""
Annual Cropland Masking Script

Script for masking biomass maps to exclude annual cropland areas using
Corine Land Cover data. Updated with recipe integration arguments.

Usage:
    python run_masking.py [OPTIONS]
    
Examples:
    # Run with default directories
    python run_masking.py

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
        description="Annual Cropland Masking for Biomass Maps",
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


class AnnualCroplandMaskingRunner:
    """
    Annual cropland masking runner with recipe integration support.
    
    Handles setup of centralized paths, configuration overrides, and
    masking execution with comprehensive error handling.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize masking runner."""
        # Setup logging
        log_level = 'INFO'
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_masking'
        )
        
        # Store arguments
        self.args = args
        
        self.logger.info("AnnualCroplandMaskingRunner initialized")

    
    def determine_input_output_dirs(self) -> tuple:
        """Determine input and output directories from arguments and defaults."""
        input_dir = BIOMASS_MAPS_RAW_DIR
        output_dir = BIOMASS_MAPS_PER_FOREST_TYPE_DIR
        
        return input_dir, output_dir
    
    def get_land_cover_file(self) -> Path:
        """Get land cover file path from arguments or default."""
        return CORINE_LAND_COVER_FILE
    
    def run_masking(self) -> bool:
        """
        Execute the annual cropland masking process.
        
        Returns:
            bool: True if masking completed successfully
        """
        try:
            self.logger.info("Starting annual cropland masking...")
            start_time = time.time()
            
            # Determine input and output directories
            input_dir, output_dir = self.determine_input_output_dirs()
            land_cover_file = self.get_land_cover_file()
            
            self.logger.info(f"Input directory: {input_dir}")
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Land cover file: {land_cover_file}")
            
            # Validate paths
            if not input_dir.exists():
                self.logger.error(f"Input directory not found: {input_dir}")
                return False
            
            if not land_cover_file.exists():
                self.logger.error(f"Land cover file not found: {land_cover_file}")
                return False
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Import masking logic here to avoid import issues
            # This would import the actual masking implementation
            try:
                from biomass_model.core.land_use_masking import LandUseMaskingPipeline
                
                # Initialize masking pipeline
                masking_pipeline = LandUseMaskingPipeline()
                
                # Run masking
                success = masking_pipeline.process_directory(
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    land_cover_file=str(land_cover_file)
                )
                
            except ImportError:
                # Fallback implementation if core masking module not available
                self.logger.warning("Core masking module not available")
                return False
            
            # Log completion
            duration = time.time() - start_time
            if success:
                self.logger.info(f"Annual cropland masking completed in {duration:.2f} seconds")
            else:
                self.logger.error(f"Annual cropland masking failed after {duration:.2f} seconds")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Masking execution failed: {str(e)}")
            return False
    
def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return False
    
    try:
        # Initialize and run masking
        runner = AnnualCroplandMaskingRunner(args)
        success = runner.run_masking()
        
        # Log completion
        if success:
            print("\n✅ Annual cropland masking completed successfully")
            return True
        else:
            print("\n❌ Annual cropland masking failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Masking interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Masking failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)