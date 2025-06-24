#!/usr/bin/env python3
"""
Bioclimatic variables calculation script.

Command-line interface for calculating bioclimatic variables (bio1-bio19) from
monthly temperature and precipitation data, and computing climate anomalies.

Usage:
    python run_bioclim_calculation.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import glob
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculator
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate bioclimatic variables and anomalies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for bioclimatic calculation script."""
    args = parse_arguments()
    
    log_level = 'INFO'
    logger = setup_logging(level=log_level, component_name='bioclim_calculation_script')
    
    try:
        # Initialize calculator
        logger.info("Initializing bioclimatic calculator...")
        calculator = BioclimCalculator(config_path=args.config)
        
        # Use config paths
        data_dir = CLIMATE_HARMONIZED_DIR
        temp_files = glob.glob(os.path.join(data_dir, calculator.climate_config['temp_pattern']))
        precip_files = glob.glob(os.path.join(data_dir, calculator.climate_config['precip_pattern']))
        
        logger.info(f"Found {len(temp_files)} temperature files and {len(precip_files)} precipitation files")
        
        if not temp_files or not precip_files:
            raise ValueError("No temperature or precipitation files found")

        
        # Step 1: Harmonize rasters (unless skipped)

        harmonized_dir = CLIMATE_HARMONIZED_DIR
        harmonized_temp_dir = harmonized_dir / "temperature"
        harmonized_precip_dir = harmonized_dir / "precipitation"
        
        logger.info("Harmonizing raster files...")
        harmonized_temp_files = calculator.harmonize_rasters(temp_files, harmonized_temp_dir)
        harmonized_precip_files = calculator.harmonize_rasters(precip_files, harmonized_precip_dir)

        
        # Step 2: Calculate reference bioclimatic variables
        logger.info("Calculating reference bioclimatic variables...")
        
        output_dir = args.output_dir or BIOCLIM_VARIABLES_DIR
        
        reference_bioclim = calculator.calculate_bioclim_variables(
            harmonized_temp_files,
            harmonized_precip_files,
            output_dir,
            start_year=time_periods['reference']['start_year'],
            end_year=time_periods['reference']['end_year'],
            rolling=time_periods['reference']['rolling']
        )
        
        if reference_bioclim:
            logger.info(f"✅ Reference bioclimatic variables calculated: {len(reference_bioclim)} variables")
        else:
            logger.error("❌ Failed to calculate reference bioclimatic variables")
            if not args.continue_on_error:
                sys.exit(1)
        
        # Step 3: Calculate anomalies (if requested)
        logger.info("Calculating bioclimatic anomalies...")
        
        reference_dir = args.output_dir or BIOCLIM_VARIABLES_DIR
        anomaly_dir = args.anomaly_dir or BIOCLIM_ANOMALIES_DIR
        
        yearly_anomalies = calculator.calculate_bioclim_anomalies(
            harmonized_temp_files,
            harmonized_precip_files,
            reference_dir,
            anomaly_dir,
            start_year=time_periods['analysis']['start_year'],
            end_year=time_periods['analysis']['end_year'],
            rolling=time_periods['analysis']['rolling']
        )
        
        if yearly_anomalies:
            total_anomalies = sum(len(year_data) for year_data in yearly_anomalies.values())
            logger.info(f"✅ Bioclimatic anomalies calculated: {len(yearly_anomalies)} years, "
                        f"{total_anomalies} total anomaly files")
        else:
            logger.error("❌ Failed to calculate bioclimatic anomalies")
            if not args.continue_on_error:
                sys.exit(1)
        
        logger.info("Bioclimatic calculation completed successfully!")
        
    except Exception as e:
        logger.error(f"Bioclimatic calculation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()