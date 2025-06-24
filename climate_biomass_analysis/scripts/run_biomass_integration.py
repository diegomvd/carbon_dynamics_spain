#!/usr/bin/env python3
"""
Biomass-climate data integration script.

Command-line interface for integrating biomass change data with climate anomalies
to create machine learning training datasets with proper spatial alignment.

Usage:
    python run_biomass_integration.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.biomass_integration import BiomassIntegrator
from shared_utils import setup_logging, ensure_directory
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integrate biomass changes with climate anomalies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    return parser.parse_args()


def validate_inputs(integrator, args):
    """Validate input directories and files."""
    logger = setup_logging()
    
    # Check biomass directory
    biomass_dir = BIOMASS_MAPS_RELDIFF_DIR
    if not Path(biomass_dir).exists():
        logger.error(f"Biomass directory not found: {biomass_dir}")
        return False
    
    # Check for biomass files
    import glob
    pattern = integrator.integration_config['pattern']
    biomass_files = glob.glob(str(Path(biomass_dir) / pattern))
    
    if not biomass_files:
        logger.error(f"No biomass files found matching pattern: {pattern}")
        logger.error(f"Searched in: {biomass_dir}")
        return False
    
    logger.info(f"✅ Found {len(biomass_files)} biomass difference files")
    
    # Check anomaly directory
    anomaly_dir = BIOCLIM_ANOMALIES_DIR
    if not Path(anomaly_dir).exists():
        logger.error(f"Anomaly directory not found: {anomaly_dir}")
        logger.error("Please run bioclimatic calculation pipeline first")
        return False
    
    # Check for anomaly subdirectories
    anomaly_subdirs = [d for d in Path(anomaly_dir).iterdir() 
                      if d.is_dir() and d.name.startswith('anomalies_')]
    
    if not anomaly_subdirs:
        logger.error(f"No anomaly subdirectories found in {anomaly_dir}")
        logger.error("Expected directories like 'anomalies_2017', 'anomalies_2018', etc.")
        return False
    
    logger.info(f"✅ Found {len(anomaly_subdirs)} anomaly year directories")
    
    # Check for anomaly files in first directory
    first_anomaly_dir = anomaly_subdirs[0]
    anomaly_files = list(first_anomaly_dir.glob("*.tif"))
    
    if not anomaly_files:
        logger.error(f"No anomaly files found in {first_anomaly_dir}")
        return False
    
    logger.info(f"✅ Found {len(anomaly_files)} anomaly files in {first_anomaly_dir.name}")
    
    return True


def check_existing_outputs(integrator, args):
    """Check if output files already exist."""
    logger = setup_logging()
    
    training_dataset = CLIMATE_BIOMASS_DATASET_FILE
    
    if Path(training_dataset).exists():
        logger.info(f"Training dataset exists, will overwrite: {training_dataset}")
    
    return True


def main():
    """Main entry point for biomass integration script."""
    args = parse_arguments()
    
    log_level = 'INFO'
    logger = setup_logging(level=log_level, component_name='biomass_integration_script')
    
    try:
        # Initialize integrator
        logger.info("Initializing biomass integrator...")
        integrator = BiomassIntegrator(config_path=args.config)
        
        # Log integration settings
        logger.info(f"Integration settings:")
        logger.info(f"  - Biomass dir: {str(BIOMASS_MAPS_RELDIFF_DIR)}")
        logger.info(f"  - Anomaly dir: {str(BIOCLIM_ANOMALIES_DIR)}")
        logger.info(f"  - Pattern: {integrator.integration_config['pattern']}")
        logger.info(f"  - Resampling: {integrator.integration_config['resampling_method']}")
        logger.info(f"  - Max points: {integrator.integration_config['max_valid_pixels']:,}")
        logger.info(f"  - Remove outliers: {integrator.integration_config['remove_outliers']}")
        
        # Validate inputs
        if not validate_inputs(integrator, args):
            logger.error("Input validation failed")
            sys.exit(1)
        
        # Check existing outputs
        if not check_existing_outputs(integrator, args):
            logger.error("Output validation failed")
            sys.exit(1)
        
        # Create output directories
        
        ensure_directory(CLIMATE_BIOMASS_TEMP_RESAMPLED_DIR)
        ensure_directory(CLIMATE_BIOMASS_DATA_DIR)
        
        # Run integration pipeline
        logger.info("🚀 Starting biomass-climate integration...")
        dataset = integrator.run_biomass_integration_pipeline()
        
        if dataset is not None and len(dataset) > 0:
            logger.info(f"✅ Biomass-climate integration completed successfully!")
            logger.info(f"📊 Dataset Statistics:")
            logger.info(f"  - Total data points: {len(dataset):,}")
            logger.info(f"  - Features: {len(dataset.columns)} columns")
            logger.info(f"  - Coordinate range:")
            logger.info(f"    - X: {dataset['x'].min():.0f} to {dataset['x'].max():.0f}")
            logger.info(f"    - Y: {dataset['y'].min():.0f} to {dataset['y'].max():.0f}")
            logger.info(f"  - Time range:")
            logger.info(f"    - Start years: {dataset['year_start'].min()} to {dataset['year_start'].max()}")
            logger.info(f"    - End years: {dataset['year_end'].min()} to {dataset['year_end'].max()}")
            logger.info(f"  - Biomass change range:")
            logger.info(f"    - Min: {dataset['biomass_rel_change'].min():.4f}")
            logger.info(f"    - Max: {dataset['biomass_rel_change'].max():.4f}")
            logger.info(f"    - Mean: {dataset['biomass_rel_change'].mean():.4f}")
            
            # Show climate variables
            climate_vars = [col for col in dataset.columns if col.startswith('bio')]
            if climate_vars:
                logger.info(f"  - Climate variables: {len(climate_vars)} bioclimatic variables")
                logger.info(f"    - Variables: {', '.join(sorted(climate_vars))}")
            
            logger.info(f"📁 Output saved: {str(CLIMATE_BIOMASS_DATASET_FILE)}")
        
        else:
            logger.error("❌ Biomass-climate integration failed - no data points created")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Biomass integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()