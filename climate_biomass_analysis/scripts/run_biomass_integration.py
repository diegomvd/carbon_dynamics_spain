#!/usr/bin/env python3
"""
Biomass-climate data integration script.

Command-line interface for integrating biomass change data with climate anomalies
to create machine learning training datasets with proper spatial alignment.

Usage:
    python run_biomass_integration.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.biomass_integration import BiomassIntegrator
from shared_utils import setup_logging


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
    
    # Processing parameters
    parser.add_argument(
        '--biomass-pattern',
        type=str,
        help='File pattern for biomass difference files (overrides config)'
    )
    
    parser.add_argument(
        '--resampling-method',
        choices=['nearest', 'bilinear', 'cubic', 'average'],
        help='Resampling method for biomass data (overrides config)'
    )
    
    parser.add_argument(
        '--max-points',
        type=int,
        help='Maximum data points to extract per dataset (overrides config)'
    )
    
    # Quality control
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        help='Remove statistical outliers from dataset'
    )
    
    parser.add_argument(
        '--no-remove-outliers',
        action='store_true',
        help='Keep all data points (no outlier removal)'
    )
    
    parser.add_argument(
        '--outlier-threshold',
        type=float,
        help='Standard deviation threshold for outlier removal (overrides config)'
    )
    
    # Processing control
    parser.add_argument(
        '--skip-resampling',
        action='store_true',
        help='Skip biomass resampling step (assume already done)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without processing'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def override_config(integrator, args):
    """Override configuration with command line arguments."""
    config_changed = False
    
    # Processing parameters
    if args.biomass_pattern:
        integrator.integration_config['pattern'] = args.biomass_pattern
        config_changed = True
    
    if args.resampling_method:
        integrator.integration_config['resampling_method'] = args.resampling_method
        config_changed = True
    
    if args.max_points:
        integrator.integration_config['max_valid_pixels'] = args.max_points
        config_changed = True
    
    # Quality control
    if args.remove_outliers:
        integrator.integration_config['remove_outliers'] = True
        config_changed = True
    
    if args.no_remove_outliers:
        integrator.integration_config['remove_outliers'] = False
        config_changed = True
    
    if args.outlier_threshold:
        integrator.integration_config['outlier_threshold'] = args.outlier_threshold
        config_changed = True
    
    if config_changed:
        integrator.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


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
        if args.overwrite:
            logger.info(f"Training dataset exists, will overwrite: {training_dataset}")
        else:
            logger.warning(f"Training dataset already exists: {training_dataset}")
            logger.warning("Use --overwrite to replace, or specify different --output-dataset")
            return False
    
    return True


def main():
    """Main entry point for biomass integration script."""
    args = parse_arguments()
    
    # Setup logging
    if args.quiet:
        log_level = 'ERROR'
    else:
        log_level = args.log_level
    
    logger = setup_logging(level=log_level, component_name='biomass_integration_script')
    
    try:
        # Initialize integrator
        logger.info("Initializing biomass integrator...")
        integrator = BiomassIntegrator(config_path=args.config)
        
        # Override configuration with command line arguments
        override_config(integrator, args)
        
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
        
        # Validation only mode
        if args.validate_only:
            logger.info("✅ Validation completed successfully")
            return
        
        # Create output directories
        from shared_utils import ensure_directory
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
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()