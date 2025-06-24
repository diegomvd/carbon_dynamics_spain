#!/usr/bin/env python3
"""
Spatial analysis script.

Command-line interface for performing spatial autocorrelation analysis and
creating spatial clusters for cross-validation in machine learning.

Usage:
    python run_spatial_analysis.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.spatial_analysis import SpatialAnalyzer
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform spatial autocorrelation analysis and clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    return parser.parse_args()


def validate_inputs(analyzer, args):
    """Validate input files and configuration."""
    logger = setup_logging()
    
    training_dataset = CLIMATE_BIOMASS_DATASET_FILE
    
    if not Path(training_dataset).exists():
        logger.error(f"Training dataset not found: {training_dataset}")
        logger.error("Please run biomass integration pipeline first or provide --input-dataset")
        return False
    
    # Validate dataset format
    try:
        import pandas as pd
        df = pd.read_csv(training_dataset)
        
        required_cols = ['x', 'y', 'biomass_rel_change']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Training dataset missing required columns: {missing_cols}")
            return False
        
        logger.info(f"✅ Training dataset validated: {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        logger.error(f"Error validating training dataset: {e}")
        return False
    
    biomass_dir = BIOMASS_CHANGE_MAPS_REL_DIFF_DIR
    
    if not Path(biomass_dir).exists():
        logger.error(f"Biomass directory not found: {biomass_dir}")
        return False
    
    # Check for raster files
    raster_files = analyzer.find_interannual_difference_files()
    
    if not raster_files:
        logger.error(f"No biomass raster files found in {biomass_dir}")
        return False
    
    logger.info(f"✅ Found {len(raster_files)} biomass raster files for autocorrelation analysis")

    return True


def main():
    """Main entry point for spatial analysis script."""
    args = parse_arguments()
    
    # Setup logging
    if args.quiet:
        log_level = 'ERROR'
    else:
        log_level = args.log_level
    
    logger = setup_logging(level=log_level, component_name='spatial_analysis_script')
    
    try:
        # Initialize analyzer
        logger.info("Initializing spatial analyzer...")
        analyzer = SpatialAnalyzer(config_path=args.config)
        
        # Validate inputs
        if not validate_inputs(analyzer, args):
            logger.error("Input validation failed")
            sys.exit(1)
        
        # Log analysis settings
        logger.info(f"Spatial analysis settings:")
        logger.info(f"  - Sample fraction: {analyzer.spatial_config['sample_fraction']}")
        logger.info(f"  - Max distance: {analyzer.spatial_config['max_distance_km']} km")
        logger.info(f"  - Distance bins: {analyzer.spatial_config['n_distance_bins']}")
        logger.info(f"  - Skip autocorrelation: {args.skip_autocorr}")
        logger.info(f"  - Skip clustering: {args.skip_clustering}")

        # Complete pipeline
        logger.info("🚀 Running complete spatial analysis pipeline...")
        
        df_clustered = analyzer.run_spatial_analysis_pipeline()
        
        if df_clustered is not None:
            logger.info(f"✅ Complete spatial analysis pipeline completed successfully!")
            logger.info(f"  - Dataset shape: {df_clustered.shape}")
            logger.info(f"  - Spatial clusters: {df_clustered['cluster_id'].nunique()}")
            logger.info(f"  - Clustered dataset: {analyzer.config['data']['clustered_dataset']}")
        else:
            logger.error("❌ Spatial analysis pipeline failed")
            sys.exit(1)
        
        logger.info("Spatial analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Spatial analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()