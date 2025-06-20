#!/usr/bin/env python3
"""
Spatial analysis script.

Command-line interface for performing spatial autocorrelation analysis and
creating spatial clusters for cross-validation in machine learning.

Usage:
    python run_spatial_analysis.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.spatial_analysis import SpatialAnalyzer
from shared_utils import setup_logging


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
    
    # Input dataset
    parser.add_argument(
        '--input-dataset',
        type=str,
        help='Path to training dataset CSV (overrides config)'
    )
    
    parser.add_argument(
        '--output-dataset',
        type=str,
        help='Path for clustered dataset output (overrides config)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--n-clusters',
        type=int,
        help='Number of spatial clusters to create (overrides config)'
    )
    
    parser.add_argument(
        '--sample-fraction',
        type=float,
        help='Fraction of pixels to sample for semivariogram (overrides config)'
    )
    
    parser.add_argument(
        '--max-distance-km',
        type=float,
        help='Maximum distance for semivariogram analysis in km (overrides config)'
    )
    
    parser.add_argument(
        '--n-distance-bins',
        type=int,
        help='Number of distance bins for semivariogram (overrides config)'
    )
    
    # Processing control
    parser.add_argument(
        '--autocorr-threshold',
        type=float,
        help='Autocorrelation threshold in km for cluster validation'
    )
    
    parser.add_argument(
        '--skip-autocorr',
        action='store_true',
        help='Skip autocorrelation analysis and use default values'
    )
    
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering step (only do autocorrelation analysis)'
    )
    
    # Biomass raster options
    parser.add_argument(
        '--biomass-dir',
        type=str,
        help='Directory containing biomass difference rasters (overrides config)'
    )
    
    parser.add_argument(
        '--raster-pattern',
        type=str,
        help='Pattern for biomass raster files (e.g., "*_rel_change_*.tif")'
    )
    
    # Output control
    parser.add_argument(
        '--output-dir',
        type=str,
        default='spatial_analysis_results',
        help='Output directory for plots and analysis results'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate analysis results'
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


def override_config(analyzer, args):
    """Override configuration with command line arguments."""
    config_changed = False
    
    # Analysis parameters
    if args.sample_fraction:
        analyzer.spatial_config['sample_fraction'] = args.sample_fraction
        config_changed = True
    
    if args.max_distance_km:
        analyzer.spatial_config['max_distance_km'] = args.max_distance_km
        config_changed = True
    
    if args.n_distance_bins:
        analyzer.spatial_config['n_distance_bins'] = args.n_distance_bins
        config_changed = True
    
    # Clustering parameters
    if args.n_clusters:
        analyzer.spatial_config['clustering']['n_clusters_range'] = [args.n_clusters, args.n_clusters]
        config_changed = True
    
    # Data paths
    if args.input_dataset:
        analyzer.config['data']['training_dataset'] = args.input_dataset
        config_changed = True
    
    if args.output_dataset:
        analyzer.config['data']['clustered_dataset'] = args.output_dataset
        config_changed = True
    
    if args.biomass_dir:
        analyzer.config['data']['biomass_diff_dir'] = args.biomass_dir
        config_changed = True
    
    if config_changed:
        analyzer.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


def validate_inputs(analyzer, args):
    """Validate input files and configuration."""
    logger = setup_logging()
    
    # Check training dataset if clustering is requested
    if not args.skip_clustering:
        training_dataset = analyzer.config['data']['training_dataset']
        
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
    
    # Check biomass rasters if autocorrelation analysis is requested
    if not args.skip_autocorr:
        biomass_dir = analyzer.config['data']['biomass_diff_dir']
        
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
        
        # Override configuration with command line arguments
        override_config(analyzer, args)
        
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
        
        # Run analysis pipeline
        if args.skip_autocorr and args.skip_clustering:
            logger.error("Cannot skip both autocorrelation analysis and clustering")
            sys.exit(1)
        
        elif args.skip_autocorr:
            # Only clustering
            logger.info("🎯 Running spatial clustering only...")
            
            training_dataset = analyzer.config['data']['training_dataset']
            import pandas as pd
            df = pd.read_csv(training_dataset)
            
            # Use default autocorrelation threshold
            default_autocorr_km = args.autocorr_threshold or 50.0
            autocorr_lengths_m = [default_autocorr_km * 1000]
            
            k = args.n_clusters or analyzer.spatial_config['clustering']['n_clusters_range'][0]
            
            df_clustered = analyzer.create_spatial_clusters_advanced(df, k, autocorr_lengths_m)
            
            # Save clustered dataset
            output_path = analyzer.config['data']['clustered_dataset']
            from shared_utils import ensure_directory
            ensure_directory(Path(output_path).parent)
            df_clustered.to_csv(output_path, index=False)
            
            logger.info(f"✅ Spatial clustering completed")
            logger.info(f"  - Created {k} clusters")
            logger.info(f"  - Clustered dataset saved: {output_path}")
        
        elif args.skip_clustering:
            # Only autocorrelation analysis
            logger.info("📊 Running spatial autocorrelation analysis only...")
            
            # Find and analyze raster files
            raster_files = analyzer.find_interannual_difference_files()
            results = []
            
            for raster_path in raster_files:
                result = analyzer.analyze_raster(raster_path)
                results.append(result)
            
            # Save results and create plots
            output_dir = Path(args.output_dir)
            summary_file, detailed_file = analyzer.save_results(results, output_dir)
            
            if not args.no_plots:
                analyzer.plot_results(results, output_dir)
            
            # Print summary
            valid_results = [r for r in results if r is not None]
            lengths = [r['autocorr_length_km'] for r in valid_results if not np.isnan(r['autocorr_length_km'])]
            
            logger.info(f"✅ Autocorrelation analysis completed")
            logger.info(f"  - Processed: {len(valid_results)}/{len(raster_files)} files")
            if lengths:
                import numpy as np
                logger.info(f"  - Mean autocorr length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} km")
                logger.info(f"  - Range: {np.min(lengths):.1f} - {np.max(lengths):.1f} km")
            logger.info(f"  - Results saved: {summary_file}")
        
        else:
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
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()