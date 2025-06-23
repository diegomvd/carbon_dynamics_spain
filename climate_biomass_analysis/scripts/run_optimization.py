#!/usr/bin/env python3
"""
Machine learning optimization script.

Command-line interface for running Bayesian optimization to select optimal
climate predictors and hyperparameters for biomass change prediction.

Usage:
    python run_optimization.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for biomass-climate modeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--n-runs',
        type=int,
        help='Number of independent optimization runs (overrides config)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        help='Number of trials per run (overrides config)'
    )
    
    parser.add_argument(
        '--random-seeds',
        nargs='+',
        type=int,
        help='Random seeds for optimization runs'
    )
    
    # Cross-validation settings
    parser.add_argument(
        '--test-blocks',
        type=int,
        help='Number of spatial blocks for testing (overrides config)'
    )
    
    parser.add_argument(
        '--validation-blocks',
        nargs=2,
        type=int,
        metavar=('MIN', 'MAX'),
        help='Range of validation blocks (min max, overrides config)'
    )
    
    # Feature selection
    parser.add_argument(
        '--exclude-bio-vars',
        nargs='+',
        help='Bioclimatic variables to exclude (e.g., bio8 bio9)'
    )
    
    parser.add_argument(
        '--no-standardization',
        action='store_true',
        help='Skip feature standardization'
    )
    
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        help='Threshold for removing correlated features (overrides config)'
    )
    
    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        help='Early stopping patience (overrides config)'
    )
    
    parser.add_argument(
        '--min-trials',
        type=int,
        help='Minimum trials before early stopping (overrides config)'
    )
    
    # Execution control
    parser.add_argument(
        '--single-run',
        action='store_true',
        help='Run only a single optimization (for testing)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume optimization from existing results'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without running optimization'
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


def validate_inputs(optimizer, args):
    """Validate input dataset and configuration."""
    logger = setup_logging()
    
    try:
        # Try to load spatial data
        df = optimizer.load_spatial_data()
        logger.info(f"✅ Dataset validation passed")
        logger.info(f"  - Dataset shape: {df.shape}")
        logger.info(f"  - Spatial clusters: {df['cluster_id'].nunique()}")
        
        # Check required columns
        required_cols = ['x', 'y', 'cluster_id', 'biomass_rel_change']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient data
        if len(df) < 1000:
            logger.warning(f"Dataset is small ({len(df)} points), results may be unreliable")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        return False


def override_config(optimizer, args):
    """Override configuration with command line arguments."""
    config_changed = False
    
    # Optimization parameters
    if args.n_runs:
        optimizer.opt_config['n_runs'] = args.n_runs
        config_changed = True
    
    if args.n_trials:
        optimizer.opt_config['n_trials'] = args.n_trials
        config_changed = True
    
    if args.random_seeds:
        optimizer.opt_config['random_seeds'] = args.random_seeds
        config_changed = True
    
    # Cross-validation
    if args.test_blocks:
        optimizer.cv_config['test_blocks'] = args.test_blocks
        config_changed = True
    
    if args.validation_blocks:
        optimizer.cv_config['validation_blocks_range'] = args.validation_blocks
        config_changed = True
    
    # Features
    if args.exclude_bio_vars:
        optimizer.features_config['exclude_bio_vars'] = args.exclude_bio_vars
        config_changed = True
    
    if args.no_standardization:
        optimizer.features_config['standardize'] = False
        config_changed = True
    
    if args.correlation_threshold:
        optimizer.features_config['correlation_threshold'] = args.correlation_threshold
        config_changed = True
    
    # Early stopping
    if args.patience:
        optimizer.early_stopping_config['patience'] = args.patience
        config_changed = True
    
    if args.min_trials:
        optimizer.early_stopping_config['min_trials'] = args.min_trials
        config_changed = True
    
    if config_changed:
        optimizer.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


def save_configuration(optimizer, output_dir):
    """Save effective configuration to output directory."""
    config_file = Path(output_dir) / "effective_config.json"
    
    # Extract relevant configuration
    effective_config = {
        'optimization': optimizer.opt_config,
        'cv_strategy': optimizer.cv_config,
        'features': optimizer.features_config,
        'early_stopping': optimizer.early_stopping_config,
    }
    
    with open(config_file, 'w') as f:
        json.dump(effective_config, f, indent=2)
    
    optimizer.logger.info(f"Effective configuration saved: {config_file}")


def main():
    """Main entry point for optimization script."""
    args = parse_arguments()
    
    # Setup logging
    if args.quiet:
        log_level = 'ERROR'
    else:
        log_level = args.log_level
    
    logger = setup_logging(level=log_level, component_name='optimization_script')
    
    try:
        # Initialize optimizer
        logger.info("Initializing optimization pipeline...")
        optimizer = OptimizationPipeline(config_path=args.config)
        
        # Override configuration with command line arguments
        override_config(optimizer, args)
        
        # Create output directory
        from shared_utils import ensure_directory
        ensure_directory(args.output_dir)
        
        # Save effective configuration
        save_configuration(optimizer, args.output_dir)
        
        # Validate inputs
        if not validate_inputs(optimizer, args):
            logger.error("Input validation failed")
            sys.exit(1)
        
        # Validation only mode
        if args.validate_only:
            logger.info("✅ Validation completed successfully")
            return
        
        # Single run mode (for testing)
        if args.single_run:
            logger.info("Running single optimization (test mode)...")
            optimizer.opt_config['n_runs'] = 1
            optimizer.opt_config['n_trials'] = min(50, optimizer.opt_config['n_trials'])
            logger.info("  - Reduced to 1 run with max 50 trials")
        
        # Log optimization settings
        logger.info(f"Optimization settings:")
        logger.info(f"  - Runs: {optimizer.opt_config['n_runs']}")
        logger.info(f"  - Trials per run: {optimizer.opt_config['n_trials']}")
        logger.info(f"  - Test blocks: {optimizer.cv_config['test_blocks']}")
        logger.info(f"  - Validation blocks: {optimizer.cv_config['validation_blocks_range']}")
        logger.info(f"  - Early stopping patience: {optimizer.early_stopping_config['patience']}")
        
        # Check for resume
        if args.resume:
            results_file = Path(args.output_dir) / "individual_run_results.pkl"
            if results_file.exists():
                logger.info("⚠️  Resume functionality not implemented")
                logger.info("   Existing results will be overwritten")
        
        # Run optimization
        logger.info("🚀 Starting Bayesian optimization...")
        results = optimizer.run_optimization_pipeline()
        
        # Move results to specified output directory
        import shutil
        default_output = Path("optimization_results")
        target_output = Path(args.output_dir)
        
        if default_output != target_output and default_output.exists():
            logger.info(f"Moving results to {target_output}")
            if target_output.exists():
                shutil.rmtree(target_output)
            shutil.move(str(default_output), str(target_output))
        
        # Log final summary
        summary = results['summary']
        logger.info(f"\n🎉 Optimization completed successfully!")
        logger.info(f"📊 Final Results:")
        logger.info(f"  - Validation R²: {summary['validation_r2']['mean']:.4f} ± {summary['validation_r2']['std']:.4f}")
        logger.info(f"  - Test R²: {summary['test_r2']['mean']:.4f} ± {summary['test_r2']['std']:.4f}")
        logger.info(f"  - Best validation R²: {summary['validation_r2']['max']:.4f}")
        logger.info(f"  - Best test R²: {summary['test_r2']['max']:.4f}")
        logger.info(f"📈 Feature Analysis:")
        logger.info(f"  - Total features: {summary['feature_analysis']['total_features']}")
        logger.info(f"  - Frequently selected: {summary['feature_analysis']['frequently_selected']}")
        logger.info(f"  - Top 5 features: {', '.join(summary['feature_analysis']['top_features'][:5])}")
        logger.info(f"📁 Results saved to: {target_output}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()