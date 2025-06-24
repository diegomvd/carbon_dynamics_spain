#!/usr/bin/env python3
"""
Machine learning optimization script.

Command-line interface for running Bayesian optimization to select optimal
climate predictors and hyperparameters for biomass change prediction.

Usage:
    python run_optimization.py

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline
from shared_utils import setup_logging, ensure_directory
from shared_utils.central_data_paths import *


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

    log_level = 'INFO'

    logger = setup_logging(level=log_level, component_name='optimization_script')
    
    try:
        # Initialize optimizer
        logger.info("Initializing optimization pipeline...")
        optimizer = OptimizationPipeline(config_path=args.config)
        
        
        # Create output directory
        ensure_directory(args.output_dir)
        
        # Save effective configuration
        save_configuration(optimizer, CLIMATE_BIOMASS_MODELS_DIR)
        
        # Validate inputs
        if not validate_inputs(optimizer, args):
            logger.error("Input validation failed")
            sys.exit(1)
        
        # Log optimization settings
        logger.info(f"Optimization settings:")
        logger.info(f"  - Runs: {optimizer.opt_config['n_runs']}")
        logger.info(f"  - Trials per run: {optimizer.opt_config['n_trials']}")
        logger.info(f"  - Test blocks: {optimizer.cv_config['test_blocks']}")
        logger.info(f"  - Validation blocks: {optimizer.cv_config['validation_blocks_range']}")
        logger.info(f"  - Early stopping patience: {optimizer.early_stopping_config['patience']}")
        
        # Run optimization
        logger.info("🚀 Starting Bayesian optimization...")
        results = optimizer.run_optimization_pipeline()
        
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
        sys.exit(1)


if __name__ == "__main__":
    main()