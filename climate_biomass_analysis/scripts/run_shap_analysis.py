#!/usr/bin/env python3
"""
SHAP analysis script.

Command-line interface for running comprehensive SHAP analysis on climate-biomass
optimization results, including feature importance, PDP analysis, and interactions.

Usage:
    python run_shap_analysis.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.shap_analysis import ShapAnalyzer
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive SHAP analysis on climate-biomass models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    

    # Model filtering options
    parser.add_argument(
        '--r2-threshold',
        type=float,
        help='Minimum R² threshold for model inclusion'
    )
    
    parser.add_argument(
        '--r2-threshold-interactions',
        type=float,
        help='R² threshold for interaction analysis'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--shap-max-samples',
        type=int,
        help='Maximum samples for SHAP importance calculation'
    )
    
    parser.add_argument(
        '--shap-max-background',
        type=int,
        help='Maximum background samples for SHAP'
    )
    
    parser.add_argument(
        '--perm-max-samples',
        type=int,
        help='Maximum samples for permutation importance'
    )
    
    parser.add_argument(
        '--perm-n-repeats',
        type=int,
        help='Number of repeats for permutation importance'
    )
    
    # PDP parameters
    parser.add_argument(
        '--pdp-max-samples',
        type=int,
        help='Maximum samples for PDP calculation'
    )
    
    parser.add_argument(
        '--pdp-n-top-features',
        type=int,
        help='Number of top features for PDP analysis'
    )
    
    parser.add_argument(
        '--pdp-lowess-frac',
        type=float,
        help='LOWESS smoothing fraction for PDP'
    )
    
    # Interaction parameters
    parser.add_argument(
        '--interaction-feature1',
        type=str,
        default='bio12',
        help='First feature for interaction analysis'
    )
    
    parser.add_argument(
        '--interaction-feature2',
        type=str,
        default='bio12_3yr',
        help='Second feature for interaction analysis'
    )
    
    parser.add_argument(
        '--interaction-max-samples',
        type=int,
        help='Maximum samples for interaction analysis'
    )
    
    # Processing control
    parser.add_argument(
        '--validate-inputs',
        action='store_true',
        help='Validate input files before processing'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without running analysis'
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


def validate_inputs(analyzer: ShapAnalyzer, args):
    """Validate input files and directories."""
    logger = analyzer.logger
    
    # Check models directory
    models_dir = args.models_dir or CLIMATE_BIOMASS_MODELS_DIR
    if not Path(models_dir).exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Check dataset file
    dataset_path = args.dataset or CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Check if output directory is writable
    output_dir = args.output_dir or CLIMATE_BIOMASS_SHAP_OUTPUT_DIR
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test write permissions
    test_file = output_path / '.write_test'
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Cannot write to output directory {output_dir}: {e}")
    
    logger.info("✅ Input validation passed")


def override_config(analyzer: ShapAnalyzer, args):
    """Override configuration with command line arguments."""
    config_changed = False
    
    # Model filtering overrides
    if args.r2_threshold:
        analyzer.shap_config['model_filtering']['r2_threshold'] = args.r2_threshold
        config_changed = True
    
    if args.r2_threshold_interactions:
        analyzer.shap_config['model_filtering']['r2_threshold_interactions'] = args.r2_threshold_interactions
        config_changed = True
    
    # Analysis parameter overrides
    if args.shap_max_samples:
        analyzer.shap_config['analysis']['shap_max_samples'] = args.shap_max_samples
        config_changed = True
    
    if args.shap_max_background:
        analyzer.shap_config['analysis']['shap_max_background'] = args.shap_max_background
        config_changed = True
    
    if args.perm_max_samples:
        analyzer.shap_config['analysis']['perm_max_samples'] = args.perm_max_samples
        config_changed = True
    
    if args.perm_n_repeats:
        analyzer.shap_config['analysis']['perm_n_repeats'] = args.perm_n_repeats
        config_changed = True
    
    # PDP parameter overrides
    if args.pdp_max_samples:
        analyzer.shap_config['analysis']['pdp_max_samples'] = args.pdp_max_samples
        config_changed = True
    
    if args.pdp_n_top_features:
        analyzer.shap_config['analysis']['pdp_n_top_features'] = args.pdp_n_top_features
        config_changed = True
    
    if args.pdp_lowess_frac:
        analyzer.shap_config['analysis']['pdp_lowess_frac'] = args.pdp_lowess_frac
        config_changed = True
    
    # Interaction parameter overrides
    if args.interaction_max_samples:
        analyzer.shap_config['analysis']['interaction_max_samples'] = args.interaction_max_samples
        config_changed = True
    
    if config_changed:
        analyzer.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


def show_dry_run_info(analyzer: ShapAnalyzer):
    """Show what would be processed in a dry run."""
    logger = analyzer.logger
    
    logger.info("🔍 DRY RUN - Analysis plan:")
    logger.info("="*50)
    
    # Show paths
    logger.info("📁 Input/Output paths:")
    logger.info(f"  Models directory: {str(CLIMATE_BIOMASS_MODELS_DIR)}")
    logger.info(f"  Dataset file: {str(CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE)}")
    logger.info(f"  Output directory: {str(CLIMATE_BIOMASS_SHAP_OUTPUT_DIR)}")
    
    # Show analysis parameters
    logger.info("⚙️ Analysis parameters:")
    logger.info(f"  R² threshold: {analyzer.shap_config['model_filtering']['r2_threshold']}")
    logger.info(f"  SHAP max samples: {analyzer.shap_config['analysis']['shap_max_samples']}")
    logger.info(f"  PDP top features: {analyzer.shap_config['analysis']['pdp_n_top_features']}")
    logger.info(f"  Interaction features: bio12 vs bio12_3yr")
    
    # Show analysis steps
    logger.info("📋 Analysis steps that would be executed:")
    logger.info("  1. Load and filter models by R² threshold")
    logger.info("  2. Calculate feature selection frequencies")
    logger.info("  3. Calculate SHAP importance values")
    logger.info("  4. Calculate permutation importance")
    logger.info("  5. Calculate PDP with LOWESS smoothing")
    logger.info("  6. Calculate 2D interaction analysis")
    logger.info("  7. Save all results and summary")
    
    logger.info("="*50)
    logger.info("🔄 Use --dry-run=false to execute the analysis")


def save_effective_config(analyzer: ShapAnalyzer):
    """Save the effective configuration used for the analysis."""
    import json
    from pathlib import Path
    
    output_dir = CLIMATE_BIOMASS_SHAP_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / "effective_shap_config.json"
    
    # Create a clean config for saving
    effective_config = {
        'model_filtering': analyzer.shap_config['model_filtering'],
        'analysis': analyzer.shap_config['analysis']
    }
    
    with open(config_file, 'w') as f:
        json.dump(effective_config, f, indent=2)
    
    analyzer.logger.info(f"Effective configuration saved: {config_file}")


def main():
    """Main entry point for SHAP analysis script."""
    args = parse_arguments()
    
    # Setup logging
    log_level = 'ERROR' if args.quiet else args.log_level
    logger = setup_logging(level=log_level, component_name='run_shap_analysis')
    
    try:
        logger.info("🧠 Starting SHAP Analysis Pipeline...")
        
        # Initialize analyzer
        analyzer = ShapAnalyzer(config_path=args.config)
        
        # Override configuration with command line arguments
        override_config(analyzer, args)
        
        # Validate inputs if requested
        if args.validate_inputs:
            logger.info("🔍 Validating inputs...")
            validate_inputs(analyzer, args)
        
        # Show dry run information and exit
        if args.dry_run:
            show_dry_run_info(analyzer)
            return
        
        # Save effective configuration
        save_effective_config(analyzer)
        
        # Run the analysis
        logger.info("🚀 Starting comprehensive SHAP analysis...")
        results = analyzer.run_comprehensive_shap_analysis()
        
        # Log success
        logger.info("✅ SHAP analysis completed successfully!")
        logger.info(f"📊 Results summary:")
        logger.info(f"   Models processed: {results['models_filtered']}/{results['models_loaded']}")
        logger.info(f"   Features analyzed: {results['features_analyzed']}")
        logger.info(f"   Analysis time: {results['analysis_time_formatted']}")
        logger.info(f"   Output directory: {results['output_directory']}")
        
        logger.info("📁 Generated files:")
        logger.info("   📈 feature_frequencies_df.csv")
        logger.info("   🧠 avg_shap_importance.pkl")
        logger.info("   🔄 avg_permutation_importance.pkl")
        logger.info("   📉 pdp_data.pkl")
        logger.info("   🎯 pdp_lowess_data.pkl")
        logger.info("   🔗 interaction_results.pkl")
        logger.info("   📋 analysis_summary.json")
        
    except KeyboardInterrupt:
        logger.error("❌ Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ SHAP analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()