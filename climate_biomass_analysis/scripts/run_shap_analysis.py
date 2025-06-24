#!/usr/bin/env python3
"""
SHAP analysis script.

Command-line interface for running comprehensive SHAP analysis on climate-biomass
optimization results, including feature importance, PDP analysis, and interactions.

Usage:
    python run_shap_analysis.py

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

    return parser.parse_args()


def validate_inputs(analyzer: ShapAnalyzer, args):
    """Validate input files and directories."""
    logger = analyzer.logger
    
    # Check models directory
    models_dir = CLIMATE_BIOMASS_MODELS_DIR
    if not Path(models_dir).exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Check dataset file
    dataset_path =  CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Check if output directory is writable
    output_dir = CLIMATE_BIOMASS_SHAP_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Input validation passed")


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
    log_level = 'INFO'
    logger = setup_logging(level=log_level, component_name='run_shap_analysis')
    
    try:
        logger.info("🧠 Starting SHAP Analysis Pipeline...")
        
        # Initialize analyzer
        analyzer = ShapAnalyzer(config_path=args.config)
        
        # Validate inputs if requested
        if args.validate_inputs:
            logger.info("🔍 Validating inputs...")
            validate_inputs(analyzer, args)
        
        # Save effective configuration
        save_effective_config(analyzer)
        
        # Run the analysis
        logger.info("🚀 Starting comprehensive SHAP analysis...")
        results = analyzer.run_comprehensive_shap_analysis()
        
        # Log success
        logger.info("SHAP analysis completed successfully!")
        logger.info(f"Results summary:")
        logger.info(f"   Models processed: {results['models_filtered']}/{results['models_loaded']}")
        logger.info(f"   Features analyzed: {results['features_analyzed']}")
        logger.info(f"   Analysis time: {results['analysis_time_formatted']}")
        logger.info(f"   Output directory: {results['output_directory']}")
        
        logger.info("Generated files:")
        logger.info("   feature_frequencies_df.csv")
        logger.info("   avg_shap_importance.pkl")
        logger.info("   avg_permutation_importance.pkl")
        logger.info("   pdp_data.pkl")
        logger.info("   pdp_lowess_data.pkl")
        logger.info("   interaction_results.pkl")
        logger.info("   analysis_summary.json")
        
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