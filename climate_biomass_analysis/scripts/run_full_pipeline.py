#!/usr/bin/env python3
"""
Complete climate-biomass analysis pipeline orchestrator.

Integrates climate data processing, bioclimatic variable calculation, biomass
integration, spatial analysis, and machine learning optimization into a single
coordinated pipeline with smart checkpointing and error recovery.

Usage:
    python run_full_pipeline.py

Author: Diego Bengochea
"""

import argparse
import sys
import time
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.climate_raster_processing import ClimateProcessingPipeline
from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculationPipeline
from climate_biomass_analysis.core.biomass_integration import BiomassIntegrationPipeline
from climate_biomass_analysis.core.spatial_analysis import SpatialAnalysisPipeline
from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline
from shared_utils.central_data_paths_constants import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete climate-biomass analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    # Stage selection
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['climate_processing', 'bioclim_calculation', 'biomass_integration', 
                'spatial_analysis', 'optimization'],
        help='Specific stages to run (default: all stages)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for complete pipeline script."""
    args = parse_arguments()

    climate_processing = ClimateProcessingPipeline(args.config)
    success_climate = ClimateProcessingPipeline.run_full_pipeline()

    bioclim_processing = BioclimCalculationPipeline(args.config)
    success_bioclim = bioclim_processing.run_full_pipeline()

    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config_path=args.config, log_level=log_level)
        
        # Determine stages to run
        if args.stages:
            stages_to_run = args.stages
        else:
            all_stages = ['climate_processing', 'bioclim_calculation', 'biomass_integration', 
                         'spatial_analysis', 'optimization']
            stages_to_run = all_stages
        
        orchestrator.logger.info(f"Pipeline stages to execute: {', '.join(stages_to_run)}")
        
        # Run pipeline
        results = orchestrator.run_full_pipeline(
            stages=stages_to_run
        )
        
        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except Exception as e:
        print(f"Pipeline orchestrator failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()