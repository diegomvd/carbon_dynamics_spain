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

from climate_biomass_analysis.core.climate_raster_conversion import ClimateRasterConversionPipeline
from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculationPipeline
from climate_biomass_analysis.core.biomass_integration import BiomassIntegrationPipeline
from climate_biomass_analysis.core.spatial_analysis import SpatialAnalysisPipeline
from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline
from climate_biomass_analysis.core.shap_analysis import ShapAnalysisPipeline


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

    climate_raster_conversion = ClimateRasterConversionPipeline(args.config)
    success_conversion = ClimateRasterConversionPipeline.run_full_pipeline()

    bioclim_processing = BioclimCalculationPipeline(args.config)
    success_bioclim = bioclim_processing.run_full_pipeline()

    biomass_integration = BiomassIntegrationPipeline(args.config)
    success_integration = biomass_integration.run_full_pipeline()

    spatial_analysis = SpatialAnalysisPipeline(args.config)
    success_spatial = spatial_analysis.run_full_pipeline()

    optimization = OptimizationPipeline(args.config)
    success_optimization = optimization.run_full_pipeline()

    shap_interpretation = ShapAnalysisPipeline(args.config)
    success_shap = shap_interpretation.run_full_pipeline()

    success = all([success_conversion,success_bioclim,success_integration,success_spatial,success_optimization,success_shap])

    return success    

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)