#!/usr/bin/env python3
"""
Complete biomass analysis pipeline orchestrator.

Integrates all biomass analysis components into a single coordinated pipeline
with smart stage detection, error recovery, and progress monitoring.

Usage:
    python run_full_analysis.py 

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.monte_carlo_analysis import MonteCarloAggregationPipeline
from biomass_analysis.core.aggregation_analysis import BiomassAggregationPipeline
from biomass_analysis.core.interannual_analysis import InterannualChangePipeline
from biomass_analysis.core.carbon_flux_analysis import CarbonFluxPipeline

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete biomass analysis pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
def main():
    """Main entry point for full analysis pipeline."""
    args = parse_arguments()
    
    biomass_aggregation = BiomassAggregationPipeline(args.config)
    success_agg = biomass_aggregation.run_full_pipeline()

    biomass_country_level = MonteCarloAggregationPipeline(args.config)
    success_country = biomass_country_level.run_full_pipeline()

    biomass_change = InterannualChangePipeline(args.config)
    
    success_change_maps = biomass_change.run_difference_mapping()
    success_transitions = biomass_change.run_transition_analysis(True)

    return all([success_agg,success_country,success_change_maps,success_transitions])    

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)