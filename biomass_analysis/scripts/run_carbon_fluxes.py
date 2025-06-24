#!/usr/bin/env python3
"""
Carbon flux analysis script.

Command-line interface for calculating interannual carbon fluxes from Monte Carlo
biomass samples. This is a thin wrapper around the core CarbonFluxAnalyzer class.

Usage:
    python run_carbon_fluxes.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.carbon_flux_analysis import CarbonFluxAnalyzer
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate interannual carbon fluxes from Monte Carlo biomass samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    return parser.parse_args()

def main():
    """Main entry point for carbon flux analysis script."""
    args = parse_arguments()
    
    # Initialize analyzer
    try:
        analyzer = CarbonFluxAnalyzer(args.config)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        sys.exit(1)
 
    analyzer.logger.setLevel('INFO')
  
    analyzer.logger.info("Starting interannual carbon flux analysis...")
    
    # Log configuration parameters
    n_combinations = analyzer.config['interannual']['carbon_fluxes']['n_combinations']
    biomass_to_carbon = analyzer.config['interannual']['carbon_fluxes']['biomass_to_carbon']
    create_diagnostics = analyzer.config['interannual']['carbon_fluxes']['create_diagnostics']
    
    analyzer.logger.info(f"Configuration:")
    analyzer.logger.info(f"  Random combinations: {n_combinations:,}")
    analyzer.logger.info(f"  Biomass to carbon factor: {biomass_to_carbon}")
    analyzer.logger.info(f"  Create diagnostic plots: {create_diagnostics}")
    
    
    # Run carbon flux analysis
    try:
        flux_df, flux_samples = analyzer.run_carbon_flux_analysis(
            create_diagnostics = False
        )
        
        if flux_df is None or flux_df.empty:
            analyzer.logger.error("No flux results calculated. Check previous errors.")
            sys.exit(1)
        
        # Save results
        output_file = analyzer.save_results(flux_df, flux_samples)
        
        # Print summary
        analyzer.logger.info(f"\n{'='*60}")
        analyzer.logger.info("CARBON FLUX ANALYSIS SUMMARY")
        analyzer.logger.info(f"{'='*60}")
        
        # Print detailed summary table
        analyzer.print_summary(flux_df)
        
        # Additional analysis summary
        analyzer.logger.info(f"\nAnalysis details:")
        analyzer.logger.info(f"  Number of year pairs: {len(flux_df)}")
        analyzer.logger.info(f"  Random combinations per pair: {n_combinations:,}")
        analyzer.logger.info(f"  Results saved to: {output_file}")
            
        # Interpretation note
        analyzer.logger.info(f"\nInterpretation:")
        analyzer.logger.info(f"  Positive flux = carbon source (biomass loss)")
        analyzer.logger.info(f"  Negative flux = carbon sink (biomass gain)")
        
        analyzer.logger.info("Carbon flux analysis complete!")
        
    except Exception as e:
        analyzer.logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
