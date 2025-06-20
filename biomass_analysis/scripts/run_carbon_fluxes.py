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
    
    # Input file
    parser.add_argument(
        '--mc-samples',
        type=str,
        default=None,
        help='Path to Monte Carlo samples NPZ file (auto-detect if not provided)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--n-combinations',
        type=int,
        help='Number of random combinations for flux calculations (override config)'
    )
    
    parser.add_argument(
        '--biomass-to-carbon',
        type=float,
        help='Biomass to carbon conversion factor (override config)'
    )
    
    # Output control
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip diagnostic plot creation'
    )
    
    parser.add_argument(
        '--save-samples',
        action='store_true',
        help='Force saving of raw flux samples'
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
    
    # Override analysis parameters
    if args.n_combinations:
        analyzer.config['interannual']['carbon_fluxes']['n_combinations'] = args.n_combinations
        config_changed = True
    
    if args.biomass_to_carbon:
        analyzer.config['interannual']['carbon_fluxes']['biomass_to_carbon'] = args.biomass_to_carbon
        config_changed = True
    
    # Override output settings
    if args.output_dir:
        analyzer.config['output']['base_output_dir'] = args.output_dir
        config_changed = True
    
    if args.no_plots:
        analyzer.config['interannual']['carbon_fluxes']['create_diagnostics'] = False
        config_changed = True
    
    if args.save_samples:
        analyzer.config['output']['save_intermediate_results'] = True
        config_changed = True
    
    if config_changed:
        analyzer.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


def main():
    """Main entry point for carbon flux analysis script."""
    args = parse_arguments()
    
    # Initialize analyzer
    try:
        analyzer = CarbonFluxAnalyzer(args.config)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    override_config(analyzer, args)
    
    # Set logging level
    if args.quiet:
        analyzer.logger.setLevel('ERROR')
    else:
        analyzer.logger.setLevel(args.log_level)
    
    analyzer.logger.info("Starting interannual carbon flux analysis...")
    
    # Log configuration parameters
    n_combinations = analyzer.config['interannual']['carbon_fluxes']['n_combinations']
    biomass_to_carbon = analyzer.config['interannual']['carbon_fluxes']['biomass_to_carbon']
    create_diagnostics = analyzer.config['interannual']['carbon_fluxes']['create_diagnostics']
    
    analyzer.logger.info(f"Configuration:")
    analyzer.logger.info(f"  Random combinations: {n_combinations:,}")
    analyzer.logger.info(f"  Biomass to carbon factor: {biomass_to_carbon}")
    analyzer.logger.info(f"  Create diagnostic plots: {create_diagnostics}")
    
    if args.mc_samples:
        analyzer.logger.info(f"  Using MC samples file: {args.mc_samples}")
    else:
        analyzer.logger.info("  Auto-detecting most recent MC samples file")
    
    # Run carbon flux analysis
    try:
        flux_df, flux_samples = analyzer.run_carbon_flux_analysis(
            mc_file_path=args.mc_samples,
            create_diagnostics=not args.no_plots
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
        
        if create_diagnostics and not args.no_plots:
            analyzer.logger.info("  Diagnostic plots created")
        
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
