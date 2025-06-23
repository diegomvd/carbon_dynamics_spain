#!/usr/bin/env python3
"""
Country-level biomass time series analysis script.

Command-line interface for country-level biomass analysis with Monte Carlo 
uncertainty quantification. This is a thin wrapper around the core 
MonteCarloAnalyzer class.

Usage:
    python run_country_trend.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.monte_carlo_analysis import MonteCarloAnalyzer
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Country-level biomass time series analysis with Monte Carlo uncertainty quantification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--biomass-types',
        nargs='+',
        default=None,
        help='Specific biomass types to process (default: all from config)'
    )
    
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=None,
        help='Specific years to process (default: all from config)'
    )
    
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip saving Monte Carlo samples (save only summary)'
    )
    
    # Processing control
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: from config)'
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
    
    # Override parallel processing settings
    if args.parallel:
        analyzer.config['monte_carlo']['parallel_processing']['enabled'] = True
        config_changed = True
    
    if args.workers:
        analyzer.config['monte_carlo']['parallel_processing']['num_workers'] = args.workers
        config_changed = True
    
    if args.no_samples:
        analyzer.config['output']['save_monte_carlo_samples'] = False
        config_changed = True
    
    if config_changed:
        analyzer.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


def main():
    """Main entry point for country trend analysis script."""
    args = parse_arguments()
    
    # Initialize analyzer
    try:
        analyzer = MonteCarloAnalyzer(args.config)
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
    
    analyzer.logger.info("Starting country-level biomass time series analysis...")
    
    # Determine biomass types and years to process
    biomass_types = args.biomass_types
    years = args.years
    
    if biomass_types:
        analyzer.logger.info(f"Processing specified biomass types: {biomass_types}")
    else:
        biomass_types = analyzer.config['file_patterns']['biomass_types']
        analyzer.logger.info(f"Processing all configured biomass types: {biomass_types}")
    
    if years:
        analyzer.logger.info(f"Processing specified years: {years}")
    else:
        years = analyzer.config['analysis']['target_years']
        analyzer.logger.info(f"Processing all configured years: {years}")
    
    # Run analysis
    try:
        results, year_samples = analyzer.run_country_analysis(biomass_types, years)
        
        if not results:
            analyzer.logger.error("No results calculated. Check previous errors.")
            sys.exit(1)
        
        # Save results
        summary_file, samples_file = analyzer.save_results(results, year_samples)
        
        # Print summary
        analyzer.logger.info(f"\n{'='*60}")
        analyzer.logger.info("COUNTRY ANALYSIS SUMMARY")
        analyzer.logger.info(f"{'='*60}")
        analyzer.logger.info(f"Total records processed: {len(results)}")
        analyzer.logger.info(f"Biomass types: {len(set(r['biomass_type'] for r in results))}")
        analyzer.logger.info(f"Years processed: {sorted(set(r['year'] for r in results))}")
        
        if summary_file:
            analyzer.logger.info(f"Summary results saved to: {summary_file}")
        
        if samples_file:
            analyzer.logger.info(f"Monte Carlo samples saved to: {samples_file}")
        
        # Show sample results
        analyzer.logger.info("\nSample results (first 6 records):")
        for result in results[:6]:
            analyzer.logger.info(f"  {result['year']} {result['biomass_type']}: "
                               f"{result['biomass_mean']:.2f} Mt "
                               f"({result['biomass_low']:.2f}-{result['biomass_high']:.2f})")
        
        analyzer.logger.info("Country-level biomass analysis complete!")
        
    except Exception as e:
        analyzer.logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
