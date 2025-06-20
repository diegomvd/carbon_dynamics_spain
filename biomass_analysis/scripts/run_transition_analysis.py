#!/usr/bin/env python3
"""
Biomass transition distribution analysis script.

Command-line interface for analyzing biomass transition distributions between
consecutive years. This is a thin wrapper around the core InterannualAnalyzer class.

Usage:
    python run_transition_analysis.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.interannual_analysis import InterannualAnalyzer
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze biomass transition distributions between consecutive years",
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
        '--years',
        nargs='+',
        type=int,
        default=None,
        help='Specific years to process (default: all from config)'
    )
    
    parser.add_argument(
        '--save-raw-data',
        action='store_true',
        help='Force saving of raw transition data for each year pair'
    )
    
    # Input directory
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Override input biomass directory from config'
    )
    
    # Output control
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    
    # Quality control
    parser.add_argument(
        '--max-biomass',
        type=float,
        help='Override maximum biomass threshold for quality control'
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
    
    # Override input directory
    if args.input_dir:
        analyzer.config['interannual']['differences']['input_biomass_dir'] = args.input_dir
        config_changed = True
    
    # Override output directory
    if args.output_dir:
        analyzer.config['output']['base_output_dir'] = args.output_dir
        config_changed = True
    
    # Override quality control threshold
    if args.max_biomass:
        analyzer.config['quality_control']['max_biomass_threshold'] = args.max_biomass
        config_changed = True
    
    if config_changed:
        analyzer.logger.info("Configuration overridden with command line arguments")
    
    return config_changed


def main():
    """Main entry point for transition analysis script."""
    args = parse_arguments()
    
    # Initialize analyzer
    try:
        analyzer = InterannualAnalyzer(args.config)
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
    
    # Determine target years
    target_years = args.years
    if target_years:
        analyzer.logger.info(f"Processing specific years: {target_years}")
    else:
        analyzer.logger.info("Processing all available years in input directory")
    
    analyzer.logger.info("Starting biomass transition distribution analysis...")
    
    # Log configuration
    max_biomass = analyzer.config['quality_control']['max_biomass_threshold']
    analyzer.logger.info(f"Maximum biomass threshold: {max_biomass} Mg/ha")
    
    if args.save_raw_data:
        analyzer.logger.info("Raw transition data will be saved for each year pair")
    
    # Run transition analysis
    try:
        results = analyzer.run_transition_analysis(target_years, args.save_raw_data)
        
        if not results:
            analyzer.logger.error("No transition results calculated. Check previous errors.")
            sys.exit(1)
        
        # Save results
        output_file = analyzer.save_results(results, 'transitions')
        
        # Print summary
        analyzer.logger.info(f"\n{'='*60}")
        analyzer.logger.info("TRANSITION ANALYSIS SUMMARY")
        analyzer.logger.info(f"{'='*60}")
        analyzer.logger.info(f"Total transition periods analyzed: {len(results)}")
        
        # Show year pairs processed
        periods = [r['period'] for r in results]
        analyzer.logger.info(f"Periods processed: {', '.join(periods)}")
        
        # Show statistics summary
        total_pixels = sum(r['n_valid_pixels'] for r in results)
        mean_raw_change = sum(r['mean_raw_diff'] for r in results) / len(results)
        mean_rel_change = sum(r['mean_rel_diff'] for r in results) / len(results)
        
        analyzer.logger.info(f"Total valid pixels processed: {total_pixels:,}")
        analyzer.logger.info(f"Average raw change: {mean_raw_change:.2f} Mg/ha")
        analyzer.logger.info(f"Average relative change: {mean_rel_change:.1f}%")
        
        # Show gain/loss summary
        total_gains = sum(r['n_gain_pixels'] for r in results)
        total_losses = sum(r['n_loss_pixels'] for r in results)
        total_stable = sum(r['n_stable_pixels'] for r in results)
        
        analyzer.logger.info(f"\nOverall transition summary:")
        analyzer.logger.info(f"  Biomass gains: {100*total_gains/total_pixels:.1f}% of pixels")
        analyzer.logger.info(f"  Biomass losses: {100*total_losses/total_pixels:.1f}% of pixels")
        analyzer.logger.info(f"  Stable biomass: {100*total_stable/total_pixels:.1f}% of pixels")
        
        if output_file:
            analyzer.logger.info(f"\nResults saved to: {output_file}")
        
        analyzer.logger.info("Biomass transition analysis complete!")
        
    except Exception as e:
        analyzer.logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
