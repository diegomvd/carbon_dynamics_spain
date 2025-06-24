#!/usr/bin/env python3
"""
Interannual biomass difference mapping script.

Command-line interface for creating interannual biomass difference maps between
consecutive years. This is a thin wrapper around the core InterannualAnalyzer class.

Usage:
    python run_interannual_differences.py 

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
        description="Create interannual biomass difference maps for consecutive years",
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
    """Main entry point for interannual differences script."""
    args = parse_arguments()
    
    # Initialize analyzer
    try:
        analyzer = InterannualAnalyzer(args.config)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        sys.exit(1)
    

    analyzer.logger.setLevel('INFO')
    
    analyzer.logger.info("Starting interannual biomass difference mapping...")
    
    # Run difference mapping analysis
    try:
        success = analyzer.run_difference_mapping(target_years)
        
        if not success:
            analyzer.logger.error("Difference mapping failed. Check previous errors.")
            sys.exit(1)
        
        analyzer.logger.info(f"\n{'='*60}")
        analyzer.logger.info("DIFFERENCE MAPPING SUMMARY")
        analyzer.logger.info(f"{'='*60}")
        analyzer.logger.info("Successfully created interannual difference maps")
        analyzer.logger.info("Two types of differences calculated:")
        analyzer.logger.info("  - Raw differences: year2 - year1 (Mg/ha)")
        analyzer.logger.info("  - Relative differences: 200*(year2-year1)/(year2+year1) (%)")
        

        # TODO: this needs correction!!
        # Show output directories
        base_dir = analyzer.config['data']['base_dir']
        raw_dir = analyzer.config['interannual']['differences']['output_raw_dir']
        rel_dir = analyzer.config['interannual']['differences']['output_relative_dir']
        
        analyzer.logger.info(f"\nOutput directories:")
        analyzer.logger.info(f"  Raw differences: {base_dir}/{raw_dir}")
        analyzer.logger.info(f"  Relative differences: {base_dir}/{rel_dir}")
        
        analyzer.logger.info("Interannual difference mapping complete!")
        
    except Exception as e:
        analyzer.logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
