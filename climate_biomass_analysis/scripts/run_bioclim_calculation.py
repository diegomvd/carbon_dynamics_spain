#!/usr/bin/env python3
"""
Bioclimatic variables calculation script.

Command-line interface for calculating bioclimatic variables (bio1-bio19) from
monthly temperature and precipitation data, and computing climate anomalies.

Usage:
    python run_bioclim_calculation.py [OPTIONS]

Author: Diego Bengochea
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculator
from shared_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate bioclimatic variables and anomalies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    # Processing options
    parser.add_argument(
        '--mode',
        choices=['reference', 'anomalies', 'both'],
        default='both',
        help='Processing mode: calculate reference period, anomalies, or both'
    )
    
    parser.add_argument(
        '--reference-start',
        type=int,
        help='Reference period start year (overrides config)'
    )
    
    parser.add_argument(
        '--reference-end',
        type=int,
        help='Reference period end year (overrides config)'
    )
    
    parser.add_argument(
        '--analysis-start',
        type=int,
        help='Analysis period start year (overrides config)'
    )
    
    parser.add_argument(
        '--analysis-end',
        type=int,
        help='Analysis period end year (overrides config)'
    )
    
    parser.add_argument(
        '--rolling-years',
        action='store_true',
        help='Use Sep-Aug rolling years instead of calendar years'
    )
    
    parser.add_argument(
        '--no-rolling-years',
        action='store_true',
        help='Use calendar years instead of Sep-Aug rolling years'
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing climate files (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for bioclimatic variables (overrides config)'
    )
    
    parser.add_argument(
        '--anomaly-dir',
        type=str,
        help='Output directory for anomalies (overrides config)'
    )
    
    # Variable selection
    parser.add_argument(
        '--variables',
        nargs='+',
        help='Specific bioclimatic variables to calculate (e.g., bio1 bio12)'
    )
    
    parser.add_argument(
        '--skip-harmonization',
        action='store_true',
        help='Skip raster harmonization step (assume already harmonized)'
    )
    
    # Quality control
    parser.add_argument(
        '--validate-inputs',
        action='store_true',
        help='Validate input files before processing'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing even if some years fail'
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


def validate_time_periods(args, calculator):
    """Validate and prepare time periods."""
    config = calculator.config
    
    # Reference period
    ref_start = args.reference_start or config['time_periods']['reference']['start_year']
    ref_end = args.reference_end or config['time_periods']['reference']['end_year']
    
    # Analysis period  
    analysis_start = args.analysis_start or config['time_periods']['analysis']['start_year']
    analysis_end = args.analysis_end or config['time_periods']['analysis']['end_year']
    
    # Rolling years setting
    if args.rolling_years:
        rolling = True
    elif args.no_rolling_years:
        rolling = False
    else:
        rolling = config['time_periods']['reference']['rolling']
    
    # Validation
    if ref_start >= ref_end:
        raise ValueError(f"Reference start year ({ref_start}) must be before end year ({ref_end})")
    
    if analysis_start >= analysis_end:
        raise ValueError(f"Analysis start year ({analysis_start}) must be before end year ({analysis_end})")
    
    return {
        'reference': {'start_year': ref_start, 'end_year': ref_end, 'rolling': rolling},
        'analysis': {'start_year': analysis_start, 'end_year': analysis_end, 'rolling': rolling}
    }


def main():
    """Main entry point for bioclimatic calculation script."""
    args = parse_arguments()
    
    # Setup logging
    if args.quiet:
        log_level = 'ERROR'
    else:
        log_level = args.log_level
    
    logger = setup_logging(level=log_level, component_name='bioclim_calculation_script')
    
    try:
        # Initialize calculator
        logger.info("Initializing bioclimatic calculator...")
        calculator = BioclimCalculator(config_path=args.config)
        
        # Validate and prepare time periods
        time_periods = validate_time_periods(args, calculator)
        logger.info(f"Reference period: {time_periods['reference']['start_year']}-{time_periods['reference']['end_year']}")
        logger.info(f"Analysis period: {time_periods['analysis']['start_year']}-{time_periods['analysis']['end_year']}")
        logger.info(f"Rolling years: {time_periods['reference']['rolling']}")
        
        # Override bioclimatic variables if specified
        if args.variables:
            calculator.bio_variables = args.variables
            logger.info(f"Processing specific variables: {args.variables}")
        
        # Get file paths
        if args.input_dir:
            import glob
            import os
            temp_files = glob.glob(os.path.join(args.input_dir, calculator.climate_config['temp_pattern']))
            precip_files = glob.glob(os.path.join(args.input_dir, calculator.climate_config['precip_pattern']))
        else:
            # Use config paths
            import glob
            import os
            data_dir = calculator.config['data']['climate_outputs']
            temp_files = glob.glob(os.path.join(data_dir, calculator.climate_config['temp_pattern']))
            precip_files = glob.glob(os.path.join(data_dir, calculator.climate_config['precip_pattern']))
        
        logger.info(f"Found {len(temp_files)} temperature files and {len(precip_files)} precipitation files")
        
        if not temp_files or not precip_files:
            raise ValueError("No temperature or precipitation files found")
        
        # Validate inputs if requested
        if args.validate_inputs:
            logger.info("Validating input files...")
            # Add validation logic here if needed
        
        # Step 1: Harmonize rasters (unless skipped)
        if not args.skip_harmonization:
            harmonized_dir = calculator.config['data']['harmonized_dir']
            harmonized_temp_dir = Path(harmonized_dir) / "temperature"
            harmonized_precip_dir = Path(harmonized_dir) / "precipitation"
            
            logger.info("Harmonizing raster files...")
            harmonized_temp_files = calculator.harmonize_rasters(temp_files, harmonized_temp_dir)
            harmonized_precip_files = calculator.harmonize_rasters(precip_files, harmonized_precip_dir)
        else:
            harmonized_temp_files = temp_files
            harmonized_precip_files = precip_files
            logger.info("Skipping harmonization step")
        
        # Step 2: Calculate reference bioclimatic variables (if requested)
        if args.mode in ['reference', 'both']:
            logger.info("Calculating reference bioclimatic variables...")
            
            output_dir = args.output_dir or calculator.config['data']['bioclim_dir']
            
            reference_bioclim = calculator.calculate_bioclim_variables(
                harmonized_temp_files,
                harmonized_precip_files,
                output_dir,
                start_year=time_periods['reference']['start_year'],
                end_year=time_periods['reference']['end_year'],
                rolling=time_periods['reference']['rolling']
            )
            
            if reference_bioclim:
                logger.info(f"✅ Reference bioclimatic variables calculated: {len(reference_bioclim)} variables")
            else:
                logger.error("❌ Failed to calculate reference bioclimatic variables")
                if not args.continue_on_error:
                    sys.exit(1)
        
        # Step 3: Calculate anomalies (if requested)
        if args.mode in ['anomalies', 'both']:
            logger.info("Calculating bioclimatic anomalies...")
            
            reference_dir = args.output_dir or calculator.config['data']['bioclim_dir']
            anomaly_dir = args.anomaly_dir or calculator.config['data']['anomaly_dir']
            
            yearly_anomalies = calculator.calculate_bioclim_anomalies(
                harmonized_temp_files,
                harmonized_precip_files,
                reference_dir,
                anomaly_dir,
                start_year=time_periods['analysis']['start_year'],
                end_year=time_periods['analysis']['end_year'],
                rolling=time_periods['analysis']['rolling']
            )
            
            if yearly_anomalies:
                total_anomalies = sum(len(year_data) for year_data in yearly_anomalies.values())
                logger.info(f"✅ Bioclimatic anomalies calculated: {len(yearly_anomalies)} years, "
                           f"{total_anomalies} total anomaly files")
            else:
                logger.error("❌ Failed to calculate bioclimatic anomalies")
                if not args.continue_on_error:
                    sys.exit(1)
        
        logger.info("Bioclimatic calculation completed successfully!")
        
    except Exception as e:
        logger.error(f"Bioclimatic calculation failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()