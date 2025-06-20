#!/usr/bin/env python3
"""
Forest Type Merging Script

Entry point for merging forest type specific biomass maps into 
country-wide merged rasters.

Usage:
    python run_merging.py [OPTIONS]
    
Examples:
    # Merge all forest types with default config
    python run_merging.py
    
    # Custom input/output directories
    python run_merging.py --input-dir ./biomass_masked --output-dir ./biomass_merged
    
    # Specific years only
    python run_merging.py --years 2020 2021 2022
    
    # Specific biomass types
    python run_merging.py --biomass-types AGBD TBD

Author: Diego Bengochea
"""

import sys
import os
import glob
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end
from shared_utils import find_files, ensure_directory, validate_directory_exists

import rasterio
import numpy as np
from rasterio.merge import merge


class ForestTypeMerger:
    """
    Pipeline for merging forest type specific biomass maps.
    
    Combines individual forest type rasters into country-wide merged maps
    for each year and biomass type.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize merging pipeline."""
        self.config = load_config(config_path, component_name="biomass_estimation")
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_merging'
        )
        
        self.logger.info("ForestTypeMerger initialized")
    
    def extract_year_from_filename(self, filepath: Path) -> Optional[int]:
        """Extract year from filename."""
        filename = filepath.stem
        
        # Common year patterns
        patterns = [
            r'(\d{4})',  # Any 4-digit number
            r'_(\d{4})_',  # Year surrounded by underscores
            r'(\d{4})_',  # Year followed by underscore
            r'_(\d{4})',  # Year preceded by underscore
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            for match in matches:
                year = int(match)
                if 2000 <= year <= 2030:  # Reasonable year range
                    return year
        
        return None
    
    def extract_file_metadata(self, filepath: Path) -> Optional[Dict[str, str]]:
        """
        Extract metadata from biomass filename.
        
        Expected format: BIOMASS_TYPE_S2_MEASURE_YEAR_100m_codeXX.tif
        """
        filename = filepath.stem
        
        try:
            # Split filename components
            parts = filename.split('_')
            
            if len(parts) < 5:
                return None
            
            metadata = {
                'biomass_type': parts[0],  # AGBD, BGBD, TBD
                'prefix': parts[1],        # S2
                'measure': parts[2],       # mean, uncertainty
                'year': parts[3],          # 2020, 2021, etc.
                'resolution': parts[4],    # 100m
                'forest_type_code': None
            }
            
            # Extract forest type code
            for part in parts:
                if part.startswith('code'):
                    metadata['forest_type_code'] = part.replace('code', '')
                    break
            
            # Validate year
            try:
                year = int(metadata['year'])
                if 2000 <= year <= 2030:
                    metadata['year'] = year
                else:
                    return None
            except ValueError:
                return None
            
            return metadata
            
        except Exception as e:
            self.logger.debug(f"Could not parse filename {filename}: {e}")
            return None
    
    def find_files_for_merging(
        self, 
        input_dir: str,
        year: int,
        biomass_type: str,
        measure: str
    ) -> List[Path]:
        """
        Find files for merging by year, biomass type, and measure.
        
        Args:
            input_dir: Input directory
            year: Year to find files for
            biomass_type: Biomass type (AGBD, BGBD, TBD)
            measure: Statistical measure (mean, uncertainty)
            
        Returns:
            List of matching files
        """
        input_path = Path(input_dir)
        
        # Search pattern
        pattern = f"{biomass_type}_*_{measure}_{year}_*.tif"
        
        matching_files = []
        
        # Search recursively if configured
        if self.config['masking']['recursive_processing']:
            files = find_files(input_path, pattern, recursive=True)
        else:
            files = find_files(input_path, pattern, recursive=False)
        
        # Filter and validate files
        for file in files:
            metadata = self.extract_file_metadata(file)
            if (metadata and 
                metadata['biomass_type'] == biomass_type and
                metadata['measure'] == measure and
                metadata['year'] == year):
                matching_files.append(file)
        
        self.logger.debug(f"Found {len(matching_files)} files for {biomass_type} {measure} {year}")
        return sorted(matching_files)
    
    def merge_rasters_by_year_and_type(
        self,
        input_dir: str,
        year: int,
        biomass_type: str,
        measure: str,
        output_dir: str
    ) -> bool:
        """
        Merge rasters for specific year, biomass type, and measure.
        
        Args:
            input_dir: Input directory containing forest type rasters
            year: Year to process
            biomass_type: Biomass type
            measure: Statistical measure
            output_dir: Output directory
            
        Returns:
            bool: True if merging succeeded
        """
        try:
            # Find files to merge
            files_to_merge = self.find_files_for_merging(
                input_dir, year, biomass_type, measure
            )
            
            if not files_to_merge:
                self.logger.warning(f"No files found for {biomass_type} {measure} {year}")
                return False
            
            self.logger.info(f"Merging {len(files_to_merge)} files for {biomass_type} {measure} {year}")
            
            # Generate output filename
            prefix = self.config['output']['prefix']
            output_filename = f"{biomass_type}_{prefix}_{measure}_{year}_100m_merged.tif"
            output_file = Path(output_dir) / output_filename
            
            # Skip if output already exists
            if output_file.exists():
                self.logger.info(f"Output file already exists, skipping: {output_filename}")
                return True
            
            # Open source files
            src_files_to_mosaic = []
            for file_path in files_to_merge:
                src = rasterio.open(file_path)
                src_files_to_mosaic.append(src)
            
            # Merge rasters
            mosaic, out_trans = merge(
                src_files_to_mosaic,
                nodata=self.config['output']['geotiff']['nodata_value']
            )
            
            # Get metadata from first file
            template_file = src_files_to_mosaic[0]
            out_meta = template_file.meta.copy()
            
            # Update metadata
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2], 
                "transform": out_trans,
                "compress": self.config['output']['geotiff']['compress'],
                "tiled": self.config['output']['geotiff']['tiled'],
                "blockxsize": self.config['output']['geotiff']['blockxsize'],
                "blockysize": self.config['output']['geotiff']['blockysize'],
                "nodata": self.config['output']['geotiff']['nodata_value']
            })
            
            # Create output directory
            ensure_directory(output_file.parent)
            
            # Write merged raster
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
                
                # Add metadata tags
                dest.update_tags(
                    MERGED_FILES=len(files_to_merge),
                    BIOMASS_TYPE=biomass_type,
                    MEASURE=measure,
                    YEAR=str(year),
                    PROCESSING='forest_types_merged'
                )
            
            # Clean up source file handles
            for src in src_files_to_mosaic:
                src.close()
            
            self.logger.info(f"Merged raster saved to {output_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging {biomass_type} {measure} {year}: {str(e)}")
            return False
    
    def process_all_combinations(
        self,
        input_dir: str,
        output_dir: str,
        years: Optional[List[int]] = None,
        biomass_types: Optional[List[str]] = None,
        measures: Optional[List[str]] = None
    ) -> bool:
        """
        Process all combinations of years, biomass types, and measures.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            years: Years to process (uses config default if None)
            biomass_types: Biomass types to process (uses config default if None)
            measures: Measures to process (uses config default if None)
            
        Returns:
            bool: True if processing succeeded
        """
        # Use defaults from config if not specified
        if years is None:
            years = self.config['processing']['target_years']
        if biomass_types is None:
            biomass_types = [bt.upper() for bt in self.config['output']['types']]
        if measures is None:
            measures = self.config['output']['measures']
        
        self.logger.info(f"Processing combinations:")
        self.logger.info(f"  Years: {years}")
        self.logger.info(f"  Biomass types: {biomass_types}")
        self.logger.info(f"  Measures: {measures}")
        
        total_combinations = len(years) * len(biomass_types) * len(measures)
        successful = 0
        failed = 0
        
        # Process each combination
        for year in years:
            for biomass_type in biomass_types:
                for measure in measures:
                    self.logger.info(f"Processing: {biomass_type} {measure} {year}")
                    
                    if self.merge_rasters_by_year_and_type(
                        input_dir, year, biomass_type, measure, output_dir
                    ):
                        successful += 1
                    else:
                        failed += 1
        
        self.logger.info(f"Merging complete: {successful}/{total_combinations} successful")
        return successful > 0
    
    def validate_inputs(self, input_dir: str) -> bool:
        """Validate input directory and files."""
        try:
            validate_directory_exists(input_dir, "Input biomass directory")
            
            # Check for biomass files
            biomass_files = find_files(input_dir, "*.tif", recursive=True)
            
            if not biomass_files:
                self.logger.error(f"No biomass files found in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(biomass_files)} biomass files for merging")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {str(e)}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Forest Type Merging Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Merge with default config
  %(prog)s --input-dir ./biomass_masked      # Custom input directory
  %(prog)s --years 2020 2021                 # Specific years only
  %(prog)s --biomass-types AGBD TBD          # Specific biomass types
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing forest type biomass maps'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for merged biomass maps'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process (default: from config)'
    )
    
    parser.add_argument(
        '--biomass-types',
        type=str,
        nargs='+',
        choices=['AGBD', 'BGBD', 'TBD'],
        help='Biomass types to process (default: from config)'
    )
    
    parser.add_argument(
        '--measures',
        type=str,
        nargs='+',
        choices=['mean', 'uncertainty'],
        help='Statistical measures to process (default: from config)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without processing'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        component_name='biomass_merging',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Forest Type Merging Pipeline...")
        merger = ForestTypeMerger(config_path=args.config)
        
        # Log pipeline start
        log_pipeline_start(logger, "Forest Type Merging", merger.config)
        
        # Determine input/output directories
        if args.input_dir:
            input_dir = args.input_dir
        else:
            # Use default from config
            base_dir = merger.config['data']['output_base_dir']
            input_dir = os.path.join(base_dir, merger.config['data']['biomass_with_mask_dir'])
        
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Use default merged directory
            base_dir = merger.config['data']['output_base_dir']
            output_dir = os.path.join(base_dir, 'biomass_maps_merged')
        
        # Validate inputs
        if not merger.validate_inputs(input_dir):
            logger.error("Input validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("✅ Validation successful - exiting")
            sys.exit(0)
        
        # Run merging
        success = merger.process_all_combinations(
            input_dir=input_dir,
            output_dir=output_dir,
            years=args.years,
            biomass_types=args.biomass_types,
            measures=args.measures
        )
        
        # Pipeline completion
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Forest Type Merging", success, elapsed_time)
        
        if success:
            logger.info("🎉 Merging completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Merging failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
