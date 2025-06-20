#!/usr/bin/env python3
"""
Annual Cropland Masking Script

Entry point for masking annual cropland areas from biomass maps using
Corine Land Cover data.

Usage:
    python run_masking.py [OPTIONS]
    
Examples:
    # Mask all biomass maps with default config
    python run_masking.py
    
    # Custom configuration
    python run_masking.py --config masking_config.yaml
    
    # Specific input directory
    python run_masking.py --input-dir ./biomass_raw --output-dir ./biomass_masked
    
    # Specific land cover values to mask
    python run_masking.py --mask-values 12 13 14

Author: Diego Bengochea
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from shared_utils import setup_logging, load_config, log_pipeline_start, log_pipeline_end
from shared_utils import find_files, ensure_directory, validate_file_exists, validate_directory_exists

import rasterio
import numpy as np
from rasterio.enums import Resampling


class AnnualCroplandMasker:
    """
    Pipeline for masking annual cropland areas from biomass maps.
    
    Uses Corine Land Cover data to identify and mask agricultural areas
    from biomass estimation results.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize masking pipeline."""
        self.config = load_config(config_path, component_name="biomass_estimation")
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_masking'
        )
        
        # Masking configuration
        self.mask_config = self.config['masking']
        self.corine_file = self.config['data']['corine_land_cover']
        self.annual_crop_values = self.config['processing']['annual_crop_values']
        
        self.logger.info("AnnualCroplandMasker initialized")
    
    def validate_inputs(self, input_dir: str) -> bool:
        """
        Validate input data for masking.
        
        Args:
            input_dir: Input directory containing biomass maps
            
        Returns:
            bool: True if inputs are valid
        """
        self.logger.info("Validating masking inputs...")
        
        try:
            # Validate Corine land cover file
            validate_file_exists(self.corine_file, "Corine Land Cover file")
            
            # Validate input directory
            validate_directory_exists(input_dir, "Input biomass directory")
            
            # Check for biomass files
            biomass_files = find_files(
                input_dir, 
                self.mask_config['target_extensions'][0],  # Use first extension
                recursive=self.mask_config['recursive_processing']
            )
            
            if not biomass_files:
                self.logger.error(f"No biomass files found in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(biomass_files)} biomass files to process")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {str(e)}")
            return False
    
    def process_raster_file(
        self, 
        raster_file: Path, 
        output_file: Path,
        mask_values: List[int]
    ) -> bool:
        """
        Mask a single raster file.
        
        Args:
            raster_file: Input raster file
            output_file: Output masked file
            mask_values: Land cover values to mask
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            self.logger.debug(f"Processing: {raster_file.name}")
            
            # Read target raster
            with rasterio.open(raster_file) as target_src:
                target_data = target_src.read(1).astype(np.float32)
                target_meta = target_src.meta.copy()
                target_nodata = target_src.nodata
            
            # Read and reproject Corine data if needed
            with rasterio.open(self.corine_file) as mask_src:
                # Check if reprojection/resampling is needed
                if (mask_src.crs != target_src.crs or 
                    mask_src.res != target_src.res or
                    mask_src.bounds != target_src.bounds):
                    
                    # Reproject/resample mask to target
                    from rasterio.warp import reproject
                    
                    mask_data = np.empty(target_data.shape, dtype=mask_src.dtypes[0])
                    
                    reproject(
                        source=rasterio.band(mask_src, 1),
                        destination=mask_data,
                        src_transform=mask_src.transform,
                        src_crs=mask_src.crs,
                        dst_transform=target_src.transform,
                        dst_crs=target_src.crs,
                        dst_nodata=0,
                        resampling=getattr(Resampling, self.mask_config['corine_resampling_method'])
                    )
                else:
                    mask_data = mask_src.read(1)
            
            # Create boolean mask for annual crop values
            combined_mask = np.isin(mask_data, mask_values)
            
            # Apply mask to biomass data
            masked_data = target_data.copy()
            if target_nodata is not None:
                nodata_value = target_nodata
            else:
                nodata_value = self.config['output']['geotiff']['nodata_value']
            
            masked_data[combined_mask] = nodata_value
            
            # Update metadata
            target_meta.update({
                'nodata': nodata_value,
                'compress': self.config['output']['geotiff']['compress']
            })
            
            # Write masked result
            ensure_directory(output_file.parent)
            with rasterio.open(output_file, 'w', **target_meta) as dst:
                dst.write(masked_data, 1)
                
                # Add metadata tags
                dst.update_tags(
                    MASKED_VALUES=','.join(map(str, mask_values)),
                    MASK_SOURCE=str(Path(self.corine_file).name),
                    PROCESSING='annual_cropland_masked'
                )
            
            self.logger.debug(f"Masked raster saved to: {output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {raster_file.name}: {str(e)}")
            return False
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        mask_values: Optional[List[int]] = None
    ) -> bool:
        """
        Process all biomass files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            mask_values: Values to mask (uses config default if None)
            
        Returns:
            bool: True if processing succeeded
        """
        if mask_values is None:
            mask_values = self.annual_crop_values
        
        self.logger.info(f"Processing directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Masking land cover values: {mask_values}")
        
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            # Ensure output directory exists
            ensure_directory(output_path)
            
            # Find biomass files
            biomass_files = []
            for pattern in self.mask_config['biomass_patterns']:
                pattern_files = find_files(
                    input_path, 
                    pattern,
                    recursive=self.mask_config['recursive_processing']
                )
                biomass_files.extend(pattern_files)
            
            if not biomass_files:
                self.logger.warning(f"No biomass files found matching patterns in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(biomass_files)} biomass files to process")
            
            # Process each file
            successful = 0
            failed = 0
            
            for biomass_file in biomass_files:
                # Calculate relative path to maintain directory structure
                if self.mask_config['recursive_processing']:
                    rel_path = biomass_file.relative_to(input_path)
                    output_file = output_path / rel_path
                else:
                    output_file = output_path / biomass_file.name
                
                # Process file
                if self.process_raster_file(biomass_file, output_file, mask_values):
                    successful += 1
                else:
                    failed += 1
            
            self.logger.info(f"Processing complete: {successful} successful, {failed} failed")
            return successful > 0
            
        except Exception as e:
            self.logger.error(f"Error processing directory: {str(e)}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Annual Cropland Masking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Mask with default config
  %(prog)s --input-dir ./biomass_raw         # Custom input directory
  %(prog)s --mask-values 12 13 14 15         # Custom mask values
  %(prog)s --validate-only                   # Validate inputs only
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
        help='Input directory containing biomass maps'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for masked biomass maps'
    )
    
    parser.add_argument(
        '--mask-values',
        type=int,
        nargs='+',
        help='Corine land cover values to mask (default: from config)'
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
        component_name='biomass_masking',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Annual Cropland Masking Pipeline...")
        masker = AnnualCroplandMasker(config_path=args.config)
        
        # Log pipeline start
        log_pipeline_start(logger, "Annual Cropland Masking", masker.config)
        
        # Determine input/output directories
        if args.input_dir:
            input_dir = args.input_dir
        else:
            # Use default from config
            base_dir = masker.config['data']['output_base_dir']
            input_dir = os.path.join(base_dir, masker.config['data']['biomass_no_masking_dir'])
        
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Use default from config
            base_dir = masker.config['data']['output_base_dir']
            output_dir = os.path.join(base_dir, masker.config['data']['biomass_with_mask_dir'])
        
        # Validate inputs
        if not masker.validate_inputs(input_dir):
            logger.error("Input validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("✅ Validation successful - exiting")
            sys.exit(0)
        
        # Run masking
        success = masker.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            mask_values=args.mask_values
        )
        
        # Pipeline completion
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Annual Cropland Masking", success, elapsed_time)
        
        if success:
            logger.info("🎉 Masking completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Masking failed!")
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
