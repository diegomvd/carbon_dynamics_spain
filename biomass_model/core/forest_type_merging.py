"""
Forest Type Merging Pipeline for Biomass Estimation

This module provides forest type merging functionality for biomass raster datasets.
Merges individual forest type biomass raster files by year to create comprehensive
maps covering all forest types. Refactored from the original merge_forest_types.py
script to fit the biomass_model component architecture with centralized path constants.

Features:
- Merges forest type specific biomass maps into country-wide maps
- Processes AGBD, BGBD, and TBD separately for both mean and uncertainty estimates
- Optimized GeoTIFF compression and tiling for efficient storage
- Automatic year detection and grouping
- Robust error handling and logging
- Integration with centralized path management

Author: Diego Bengochea
"""

import os
import glob
import re
import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import tempfile

# Shared utilities
from shared_utils import get_logger, ensure_directory
from shared_utils.central_data_paths_constants import *


class ForestTypeMergingPipeline:
    """
    Pipeline for merging forest type specific biomass maps into country-wide maps.
    
    Combines individual forest type biomass raster files by year to create
    comprehensive maps covering all forest types. Handles AGBD, BGBD, and TBD
    (above-ground, below-ground, and total biomass density) separately for both
    mean and uncertainty estimates.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the forest type merging pipeline.
        
        Args:
            temp_dir: Optional custom temporary directory for intermediate files
        """
        # Store configuration and data paths
        self.config = load_config(config_path, component='biomass_estimation')

        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_estimation',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Create temp directory under biomass_maps
        self.temp_dir = BIOMASS_MAPS_RAW_DIR.parent / "temp" / "merging"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Target file extensions
        self.target_extensions = ['.tif', '.tiff']
        
        # Biomass type patterns
        self.biomass_patterns = self.config['processing']['merged_naming_patterns'] 
        
        self.logger.info(f"ForestTypeMergingPipeline initialized")
        self.logger.info(f"Temporary directory: {self.temp_dir}")
    
    def run_full_pipeline(self) -> bool:
        """
        Process a directory of forest type specific biomass rasters with merging.
        Returns:
            True if processing completed successfully, False otherwise
        """
        input_dir = BIOMASS_MAPS_PER_FOREST_TYPE_DIR
        output_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR

        # Create output directory structure
        ensure_directory(input_dir)
        ensure_directory(output_dir)
        mean_output_dir = output_dir / "mean"
        uncertainty_output_dir = output_dir / "uncertainty"
        ensure_directory(mean_output_dir)
        ensure_directory(uncertainty_output_dir)
        
        # Find biomass directories to process
        biomass_dirs = self._find_biomass_directories(input_dir, self.config['output']['types'])    
            
        if not biomass_dirs:
            self.logger.warning("No biomass directories found to process")
            return True
            
        self.logger.info(f"Found {len(biomass_dirs)} biomass directories to process")
            
        # Process each biomass directory
        success_count = 0
        error_count = 0
            
        for biomass_dir in biomass_dirs:
            try:
                self.logger.info(f"Processing directory: {biomass_dir}")
                
                # Auto-discover years if not specified
                years_to_process = self.config['processing']['target_years']
                measures_to_process = self.config['output']['measures']
                
                self.logger.debug(f"Years to process: {years_to_process}")
                self.logger.debug(f"Measures to process: {measures_to_process}")
                
                resolution = '100'
                compression = self.config['output']['geotiff']['compress']    

                # Process each year and measure combination
                for year in years_to_process:
                    for measure in measures_to_process:
                        try:
                            # Determine output directory based on measure
                            if measure == 'mean':
                                target_output_dir = mean_output_dir
                            elif measure == 'uncertainty':
                                target_output_dir = uncertainty_output_dir
                            else:
                                # For other measures, create subdirectory
                                target_output_dir = output_dir / measure
                                ensure_directory(target_output_dir)
                            
                            # Merge rasters for this year and measure
                            success = self._merge_rasters_by_year_and_type(
                                biomass_dir, year, target_output_dir, measure,
                                resolution, compression
                            )
                            
                            if success:
                                success_count += 1
                            else:
                                error_count += 1
                                if not continue_on_error:
                                    return False
                        
                        except Exception as e:
                            error_count += 1
                            self.logger.error(f"Error processing {biomass_dir} year {year} measure {measure}: {str(e)}")
                            if not continue_on_error:
                                raise
                
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing directory {biomass_dir}: {str(e)}")
                if not continue_on_error:
                    raise
        
        # Log completion
        self.logger.info(f"Merging completed: {success_count} succeeded, {error_count} failed")
        
        # Cleanup temporary files
        self._cleanup_temp_files()
        
        return error_count == 0 or continue_on_error
        

    
    def _find_biomass_directories(
        self,
        input_dir: Path,
        biomass_types: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Find biomass directories in input directory.
        
        Args:
            input_dir: Directory to search
            biomass_types: Optional biomass type filter
            
        Returns:
            List of biomass directory paths
        """
        biomass_dirs = []
        
        # Search for biomass type directories
        for pattern in self.biomass_patterns:
            dirs = list(input_dir.glob(pattern))
            biomass_dirs.extend([d for d in dirs if d.is_dir()])
        
        # Apply biomass type filter if specified
        if biomass_types:
            filtered_dirs = []
            for biomass_dir in biomass_dirs:
                dir_name = biomass_dir.name
                if any(btype in dir_name for btype in biomass_types):
                    filtered_dirs.append(biomass_dir)
            biomass_dirs = filtered_dirs
        
        return sorted(biomass_dirs)
    
    def _extract_year_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract the year from a biomass raster filename using regex pattern matching.
        
        Args:
            filename: Input filename containing year information
            
        Returns:
            Extracted 4-digit year string, or None if no year found
        """
        match = re.search(r'_(\d{4})_', filename)
        if match:
            return match.group(1)
        return None
    
    def _get_base_filename(self, filename: str) -> str:
        """
        Extract the base filename without the forest type code suffix.
        
        Removes the "_codeXX" or similar suffix from biomass raster filenames
        to get the common base name for merging files from different forest types.
        
        Args:
            filename: Input filename with forest type code
            
        Returns:
            Base filename without extension and code suffix
        """
        # Extract basename without path and extension
        basename = os.path.splitext(os.path.basename(filename))[0]
        
        # Remove various forest type code suffixes
        patterns_to_remove = [
            r'_code\d+$',           # _code12, _code23, etc.
            r'_genus_[^_]+$',       # _genus_Pinus, etc.
            r'_family_[^_]+$',      # _family_Pinaceae, etc.
            r'_clade_[^_]+$',       # _clade_Gymnosperm, etc.
        ]
        
        base_name = basename
        for pattern in patterns_to_remove:
            base_name = re.sub(pattern, '', base_name)
        
        return base_name
    
    def _merge_rasters_by_year_and_type(
        self,
        biomass_dir: Path,
        year: str,
        output_dir: Path,
        measure: str,
        resolution: str,
        compression: str,
        
    ) -> bool:
        """
        Merge all raster files for a specific year and statistical measure.
        
        Args:
            biomass_dir: Directory containing biomass rasters
            year: Target year for file filtering
            output_dir: Directory to save merged output file
            measure: Statistical measure (mean, uncertainty, etc.)
            resolution: Resolution string for output naming
            compression: Compression method for output
            
        Returns:
            True if merging succeeded, False otherwise
        """
        try:
            self.logger.debug(f"Processing {measure} files for year {year} in {biomass_dir}...")
            
            # Build search pattern for target files
            pattern = f"*_{measure}_{year}_*.tif"
            raster_files = list(biomass_dir.glob(pattern))
            
            if not raster_files:
                self.logger.debug(f"No {measure} raster files found for year {year} with pattern {pattern}")
                return True  # Not an error, just no files to process
            
            self.logger.debug(f"Found {len(raster_files)} {measure} raster files for year {year}")
            
            # Extract base filename and biomass type for output naming
            first_file = raster_files[0]
            base_name = self._get_base_filename(first_file.name)
            
            # Extract biomass type from directory name
            biomass_type = self._extract_biomass_type_from_dir(biomass_dir.name)
            
            # Generate output filename
            output_filename = f"{base_name}_{biomass_type}_merged_{resolution}m.tif"
            output_file = output_dir / output_filename
            
            # Perform merging
            success = self._merge_raster_files(
                raster_files, output_file, compression
            )
            
            if success:
                self.logger.debug(f"Merged {measure} raster saved to {output_filename}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error merging {measure} files for year {year}: {str(e)}")
            return False
    
    def _extract_biomass_type_from_dir(self, dir_name: str) -> str:
        """Extract biomass type from directory name."""
        if 'AGBD' in dir_name:
            return 'AGBD'
        elif 'BGBD' in dir_name:
            return 'BGBD'
        elif 'TBD' in dir_name:
            return 'TBD'
        else:
            # Fallback: extract first word
            return dir_name.split('_')[0]
    
    def _merge_raster_files(
        self,
        raster_files: List[Path],
        output_file: Path,
        compression: str
    ) -> bool:
        """
        Merge a list of raster files into a single output raster.
        
        Args:
            raster_files: List of input raster files
            output_file: Output merged raster file
            compression: Compression method
            
        Returns:
            True if merging succeeded, False otherwise
        """
        try:
            # Open all source raster files for merging
            src_files_to_mosaic = []
            for raster_file in raster_files:
                src = rasterio.open(raster_file)
                src_files_to_mosaic.append(src)
            
            # Perform raster merging operation
            mosaic, out_trans = merge(src_files_to_mosaic)
            
            # Copy metadata from first file and update for output
            out_meta = src_files_to_mosaic[0].meta.copy()
            
            # Configure optimized GeoTIFF output settings
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": compression,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            })
            
            # Ensure output directory exists
            ensure_directory(output_file.parent)
            
            # Write merged raster to output file
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Clean up source file handles
            for src in src_files_to_mosaic:
                src.close()
            
            # Log statistics
            total_pixels = mosaic.shape[1] * mosaic.shape[2]
            valid_pixels = np.sum(~np.isnan(mosaic[0])) if mosaic.dtype.kind == 'f' else total_pixels
            
            self.logger.debug(
                f"Merged {len(raster_files)} files into {output_file.name}: "
                f"{mosaic.shape[2]}x{mosaic.shape[1]} pixels, "
                f"{valid_pixels:,} valid pixels ({valid_pixels/total_pixels*100:.1f}%)"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging raster files: {str(e)}")
            return False
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            if self.temp_dir.exists():
                # Remove temporary files but keep directory
                for temp_file in self.temp_dir.glob("*"):
                    if temp_file.is_file():
                        temp_file.unlink()
                        self.logger.debug(f"Cleaned up temporary file: {temp_file.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary files: {str(e)}")
    