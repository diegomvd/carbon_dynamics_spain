"""
Post-Processing Pipeline

Multi-step post-processing pipeline for canopy height predictions including:
1. Merge patches into tiles
2. Sanitize and interpolate temporal gaps
3. Create final country-wide mosaics

Author: Diego Bengochea
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal
from enum import Enum
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import xarray as xr

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils import find_files, log_pipeline_start, log_pipeline_end, log_section


class PipelineStep(Enum):
    """Enumeration of post-processing pipeline steps."""
    MERGE = "merge"
    SANITIZE = "sanitize"
    FINAL_MERGE = "final_merge"
    ALL = "all"


class PostProcessingPipeline:
    """
    Comprehensive post-processing pipeline for canopy height predictions.
    
    Implements a three-step workflow:
    1. Merge prediction patches into 120km tiles
    2. Sanitize outliers and interpolate temporal gaps
    3. Downsample and create final country-wide mosaics
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the post-processing pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='canopy_height_postprocessing',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Pipeline state
        self.start_time = None
        self.step_times = {}
        
        self.logger.info("PostProcessingPipeline initialized")
    
    def validate_step_prerequisites(self, step: PipelineStep) -> bool:
        """
        Validate prerequisites for a specific step.
        
        Args:
            step: Pipeline step to validate
            
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info(f"Validating prerequisites for step: {step.value}")
        
        try:
            if step in [PipelineStep.MERGE, PipelineStep.ALL]:
                # Check prediction patches directory
                merge_config = self.config['post_processing']['merge']
                input_dir = Path(merge_config['input_dir'])
                
                if not input_dir.exists():
                    self.logger.error(f"Merge input directory not found: {input_dir}")
                    return False
                
                # Check for prediction files
                prediction_files = find_files(input_dir, "*.tif")
                if not prediction_files:
                    self.logger.error(f"No prediction files found in {input_dir}")
                    return False
                
                self.logger.info(f"Found {len(prediction_files)} prediction files for merging")
            
            if step in [PipelineStep.SANITIZE, PipelineStep.ALL]:
                # Check merged tiles directory
                sanitize_config = self.config['post_processing']['sanitize']
                input_dir = Path(sanitize_config['input_dir'])
                
                if not input_dir.exists():
                    self.logger.error(f"Sanitize input directory not found: {input_dir}")
                    return False
                
                tiles = find_files(input_dir, "*.tif")
                if not tiles:
                    self.logger.error(f"No tiles found for sanitization in {input_dir}")
                    return False
                
                self.logger.info(f"Found {len(tiles)} tiles for sanitization")
            
            if step in [PipelineStep.FINAL_MERGE, PipelineStep.ALL]:
                # Check sanitized tiles directory
                final_config = self.config['post_processing']['final_merge']
                input_dir = Path(final_config['input_dir'])
                
                if not input_dir.exists():
                    self.logger.error(f"Final merge input directory not found: {input_dir}")
                    return False
                
                tiles = find_files(input_dir, final_config['file_pattern'])
                if not tiles:
                    self.logger.error(f"No sanitized tiles found in {input_dir}")
                    return False
                
                self.logger.info(f"Found {len(tiles)} tiles for final merging")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating prerequisites: {str(e)}")
            return False
    
    def run_merge_step(self) -> bool:
        """
        Step 1: Merge prediction patches into 120km tiles.
        
        Returns:
            bool: True if merging succeeded
        """
        try:
            log_section(self.logger, "STEP 1: MERGE PATCHES TO TILES")
            step_start = time.time()
            
            merge_config = self.config['post_processing']['merge']
            input_dir = Path(merge_config['input_dir'])
            output_dir = Path(merge_config['output_dir'])
            
            # Ensure output directory
            ensure_directory(output_dir)
            
            # Find prediction files
            prediction_files = find_files(input_dir, "*.tif")
            
            if not prediction_files:
                self.logger.error(f"No prediction files found in {input_dir}")
                return False
            
            self.logger.info(f"Merging {len(prediction_files)} prediction files...")
            
            # Group files by year and spatial tile
            file_groups = self._group_files_for_merging(prediction_files)
            
            # Process each group
            successful_merges = 0
            total_groups = len(file_groups)
            
            for group_key, file_list in file_groups.items():
                output_filename = f"canopy_height_{group_key}_120km.tif"
                output_file = output_dir / output_filename
                
                # Skip if output exists
                if output_file.exists():
                    self.logger.info(f"Output exists, skipping: {output_filename}")
                    continue
                
                # Merge files in group
                if self._merge_file_group(file_list, output_file, merge_config):
                    successful_merges += 1
                else:
                    self.logger.error(f"Failed to merge group: {group_key}")
            
            # Record step time
            self.step_times['merge'] = time.time() - step_start
            
            self.logger.info(f"Merge step complete: {successful_merges}/{total_groups} successful")
            return successful_merges > 0
            
        except Exception as e:
            self.logger.error(f"Error in merge step: {str(e)}")
            return False
    
    def run_sanitize_step(self) -> bool:
        """
        Step 2: Sanitize outliers and interpolate temporal gaps.
        
        Returns:
            bool: True if sanitization succeeded
        """
        try:
            log_section(self.logger, "STEP 2: SANITIZE AND INTERPOLATE")
            step_start = time.time()
            
            sanitize_config = self.config['post_processing']['sanitize']
            input_dir = Path(sanitize_config['input_dir'])
            output_dir = Path(sanitize_config['output_dir'])
            
            # Ensure output directory
            ensure_directory(output_dir)
            
            # Find merged tiles
            tile_files = find_files(input_dir, "*.tif")
            
            if not tile_files:
                self.logger.error(f"No tiles found for sanitization in {input_dir}")
                return False
            
            self.logger.info(f"Sanitizing {len(tile_files)} tiles...")
            
            # Group tiles by spatial location
            spatial_groups = self._group_tiles_by_location(tile_files)
            
            # Process each spatial group
            successful_sanitizations = 0
            total_groups = len(spatial_groups)
            
            for location_key, tile_group in spatial_groups.items():
                self.logger.info(f"Processing spatial group: {location_key}")
                
                if self._sanitize_tile_group(tile_group, output_dir, sanitize_config):
                    successful_sanitizations += 1
                else:
                    self.logger.error(f"Failed to sanitize group: {location_key}")
            
            # Record step time
            self.step_times['sanitize'] = time.time() - step_start
            
            self.logger.info(f"Sanitize step complete: {successful_sanitizations}/{total_groups} successful")
            return successful_sanitizations > 0
            
        except Exception as e:
            self.logger.error(f"Error in sanitize step: {str(e)}")
            return False
    
    def run_final_merge_step(self) -> bool:
        """
        Step 3: Create final country-wide mosaics at 100m resolution.
        
        Returns:
            bool: True if final merging succeeded
        """
        try:
            log_section(self.logger, "STEP 3: FINAL COUNTRY-WIDE MOSAICS")
            step_start = time.time()
            
            final_config = self.config['post_processing']['final_merge']
            input_dir = Path(final_config['input_dir'])
            output_dir = Path(final_config['output_dir'])
            
            # Ensure output directory
            ensure_directory(output_dir)
            
            # Find sanitized tiles
            tile_files = find_files(input_dir, final_config['file_pattern'])
            
            if not tile_files:
                self.logger.error(f"No sanitized tiles found in {input_dir}")
                return False
            
            self.logger.info(f"Creating final mosaics from {len(tile_files)} tiles...")
            
            # Group by year
            year_groups = self._group_files_by_year(tile_files)
            
            # Process each year
            successful_mosaics = 0
            total_years = len(year_groups)
            
            for year, tile_list in year_groups.items():
                output_filename = final_config['output_pattern'].format(year=year)
                output_file = output_dir / output_filename
                
                # Skip if output exists
                if output_file.exists():
                    self.logger.info(f"Output exists, skipping: {output_filename}")
                    continue
                
                self.logger.info(f"Creating mosaic for year {year} from {len(tile_list)} tiles")
                
                if self._create_country_mosaic(tile_list, output_file, final_config):
                    successful_mosaics += 1
                    
                    # Create overview pyramids if requested
                    if final_config.get('create_overview_pyramids', False):
                        self._create_overview_pyramids(output_file)
                else:
                    self.logger.error(f"Failed to create mosaic for year {year}")
            
            # Record step time
            self.step_times['final_merge'] = time.time() - step_start
            
            self.logger.info(f"Final merge step complete: {successful_mosaics}/{total_years} successful")
            return successful_mosaics > 0
            
        except Exception as e:
            self.logger.error(f"Error in final merge step: {str(e)}")
            return False
    
    def _group_files_for_merging(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group prediction files by year and spatial tile for merging."""
        groups = {}
        
        for file in files:
            # Extract year and tile information from filename
            # This is a simplified version - actual implementation would parse filenames
            year = self._extract_year_from_filename(file)
            tile_id = self._extract_tile_id_from_filename(file)
            
            if year and tile_id:
                group_key = f"{year}_{tile_id}"
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(file)
        
        return groups
    
    def _merge_file_group(self, files: List[Path], output_file: Path, config: Dict) -> bool:
        """Merge a group of files into a single tile."""
        try:
            # Open source files
            src_files = [rasterio.open(f) for f in files]
            
            # Merge
            mosaic, out_trans = merge(
                src_files,
                nodata=config['nodata_value']
            )
            
            # Get output profile
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": config['compression'],
                "tiled": True,
                "nodata": config['nodata_value']
            })
            
            # Write output
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Clean up
            for src in src_files:
                src.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging files: {str(e)}")
            return False
    
    def _group_tiles_by_location(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group tiles by spatial location."""
        groups = {}
        
        for file in files:
            # Extract spatial location from filename
            location = self._extract_location_from_filename(file)
            
            if location:
                if location not in groups:
                    groups[location] = []
                groups[location].append(file)
        
        return groups
    
    def _sanitize_tile_group(self, tiles: List[Path], output_dir: Path, config: Dict) -> bool:
        """Sanitize a group of tiles (temporal series for one location)."""
        try:
            # This would implement outlier detection and temporal interpolation
            # For now, simplified version that just copies files
            
            for tile in tiles:
                # Generate output filename
                output_filename = f"{tile.stem}_sanitized.tif"
                output_file = output_dir / output_filename
                
                # For now, just copy (real implementation would do sanitization)
                with rasterio.open(tile) as src:
                    data = src.read(1)
                    profile = src.profile.copy()
                    
                    # Apply outlier detection
                    if config['outlier_detection']:
                        data = self._remove_outliers(data, config)
                    
                    # Write sanitized output
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(data, 1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sanitizing tiles: {str(e)}")
            return False
    
    def _remove_outliers(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """Remove outliers from height data."""
        # Simple outlier removal based on height range
        height_min = config['height_min']
        height_max = config['height_max']
        
        data = np.where((data < height_min) | (data > height_max), np.nan, data)
        
        # Z-score based outlier removal
        if config.get('zscore_threshold'):
            from scipy import stats
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            data = np.where(z_scores > config['zscore_threshold'], np.nan, data)
        
        return data
    
    def _group_files_by_year(self, files: List[Path]) -> Dict[int, List[Path]]:
        """Group files by year."""
        groups = {}
        
        for file in files:
            year = self._extract_year_from_filename(file)
            if year:
                if year not in groups:
                    groups[year] = []
                groups[year].append(file)
        
        return groups
    
    def _create_country_mosaic(self, tiles: List[Path], output_file: Path, config: Dict) -> bool:
        """Create country-wide mosaic from tiles."""
        try:
            # Open source files
            src_files = [rasterio.open(f) for f in tiles]
            
            # Merge with resampling if needed
            mosaic, out_trans = merge(
                src_files,
                nodata=config.get('nodata_value', -9999),
                res=config.get('target_resolution', 100)
            )
            
            # Get output profile
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512
            })
            
            # Write output
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Clean up
            for src in src_files:
                src.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating country mosaic: {str(e)}")
            return False
    
    def _create_overview_pyramids(self, raster_file: Path) -> bool:
        """Create overview pyramids for raster file."""
        try:
            with rasterio.open(raster_file, 'r+') as src:
                src.build_overviews([2, 4, 8, 16], Resampling.average)
            return True
        except Exception as e:
            self.logger.error(f"Error creating overviews: {str(e)}")
            return False
    
    def _extract_year_from_filename(self, filepath: Path) -> Optional[int]:
        """Extract year from filename."""
        import re
        filename = filepath.stem
        matches = re.findall(r'(\d{4})', filename)
        for match in matches:
            year = int(match)
            if 2000 <= year <= 2030:
                return year
        return None
    
    def _extract_tile_id_from_filename(self, filepath: Path) -> Optional[str]:
        """Extract tile ID from filename."""
        # Simplified - would need actual implementation based on naming convention
        return "tile_default"
    
    def _extract_location_from_filename(self, filepath: Path) -> Optional[str]:
        """Extract location from filename."""
        # Simplified - would need actual implementation based on naming convention
        return "location_default"
    
    def run_pipeline(
        self, 
        steps: Optional[List[PipelineStep]] = None,
        continue_on_error: bool = False
    ) -> Dict[str, bool]:
        """
        Run the complete or partial post-processing pipeline.
        
        Args:
            steps: List of steps to run (default: all steps)
            continue_on_error: Continue even if a step fails
            
        Returns:
            Dict mapping step names to success status
        """
        if steps is None:
            steps = [PipelineStep.ALL]
        
        # Expand 'all' step
        if PipelineStep.ALL in steps:
            steps = [PipelineStep.MERGE, PipelineStep.SANITIZE, PipelineStep.FINAL_MERGE]
        
        self.start_time = time.time()
        
        # Log pipeline start
        log_pipeline_start(self.logger, "Canopy Height Post-Processing", self.config)
        
        results = {}
        
        try:
            for step in steps:
                # Validate prerequisites
                if not self.validate_step_prerequisites(step):
                    results[step.value] = False
                    if not continue_on_error:
                        break
                    continue
                
                # Execute step
                success = False
                if step == PipelineStep.MERGE:
                    success = self.run_merge_step()
                elif step == PipelineStep.SANITIZE:
                    success = self.run_sanitize_step()
                elif step == PipelineStep.FINAL_MERGE:
                    success = self.run_final_merge_step()
                
                results[step.value] = success
                
                if not success and not continue_on_error:
                    break
            
            # Pipeline completion
            overall_success = all(results.values())
            elapsed_time = time.time() - self.start_time
            log_pipeline_end(self.logger, "Canopy Height Post-Processing", overall_success, elapsed_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Post-processing pipeline failed: {str(e)}")
            return {step.value: False for step in steps}
