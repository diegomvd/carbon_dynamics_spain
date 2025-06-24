"""
I/O Utilities for Biomass Estimation

This module provides efficient raster I/O operations with support for
large-scale processing, chunking, and distributed computing.
Updated to use CentralDataPaths instead of config file paths.

Author: Diego Bengochea
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import rasterio
import xarray as xr
import dask.array as da
from rasterio.windows import Window
from rasterio.enums import Resampling

# Shared utilities
from shared_utils import get_logger, find_files, validate_file_exists
from shared_utils.central_data_paths_constants import *


class RasterManager:
    """
    Manager for raster I/O operations in biomass estimation pipeline.
    
    Handles efficient loading, processing, and saving of raster data
    with support for chunking and distributed processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the raster manager.
        
        Args:
            config: Configuration dictionary (processing parameters only)
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.io')
        
        # Processing parameters
        self.chunk_size = config['compute']['chunk_size']
        self.nodata_value = config['output']['geotiff']['nodata_value']
        
        self.logger.info("RasterManager initialized")
    
    def find_height_rasters_for_year(self, year: int, resolution: str = '100m') -> List[Path]:
        """
        Find height raster files for a specific year.
        
        Args:
            year: Year to search for
            resolution: Resolution to use ('10m' or '100m')
            
        Returns:
            List of matching height raster files
        """
        # UPDATED: Use CentralDataPaths instead of config
        if resolution == '10m':
            height_maps_dir = HEIGHT_MAPS_10M_DIR
        else:
            height_maps_dir = HEIGHT_MAPS_100M_DIR
        
        year_dir = height_maps_dir / str(year)
        
        if not year_dir.exists():
            self.logger.warning(f"Height maps directory for {year} not found: {year_dir}")
            return []
        
        # Find all .tif files in the year directory
        pattern = self.config['processing']['file_pattern']
        all_files = list(year_dir.glob(pattern))
        
        self.logger.debug(f"Found {len(all_files)} height rasters for year {year}")
        return sorted(all_files)
    
    def find_mask_files_for_year(self, year: int) -> List[Path]:
        """
        Find forest type mask files for a specific year using CentralDataPaths.
        
        Args:
            year: Year to search for
            
        Returns:
            List of matching mask files
        """
        # UPDATED: Use CentralDataPaths instead of config
        masks_dir = FOREST_TYPE_MASKS_DIR
        
        if not masks_dir.exists():
            self.logger.warning(f"Forest type maps directory not found: {masks_dir}")
            return []
        
        # Look for mask files with year in filename
        mask_files = []
        for pattern in ["*.shp", "*.tif"]:
            files = list(masks_dir.glob(pattern))
            # Filter by year if year is in filename
            year_files = [f for f in files if str(year) in f.name]
            mask_files.extend(year_files)
        
        # If no year-specific files, return all mask files
        if not mask_files:
            for pattern in ["*.shp", "*.tif"]:
                mask_files.extend(list(masks_dir.glob(pattern)))
        
        self.logger.debug(f"Found {len(mask_files)} mask files for year {year}")
        return sorted(mask_files)
    
    def _extract_year_from_filename(self, filepath: Path) -> Optional[int]:
        """
        Extract year from filename using regex patterns.
        
        Args:
            filepath: File path to analyze
            
        Returns:
            Year if found, None otherwise
        """
        filename = filepath.name
        
        # Common year patterns in filenames
        patterns = [
            r'_(\d{4})_',  # _YYYY_
            r'_(\d{4})\.',  # _YYYY.
            r'^(\d{4})_',   # YYYY_
            r'(\d{4})$'     # YYYY at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                year = int(match.group(1))
                # Validate year range
                if 2000 <= year <= 2030:
                    return year
        
        return None
    
    def load_raster_as_array(
        self, 
        filepath: Path,
        chunks: Optional[Tuple[int, int]] = None,
        window: Optional[Window] = None
    ) -> Optional[xr.DataArray]:
        """
        Load raster as xarray DataArray with optional chunking.
        
        Args:
            filepath: Raster file path
            chunks: Chunk size tuple (height, width)
            window: Rasterio window for partial reading
            
        Returns:
            DataArray or None if loading failed
        """
        try:
            validate_file_exists(filepath, "Raster file")
            
            with rasterio.open(filepath) as src:
                # Read data
                if window:
                    data = src.read(1, window=window)
                    transform = src.window_transform(window)
                else:
                    data = src.read(1)
                    transform = src.transform
                
                # Convert to DataArray
                da_data = xr.DataArray(
                    data,
                    dims=['y', 'x'],
                    attrs={
                        'crs': src.crs,
                        'transform': transform,
                        'nodata': src.nodata
                    }
                )
                
                # Apply chunking if requested
                if chunks:
                    da_data = da_data.chunk(chunks)
                
                return da_data
                
        except Exception as e:
            self.logger.error(f"Error loading raster {filepath}: {str(e)}")
            return None
    
    def load_raster_as_dask_array(
        self, 
        filepath: Path,
        chunks: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[da.Array, Dict]]:
        """
        Load raster as dask array for distributed processing.
        
        Args:
            filepath: Raster file path
            chunks: Chunk size tuple
            
        Returns:
            Tuple of (dask_array, metadata) or None if failed
        """
        try:
            validate_file_exists(filepath, "Raster file")
            
            with rasterio.open(filepath) as src:
                # Get metadata
                metadata = {
                    'shape': (src.height, src.width),
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'nodata': src.nodata
                }
                
                # Create dask array
                if chunks is None:
                    chunks = (self.chunk_size, self.chunk_size)
                
                # Use rasterio's built-in dask support
                dask_array = da.from_array(
                    src.read(1),
                    chunks=chunks
                )
                
                return dask_array, metadata
                
        except Exception as e:
            self.logger.error(f"Error loading raster as dask array {filepath}: {str(e)}")
            return None
    
    def save_raster_data(
        self,
        data: Union[np.ndarray, da.Array, xr.DataArray],
        output_path: Path,
        template_file: Optional[Path] = None,
        profile_overrides: Optional[Dict] = None
    ) -> bool:
        """
        Save raster data to file with optimized settings.
        
        Args:
            data: Data to save
            output_path: Output file path
            template_file: Template file for spatial reference
            profile_overrides: Override profile parameters
            
        Returns:
            bool: True if save succeeded
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            if isinstance(data, xr.DataArray):
                data_array = data.values
                if hasattr(data, 'attrs') and 'transform' in data.attrs:
                    transform = data.attrs['transform']
                    crs = data.attrs.get('crs')
                else:
                    transform = None
                    crs = None
            elif isinstance(data, da.Array):
                data_array = data.compute()
                transform = None
                crs = None
            else:
                data_array = data
                transform = None
                crs = None
            
            # Get profile from template or create default
            if template_file and template_file.exists():
                with rasterio.open(template_file) as src:
                    profile = src.profile.copy()
            else:
                profile = self._create_default_profile(data_array.shape)
            
            # Update profile with data characteristics
            profile.update({
                'height': data_array.shape[0],
                'width': data_array.shape[1],
                'dtype': data_array.dtype,
                'count': 1
            })
            
            # Apply configuration overrides
            profile.update(self.config['output']['geotiff'])
            
            # Apply custom overrides
            if profile_overrides:
                profile.update(profile_overrides)
            
            # Update spatial reference if available
            if transform:
                profile['transform'] = transform
            if crs:
                profile['crs'] = crs
            
            # Write data
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data_array, 1)
            
            self.logger.debug(f"Saved raster data to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving raster data to {output_path}: {str(e)}")
            return False
    
    def _create_default_profile(self, shape: Tuple[int, int]) -> Dict:
        """
        Create default raster profile.
        
        Args:
            shape: Data shape (height, width)
            
        Returns:
            Default profile dictionary
        """
        return {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:25830',  # Default CRS for Spain
            'transform': rasterio.transform.from_bounds(0, 0, shape[1], shape[0], shape[1], shape[0]),
            'nodata': self.nodata_value,
            **self.config['output']['geotiff']
        }
    
    def get_raster_info(self, filepath: Path) -> Optional[Dict]:
        """
        Get comprehensive information about a raster file.
        
        Args:
            filepath: Raster file path
            
        Returns:
            Dictionary with raster information
        """
        try:
            validate_file_exists(filepath, "Raster file")
            
            with rasterio.open(filepath) as src:
                info = {
                    'filename': filepath.name,
                    'driver': src.driver,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': src.nodata,
                    'pixel_size': (src.res[0], src.res[1]),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024)
                }
                
                # Add statistics if data is reasonable size
                if info['width'] * info['height'] < 50_000_000:  # ~50M pixels
                    data = src.read(1)
                    valid_data = data[data != src.nodata] if src.nodata else data
                    if len(valid_data) > 0:
                        info.update({
                            'min_value': float(np.min(valid_data)),
                            'max_value': float(np.max(valid_data)),
                            'mean_value': float(np.mean(valid_data)),
                            'std_value': float(np.std(valid_data)),
                            'valid_pixels': len(valid_data),
                            'total_pixels': data.size,
                            'valid_percentage': (len(valid_data) / data.size) * 100
                        })
                
                return info
                
        except Exception as e:
            self.logger.error(f"Error getting raster info for {filepath}: {str(e)}")
            return None
    
    def check_raster_alignment(
        self, 
        raster_files: List[Path],
        check_crs: bool = True,
        check_transform: bool = True,
        check_shape: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Check if multiple rasters are properly aligned.
        
        Args:
            raster_files: List of raster files to check
            check_crs: Whether to check CRS alignment
            check_transform: Whether to check transform alignment
            check_shape: Whether to check shape alignment
            
        Returns:
            Tuple of (is_aligned, list_of_issues)
        """
        if len(raster_files) < 2:
            return True, []
        
        issues = []
        reference_info = None
        
        try:
            # Get reference info from first file
            with rasterio.open(raster_files[0]) as ref_src:
                reference_info = {
                    'crs': ref_src.crs,
                    'transform': ref_src.transform,
                    'shape': (ref_src.height, ref_src.width)
                }
            
            # Check each file against reference
            for i, raster_file in enumerate(raster_files[1:], 1):
                with rasterio.open(raster_file) as src:
                    # Check CRS
                    if check_crs and src.crs != reference_info['crs']:
                        issues.append(f"File {i}: CRS mismatch ({src.crs} vs {reference_info['crs']})")
                    
                    # Check transform
                    if check_transform and src.transform != reference_info['transform']:
                        issues.append(f"File {i}: Transform mismatch")
                    
                    # Check shape
                    if check_shape and (src.height, src.width) != reference_info['shape']:
                        issues.append(f"File {i}: Shape mismatch ({src.height}x{src.width} vs {reference_info['shape']})")
            
            is_aligned = len(issues) == 0
            
            if is_aligned:
                self.logger.debug(f"All {len(raster_files)} rasters are properly aligned")
            else:
                self.logger.warning(f"Raster alignment issues found: {len(issues)} problems")
            
            return is_aligned, issues
            
        except Exception as e:
            self.logger.error(f"Error checking raster alignment: {str(e)}")
            return False, [f"Error during alignment check: {str(e)}"]
    
    def resample_raster_to_match(
        self, 
        source_file: Path, 
        reference_file: Path, 
        output_file: Path,
        resampling_method: Resampling = Resampling.bilinear
    ) -> bool:
        """
        Resample a raster to match the spatial properties of a reference raster.
        
        Args:
            source_file: Source raster to resample
            reference_file: Reference raster for target properties
            output_file: Output path for resampled raster
            resampling_method: Resampling algorithm to use
            
        Returns:
            bool: True if resampling succeeded
        """
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(reference_file) as ref_src:
                # Get target properties from reference
                target_transform = ref_src.transform
                target_crs = ref_src.crs
                target_width = ref_src.width
                target_height = ref_src.height
            
            with rasterio.open(source_file) as src:
                # Read source data
                source_data = src.read(1)
                
                # Create output array
                resampled_data = np.empty((target_height, target_width), dtype=source_data.dtype)
                
                # Perform resampling
                rasterio.warp.reproject(
                    source=source_data,
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )
                
                # Create output profile
                output_profile = src.profile.copy()
                output_profile.update({
                    'crs': target_crs,
                    'transform': target_transform,
                    'width': target_width,
                    'height': target_height
                })
                
                # Write output
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    dst.write(resampled_data, 1)
            
            self.logger.info(f"Successfully resampled {source_file.name} to match {reference_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resampling raster: {str(e)}")
            return False