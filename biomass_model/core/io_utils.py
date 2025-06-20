"""
I/O Utilities for Biomass Estimation

This module provides efficient raster I/O operations with support for
large-scale processing, chunking, and distributed computing.

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
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.io')
        
        # Processing parameters
        self.chunk_size = config['compute']['chunk_size']
        self.nodata_value = config['output']['geotiff']['nodata_value']
        
        self.logger.info("RasterManager initialized")
    
    def find_height_rasters_for_year(self, year: int) -> List[Path]:
        """
        Find height raster files for a specific year.
        
        Args:
            year: Year to search for
            
        Returns:
            List of matching height raster files
        """
        input_dir = Path(self.config['data']['input_data_dir'])
        pattern = self.config['processing']['file_pattern']
        
        # Find all files matching pattern
        all_files = find_files(input_dir, pattern)
        
        # Filter by year
        year_files = []
        for file in all_files:
            if self._extract_year_from_filename(file) == year:
                year_files.append(file)
        
        self.logger.debug(f"Found {len(year_files)} height rasters for year {year}")
        return sorted(year_files)
    
    def find_mask_files_for_year(self, year: int) -> List[Path]:
        """
        Find forest type mask files for a specific year.
        
        Args:
            year: Year to search for
            
        Returns:
            List of matching mask files
        """
        masks_dir = Path(self.config['data']['masks_dir'])
        
        # Find all mask files
        all_masks = find_files(masks_dir, '*.tif')
        
        # Filter by year
        year_masks = []
        for mask_file in all_masks:
            if self._extract_year_from_filename(mask_file) == year:
                year_masks.append(mask_file)
        
        self.logger.debug(f"Found {len(year_masks)} mask files for year {year}")
        return sorted(year_masks)
    
    def _extract_year_from_filename(self, filepath: Path) -> Optional[int]:
        """
        Extract year from filename using common patterns.
        
        Args:
            filepath: File path to extract year from
            
        Returns:
            Year as integer or None if not found
        """
        filename = filepath.stem
        
        # Common year patterns in filenames
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
                # Reasonable year range
                if 2000 <= year <= 2030:
                    return year
        
        return None
    
    def extract_forest_type_from_filename(self, filepath: Path) -> Optional[str]:
        """
        Extract forest type code from filename.
        
        Args:
            filepath: File path to extract forest type from
            
        Returns:
            Forest type code or None if not found
        """
        filename = filepath.stem
        
        # Common forest type patterns
        patterns = [
            r'code(\d+)',  # code123
            r'type(\d+)',  # type123
            r'_(\d+)$',    # ending with underscore and number
            r'(\d+)\.tif$'  # number before .tif extension
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        self.logger.warning(f"Could not extract forest type from filename: {filename}")
        return None
    
    def load_raster_data(
        self, 
        filepath: Path, 
        chunks: Optional[Tuple[int, int]] = None,
        window: Optional[Window] = None
    ) -> Optional[xr.DataArray]:
        """
        Load raster data as xarray DataArray with optional chunking.
        
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
        issues = []
        
        if len(raster_files) < 2:
            return True, []
        
        try:
            # Get reference properties from first raster
            with rasterio.open(raster_files[0]) as ref:
                ref_crs = ref.crs
                ref_transform = ref.transform
                ref_shape = (ref.height, ref.width)
            
            # Check each subsequent raster
            for i, raster_file in enumerate(raster_files[1:], 1):
                with rasterio.open(raster_file) as src:
                    if check_crs and src.crs != ref_crs:
                        issues.append(f"CRS mismatch in {raster_file.name}: {src.crs} vs {ref_crs}")
                    
                    if check_transform and src.transform != ref_transform:
                        issues.append(f"Transform mismatch in {raster_file.name}")
                    
                    if check_shape and (src.height, src.width) != ref_shape:
                        issues.append(f"Shape mismatch in {raster_file.name}: {(src.height, src.width)} vs {ref_shape}")
            
            is_aligned = len(issues) == 0
            
            if is_aligned:
                self.logger.debug(f"All {len(raster_files)} rasters are properly aligned")
            else:
                self.logger.warning(f"Found {len(issues)} alignment issues")
            
            return is_aligned, issues
            
        except Exception as e:
            self.logger.error(f"Error checking raster alignment: {str(e)}")
            return False, [f"Error during alignment check: {str(e)}"]
    
    def create_overview_pyramids(self, raster_file: Path, levels: List[int] = None) -> bool:
        """
        Create overview pyramids for faster visualization.
        
        Args:
            raster_file: Raster file to create overviews for
            levels: Overview levels (default: [2, 4, 8, 16])
            
        Returns:
            bool: True if successful
        """
        try:
            if levels is None:
                levels = [2, 4, 8, 16]
            
            with rasterio.open(raster_file, 'r+') as src:
                # Calculate overview levels that make sense for this raster
                valid_levels = []
                for level in levels:
                    if src.width // level > 256 and src.height // level > 256:
                        valid_levels.append(level)
                
                if valid_levels:
                    src.build_overviews(valid_levels, Resampling.average)
                    src.update_tags(ns='rio_overview', resampling='average')
                    self.logger.debug(f"Created overviews for {raster_file.name}: {valid_levels}")
                    return True
                else:
                    self.logger.debug(f"Raster too small for overviews: {raster_file.name}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error creating overviews for {raster_file}: {str(e)}")
            return False
    
    def get_processing_chunks(
        self, 
        raster_shape: Tuple[int, int],
        chunk_size: Optional[int] = None,
        overlap: int = 0
    ) -> List[Tuple[Window, Tuple[int, int]]]:
        """
        Generate processing chunks for large rasters.
        
        Args:
            raster_shape: Shape of raster (height, width)
            chunk_size: Size of chunks (uses config default if None)
            overlap: Overlap between chunks in pixels
            
        Returns:
            List of (window, chunk_shape) tuples
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        height, width = raster_shape
        chunks = []
        
        for row in range(0, height, chunk_size - overlap):
            for col in range(0, width, chunk_size - overlap):
                # Calculate actual chunk dimensions
                chunk_height = min(chunk_size, height - row)
                chunk_width = min(chunk_size, width - col)
                
                # Create window
                window = Window(col, row, chunk_width, chunk_height)
                chunk_shape = (chunk_height, chunk_width)
                
                chunks.append((window, chunk_shape))
        
        self.logger.debug(f"Generated {len(chunks)} processing chunks for shape {raster_shape}")
        return chunks
    
    def validate_output_directory(self, output_dir: Path) -> bool:
        """
        Validate output directory is writable and has sufficient space.
        
        Args:
            output_dir: Output directory to validate
            
        Returns:
            bool: True if directory is suitable for output
        """
        try:
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = output_dir / '.write_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
            except Exception:
                self.logger.error(f"No write permission for output directory: {output_dir}")
                return False
            
            # Check available space (warn if less than 10GB)
            from shared_utils.path_utils import get_available_space
            available_gb = get_available_space(output_dir)
            
            if available_gb < 10:
                self.logger.warning(f"Low disk space in output directory: {available_gb:.1f} GB available")
                if available_gb < 1:
                    self.logger.error("Insufficient disk space (< 1GB)")
                    return False
            
            self.logger.debug(f"Output directory validated: {output_dir} ({available_gb:.1f} GB available)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating output directory {output_dir}: {str(e)}")
            return False