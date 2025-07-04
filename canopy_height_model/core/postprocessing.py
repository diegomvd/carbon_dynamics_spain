"""
Post-Processing Pipeline

Complete multi-step post-processing pipeline for canopy height predictions including:
1. Merge patches into 120km tiles with geographic filtering
2. Sanitize outliers and interpolate temporal gaps  
3. Create final country-wide mosaics at target resolution

Contains the complete refactored logic from the original merge_predictions.py,
sanitize_predictions.py, and downsample_merge.py implementations.

Author: Diego Bengochea
"""

import os
import glob
import re
import time
import itertools
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal, Tuple
from enum import Enum
from functools import partial
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import interpolate
import geopandas as gpd
import shapely
import odc.geo
import odc.geo.geobox
from tqdm import tqdm

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils import find_files, log_pipeline_start, log_pipeline_end, log_section
from shared_utils.central_data_paths_constants import HEIGHT_MAPS_TMP_RAW_DIR, HEIGHT_MAPS_TMP_120KM_DIR, HEIGHT_MAPS_10M_DIR, HEIGHT_MAPS_100M_DIR


class PipelineStep(Enum):
    """Enumeration of post-processing pipeline steps."""
    MERGE = "merge"
    SANITIZE = "sanitize"
    FINAL_MERGE = "final_merge"
    ALL = "all"


class PredictionMerger:
    """
    Parallel processor for merging canopy height prediction tiles.
    
    This class handles the conversion of small prediction patches into larger
    geographic tiles suitable for further processing and analysis.
    """
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize the prediction merger."""
        self.config = config
        self.logger = logger
        self.merge_config = config['post_processing']['merge']
        
        # Create output directory
        ensure_directory(HEIGHT_MAPS_TMP_120KM_DIR)
        
        # Load Spain boundaries for masking
        self._load_spain_boundaries()
        
        # Create geographic grid
        self._create_geographic_grid()
        
        self.logger.info("PredictionMerger initialized")
        
    def _load_spain_boundaries(self) -> None:
        """Load and prepare Spain boundaries for geographic masking."""
        try:
            # Get shapefile path from configuration
            spain_shapefile = self.merge_config.get('spain_shapefile')
            if not spain_shapefile:
                raise ValueError("Spain shapefile path not found in configuration")
            
            # Load Spain boundaries
            self.spain = gpd.read_file(spain_shapefile)
            self.spain = self.spain[['geometry', 'COUNTRY']].to_crs(
                epsg=self.merge_config['target_crs'].split(':')[1]
            )
            
            # Create geometry for odc.geo 
            self.spain_geometry = odc.geo.geom.Geometry(
                self.spain.geometry[0], 
                crs=self.merge_config['target_crs']
            )
            
            self.logger.info(f"Spain boundaries loaded from: {spain_shapefile}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Spain boundaries: {str(e)}")
            raise  # Don't continue with invalid boundaries!

    def _create_geographic_grid(self) -> None:
        """Create geographic grid using Spain boundaries."""
        try:
            # Create a GeoBox for all continental Spain using boundaries
            geobox_spain = odc.geo.geobox.GeoBox.from_geopolygon(
                self.spain_geometry,  # Use REAL Spain geometry, not hardcoded bounds!
                resolution=self.merge_config['resolution_meters']
            )
            
            # Calculate tile size in grid units
            tile_size_meters = self.merge_config['tile_size_km'] * 1000
            tile_size_grid = tile_size_meters // self.merge_config['resolution_meters']
            
            # Divide the full geobox into tiles
            self.geotiles = odc.geo.geobox.GeoboxTiles(
                geobox_spain, 
                (tile_size_grid, tile_size_grid)
            )
            
            # Extract all tiles
            self.geotiles_list = [
                self.geotiles.__getitem__(tile) 
                for tile in self.geotiles._all_tiles()
            ]
            
            self.logger.info(f"Created {len(self.geotiles_list)} geographic tiles")
            
        except Exception as e:
            self.logger.error(f"Failed to create geographic grid: {str(e)}")
            raise  # Don't continue with invalid grid!

    def _convert_timestamp_to_year(self, timestamp: int) -> int:
        """Convert timestamp to year using mapping."""
        year_mapping = self.merge_config.get('year_timestamps', {})
        return year_mapping.get(timestamp, timestamp)
    
    def _generate_output_filename(self, year: int, lat: float, lon: float) -> str:
        """Generate output filename for merged tile."""
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        abs_lat = abs(lat)
        abs_lon = abs(lon)
        
        filename = f"canopy_height_{year}_{lat_dir}{abs_lat:.1f}_{lon_dir}{abs_lon:.1f}.tif"
        return str(HEIGHT_MAPS_TMP_120KM_DIR / filename)
    
    def _find_intersecting_files(self, tile_shapely_box: shapely.geometry.box, year: int) -> List[str]:
        """Find prediction files that intersect with the given tile using spatial intersection."""
        # Use the original directory structure: input_dir/year.0/files
        predictions_dir = HEIGHT_MAPS_TMP_RAW_DIR / f'{year}.0'
        
        if not predictions_dir.exists():
            return []
        
        # Find all prediction files for this year
        all_files = list(predictions_dir.glob(f'*{year}.0.tif'))
        files_to_merge = []
        
        for fname in all_files:
            try:
                with rasterio.open(fname) as src:
                    bounds = src.bounds
                    prediction_shapely_box = shapely.box(
                        bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                    
                    # REAL spatial intersection test (not placeholder!)
                    if shapely.intersects(tile_shapely_box, prediction_shapely_box):
                        files_to_merge.append(str(fname))
                        
            except Exception as e:
                self.logger.warning(f"Error reading {fname}: {str(e)}")
                continue
        
        return files_to_merge

    def _merge_tile_files(self, files_to_merge: List[str], original_bounds: Tuple[float, float, float, float]):
        """Merge files for a single tile using overlap averaging."""
        try:
            datasets = [rasterio.open(f) for f in files_to_merge]
            
            try:
                # Use SUM and COUNT methods to handle overlaps properly (not 'first'!)
                image_sum, transform_sum = merge(
                    datasets, 
                    bounds=original_bounds, 
                    method='sum'
                )
                image_count, transform_count = merge(
                    datasets, 
                    bounds=original_bounds, 
                    method='count'
                )
                
                # Average overlapping areas - PROPER overlap handling
                image = image_sum / image_count
                image = image[0, :, :]  # Remove band dimension
                
                return image, transform_count
                
            finally:
                # Close datasets
                for ds in datasets:
                    ds.close()
                
        except Exception as e:
            self.logger.error(f"Error merging files: {e}")
            return None, None
    
    def _apply_spain_mask(self, image: np.ndarray, transform) -> np.ndarray:
        """Apply Spain boundary mask to image."""
        if self.spain is None:
            return image
        
        try:
            # Create mask from Spain boundaries
            mask = rasterio.features.geometry_mask(
                self.spain.geometry,
                transform=transform,
                invert=True,
                out_shape=image.shape
            )
            
            # Apply mask
            masked_image = np.where(mask, image, self.merge_config['nodata_value'])
            return masked_image
            
        except Exception as e:
            self.logger.warning(f"Could not apply Spain mask: {e}")
            return image
    
    def _write_output_raster(self, image: np.ndarray, transform, savepath: str, year: int) -> None:
        """Write the processed image to a raster file."""
        try:
            with rasterio.open(
                savepath,
                mode="w",
                driver="GTiff",
                height=image.shape[-2],
                width=image.shape[-1],
                count=1,
                dtype='float32',
                crs=self.merge_config['target_crs'],
                transform=transform,
                nodata=self.merge_config['nodata_value'],
                compress=self.merge_config['compression'],
                tiled=True
            ) as new_dataset:
                new_dataset.write(image, 1)
                new_dataset.update_tags(DATE=str(year))
                
        except Exception as e:
            self.logger.error(f"Error writing raster {savepath}: {e}")
            raise
    
    def process_tile_year(self, args) -> str:
        """Process a single tile for a specific year."""
        i, (tile, year) = args
        
        try:
            # Buffer the tile to prevent stitching artifacts
            original_tile = tile
            buffer_meters = self.merge_config.get('buffer_meters', 1000)
            tile = original_tile.buffered(buffer_meters)
            
            # Get tile coordinates
            tile_bbox = tile.boundingbox
            tile_shapely_box = shapely.box(
                tile_bbox.left, tile_bbox.bottom, 
                tile_bbox.right, tile_bbox.top
            )
            
            # Convert to lat/lon for filename
            tile_bbox_latlon = tile_bbox.to_crs('EPSG:4326')
            lon = tile_bbox_latlon.left
            lat = tile_bbox_latlon.top
            
            # Convert year and generate filename
            year_converted = self._convert_timestamp_to_year(year)
            savepath = self._generate_output_filename(year_converted, lat, lon)
            
            # Skip if file already exists
            if Path(savepath).exists():
                return f"Skipped {i} (file already exists): {savepath}"
            
            # Find intersecting prediction files
            files_to_merge = self._find_intersecting_files(tile_shapely_box, year)
            
            if not files_to_merge:
                return f"No files in tile {tile_bbox} for year {year_converted}"
            
            # Get original tile bounds (without buffer)
            original_bbox = original_tile.boundingbox
            original_bounds = (
                original_bbox.left, original_bbox.bottom,
                original_bbox.right, original_bbox.top
            )
            
            # Merge files
            image, transform = self._merge_tile_files(files_to_merge, original_bounds)
            
            if image is None:
                return f"Error merging files for task {i}"
            
            # Apply Spain mask
            masked_image = self._apply_spain_mask(image, transform)
            
            # Write output raster
            self._write_output_raster(masked_image, transform, savepath, year_converted)
            
            return f"Processed {i}: {savepath}"
            
        except Exception as e:
            return f"Error processing task {i} (year={year}): {str(e)}"
    
    def run_parallel_merge(self) -> bool:
        """Run the parallel merging process."""
        self.logger.info("Starting parallel prediction merging...")
        
        if not self.geotiles_list:
            self.logger.error("No geographic tiles available")
            return False
        
        # Create task list
        years = list(self.merge_config.get('year_timestamps', {}).keys())
        if not years:
            years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  # Default years
            
        args_list = list(itertools.product(self.geotiles_list, years))
        
        # Prepare indexed arguments for progress tracking
        indexed_args = [(i, args) for i, args in enumerate(args_list)]
        
        # Use multiprocessing
        num_cores = min(
            self.merge_config.get('num_workers', 4), 
            multiprocessing.cpu_count()
        )
        
        self.logger.info(f"Using {num_cores} cores for processing {len(args_list)} tasks")
        
        try:
            with multiprocessing.Pool(processes=num_cores) as pool:
                results = list(tqdm(
                    pool.imap(self.process_tile_year, indexed_args),
                    total=len(indexed_args),
                    desc="Merging tiles"
                ))
            
            # Print summary statistics
            successful = sum(1 for r in results if r.startswith("Processed"))
            skipped = sum(1 for r in results if r.startswith("Skipped"))
            no_files = sum(1 for r in results if r.startswith("No files"))
            errors = sum(1 for r in results if r.startswith("Error"))
            
            self.logger.info(f"Merging summary:")
            self.logger.info(f"Total tasks: {len(results)}")
            self.logger.info(f"Successfully processed: {successful}")
            self.logger.info(f"Skipped (already exist): {skipped}")
            self.logger.info(f"No files found: {no_files}")
            self.logger.info(f"Errors: {errors}")
            
            return successful > 0
            
        except Exception as e:
            self.logger.error(f"Parallel merging failed: {e}")
            return False


class HeightSanitizer:
    """
    Comprehensive height prediction sanitizer with outlier detection and interpolation.
    
    This class handles the identification and removal of outliers in canopy height
    predictions, followed by temporal interpolation to fill gaps using adjacent
    years when reliable data is available.
    """
    
    # TODO: this logic of output dirs is much more complex than previously thought!!!!! Careful oversimplification here.
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize the height sanitizer."""
        self.config = config
        self.logger = logger
        self.sanitize_config = config['post_processing']['sanitize']
        
        # Create output directories
        output_dirs = [
            HEIGHT_MAPS_TMP_120KM_DIR,
            HEIGHT_MAPS_10M_DIR,
            HEIGHT_MAPS_TMP_INTERPOLATION_MASKS_DIR
        ]
        
        for directory in output_dirs:
            ensure_directory(Path(directory))
        
        self.logger.info("HeightSanitizer initialized")
    
    def extract_info_from_filename(self, filepath: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """Extract year, latitude, and longitude from filename."""
        filename = os.path.basename(filepath)
        match = re.match(
            r'canopy_height_(\d{4})_(N|S)([\d\.]+)_(E|W)([\d\.]+)\.tif', 
            filename
        )
        
        if match:
            year = int(match.group(1))
            lat_dir = match.group(2)
            lat = float(match.group(3))
            lon_dir = match.group(4)
            lon = float(match.group(5))
            
            # Adjust for S and W being negative
            if lat_dir == 'S':
                lat = -lat
            if lon_dir == 'W':
                lon = -lon
                
            return year, lat, lon
        else:
            self.logger.warning(f"Could not extract information from filename: {filename}")
            return None, None, None
    
    def identify_outliers(self, data: np.ndarray, nodata_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Identify outlier values in raster data."""
        if nodata_value is not None:
            nodata_mask = np.isclose(data, nodata_value, rtol=1e-5)
        else:
            nodata_mask = np.zeros_like(data, dtype=bool)
        
        # Height thresholds
        upper_threshold = self.sanitize_config.get('upper_threshold', 60.0)
        lower_threshold = self.sanitize_config.get('lower_threshold', 0.0)
        
        # Identify outliers
        outlier_mask = (
            (data > upper_threshold) | 
            (data < lower_threshold)
        ) & ~nodata_mask
        
        return outlier_mask, nodata_mask
    
    def calculate_statistics(
        self, 
        data: np.ndarray, 
        nodata_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for raster data.
        
        Args:
            data (np.ndarray): Raster data array
            nodata_mask (np.ndarray): Boolean mask for NoData values
            
        Returns:
            Dict[str, Any]: Dictionary containing comprehensive statistics
        """
        valid_data = data[~nodata_mask]
        
        if len(valid_data) == 0:
            return {
                "min": None, 
                "max": None, 
                "mean": None, 
                "std": None, 
                "num_pixels": 0
            }
        
        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "num_pixels": int(valid_data.size)
        }
    
    def process_raster_file(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Process a single raster file for outlier detection."""
        try:
            year, lat, lon = self.extract_info_from_filename(filepath)
            if year is None:
                return filepath, {"error": "Could not extract info from filename"}
            
            # Create output filename
            processed_dir = self.sanitize_config.get('processed_dir', 'processed')
            output_filename = os.path.join(
                processed_dir, 
                os.path.basename(filepath).replace('.tif', '_processed.tif')
            )
            
            # Skip if already processed
            if os.path.exists(output_filename):
                return filepath, {
                    "year": year, "lat": lat, "lon": lon,
                    "output_file": output_filename, "skipped": True
                }
            
            # Process the raster
            with rasterio.open(filepath) as src:
                data = src.read(1)
                meta = src.meta.copy()
                nodata_value = src.nodata
                
                # Calculate original statistics
                nodata_mask = (
                    np.isclose(data, nodata_value, rtol=1e-5) 
                    if nodata_value is not None 
                    else np.zeros_like(data, dtype=bool)
                )
                original_stats = self.calculate_statistics(data, nodata_mask)
                
                # Identify outliers
                outlier_mask, nodata_mask = self.identify_outliers(data, nodata_value)
                
                # Count outliers and negative values
                negative_mask = (data < 0) & ~nodata_mask
                num_negative = np.sum(negative_mask)
                num_outliers = np.sum(outlier_mask)
                total_valid = data.size - np.sum(nodata_mask)
                
                negative_percent = (
                    100 * num_negative / total_valid if total_valid > 0 else 0
                )
                outlier_percent = (
                    100 * num_outliers / total_valid if total_valid > 0 else 0
                )
                
                # Convert to float32 and set outliers to NaN
                data = data.astype(np.float32)
                data[outlier_mask] = np.nan
                
                # Preserve original NoData values
                if nodata_value is not None:
                    data[nodata_mask] = nodata_value
                
                # Calculate processed statistics
                # Create updated nodata mask for processed data
                if nodata_value is not None:
                    valid_processed = data[
                        ~np.isnan(data) & ~np.isclose(data, nodata_value, rtol=1e-5)
                    ]
                    processed_nodata_mask = (
                        np.isnan(data) | np.isclose(data, nodata_value, rtol=1e-5)
                    )
                else:
                    valid_processed = data[~np.isnan(data)]
                    processed_nodata_mask = np.isnan(data)
                
                processed_stats = self.calculate_statistics(data, processed_nodata_mask)
                processed_stats.update({
                    "num_outliers": int(num_outliers),
                    "outlier_percent": float(outlier_percent),
                    "num_negative": int(num_negative),
                    "negative_percent": float(negative_percent)
                })
                
                # Update metadata - use rasterio constant instead of string
                meta.update(dtype=rasterio.float32)
                if nodata_value is not None:
                    meta.update(nodata=nodata_value)
                
                # Create output directory before writing
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                
                # Write processed file
                with rasterio.open(output_filename, 'w', **meta) as dst:
                    dst.write(data, 1)
                
                return filepath, {
                    "year": year, "lat": lat, "lon": lon,
                    "original_stats": original_stats,
                    "processed_stats": processed_stats,
                    "output_file": output_filename
                }
                
        except Exception as e:
            return filepath, {"error": str(e)}
    
    def _perform_temporal_interpolation(
        self,
        raster_data: Dict[int, np.ndarray],
        nodata_values: Dict[int, float],
        meta: Dict[str, Any],
        years: List[int],
        lat: float,
        lon: float
    ) -> List[Tuple[int, str]]:
        """
        Perform temporal interpolation on raster time series (pixel-wise).
        
        This is the sophisticated algorithm that was replaced with a simple file copy!
        """
        years_array = np.array(sorted(years))
        shape = next(iter(raster_data.values())).shape
        
        # Create data and mask stacks (3D: time × height × width)
        data_stack = np.zeros((len(years), shape[0], shape[1]), dtype=np.float32)
        mask_stack = np.zeros((len(years), shape[0], shape[1]), dtype=bool)
        
        # Fill stacks with data and validity masks
        for i, year in enumerate(years_array):
            data = raster_data[year]
            nodata_value = nodata_values[year]
            
            # Create valid data mask
            valid_mask = ~np.isnan(data)
            if nodata_value is not None:
                valid_mask &= ~np.isclose(data, nodata_value, rtol=1e-5)
            
            data_stack[i] = data
            mask_stack[i] = valid_mask
        
        # Initialize interpolated stack
        interpolated_stack = data_stack.copy()
        
        # Identify pixels needing interpolation
        has_valid_data = np.any(mask_stack, axis=0)
        
        # Identify pixels that are NoData in all years
        all_nodata = np.zeros(shape, dtype=bool)
        for year, data in raster_data.items():
            nodata_value = nodata_values[year]
            if nodata_value is not None:
                all_nodata |= np.isclose(data, nodata_value, rtol=1e-5)
        
        # Get pixels to interpolate
        pixels_to_interpolate = has_valid_data & ~all_nodata
        y_coords, x_coords = np.where(pixels_to_interpolate)
        
        self.logger.info(f"Interpolating {len(y_coords)} pixels for location: {lat}, {lon}")
        
        # Perform PIXEL-WISE interpolation
        for idx in tqdm(range(len(y_coords)), desc=f"Interpolating pixels for {lat},{lon}", leave=False):
            y, x = y_coords[idx], x_coords[idx]
            
            for i, year in enumerate(years_array):
                # Skip if already valid or NoData
                if (mask_stack[i, y, x] or 
                    (nodata_values[year] is not None and 
                    np.isclose(data_stack[i, y, x], nodata_values[year], rtol=1e-5))):
                    continue
                
                # Interpolation logic for edge and interior years
                if i == 0 and i+1 < len(years_array) and mask_stack[i+1, y, x]:
                    # First year: use next year
                    interpolated_stack[i, y, x] = data_stack[i+1, y, x]
                elif (i == len(years_array) - 1 and i-1 >= 0 and 
                    mask_stack[i-1, y, x]):
                    # Last year: use previous year
                    interpolated_stack[i, y, x] = data_stack[i-1, y, x]
                elif (i > 0 and i < len(years_array) - 1 and 
                    mask_stack[i-1, y, x] and mask_stack[i+1, y, x]):
                    # Interior years: average adjacent years
                    interpolated_stack[i, y, x] = (
                        data_stack[i-1, y, x] + data_stack[i+1, y, x]
                    ) / 2
        
        # Save interpolated rasters
        return self._save_interpolated_rasters(
            interpolated_stack, years_array, raster_data, 
            nodata_values, meta, lat, lon
        )

    def interpolate_location(
        self, 
        location_data: Tuple[Tuple[float, float], List[Tuple[int, str]]]
    ) -> Tuple[Tuple[float, float], List[Tuple[int, str]]]:
        """
        Interpolate NaN values in time series for a single location using pixel-wise interpolation.
        
        Args:
            location_data: Tuple of (location, tiles) where tiles are (year, filepath) tuples
            
        Returns:
            Tuple of (location, interpolated_files)
        """
        location, tiles = location_data
        lat, lon = location
        
        self.logger.info(f"Interpolating time series for location: {lat}, {lon}")
        
        # Sort tiles by year
        tiles.sort()
        
        # Load all rasters for this location
        years = []
        raster_data = {}
        nodata_values = {}
        meta = None
        
        for year, filepath in tqdm(tiles, desc=f"Loading tiles for {lat},{lon}", leave=False):
            try:
                with rasterio.open(filepath) as src:
                    data = src.read(1)
                    raster_data[year] = data
                    nodata_values[year] = src.nodata
                    if meta is None:
                        meta = src.meta.copy()
                    years.append(year)
            except Exception as e:
                self.logger.error(f"Error loading {filepath}: {str(e)}")
                continue
        
        if not raster_data or meta is None:
            self.logger.warning(f"No valid data for location: {lat}, {lon}")
            return location, []
        
        # Perform temporal interpolation
        interpolated_files = self._perform_temporal_interpolation(
            raster_data, nodata_values, meta, years, lat, lon
        )
        
        return location, interpolated_files
    
    def get_interpolated_filename(self, year: int, lat: float, lon: float) -> str:
        """Get the expected interpolated filename for given parameters."""
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        abs_lat = abs(lat)
        abs_lon = abs(lon)
        
        interpolated_dir = self.sanitize_config.get('interpolated_dir', 'interpolated')
        return os.path.join(
            interpolated_dir, 
            f"canopy_height_{year}_{lat_dir}{abs_lat:.1f}_{lon_dir}{abs_lon:.1f}_interpolated.tif"
        )

    def _create_interpolation_mask(
        self,
        data: np.ndarray,
        original_data: np.ndarray,
        nodata_value: Optional[float]
    ) -> np.ndarray:
        """
        Create mask showing interpolation results.
        
        Mask values: 0=Original, 1=NoData, 2=Interpolated, 3=Failed
        """
        interp_mask = np.zeros_like(data, dtype=np.uint8)
        
        if nodata_value is not None:
            # NoData values
            nodata_mask = np.isclose(original_data, nodata_value, rtol=1e-5)
            interp_mask[nodata_mask] = 1
            
            # Successfully interpolated pixels
            successful_interp = (
                ~np.isclose(data, nodata_value, rtol=1e-5) & 
                ~nodata_mask & 
                np.isnan(original_data)
            )
            interp_mask[successful_interp] = 2
            
            # Failed to interpolate
            failed_interp = np.isnan(data) & ~nodata_mask
            interp_mask[failed_interp] = 3
        else:
            # Successfully interpolated pixels
            successful_interp = ~np.isnan(data) & np.isnan(original_data)
            interp_mask[successful_interp] = 2
            
            # Failed to interpolate
            failed_interp = np.isnan(data) & np.isnan(original_data)
            interp_mask[failed_interp] = 3
        
        return interp_mask

    def _save_interpolation_mask(
        self,
        interp_mask: np.ndarray,
        meta: Dict[str, Any],
        year: int,
        lat: float,
        lon: float
    ) -> None:
        """Save interpolation mask to file."""
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        abs_lat = abs(lat)
        abs_lon = abs(lon)
        
        masks_dir = self.sanitize_config.get('interpolation_masks_dir', 'masks')
        mask_filename = os.path.join(
            masks_dir,
            f"canopy_height_{year}_{lat_dir}{abs_lat:.1f}_{lon_dir}{abs_lon:.1f}_interp_mask.tif"
        )
        
        mask_meta = meta.copy()
        mask_meta.update(dtype=rasterio.uint8, nodata=None)
        
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        with rasterio.open(mask_filename, 'w', **mask_meta) as dst:
            dst.write(interp_mask, 1)

    def _save_interpolated_rasters(
        self,
        interpolated_stack: np.ndarray,
        years_array: np.ndarray,
        raster_data: Dict[int, np.ndarray],
        nodata_values: Dict[int, float],
        meta: Dict[str, Any],
        lat: float,
        lon: float
    ) -> List[Tuple[int, str]]:
        """
        Save interpolated rasters and create interpolation masks.
        """
        interpolated_files = []
        
        for i, year in enumerate(tqdm(years_array, desc=f"Saving interpolated tiles for {lat},{lon}", leave=False)):
            data = interpolated_stack[i]
            original_data = raster_data[year]
            nodata_value = nodata_values[year]
            
            # Generate output filename
            output_filename = self.get_interpolated_filename(year, lat, lon)
            
            # Create interpolation mask
            interp_mask = self._create_interpolation_mask(
                data, original_data, nodata_value
            )
            
            # Save interpolated data
            try:
                interp_meta = meta.copy()
                if nodata_value is not None:
                    interp_meta.update(nodata=nodata_value)
                
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                with rasterio.open(output_filename, 'w', **interp_meta) as dst:
                    dst.write(data, 1)
                
                # Save interpolation mask
                self._save_interpolation_mask(
                    interp_mask, meta, year, lat, lon
                )
                
                interpolated_files.append((year, output_filename))
                
            except Exception as e:
                self.logger.error(f"Error saving {output_filename}: {str(e)}")
        
        return interpolated_files

    def run_sanitization_pipeline(self) -> bool:
        """Run the complete sanitization and interpolation pipeline."""
        try:
            self.logger.info("Starting height sanitization pipeline...")
            
            # Step 1: Find and process all raster files
            input_dir = HEIGHT_MAPS_TMP_120KM_DIR
            raster_files = list(input_dir.glob("*.tif"))
            
            if not raster_files:
                self.logger.error(f"No raster files found in {input_dir}")
                return False
            
            self.logger.info(f"Processing {len(raster_files)} raster files for outlier detection")
            
            # Process files in parallel
            num_workers = self.sanitize_config.get('num_workers', 4)
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.process_raster_file, [str(f) for f in raster_files]),
                    total=len(raster_files),
                    desc="Processing outliers"
                ))
            
            # Create summary DataFrame
            summary_data = []
            for filepath, stats in results:
                if "error" not in stats and not stats.get("skipped", False):
                    summary_data.append({
                        "filename": os.path.basename(filepath),
                        "year": stats["year"],
                        "lat": stats["lat"],
                        "lon": stats["lon"],
                        "outlier_percent": stats["processed_stats"]["outlier_percent"],
                        "processed_file": stats["output_file"]
                    })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary
            summary_file = self.sanitize_config.get('summary_file', 'sanitization_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            
            # Step 2: Organize files by location for temporal interpolation
            processed_by_location = defaultdict(list)
            
            for _, row in summary_df.iterrows():
                location = (row["lat"], row["lon"])
                year = row["year"]
                processed_by_location[location].append((year, row["processed_file"]))
            
            # Step 3: Perform temporal interpolation in parallel
            self.logger.info("Performing temporal interpolation to fill gaps")
            interpolation_data = list(processed_by_location.items())
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                interpolation_results = list(tqdm(
                    pool.imap(self.interpolate_location, interpolation_data),
                    total=len(interpolation_data),
                    desc="Interpolating locations"
                ))
            
            # Count interpolated files
            total_interpolated = sum(len(files) for _, files in interpolation_results)
            self.logger.info(f"Completed temporal interpolation. Total interpolated tiles: {total_interpolated}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sanitization pipeline failed: {e}")
            return False


class FinalMerger:
    """
    Final merger for creating country-wide canopy height mosaics.
    
    This class handles the downsampling and merging of processed height tiles
    into final country-wide products at the target resolution.
    """
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize the final merger."""
        self.config = config
        self.logger = logger
        self.merge_config = config['post_processing']['final_merge']
        
        # Create output directory
        ensure_directory(HEIGHT_MAPS_100M_DIR)
        
        # Create temporary directory for processing
        self.temp_dir = HEIGHT_MAPS_100M_DIR / 'temp'
        ensure_directory(self.temp_dir)
        
        # Map resampling method string to enum
        resampling_method = self.merge_config.get('resampling_method', 'average')
        self.resampling_method = getattr(Resampling, resampling_method.upper())
        
        self.logger.info("FinalMerger initialized")
    
    def find_rasters_by_year(self) -> Dict[str, List[str]]:
        """Group raster files by year."""
        input_dir = HEIGHT_MAPS_10M_DIR
        file_pattern = self.merge_config.get('file_pattern', '*.tif')
        
        files = list(input_dir.glob(file_pattern))
        rasters_by_year = defaultdict(list)
        
        for file_path in files:
            # Extract year from filename
            try:
                year_match = re.search(r'(\d{4})', file_path.name)
                if year_match:
                    year = year_match.group(1)
                    rasters_by_year[year].append(str(file_path))
            except Exception as e:
                self.logger.warning(f"Could not extract year from {file_path.name}: {e}")
        
        return dict(rasters_by_year)
    
    def downsample_raster(self, input_file: str, output_file: str) -> None:
        """Downsample a single raster to target resolution."""
        target_resolution = self.merge_config['target_resolution']
        
        with rasterio.open(input_file) as src:
            # Calculate new dimensions
            scale_factor = src.res[0] / target_resolution
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)
            
            # Create new transform
            new_transform = rasterio.Affine(
                target_resolution, 0, src.bounds.left,
                0, -target_resolution, src.bounds.top
            )
            
            # Update profile for output using ORIGINAL approach
            profile = src.profile.copy()
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': new_transform,
                'compress': self.merge_config.get('compression', 'lzw')
            })

            # Use WarpedVRT for resampling
            with WarpedVRT(
                src,
                width=new_width,
                height=new_height,
                transform=new_transform,
                resampling=self.resampling_method
            ) as vrt:
                data = vrt.read()
                # Write downsampled raster
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)
    
    def merge_rasters(self, raster_files: List[str], output_file: str) -> None:
        """Merge multiple rasters into a single output."""
        datasets = [rasterio.open(f) for f in raster_files]
        
        try:
            # Merge datasets
            merged_data, merged_transform = merge(datasets)
            
            # Get metadata from first dataset
            meta = datasets[0].meta.copy()
            meta.update({
                'height': merged_data.shape[1],
                'width': merged_data.shape[2],
                'transform': merged_transform,
                'compress': self.merge_config.get('compression', 'lzw'),
                'tiled': True,
                "blockxsize": 256, 
                "blockysize": 256   
            })
            
            # Write merged raster
            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(merged_data)
                
        finally:
            # Close datasets
            for ds in datasets:
                ds.close()
    
    def cleanup_temporary_files(self, temp_files: List[str]) -> None:
        """Clean up temporary files."""
        for file_path in temp_files:
            try:
                os.remove(file_path)
                self.logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Error removing temporary file {file_path}: {e}")
    
    def process_year(self, year: str, raster_files: List[str]) -> bool:
        """Process all rasters for a single year."""
        self.logger.info(f"Processing year {year} with {len(raster_files)} files")
        
        # Check if final output already exists
        output_pattern = self.merge_config.get('output_pattern', 'canopy_height_{year}_100m.tif')
        final_output_name = output_pattern.format(year=year)
        final_output_path = Path(HEIGHT_MAPS_100M_DIR) / final_output_name
        
        if final_output_path.exists():
            self.logger.info(f"Final output already exists for year {year}: {final_output_path}")
            return True
        
        # Downsample each raster
        downsampled_files = []
        
        for i, raster_file in enumerate(raster_files):
            temp_output_name = f"downsampled_{year}_{i}.tif"
            temp_output_path = self.temp_dir / temp_output_name
            
            self.logger.debug(f"Downsampling {Path(raster_file).name}")
            
            try:
                self.downsample_raster(str(raster_file), str(temp_output_path))
                downsampled_files.append(str(temp_output_path))
                
            except Exception as e:
                self.logger.error(f"Error processing {raster_file}: {e}")
                continue
        
        if not downsampled_files:
            self.logger.warning(f"No files successfully processed for year {year}")
            return False
        
        # Merge downsampled rasters
        self.logger.info(f"Merging {len(downsampled_files)} downsampled rasters for year {year}")
        
        try:
            self.merge_rasters(downsampled_files, str(final_output_path))
            self.logger.info(f"Successfully created final mosaic for year {year}: {final_output_path}")
            success = True
            
        except Exception as e:
            self.logger.error(f"Error creating final mosaic for year {year}: {e}")
            success = False
        
        # Clean up temporary files
        self.cleanup_temporary_files(downsampled_files)
        return success
    
    def run_final_merge_pipeline(self) -> bool:
        """Run the complete final merge pipeline."""
        self.logger.info("Starting final merge pipeline...")
        
        try:
            # Find all rasters grouped by year
            rasters_by_year = self.find_rasters_by_year()
            
            if not rasters_by_year:
                self.logger.error("No raster files found to process")
                return False
            
            successful_years = 0
            
            # Process each year
            for year, raster_files in rasters_by_year.items():
                try:
                    if self.process_year(year, raster_files):
                        successful_years += 1
                except Exception as e:
                    self.logger.error(f"Error processing year {year}: {e}")
                    continue
            
            # Remove temporary directory if empty
            try:
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    self.logger.info("Removed empty temporary directory")
            except Exception as e:
                self.logger.warning(f"Error removing temporary directory: {e}")
            
            self.logger.info(f"Final merge pipeline completed. Processed {successful_years}/{len(rasters_by_year)} years successfully")
            return successful_years > 0
            
        except Exception as e:
            self.logger.error(f"Final merge pipeline failed: {e}")
            return False


class PostProcessingPipeline:
    """
    Comprehensive post-processing pipeline for canopy height predictions.
    
    Implements a three-step workflow:
    1. Merge prediction patches into 120km tiles
    2. Sanitize outliers and interpolate temporal gaps
    3. Downsample and create final country-wide mosaics
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        steps: Optional[str] = None
    ):
        """
        Initialize the post-processing pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_model")
        
        # Setup logging
        self.logger = get_logger('canopy_height_postprocessing')
        
        # Pipeline state
        self.start_time = None
        self.step_times = {}
        
        # Initialize processing components (lazy initialization)
        self.merger = None
        self.sanitizer = None
        self.final_merger = None

        if steps:
            step_names = [s.strip().lower() for s in steps.split(',')]
            steps_list = []
            
            for step_name in step_names:
                try:
                    if step_name == 'all':
                        step = PipelineStep.ALL
                    elif step_name == 'merge':
                        step = PipelineStep.MERGE
                    elif step_name == 'sanitize':
                        step = PipelineStep.SANITIZE
                    elif step_name in ['final_merge', 'final-merge', 'finalize']:
                        step = PipelineStep.FINAL_MERGE
                    else:
                        raise ValueError(f"Unknown step: {step_name}")
                    
                    steps_list.append(step)
                except ValueError as e:
                    print(f"Warning: {e}")
                    print(f"Available steps: merge, sanitize, final_merge, all")
            self.steps = steps_list
        else:
            self.steps = [PipelineStep.ALL]
        
        self.logger.info("PostProcessingPipeline initialized")
    
    def run_full_pipeline(self) -> bool:
        all_results = self.run_pipeline()
        overall_success = all(results.values())
        return overall_success

    def validate_step_prerequisites(self, step: PipelineStep) -> bool:
        """Validate prerequisites for a specific step."""
        try:
            if step in [PipelineStep.MERGE, PipelineStep.ALL]:
                # Check if prediction files exist
                merge_config = self.config['post_processing']['merge']
                input_dir = HEIGHT_MAPS_TMP_RAW_DIR
                
                if not input_dir.exists():
                    self.logger.error(f"Merge input directory does not exist: {input_dir}")
                    return False
                
                # Check for prediction files
                prediction_files = list(input_dir.glob("*.tif"))
                if not prediction_files:
                    self.logger.error(f"No prediction files found in {input_dir}")
                    return False
                
                self.logger.info(f"Found {len(prediction_files)} prediction files for merging")
            
            if step in [PipelineStep.SANITIZE, PipelineStep.ALL]:
                # Check if merged tiles exist
                sanitize_config = self.config['post_processing']['sanitize']
                input_dir = HEIGHT_MAPS_TMP_120KM_DIR
                
                if not input_dir.exists():
                    self.logger.error(f"Sanitize input directory does not exist: {input_dir}")
                    return False
                
                tiles = list(input_dir.glob("*.tif"))
                if not tiles:
                    self.logger.error(f"No tiles found for sanitization in {input_dir}")
                    return False
                
                self.logger.info(f"Found {len(tiles)} tiles for sanitization")
            
            if step in [PipelineStep.FINAL_MERGE, PipelineStep.ALL]:
                # Check if sanitized/interpolated tiles exist
                final_config = self.config['post_processing']['final_merge']
                input_dir = HEIGHT_MAPS_10M_DIR
                
                if not input_dir.exists():
                    self.logger.error(f"Final merge input directory does not exist: {input_dir}")
                    return False
                
                file_pattern = final_config.get('file_pattern', '*.tif')
                tiles = list(input_dir.glob(file_pattern))
                if not tiles:
                    self.logger.error(f"No tiles found for final merging in {input_dir}")
                    return False
                
                self.logger.info(f"Found {len(tiles)} tiles for final merging")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating prerequisites for {step.value}: {e}")
            return False
    
    def run_merge_step(self) -> bool:
        """Step 1: Merge prediction patches into 120km tiles."""
        try:
            log_section(self.logger, "STEP 1: MERGE PATCHES TO TILES")
            step_start = time.time()
            
            # Initialize merger
            self.merger = PredictionMerger(self.config, self.logger)
            
            # Run merging
            success = self.merger.run_parallel_merge()
            
            # Record step time
            self.step_times['merge'] = time.time() - step_start
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in merge step: {e}")
            return False
    
    def run_sanitize_step(self) -> bool:
        """Step 2: Sanitize outliers and interpolate temporal gaps."""
        try:
            log_section(self.logger, "STEP 2: SANITIZE AND INTERPOLATE")
            step_start = time.time()
            
            # Initialize sanitizer
            self.sanitizer = HeightSanitizer(self.config, self.logger)
            
            # Run sanitization
            success = self.sanitizer.run_sanitization_pipeline()
            
            # Record step time
            self.step_times['sanitize'] = time.time() - step_start
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in sanitize step: {e}")
            return False
    
    def run_final_merge_step(self) -> bool:
        """Step 3: Create final country-wide mosaics."""
        try:
            log_section(self.logger, "STEP 3: FINAL COUNTRY-WIDE MOSAICS")
            step_start = time.time()
            
            # Initialize final merger
            self.final_merger = FinalMerger(self.config, self.logger)
            
            # Run final merging
            success = self.final_merger.run_final_merge_pipeline()
            
            # Record step time
            self.step_times['final_merge'] = time.time() - step_start
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in final merge step: {e}")
            return False
    
    def run_pipeline(
        self, 
        continue_on_error: bool = False
    ) -> bool:
        """
        Run the complete or partial post-processing pipeline.
        
        Args:
            steps: List of steps to run (default: all steps)
            continue_on_error: Continue even if a step fails
            
        Returns:
            Dict mapping step names to success status
        """

        steps = self.steps

        if steps is None:
            steps = [PipelineStep.ALL]
        
        # Expand 'all' step
        if PipelineStep.ALL in steps:
            steps = [PipelineStep.MERGE, PipelineStep.SANITIZE, PipelineStep.FINAL_MERGE]
        
        self.start_time = time.time()
        
        # Log pipeline start
        log_pipeline_start(self.logger, "Canopy Height Post-Processing Pipeline")
        
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
            self.logger.info(f'Pipeline success: {results}')
            elapsed_time = time.time() - self.start_time
            log_pipeline_end(self.logger, "Canopy Height Post-Processing Pipeline", overall_success, elapsed_time)
            
            return overall_success
            
        except Exception as e:
            self.logger.error(f"Post-processing pipeline failed: {e}")
            return {step.value: False for step in steps}
