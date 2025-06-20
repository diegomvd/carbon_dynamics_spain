"""
Sanitization and temporal interpolation of canopy height predictions.

This script provides comprehensive post-processing for canopy height predictions,
including outlier detection, data cleaning, and temporal interpolation to fill
gaps in the time series. Implements parallel processing for efficient handling
of large datasets and generates detailed statistics and interpolation masks.

The sanitization process identifies unrealistic height values, marks them as
outliers, and performs temporal interpolation using adjacent years when possible.
Creates comprehensive summaries and quality control outputs.

Author: Diego Bengochea
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from functools import partial
import multiprocessing

# Third-party imports
import numpy as np
import pandas as pd
import rasterio
from scipy import interpolate
from tqdm import tqdm

# Local imports
from config import load_config, setup_logging, create_output_directory


class HeightSanitizer:
    """
    Comprehensive height prediction sanitizer with outlier detection and interpolation.
    
    This class handles the identification and removal of outliers in canopy height
    predictions, followed by temporal interpolation to fill gaps using adjacent
    years when reliable data is available.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the height sanitizer with configuration.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging']['level'])
        
        # Get sanitization configuration
        self.sanitize_config = self.config['post_processing']['sanitize']
        
        # Create output directories
        for directory in [
            self.sanitize_config['processed_dir'],
            self.sanitize_config['interpolated_dir'],
            self.sanitize_config['interpolation_masks_dir']
        ]:
            create_output_directory(Path(directory))
        
        self.logger.info("Height sanitizer initialized")
        self.logger.info(f"Input directory: {self.sanitize_config['input_dir']}")
        self.logger.info(f"Upper threshold: {self.sanitize_config['upper_threshold']}m")
        self.logger.info(f"Number of workers: {self.sanitize_config['num_workers']}")
    
    def extract_info_from_filename(self, filepath: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """
        Extract year, latitude, and longitude from filename.
        
        Args:
            filepath (str): Path to the raster file
            
        Returns:
            Tuple[Optional[int], Optional[float], Optional[float]]: Year, latitude, longitude
        """
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
    
    def get_processed_filename(self, filepath: str) -> str:
        """
        Get the expected processed filename for a given input filepath.
        
        Args:
            filepath (str): Input file path
            
        Returns:
            str: Expected processed filename
        """
        return os.path.join(
            self.sanitize_config['processed_dir'], 
            os.path.basename(filepath).replace('.tif', '_processed.tif')
        )
    
    def get_interpolated_filename(self, year: int, lat: float, lon: float) -> str:
        """
        Get the expected interpolated filename for given parameters.
        
        Args:
            year (int): Year of the data
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            
        Returns:
            str: Expected interpolated filename
        """
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        abs_lat = abs(lat)
        abs_lon = abs(lon)
        
        return os.path.join(
            self.sanitize_config['interpolated_dir'], 
            f"canopy_height_{year}_{lat_dir}{abs_lat:.1f}_{lon_dir}{abs_lon:.1f}_interpolated.tif"
        )
    
    def identify_outliers(
        self, 
        data: np.ndarray, 
        nodata_value: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify outlier values in raster data.
        
        Args:
            data (np.ndarray): Raster data array
            nodata_value (Optional[float]): NoData value in the raster
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Outlier mask and NoData mask
        """
        # Initialize masks
        outlier_mask = np.zeros_like(data, dtype=bool)
        
        if nodata_value is not None:
            nodata_mask = np.isclose(data, nodata_value, rtol=1e-5)
        else:
            nodata_mask = np.zeros_like(data, dtype=bool)
        
        # Mark negative values as outliers
        outlier_mask = np.logical_or(
            outlier_mask, 
            (data < self.sanitize_config['lower_threshold']) & ~nodata_mask
        )
        
        # Mark values above upper threshold as outliers
        outlier_mask = np.logical_or(
            outlier_mask, 
            (data > self.sanitize_config['upper_threshold']) & ~nodata_mask
        )
        
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
            Dict[str, Any]: Dictionary containing statistics
        """
        valid_data = data[~nodata_mask]
        
        if len(valid_data) == 0:
            return {
                "min": None, "max": None, "mean": None, 
                "std": None, "num_pixels": 0
            }
        
        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "num_pixels": int(valid_data.size)
        }
    
    def process_tile(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a single tile: identify outliers and mark them as NaN.
        
        Args:
            filepath (str): Path to the input raster file
            
        Returns:
            Tuple[str, Dict[str, Any]]: Filepath and processing statistics
        """
        try:
            year, lat, lon = self.extract_info_from_filename(filepath)
            if year is None:
                return filepath, {"error": "Could not extract info from filename"}
            
            # Create output filename
            output_filename = self.get_processed_filename(filepath)
            
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
                if nodata_value is not None:
                    valid_processed = data[
                        ~np.isnan(data) & ~np.isclose(data, nodata_value, rtol=1e-5)
                    ]
                else:
                    valid_processed = data[~np.isnan(data)]
                
                processed_stats = self.calculate_statistics(
                    valid_processed, np.zeros_like(valid_processed, dtype=bool)
                )
                processed_stats.update({
                    "num_negative": int(num_negative),
                    "negative_percent": float(negative_percent),
                    "num_outliers": int(num_outliers),
                    "outlier_percent": float(outlier_percent)
                })
                
                # Update metadata and write output
                meta.update(dtype=rasterio.float32)
                if nodata_value is not None:
                    meta.update(nodata=nodata_value)
                
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                with rasterio.open(output_filename, 'w', **meta) as dst:
                    dst.write(data, 1)
            
            return filepath, {
                "year": year, "lat": lat, "lon": lon,
                "original_stats": original_stats,
                "processed_stats": processed_stats,
                "output_file": output_filename
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {str(e)}")
            return filepath, {"error": str(e)}
    
    def process_location(self, location_data: Tuple[Tuple[float, float], List[str]]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process all years for a single location serially.
        
        Args:
            location_data: Tuple of (location, files) where location is (lat, lon)
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of processing results
        """
        location, files = location_data
        lat, lon = location
        
        self.logger.info(f"Processing location: {lat}, {lon} with {len(files)} tiles")
        
        results = []
        for filepath in tqdm(files, desc=f"Location {lat},{lon}", leave=False):
            result = self.process_tile(filepath)
            results.append(result)
        
        return results
    
    def interpolate_location(
        self, 
        location_data: Tuple[Tuple[float, float], List[Tuple[int, str]]]
    ) -> Tuple[Tuple[float, float], List[Tuple[int, str]]]:
        """
        Interpolate NaN values in time series for a single location.
        
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
        Perform temporal interpolation on raster time series.
        
        Args:
            raster_data: Dictionary mapping years to raster arrays
            nodata_values: Dictionary mapping years to nodata values
            meta: Raster metadata
            years: List of years
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            List[Tuple[int, str]]: List of (year, filepath) tuples for interpolated files
        """
        years_array = np.array(sorted(years))
        shape = next(iter(raster_data.values())).shape
        
        # Create data and mask stacks
        data_stack = np.zeros((len(years), shape[0], shape[1]), dtype=np.float32)
        mask_stack = np.zeros((len(years), shape[0], shape[1]), dtype=bool)
        
        # Fill stacks
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
        
        # Perform interpolation
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
        
        Args:
            interpolated_stack: Stack of interpolated data arrays
            years_array: Array of years
            raster_data: Original raster data
            nodata_values: NoData values for each year
            meta: Raster metadata
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            List[Tuple[int, str]]: List of saved file information
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
    
    def _create_interpolation_mask(
        self,
        data: np.ndarray,
        original_data: np.ndarray,
        nodata_value: Optional[float]
    ) -> np.ndarray:
        """
        Create mask showing interpolation results.
        
        Args:
            data: Interpolated data array
            original_data: Original data array
            nodata_value: NoData value
            
        Returns:
            np.ndarray: Interpolation mask array
        """
        # Mask values: 0=Original, 1=NoData, 2=Interpolated, 3=Failed
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
        """
        Save interpolation mask to file.
        
        Args:
            interp_mask: Interpolation mask array
            meta: Raster metadata
            year: Year of the data
            lat: Latitude coordinate
            lon: Longitude coordinate
        """
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        abs_lat = abs(lat)
        abs_lon = abs(lon)
        
        mask_filename = os.path.join(
            self.sanitize_config['interpolation_masks_dir'],
            f"canopy_height_{year}_{lat_dir}{abs_lat:.1f}_{lon_dir}{abs_lon:.1f}_interp_mask.tif"
        )
        
        mask_meta = meta.copy()
        mask_meta.update(dtype=rasterio.uint8, nodata=None)
        
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        with rasterio.open(mask_filename, 'w', **mask_meta) as dst:
            dst.write(interp_mask, 1)
    
    def analyze_results(self, results_list: List[List[Tuple[str, Dict[str, Any]]]]) -> pd.DataFrame:
        """
        Analyze and summarize processing results.
        
        Args:
            results_list: List of lists of (filepath, stats) tuples
            
        Returns:
            pd.DataFrame: Summary statistics DataFrame
        """
        # Flatten results
        results = [item for sublist in results_list for item in sublist]
        
        data = []
        for filepath, stats in results:
            if "error" in stats:
                self.logger.warning(f"Skipping {filepath} due to error: {stats['error']}")
                continue
            
            if "skipped" in stats and stats["skipped"]:
                row = {
                    "filename": os.path.basename(filepath),
                    "year": stats["year"],
                    "lat": stats["lat"],
                    "lon": stats["lon"],
                    "skipped": True,
                    "filepath": filepath,
                    "processed_file": stats["output_file"]
                }
            else:
                row = {
                    "filename": os.path.basename(filepath),
                    "year": stats["year"],
                    "lat": stats["lat"],
                    "lon": stats["lon"],
                    "original_min": stats["original_stats"]["min"],
                    "original_max": stats["original_stats"]["max"],
                    "original_mean": stats["original_stats"]["mean"],
                    "original_std": stats["original_stats"]["std"],
                    "processed_min": stats["processed_stats"]["min"],
                    "processed_max": stats["processed_stats"]["max"],
                    "processed_mean": stats["processed_stats"]["mean"],
                    "processed_std": stats["processed_stats"]["std"],
                    "num_outliers": stats["processed_stats"]["num_outliers"],
                    "outlier_percent": stats["processed_stats"]["outlier_percent"],
                    "num_negative": stats["processed_stats"]["num_negative"],
                    "negative_percent": stats["processed_stats"]["negative_percent"],
                    "filepath": filepath,
                    "processed_file": stats["output_file"],
                    "skipped": False
                }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values("outlier_percent", ascending=False)
    
    def run_sanitization_pipeline(self) -> None:
        """Run the complete sanitization and interpolation pipeline."""
        self.logger.info("Starting height prediction sanitization pipeline...")
        
        # Find all tile files
        tile_pattern = os.path.join(
            self.sanitize_config['input_dir'], 
            "canopy_height_*.tif"
        )
        tile_files = glob.glob(tile_pattern)
        
        if not tile_files:
            self.logger.error(f"No files found matching pattern in {self.sanitize_config['input_dir']}")
            return
        
        self.logger.info(f"Found {len(tile_files)} tiles to process")
        
        # Group tiles by location
        tiles_by_location = {}
        for filepath in tqdm(tile_files, desc="Grouping tiles by location"):
            year, lat, lon = self.extract_info_from_filename(filepath)
            if year is None:
                continue
            
            location = (lat, lon)
            if location not in tiles_by_location:
                tiles_by_location[location] = []
            tiles_by_location[location].append(filepath)
        
        self.logger.info(f"Grouped tiles into {len(tiles_by_location)} unique locations")
        
        # Step 1: Process tiles in parallel by location
        self.logger.info("Step 1: Processing tiles to identify and handle outliers")
        location_data = list(tiles_by_location.items())
        
        with multiprocessing.Pool(processes=self.sanitize_config['num_workers']) as pool:
            results_list = list(tqdm(
                pool.imap(self.process_location, location_data),
                total=len(location_data),
                desc="Processing locations"
            ))
        
        # Analyze and save summary
        self.logger.info("Analyzing results and generating summary...")
        summary_df = self.analyze_results(results_list)
        summary_df.to_csv(self.sanitize_config['summary_file'], index=False)
        self.logger.info(f"Saved processing summary to {self.sanitize_config['summary_file']}")
        
        # Print summary statistics
        self._print_summary_statistics(summary_df)
        
        # Step 2: Organize data for temporal interpolation
        processed_by_location = {}
        for _, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Organizing for interpolation"):
            location = (row["lat"], row["lon"])
            year = row["year"]
            
            if location not in processed_by_location:
                processed_by_location[location] = []
            processed_by_location[location].append((year, row["processed_file"]))
        
        # Step 3: Perform temporal interpolation in parallel
        self.logger.info("Step 2: Performing temporal interpolation to fill gaps")
        interpolation_data = list(processed_by_location.items())
        
        with multiprocessing.Pool(processes=self.sanitize_config['num_workers']) as pool:
            interpolation_results = list(tqdm(
                pool.imap(self.interpolate_location, interpolation_data),
                total=len(interpolation_data),
                desc="Interpolating locations"
            ))
        
        # Count interpolated files
        total_interpolated = sum(len(files) for _, files in interpolation_results)
        self.logger.info(f"Completed temporal interpolation. Total interpolated tiles: {total_interpolated}")
        
        self.logger.info("Sanitization pipeline completed successfully!")
    
    def _print_summary_statistics(self, summary_df: pd.DataFrame) -> None:
        """Print comprehensive summary statistics."""
        total_tiles = len(summary_df)
        tiles_with_outliers = len(summary_df[summary_df.num_outliers > 0])
        avg_outlier_percent = summary_df.outlier_percent.mean()
        max_outlier_percent = summary_df.outlier_percent.max()
        
        self.logger.info(f"Total tiles processed: {total_tiles}")
        self.logger.info(f"Tiles with outliers: {tiles_with_outliers} ({100*tiles_with_outliers/total_tiles:.1f}%)")
        self.logger.info(f"Average outlier percentage: {avg_outlier_percent:.2f}%")
        self.logger.info(f"Maximum outlier percentage: {max_outlier_percent:.2f}%")


def main():
    """Main entry point for the sanitization pipeline."""
    try:
        sanitizer = HeightSanitizer()
        sanitizer.run_sanitization_pipeline()
        
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Sanitization pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()