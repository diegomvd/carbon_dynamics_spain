"""
Climate raster processing pipeline for GRIB to GeoTIFF conversion.

Processes climate anomaly GRIB files for Spain and saves as properly georeferenced TIFFs
with optional reference grid alignment. Handles coordinate transformations, clipping,
and reprojection to EPSG:25830.

Author: Diego Bengochea
"""

import os
import glob
import earthkit.data as ek
import xarray as xr
import geopandas as gpd
import rioxarray
import numpy as np
from pyproj import CRS
import warnings
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils.central_data_paths_constants import *

warnings.filterwarnings('ignore')


class ClimateRasterConversionPipeline:
    """
    Climate raster processing pipeline for GRIB to GeoTIFF conversion.
    
    This class handles the conversion of climate data from GRIB format to properly
    georeferenced GeoTIFF files, with coordinate transformations, clipping to Spain,
    and reprojection to the target CRS.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the climate processor.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='climate_processing',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration
        self.climate_config = self.config['climate_processing']
        self.target_crs = self.config['geographic']['target_crs']
        self.resampling_method = getattr(Resampling, self.climate_config['resampling_method'])
        
        self.input_dir = CLIMATE_RAW_DIR
        self.output_dir = CLIMATE_RASTERS_RAW_DIR

        self.logger.info(f"Initialized ClimateProcessingPipeline with target CRS: {self.target_crs}")

    def run_full_pipeline(self):

        self.logger.info(f"Processing directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")   
    
        results = self.process_directory(self.input_dir,self.output_dir)

        # Report results
        successful = sum(results.values())
        total = len(results)
        self.logger.info(f"Processing completed: {successful}/{total} files successful")
        
        if successful < total:
            failed_files = [f for f, success in results.items() if not success]
            self.logger.warning(f"Failed files: {failed_files}")
    
        self.logger.info("Validating output files...")
        validation_result = self.validate_outputs(self.output_dir)
        
        if validation_result:
            self.logger.info("Climate processing completed: all output files validated successfully")
        else:
            self.logger.error("Output validation failed")
        
        return validation_result 

    def estimate_resolution_meters(self, data: xr.DataArray) -> float:
        """
        Estimate a suitable resolution in meters for reprojection based on the original data.
        
        Args:
            data: Input data array with x,y coordinates
            
        Returns:
            Estimated resolution in meters
        """
        try:
            # Get coordinate spacing
            x_coords = data.x.values
            y_coords = data.y.values
            
            # Calculate average spacing in degrees
            x_spacing = np.mean(np.abs(np.diff(x_coords)))
            y_spacing = np.mean(np.abs(np.diff(y_coords)))
            
            # Convert degrees to meters (approximate for Spain ~40°N)
            longitude_factor = 85000  # meters per degree longitude at ~40°N
            latitude_factor = 111000  # meters per degree latitude
            
            x_meters = x_spacing * longitude_factor
            y_meters = y_spacing * latitude_factor
            
            # Use the average as our resolution
            resolution = (x_meters + y_meters) / 2
            
            # Round to a nice number
            if resolution < 100:
                return round(resolution / 10) * 10  # Round to nearest 10m
            elif resolution < 1000:
                return round(resolution / 100) * 100  # Round to nearest 100m
            else:
                return round(resolution / 1000) * 1000  # Round to nearest 1000m
                
        except Exception as e:
            self.logger.warning(f"Error estimating resolution: {e}")
            # Default resolution if estimation fails (1km)
            return 1000
    
    def standard_reprojection(self, data: xr.DataArray, output_crs: str) -> xr.DataArray:
        """
        Standard reprojection without reference grid.
        
        Args:
            data: Input data array
            output_crs: Target coordinate reference system
            
        Returns:
            Reprojected data array
        """
        # Estimate resolution
        resolution = self.estimate_resolution_meters(data)
        self.logger.debug(f"Using estimated resolution: {resolution}m")
        
        # Reproject to target CRS
        reprojected = data.rio.reproject(
            output_crs,
            resolution=resolution,
            resampling=self.resampling_method
        )
        
        return reprojected
    
    def reference_grid_reprojection(
        self, 
        data: xr.DataArray, 
        reference_path: Union[str, Path]
    ) -> xr.DataArray:
        """
        Reproject data to match a reference grid exactly.
        
        Args:
            data: Input data array
            reference_path: Path to reference raster file
            
        Returns:
            Reprojected data array matching reference grid
        """
        with rasterio.open(reference_path) as ref:
            # Get reference grid properties
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            
            self.logger.debug(f"Reference grid: {ref_width}x{ref_height}, CRS: {ref_crs}")
            
            # Reproject to match reference exactly
            reprojected = data.rio.reproject_match(
                rioxarray.open_rasterio(reference_path),
                resampling=self.resampling_method
            )
            
            return reprojected
    
    def load_spain_boundary(self, boundary_path: Optional[Union[str, Path]] = None) -> gpd.GeoDataFrame:
        """
        Load Spain boundary shapefile for clipping.
        
        Args:
            boundary_path: Path to boundary shapefile. If None, uses config default.
            
        Returns:
            Spain boundary GeoDataFrame
        """
        if boundary_path is None:
            boundary_path = SPAIN_BOUNDARIES_FILE
        
        if not Path(boundary_path).exists():
            self.logger.warning(f"Spain boundary file not found: {boundary_path}")
            return None
        
        spain = gpd.read_file(boundary_path)
        self.logger.debug(f"Loaded Spain boundary with {len(spain)} features")
        return spain
    
    def clip_to_spain(
        self, 
        data: xr.DataArray, 
        spain_boundary: Optional[gpd.GeoDataFrame] = None
    ) -> xr.DataArray:
        """
        Clip data to Spain boundary.
        
        Args:
            data: Input data array
            spain_boundary: Spain boundary GeoDataFrame. If None, loads from config.
            
        Returns:
            Clipped data array
        """
        if spain_boundary is None:
            spain_boundary = self.load_spain_boundary()
        
        if spain_boundary is None:
            self.logger.warning("No Spain boundary available, skipping clipping")
            return data
        
        try:
            # Ensure CRS match
            if data.rio.crs != spain_boundary.crs:
                spain_boundary = spain_boundary.to_crs(data.rio.crs)
            
            # Clip to Spain
            clipped = data.rio.clip(spain_boundary.geometry.values, data.rio.crs)
            self.logger.debug("Successfully clipped data to Spain boundary")
            return clipped
            
        except Exception as e:
            self.logger.warning(f"Error clipping to Spain boundary: {e}")
            return data
    
    def process_grib_file(
        self,
        grib_path: Union[str, Path],
        output_dir: Union[str, Path],  # Changed from single output_path to output_dir
        reference_grid: Optional[Union[str, Path]] = None,
        clip_to_spain: bool = True
    ) -> bool:
        """
        Process a single GRIB file to multiple GeoTIFFs (one per variable per time).
        
        Args:
            grib_path: Path to input GRIB file
            output_dir: Directory for output GeoTIFF files
            reference_grid: Optional reference grid for exact alignment
            clip_to_spain: Whether to clip output to Spain boundary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Processing: {Path(grib_path).name}")
            
            # Load GRIB data
            data = ek.from_source("file", grib_path)
            xr_data = data.to_xarray()
            
            self.logger.debug(f"Available variables: {list(xr_data.data_vars)}")
            self.logger.debug(f"Available coordinates: {list(xr_data.coords)}")
            
            # Process by forecast reference time if available
            if 'forecast_reference_time' in xr_data.coords:
                for date in xr_data.forecast_reference_time:
                    ds_xr_date = xr_data.sel(forecast_reference_time=date)
                    year = str(date.values)[:4]
                    month = str(date.values)[5:7]
                    self.logger.debug(f"Processing date: {year}-{month}")
                    
                    # Process each variable in the dataset
                    for var in xr_data.data_vars:
                        self.logger.debug(f"Processing variable: {var}")
                        
                        success = self._process_variable(
                            ds_xr_date[var], 
                            var, 
                            f"{year}-{month}",
                            output_dir,
                            reference_grid,
                            clip_to_spain
                        )
                        
                        if not success:
                            self.logger.warning(f"Failed to process {var} for {year}-{month}")
            else:
                # No forecast_reference_time, process each variable as-is
                self.logger.debug("No 'forecast_reference_time' found, processing variables directly")
                
                for var in xr_data.data_vars:
                    self.logger.debug(f"Processing variable: {var}")
                    
                    success = self._process_variable(
                        xr_data[var],
                        var,
                        "processed",  # Default suffix
                        output_dir,
                        reference_grid,
                        clip_to_spain
                    )
                    
                    if not success:
                        self.logger.warning(f"Failed to process {var}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {grib_path}: {e}")
            return False
    
    def _process_variable(
        self,
        var_data: xr.DataArray,
        var_name: str,
        time_suffix: str,
        output_dir: Union[str, Path],
        reference_grid: Optional[Union[str, Path]] = None,
        clip_to_spain: bool = True
    ) -> bool:
        """
        Process a single variable to GeoTIFF.
        
        Args:
            var_data: Variable data array
            var_name: Variable name
            time_suffix: Time suffix for filename
            output_dir: Output directory
            reference_grid: Optional reference grid
            clip_to_spain: Whether to clip to Spain
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle coordinate names
            lon_names = ['longitude', 'lon', 'x']
            lat_names = ['latitude', 'lat', 'y']
            
            lon_dim = next((dim for dim in lon_names if dim in var_data.dims), None)
            lat_dim = next((dim for dim in lat_names if dim in var_data.dims), None)
            
            if not lon_dim or not lat_dim:
                self.logger.warning(f"Could not identify lat/lon dimensions in {var_name}")
                return False
            
            # Rename coordinates if needed
            if lon_dim != 'x' or lat_dim != 'y':
                var_data = var_data.rename({lon_dim: 'x', lat_dim: 'y'})
            
            # Handle longitude range
            if 'x' in var_data.coords and var_data.coords['x'].max() > 180:
                self.logger.debug("Converting longitudes from 0-360 to -180-180 range")
                var_data.coords['x'] = (var_data.coords['x'] + 180) % 360 - 180
                var_data = var_data.sortby('x')
            
            # Set CRS
            var_data = var_data.rio.write_crs("EPSG:4326")
            
            # Average time dimension if present and has multiple values
            if 'time' in var_data.dims and var_data.time.size > 1:
                var_data = var_data.mean(dim='time', keepdims=False)
            
            # Reproject
            if reference_grid:
                reprojected = self.reference_grid_reprojection(var_data, reference_grid)
            else:
                reprojected = self.standard_reprojection(var_data, self.target_crs)
            
            # Clip to Spain if requested
            if clip_to_spain:
                reprojected = self.clip_to_spain(reprojected)
            
            # Create output filename
            output_filename = f"{var_name}_{time_suffix}.tif"
            output_path = Path(output_dir) / output_filename
            
            # Ensure output directory exists
            ensure_directory(Path(output_path).parent)
            
            # Save as GeoTIFF
            reprojected.rio.to_raster(
                output_path,
                compress=self.climate_config.get('compress', 'lzw')
            )
            
            self.logger.debug(f"Saved: {output_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing variable {var_name}: {e}")
            return False

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.grib",
        reference_grid: Optional[Union[str, Path]] = None,
        clip_to_spain: bool = True
    ) -> Dict[str, bool]:
        """
        Process all GRIB files in a directory.
        
        Args:
            input_dir: Input directory containing GRIB files
            output_dir: Output directory for GeoTIFF files
            pattern: File pattern to match (default: "*.grib")
            reference_grid: Optional reference grid for exact alignment
            clip_to_spain: Whether to clip outputs to Spain boundary
            
        Returns:
            Dictionary mapping input files to processing success status
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find GRIB files
        grib_files = list(input_dir.glob(pattern))
        
        if not grib_files:
            self.logger.warning(f"No GRIB files found matching pattern: {pattern}")
            return {}
        
        self.logger.info(f"Found {len(grib_files)} GRIB files to process")
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Process each file
        results = {}
        for grib_file in grib_files:
            success = self.process_grib_file(
                grib_file, 
                output_dir,  # ← Pass directory, not single file path
                reference_grid=reference_grid,
                clip_to_spain=clip_to_spain
            )
            
            results[str(grib_file)] = success
        
        # Summary
        successful = sum(results.values())
        self.logger.info(f"Processing complete: {successful}/{len(grib_files)} files successful")
        
        return results
    
    def validate_outputs(self, output_dir: Union[str, Path]) -> bool:
        """
        Validate that output files are properly formatted GeoTIFFs.
        
        Args:
            output_dir: Directory containing output files
            
        Returns:
            True if all files are valid, False otherwise
        """
        output_dir = Path(output_dir)
        tif_files = list(output_dir.glob("*.tif"))
        
        if not tif_files:
            self.logger.warning("No GeoTIFF files found for validation")
            return False
        
        invalid_files = []
        for tif_file in tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    # Check basic properties
                    if src.crs is None:
                        invalid_files.append(f"{tif_file.name}: No CRS")
                    elif str(src.crs) != self.target_crs:
                        invalid_files.append(f"{tif_file.name}: Wrong CRS ({src.crs})")
                    elif src.width == 0 or src.height == 0:
                        invalid_files.append(f"{tif_file.name}: Invalid dimensions")
                        
            except Exception as e:
                invalid_files.append(f"{tif_file.name}: Read error - {e}")
        
        if invalid_files:
            self.logger.error(f"Validation failed for {len(invalid_files)} files:")
            for error in invalid_files:
                self.logger.error(f"  {error}")
            return False
        
        self.logger.info(f"Validation successful: {len(tif_files)} files are valid")
        return True