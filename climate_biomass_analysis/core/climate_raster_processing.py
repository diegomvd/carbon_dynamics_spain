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

warnings.filterwarnings('ignore')


class ClimateProcessor:
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
        self.target_crs = self.climate_config['target_crs']
        self.resampling_method = getattr(Resampling, self.climate_config['resampling_method'].upper())
        
        self.logger.info(f"Initialized ClimateProcessor with target CRS: {self.target_crs}")
    
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
            boundary_path = self.config['data']['spain_boundary']
        
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
        output_path: Union[str, Path],
        reference_grid: Optional[Union[str, Path]] = None,
        clip_to_spain: bool = True
    ) -> bool:
        """
        Process a single GRIB file to GeoTIFF.
        
        Args:
            grib_path: Path to input GRIB file
            output_path: Path for output GeoTIFF file
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
            
            # Get the data variable (assume first non-coordinate variable)
            data_vars = [var for var in xr_data.data_vars]
            if not data_vars:
                raise ValueError("No data variables found in GRIB file")
            
            var_name = data_vars[0]
            da = xr_data[var_name]
            
            # Ensure proper coordinate names
            if 'longitude' in da.dims:
                da = da.rename({'longitude': 'x', 'latitude': 'y'})
            elif 'lon' in da.dims:
                da = da.rename({'lon': 'x', 'lat': 'y'})
            
            # Set CRS (assume WGS84 for GRIB files)
            da = da.rio.write_crs("EPSG:4326")
            
            # Reproject to target CRS
            if reference_grid:
                reprojected = self.reference_grid_reprojection(da, reference_grid)
            else:
                reprojected = self.standard_reprojection(da, self.target_crs)
            
            # Clip to Spain if requested
            if clip_to_spain:
                reprojected = self.clip_to_spain(reprojected)
            
            # Ensure output directory exists
            ensure_directory(Path(output_path).parent)
            
            # Save as GeoTIFF
            reprojected.rio.to_raster(
                output_path,
                compress=self.climate_config['harmonization']['compress']
            )
            
            self.logger.info(f"Saved: {Path(output_path).name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {grib_path}: {e}")
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
            # Generate output filename
            output_name = grib_file.stem + ".tif"
            output_path = output_dir / output_name
            
            # Process file
            success = self.process_grib_file(
                grib_file, 
                output_path, 
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