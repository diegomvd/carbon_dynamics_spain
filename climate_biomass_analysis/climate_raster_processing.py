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
import yaml
import logging
from pathlib import Path


def setup_logging():
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path="climate_biomass_config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise


def estimate_resolution_meters(data):
    """
    Estimate a suitable resolution in meters for reprojection based on the original data.
    
    Args:
        data (xarray.DataArray): Input data array with x,y coordinates
        
    Returns:
        float: Estimated resolution in meters
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
        print(f"Error estimating resolution: {e}")
        # Default resolution if estimation fails (1km)
        return 1000


def standard_reprojection(data, output_crs):
    """
    Standard reprojection without reference grid.
    
    Args:
        data (xarray.DataArray): Input data array
        output_crs (str): Target CRS (e.g., "EPSG:25830")
        
    Returns:
        tuple: (reprojected_data, output_crs)
    """
    try:
        # Estimate resolution
        resolution = estimate_resolution_meters(data)
        print(f"Estimated resolution for reprojection: {resolution}m")
        
        # Reproject using rioxarray
        reprojected = data.rio.reproject(
            output_crs,
            resolution=resolution,
            resampling=Resampling.bilinear
        )
        return reprojected, output_crs
    except Exception as e:
        print(f"Standard reprojection error: {e}")
        print("Returning original data with WGS84 CRS")
        return data, "EPSG:4326"


def process_climate_anomalies(config):
    """
    Process climate anomaly GRIB files for Spain and save as properly georeferenced TIFFs.
    
    Args:
        config (dict): Configuration dictionary containing paths and parameters
    """
    logger = setup_logging()
    logger.info("Starting climate anomaly processing...")
    
    # Extract config parameters
    grib_files_pattern = config['paths']['climate_grib_pattern']
    spain_shapefile = config['paths']['spain_shapefile']
    output_folder = config['paths']['climate_outputs']
    output_crs = config['climate']['output_crs']
    reference_raster = config['paths'].get('reference_raster')
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Reference grid parameters
    reference_transform = None
    reference_height = None
    reference_width = None
    reference_crs = None
    
    # Load reference raster if provided for perfect alignment
    if reference_raster:
        try:
            logger.info(f"Loading reference raster for alignment: {reference_raster}")
            with rasterio.open(reference_raster) as src:
                reference_transform = src.transform
                reference_height = src.height
                reference_width = src.width
                reference_crs = src.crs
                
                # Override output_crs with reference raster's CRS if it exists
                if reference_crs:
                    output_crs = reference_crs
                    logger.info(f"Using reference raster's CRS: {output_crs}")
        except Exception as e:
            logger.warning(f"Could not load reference raster: {e}")
            logger.info("Continuing without reference grid alignment")
    
    # Load Spain shapefile
    spain = gpd.read_file(spain_shapefile)
    spain = spain.to_crs("EPSG:4326")  # Ensure shapefile is in WGS84 for initial processing
    
    # Find all GRIB files
    grib_files = glob.glob(grib_files_pattern)
    logger.info(f"Found {len(grib_files)} GRIB files")
    
    # Process each GRIB file
    for grib_file in grib_files:
        logger.info(f"Processing {grib_file}")
        
        try:
            # Load GRIB file using earthkit-data
            ds = ek.from_source("file", grib_file)
            
            # Convert to xarray Dataset
            ds_xr = ds.to_xarray()
            
            # Print available variables and dimensions for debugging
            logger.info(f"Available variables: {list(ds_xr.data_vars)}")
            logger.info(f"Available dimensions: {list(ds_xr.dims)}")
            logger.info(f"Available coordinates: {list(ds_xr.coords)}")
            
            # Process by forecast reference time
            if 'forecast_reference_time' in ds_xr.coords:
                for date in ds_xr.forecast_reference_time:
                    ds_xr_date = ds_xr.sel(forecast_reference_time=date)
                    year = str(date.values)[:4]
                    month = str(date.values)[5:7]
                    logger.info(f"Processing date: {year}-{month}")
                    
                    # Process each variable in the dataset
                    for var in ds_xr.data_vars:
                        logger.info(f"Processing variable: {var}")
                        var_data = ds_xr_date[var]
                        
                        # Handle coordinate names - GRIB files might use different conventions
                        lon_names = ['longitude', 'lon', 'x']
                        lat_names = ['latitude', 'lat', 'y']
                        
                        lon_dim = next((dim for dim in lon_names if dim in var_data.dims), None)
                        lat_dim = next((dim for dim in lat_names if dim in var_data.dims), None)
                        
                        if not lon_dim or not lat_dim:
                            logger.warning(f"Could not identify lat/lon dimensions in {var}")
                            logger.info(f"Available dimensions: {var_data.dims}")
                            continue
                        
                        # Explicitly set coordinate names for rioxarray if needed
                        if lon_dim != 'x' or lat_dim != 'y':
                            var_data = var_data.rename({lon_dim: 'x', lat_dim: 'y'})
                        
                        # Ensure longitudes are within -180 to 180 range
                        if 'x' in var_data.coords and var_data.coords['x'].max() > 180:
                            logger.info("Converting longitudes from 0-360 to -180-180 range")
                            var_data.coords['x'] = (var_data.coords['x'] + 180) % 360 - 180
                            var_data = var_data.sortby('x')
                        
                        # Set CRS explicitly to WGS84 (EPSG:4326)
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning)
                                var_data = var_data.rio.write_crs(4326)
                        except AttributeError:
                            logger.error(f"rioxarray could not set CRS for {var}. Skipping.")
                            continue
                        
                        # If there's still a time dimension, average it (e.g., for monthly averages)
                        if 'time' in var_data.dims and var_data.time.size > 1:
                            var_data = var_data.mean(dim='time', keepdims=False)
                        
                        # Clip to Spain, include pixels that partially intersect with the boundary
                        try:
                            clipped_data = var_data.rio.clip(
                                spain.geometry, 
                                spain.crs, 
                                drop=False,
                                all_touched=True
                            )
                        except Exception as e:
                            logger.warning(f"Clipping error: {e}")
                            clipped_data = var_data
                            logger.info("Using full extent instead")
                        
                        # Reproject using reference grid if available, otherwise standard reprojection
                        if reference_transform is not None and reference_height is not None and reference_width is not None:
                            try:
                                logger.info("Reprojecting to match reference grid exactly")
                                
                                # Reproject to match the reference grid exactly
                                reprojected_data = clipped_data.rio.reproject(
                                    output_crs,
                                    transform=reference_transform,
                                    height=reference_height,
                                    width=reference_width,
                                    resampling=Resampling.bilinear
                                )
                                
                                output_crs_to_use = output_crs
                                
                            except Exception as e:
                                logger.warning(f"Error in reference grid reprojection: {e}")
                                logger.info("Falling back to standard reprojection")
                                reprojected_data, output_crs_to_use = standard_reprojection(clipped_data, output_crs)
                        else:
                            # Standard reprojection without reference grid
                            reprojected_data, output_crs_to_use = standard_reprojection(clipped_data, output_crs)
                        
                        # Save as GeoTIFF
                        output_file = os.path.join(output_folder, f"{var}_{year}_{month}.tif")
                        
                        # Save with rioxarray
                        reprojected_data.rio.to_raster(
                            output_file,
                            driver="GTiff",
                            dtype="float32",
                            crs=output_crs_to_use,
                            nodata=np.nan
                        )
                        logger.info(f"Saved {output_file} in {output_crs_to_use}")
            else:
                logger.info("No 'forecast_reference_time' dimension found. Processing all variables as-is.")
                
                # Process each variable in the dataset
                for var in ds_xr.data_vars:
                    logger.info(f"Processing variable: {var}")
                    var_data = ds_xr[var]
                    
                    # Handle coordinate names - GRIB files might use different conventions
                    lon_names = ['longitude', 'lon', 'x']
                    lat_names = ['latitude', 'lat', 'y']
                    
                    lon_dim = next((dim for dim in lon_names if dim in var_data.dims), None)
                    lat_dim = next((dim for dim in lat_names if dim in var_data.dims), None)
                    
                    if not lon_dim or not lat_dim:
                        logger.warning(f"Could not identify lat/lon dimensions in {var}")
                        logger.info(f"Available dimensions: {var_data.dims}")
                        continue
                    
                    # Explicitly set coordinate names for rioxarray if needed
                    if lon_dim != 'x' or lat_dim != 'y':
                        var_data = var_data.rename({lon_dim: 'x', lat_dim: 'y'})
                    
                    # Ensure longitudes are within -180 to 180 range
                    if 'x' in var_data.coords and var_data.coords['x'].max() > 180:
                        logger.info("Converting longitudes from 0-360 to -180-180 range")
                        var_data.coords['x'] = (var_data.coords['x'] + 180) % 360 - 180
                        var_data = var_data.sortby('x')
                    
                    # Set CRS explicitly to WGS84 (EPSG:4326)
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            var_data = var_data.rio.write_crs(4326)
                    except AttributeError:
                        logger.error(f"rioxarray could not set CRS for {var}. Skipping.")
                        continue
                    
                    # If there's a time dimension, process each year
                    if 'time' in var_data.dims:
                        # Get unique years
                        years = np.unique([str(pd.Timestamp(t.values).year) for t in var_data.time])
                        
                        for year in years:
                            # Filter data for this year
                            year_data = var_data.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                            
                            if year_data.size == 0:
                                logger.info(f"No data found for {var} in {year}")
                                continue
                                
                            # Take the average for the year
                            if year_data.time.size > 1:
                                year_data = year_data.mean(dim='time', keepdims=False)
                            
                            # Process like above
                            try:
                                clipped_data = year_data.rio.clip(
                                    spain.geometry, 
                                    spain.crs, 
                                    drop=False,
                                    all_touched=True
                                )
                            except Exception as e:
                                logger.warning(f"Clipping error: {e}")
                                clipped_data = year_data
                            
                            if reference_transform is not None:
                                try:
                                    reprojected_data = clipped_data.rio.reproject(
                                        output_crs,
                                        transform=reference_transform,
                                        height=reference_height,
                                        width=reference_width,
                                        resampling=Resampling.bilinear
                                    )
                                    output_crs_to_use = output_crs
                                except Exception as e:
                                    reprojected_data, output_crs_to_use = standard_reprojection(clipped_data, output_crs)
                            else:
                                reprojected_data, output_crs_to_use = standard_reprojection(clipped_data, output_crs)
                            
                            output_file = os.path.join(output_folder, f"{var}_{year}.tif")
                            reprojected_data.rio.to_raster(
                                output_file,
                                driver="GTiff",
                                dtype="float32",
                                crs=output_crs_to_use,
                                nodata=np.nan
                            )
                            logger.info(f"Saved {output_file} in {output_crs_to_use}")
                    else:
                        # No time dimension, just process as is
                        try:
                            clipped_data = var_data.rio.clip(
                                spain.geometry, 
                                spain.crs, 
                                drop=False,
                                all_touched=True
                            )
                        except Exception as e:
                            logger.warning(f"Clipping error: {e}")
                            clipped_data = var_data
                        
                        if reference_transform is not None:
                            try:
                                reprojected_data = clipped_data.rio.reproject(
                                    output_crs,
                                    transform=reference_transform,
                                    height=reference_height,
                                    width=reference_width,
                                    resampling=Resampling.bilinear
                                )
                                output_crs_to_use = output_crs
                            except Exception as e:
                                reprojected_data, output_crs_to_use = standard_reprojection(clipped_data, output_crs)
                        else:
                            reprojected_data, output_crs_to_use = standard_reprojection(clipped_data, output_crs)
                        
                        output_file = os.path.join(output_folder, f"{var}.tif")
                        reprojected_data.rio.to_raster(
                            output_file,
                            driver="GTiff",
                            dtype="float32",
                            crs=output_crs_to_use,
                            nodata=np.nan
                        )
                        logger.info(f"Saved {output_file} in {output_crs_to_use}")
                
        except Exception as e:
            logger.error(f"Error processing file {grib_file}: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run climate raster processing."""
    logger = setup_logging()
    logger.info("Starting climate raster processing pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Process climate anomalies
        process_climate_anomalies(config)
        
        logger.info("Climate raster processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
