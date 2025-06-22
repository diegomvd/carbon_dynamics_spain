#!/usr/bin/env python3
"""
Data Coverage Processing Utility

Processes PNOA/Sentinel-2 raster files into binary coverage maps for visualization.
This utility is specifically for Figure S1 data coverage processing.

Author: Diego Bengochea
"""

import os
import glob
import numpy as np
import rasterio
from rasterio import features, warp
from rasterio.enums import Resampling
import geopandas as gpd
from pathlib import Path
import logging
from tqdm import tqdm
import dask
from dask import delayed
from dask.distributed import Client
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# Default configuration
DEFAULT_TARGET_RESOLUTION = 100  # meters
DEFAULT_TARGET_CRS = "EPSG:25830"  # ETRS89 / UTM zone 30N (Spain)
DEFAULT_N_WORKERS = 14
DEFAULT_CHUNK_SIZE = 32

def setup_dask_client(n_workers=DEFAULT_N_WORKERS, threads_per_worker=1, memory_limit="12GB"):
    """Initialize Dask client for parallel processing."""
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=':8787'
    )
    logging.info(f"Dask client initialized: {client}")
    return client

def find_raster_files(raster_dir):
    """Find all TIFF files in directory."""
    logging.info(f"Searching for TIFF files in: {raster_dir}")
    
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    raster_files = []
    
    for pattern in tiff_patterns:
        raster_files.extend(glob.glob(os.path.join(raster_dir, pattern)))
    
    logging.info(f"Found {len(raster_files)} TIFF files")
    
    if len(raster_files) == 0:
        raise FileNotFoundError(f"No TIFF files found in {raster_dir}")
    
    return sorted(raster_files)

def get_combined_bounds(raster_files):
    """Get combined bounds from all raster files."""
    logging.info(f"Calculating bounds from {len(raster_files)} rasters...")
    
    bounds_list = []
    for raster_file in tqdm(raster_files, desc="Getting bounds"):
        try:
            with rasterio.open(raster_file) as src:
                bounds_list.append(src.bounds)
        except Exception as e:
            logging.warning(f"Could not read {raster_file}: {e}")
    
    if bounds_list:
        left = min(bounds.left for bounds in bounds_list)
        bottom = min(bounds.bottom for bounds in bounds_list)
        right = max(bounds.right for bounds in bounds_list)
        top = max(bounds.top for bounds in bounds_list)
        combined_bounds = (left, bottom, right, top)
        
        logging.info(f"Combined bounds: {combined_bounds}")
        return combined_bounds
    else:
        raise ValueError("Could not determine bounds from raster files")

def calculate_target_grid(bounds, target_resolution=DEFAULT_TARGET_RESOLUTION, target_crs=DEFAULT_TARGET_CRS):
    """Calculate target grid parameters."""
    left, bottom, right, top = bounds
    
    # Calculate dimensions at target resolution
    width = int(np.ceil((right - left) / target_resolution))
    height = int(np.ceil((top - bottom) / target_resolution))
    
    # Create transform
    target_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    
    logging.info(f"Target grid: {width} x {height} at {target_resolution}m resolution")
    
    return target_transform, (width, height)

@delayed
def process_raster_chunk(raster_files_chunk, target_transform, target_shape, target_crs):
    """Process a chunk of raster files in parallel."""
    width, height = target_shape
    chunk_coverage = np.zeros((height, width), dtype=np.uint8)
    
    processed_count = 0
    error_count = 0
    
    for raster_file in raster_files_chunk:
        try:
            with rasterio.open(raster_file) as src:
                # Reproject and resample
                resampled_data = np.zeros((height, width), dtype=np.float32)
                
                warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                    num_threads=1
                )
                
                # Create binary mask: 1 where valid data exists
                if src.nodata is not None:
                    valid_mask = (resampled_data != src.nodata) & ~np.isnan(resampled_data)
                else:
                    valid_mask = ~np.isnan(resampled_data)
                
                # Update coverage array (union operation)
                chunk_coverage = np.logical_or(chunk_coverage, valid_mask).astype(np.uint8)
                processed_count += 1
                
        except Exception as e:
            error_count += 1
            continue
    
    return chunk_coverage, processed_count, error_count

def process_all_rasters(raster_files, target_transform, target_shape, target_crs, chunk_size=DEFAULT_CHUNK_SIZE):
    """Process all rasters using Dask parallelization."""
    logging.info(f"Processing {len(raster_files)} rasters in parallel...")
    
    # Split files into chunks
    chunks = [raster_files[i:i+chunk_size] for i in range(0, len(raster_files), chunk_size)]
    logging.info(f"Created {len(chunks)} chunks of ~{chunk_size} files each")
    
    # Create delayed tasks
    delayed_tasks = [
        process_raster_chunk(chunk, target_transform, target_shape, target_crs) 
        for chunk in chunks
    ]
    
    # Execute tasks and collect results
    logging.info("Executing parallel processing...")
    results = dask.compute(*delayed_tasks)
    
    # Combine results
    width, height = target_shape
    final_coverage = np.zeros((height, width), dtype=np.uint8)
    total_processed = 0
    total_errors = 0
    
    for chunk_coverage, processed_count, error_count in results:
        final_coverage = np.logical_or(final_coverage, chunk_coverage).astype(np.uint8)
        total_processed += processed_count
        total_errors += error_count
    
    logging.info(f"Successfully processed: {total_processed} rasters")
    logging.info(f"Errors encountered: {total_errors} rasters")
    
    return final_coverage

def apply_polygon_mask(coverage_array, shapefile_path, target_transform, target_crs):
    """Apply polygon mask to limit coverage to study area."""
    logging.info("Applying polygon mask...")
    
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    logging.info(f"Loaded shapefile with {len(gdf)} polygons")
    
    # Reproject to match raster CRS if needed
    if gdf.crs != target_crs:
        logging.info(f"Reprojecting shapefile from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    
    # Create mask from polygon(s)
    polygon_geoms = [geom for geom in gdf.geometry if geom is not None]
    
    if not polygon_geoms:
        logging.warning("No valid geometries found in shapefile")
        return coverage_array
    
    # Rasterize polygon
    polygon_mask = features.rasterize(
        polygon_geoms,
        out_shape=coverage_array.shape,
        transform=target_transform,
        fill=0,  # Outside polygon
        default_value=1,  # Inside polygon
        dtype=np.uint8
    )
    
    # Apply mask: keep coverage data only where polygon_mask = 1
    masked_coverage = coverage_array * polygon_mask
    
    logging.info(f"Pixels with data inside polygon: {np.sum(masked_coverage):,}")
    logging.info(f"Coverage within polygon: {np.sum(masked_coverage)/np.sum(polygon_mask)*100:.2f}%")
    
    return masked_coverage

def save_coverage_raster(coverage_array, output_path, target_transform, target_crs):
    """Save the final coverage array as a GeoTIFF."""
    logging.info(f"Saving output to {output_path}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    height, width = coverage_array.shape
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=coverage_array.dtype,
        crs=target_crs,
        transform=target_transform,
        compress='lzw',
        tiled=True,
        blockxsize=512,
        blockysize=512,
        predictor=2
    ) as dst:
        dst.write(coverage_array, 1)
        dst.set_band_description(1, "Data Coverage: 1=Data Present, 0=No Data")

def process_data_coverage(raster_dir, shapefile_path, output_path, 
                         target_resolution=DEFAULT_TARGET_RESOLUTION,
                         target_crs=DEFAULT_TARGET_CRS,
                         n_workers=DEFAULT_N_WORKERS):
    """
    Main function to process data coverage.
    
    Args:
        raster_dir: Directory containing TIFF files
        shapefile_path: Path to boundary shapefile
        output_path: Output path for coverage raster
        target_resolution: Target resolution in meters
        target_crs: Target CRS
        n_workers: Number of parallel workers
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize Dask
    client = setup_dask_client(n_workers=n_workers)
    
    try:
        # Find raster files
        raster_files = find_raster_files(raster_dir)
        
        # Get combined bounds
        combined_bounds = get_combined_bounds(raster_files)
        
        # Calculate target grid
        target_transform, target_shape = calculate_target_grid(
            combined_bounds, target_resolution, target_crs
        )
        
        # Process all rasters
        coverage_array = process_all_rasters(
            raster_files, target_transform, target_shape, target_crs
        )
        
        # Apply polygon mask if shapefile provided
        if shapefile_path and os.path.exists(shapefile_path):
            coverage_array = apply_polygon_mask(
                coverage_array, shapefile_path, target_transform, target_crs
            )
        
        # Save output
        save_coverage_raster(coverage_array, output_path, target_transform, target_crs)
        
        # Summary statistics
        total_pixels = coverage_array.size
        data_pixels = np.sum(coverage_array)
        coverage_percentage = (data_pixels / total_pixels) * 100
        
        logging.info("=" * 60)
        logging.info("PROCESSING COMPLETE")
        logging.info(f"Output file: {output_path}")
        logging.info(f"Total pixels: {total_pixels:,}")
        logging.info(f"Pixels with data: {data_pixels:,}")
        logging.info(f"Data coverage: {coverage_percentage:.2f}%")
        logging.info("=" * 60)
        
    finally:
        # Clean up Dask client
        client.close()

if __name__ == "__main__":
    # Example usage
    raster_dir = "/path/to/raster/files"
    shapefile_path = "/path/to/boundary.shp"
    output_path = "/path/to/output/coverage.tif"
    
    process_data_coverage(raster_dir, shapefile_path, output_path)