"""
Utility functions for Sentinel-2 satellite imagery processing.

This module provides comprehensive utilities for Sentinel-2 L2A data processing including:
- Scene Classification Layer (SCL) masking and scene selection algorithms
- STAC catalog integration with recursive cloud threshold handling
- Distributed computing with optimized Dask cluster management
- Geospatial operations for tiling and coordinate transformations
- Data loading and preprocessing for large-scale satellite imagery

Key algorithms preserve exact logic for performance-critical operations including
best scene selection based on valid pixel percentage and distributed processing
patterns optimized for memory management.

Author: Diego Bengochea
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import odc.geo
import itertools
import dask.distributed
from odc.stac import load
from contextlib import contextmanager
import gc
import time
import psutil
from pathlib import Path

# Shared utilities - replacing custom implementations
from shared_utils import setup_logging, get_logger, load_config, ensure_directory


def create_scl_mask(scl, valid_classes):
    """
    Create mask from Scene Classification Layer (SCL).
    
    Args:
        scl: SCL data array from Sentinel-2
        valid_classes: List of valid SCL class codes for masking
        
    Returns:
        Boolean mask array where True indicates valid pixels
        
    Examples:
        >>> mask = create_scl_mask(dataset.scl, [4, 5, 6, 11])
    """
    # CRITICAL: Preserve exact algorithmic logic
    mask = scl.isin(valid_classes)
    return mask


def mask_scene(dataset, valid_classes):
    """
    Apply SCL-based masking to a dataset.
    
    Args:
        dataset: Input xarray.Dataset with SCL band
        valid_classes: List of valid SCL class codes for masking
        
    Returns:
        tuple: (masked_dataset, mask) where masked_dataset has invalid pixels as NaN
        
    Examples:
        >>> masked_data, mask = mask_scene(dataset, [4, 5, 6, 11])
    """
    # CRITICAL: Preserve exact algorithmic logic
    mask = create_scl_mask(dataset.scl, valid_classes)
    masked_dataset = dataset.where(mask)
    return masked_dataset, mask


def select_best_scenes(dataset, n_scenes, valid_classes, bands_to_drop):
    """
    Select n best scenes based on percentage of valid pixels obtained after masking.
    
    This function implements the core scene selection algorithm that identifies
    the optimal scenes for mosaic creation based on valid pixel coverage after
    applying SCL masking. The algorithm preserves the exact mathematical logic
    for calculating valid pixel percentages and scene ranking.
    
    Args:
        dataset: Input xarray.Dataset with multiple time steps
        n_scenes: Number of best scenes to select
        valid_classes: List of valid SCL class codes for masking
        bands_to_drop: List of band names to drop from output dataset
        
    Returns:
        tuple: (selected_dataset, time_span_string) where selected_dataset contains
               only the best n_scenes and time_span_string describes temporal coverage
        
    Examples:
        >>> dataset, time_span = select_best_scenes(data, 12, [4,5,6,11], ['scl'])
    """
    # CRITICAL: Preserve exact algorithmic logic - this is performance critical
    masked_dataset, mask = mask_scene(dataset, valid_classes)

    # Calculate valid pixel percentage for each scene - EXACT LOGIC PRESERVED
    valid_percentage = (
        xr.where(mask, 1, np.nan)
        .count(dim=["x", "y"]) / (mask.shape[-1] * mask.shape[-2]) * 100
    )

    # Select top n dates based on valid pixel percentage - EXACT LOGIC PRESERVED
    best_dates = (
        valid_percentage
        .sortby(valid_percentage, ascending=False)
        .head(n_scenes)
        .time
    )

    # Calculate time span between best scenes - EXACT LOGIC PRESERVED
    time_span = str(
        (best_dates.values.max() - best_dates.values.min()).astype('timedelta64[D]')
    )

    return masked_dataset.drop_vars(bands_to_drop).sel(time=best_dates), time_span


def create_processing_tiles(spain_polygon_path, tile_size):
    """
    Create the series of tiles to process Iberian territory.
    
    Generates processing tiles by dividing Spain's territory into regular grid tiles
    of specified size for distributed processing. Uses EPSG:25830 (UTM zone 30N)
    as the processing coordinate system.
    
    Args:
        spain_polygon_path: Path to Spain polygon shapefile
        tile_size: Size of processing tiles in coordinate units
        
    Returns:
        list: List of GeoBox tiles covering Spain territory
        
    Examples:
        >>> tiles = create_processing_tiles("/path/to/spain.shp", 12288)
    """
    # CRITICAL: Preserve exact geospatial processing logic
    spain = (
        gpd.read_file(spain_polygon_path)
        .to_crs(epsg='25830')
        .iloc[0]
        .geometry
    )
    spain = odc.geo.geom.Geometry(spain, crs='EPSG:25830')
    spain = odc.geo.geobox.GeoBox.from_geopolygon(spain, resolution=10)

    # Divide the full geobox in Geotiles for processing - EXACT LOGIC PRESERVED
    geotiles_spain = odc.geo.geobox.GeoboxTiles(spain, (tile_size, tile_size))
    geotiles_spain = [
        geotiles_spain.__getitem__(tile) for tile in geotiles_spain._all_tiles()
    ]
    return geotiles_spain


def create_processing_list(spain_polygon_path, tile_size, years):
    """
    Create list of all tile-year combinations for processing.
    
    Args:
        spain_polygon_path: Path to Spain polygon shapefile
        tile_size: Size of processing tiles in coordinate units
        years: List of years to process
        
    Returns:
        list: List of (tile, year) tuples for processing pipeline
        
    Examples:
        >>> processing_list = create_processing_list("/path/to/spain.shp", 12288, [2019, 2021])
    """
    # CRITICAL: Preserve exact tile generation and combination logic
    tiles = create_processing_tiles(spain_polygon_path, tile_size)
    return list(itertools.product(tiles, years))


def get_tile_bounding_box(tile):
    """
    Get bounding box of a tile in EPSG:4326.
    
    Args:
        tile: Processing tile (odc.geo.geobox.GeoBox)
        
    Returns:
        tuple: (left, bottom, right, top) coordinates in EPSG:4326
        
    Examples:
        >>> bbox = get_tile_bounding_box(tile)
        >>> left, bottom, right, top = bbox
    """
    # CRITICAL: Preserve exact coordinate transformation logic
    bbox = tile.boundingbox.to_crs('EPSG:4326')
    return (bbox.left, bbox.bottom, bbox.right, bbox.top)


def search_catalog(catalog, bbox, time_range, cloud_threshold, min_n_items, max_cloud_threshold):
    """
    Search STAC catalog for Sentinel-2 scenes with recursive cloud threshold increase.
    
    This function implements recursive cloud threshold adjustment to ensure sufficient
    scenes are found for processing. If initial cloud threshold yields insufficient
    results, it automatically increases the threshold and retries until either
    enough scenes are found or maximum threshold is reached.
    
    Args:
        catalog: STAC catalog client instance
        bbox: Bounding box coordinates (left, bottom, right, top)
        time_range: Time range string for temporal filtering
        cloud_threshold: Initial cloud cover threshold percentage
        min_n_items: Minimum number of items required for processing
        max_cloud_threshold: Maximum cloud threshold to attempt
        
    Returns:
        Collection of STAC items matching search criteria
        
    Examples:
        >>> items = search_catalog(catalog, bbox, "2022-06-01/2022-09-01", 1, 40, 61)
    """
    # CRITICAL: Preserve exact STAC search and recursive logic
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        bbox=bbox,
        datetime=time_range,
        query=[f'eo:cloud_cover<{cloud_threshold}']
    )

    item_collection = search.item_collection()

    # Recursive cloud threshold increase - EXACT LOGIC PRESERVED
    if len(item_collection) < min_n_items and cloud_threshold < max_cloud_threshold:
        return search_catalog(
            catalog, bbox, time_range, cloud_threshold + 10,
            min_n_items, max_cloud_threshold
        )
    else:
        return item_collection


def load_dataset(item_collection, tile, bands, chunk_size):
    """
    Load dataset from STAC items for a given tile.
    
    Args:
        item_collection: Collection of STAC items to load
        tile: Processing tile (GeoBox) defining spatial extent
        bands: List of bands to load from Sentinel-2 data
        chunk_size: Chunk size for dask processing optimization
        
    Returns:
        Loaded xarray.Dataset with specified bands and spatial extent
        
    Examples:
        >>> dataset = load_dataset(items, tile, ['red', 'green', 'blue'], 2048)
    """
    # CRITICAL: Preserve exact data loading parameters and logic
    dataset = load(
        item_collection,
        bands=bands,
        geobox=tile,
        chunks={'x': chunk_size, 'y': chunk_size},
        groupby='solar_day',
        resampling='bilinear'
    )
    return dataset


def create_dataset(catalog, tile, time_range, config):
    """
    Create dataset by searching catalog and loading data for a tile.
    
    High-level function that combines STAC catalog search with data loading
    to create a complete dataset ready for processing. Handles all the
    configuration-driven parameters for catalog search and data loading.
    
    Args:
        catalog: STAC catalog client instance
        tile: Processing tile (GeoBox) defining spatial extent
        time_range: Time range string for temporal filtering
        config: Configuration dictionary with processing parameters
        
    Returns:
        Complete xarray.Dataset ready for mosaic processing
        
    Examples:
        >>> dataset = create_dataset(catalog, tile, "2022-06-01/2022-09-01", config)
    """
    # CRITICAL: Preserve exact dataset creation workflow
    bounding_box = get_tile_bounding_box(tile)
    item_collection = search_catalog(
        catalog,
        bounding_box,
        time_range,
        config['processing']['cloud_threshold'],
        config['data']['min_n_items'],
        config['data']['max_cloud_threshold']
    )
    dataset = load_dataset(
        item_collection,
        tile,
        config['data']['bands'],
        config['processing']['chunk_size']
    )
    return dataset


@contextmanager
def setup_optimized_cluster(n_workers, threads_per_worker, memory_per_worker):
    """
    Setup Dask cluster optimized for Sentinel-2 processing.
    
    This context manager creates an optimized Dask cluster for large-scale
    satellite imagery processing with proper resource management and cleanup.
    The cluster configuration is critical for memory-intensive operations
    and must be preserved exactly to maintain performance characteristics.
    
    Args:
        n_workers: Number of worker processes
        threads_per_worker: Number of threads per worker process
        memory_per_worker: Memory limit per worker (e.g., "20GB")
        
    Yields:
        dask.distributed.Client: Configured Dask client for distributed processing
        
    Examples:
        >>> with setup_optimized_cluster(8, 3, "20GB") as client:
        ...     result = client.compute(dataset)
    """
    logger = get_logger('sentinel2_processing')
    cluster = None
    client = None
    
    try:
        # CRITICAL: Preserve exact cluster configuration parameters
        cluster = dask.distributed.LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_per_worker
        )
        client = dask.distributed.Client(cluster)
        yield client

    except Exception as e:
        logger.error(f"Error in cluster operations: {str(e)}")
        raise
    finally:
        # CRITICAL: Preserve exact cleanup sequence for memory management
        try:
            if client is not None:
                logger.info("Closing client...")
                client.close()

            if cluster is not None:
                logger.info("Closing cluster...")
                cluster.close()

            # Force garbage collection - CRITICAL for memory management
            gc.collect()
            time.sleep(2)

            # Log memory usage after cleanup - PRESERVE monitoring logic
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")

        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")


def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
        
    Examples:
        >>> create_output_directory("/path/to/output")
    """
    # Use shared utilities function for consistency
    ensure_directory(output_dir)
    logger = get_logger('sentinel2_processing')
    logger.info(f"Output directory ready: {output_dir}")


def generate_output_filename(tile, year, base_dir):
    """
    Generate standardized output filename for a tile and year.
    
    Creates consistent naming convention for Sentinel-2 mosaic outputs
    based on tile geometry and processing year. The filename format
    must be preserved exactly to maintain compatibility with existing
    post-processing workflows.
    
    Args:
        tile: Processing tile (GeoBox) for spatial reference
        year: Processing year for temporal reference
        base_dir: Base directory for output files
        
    Returns:
        str: Complete path to output file with standardized naming
        
    Examples:
        >>> filename = generate_output_filename(tile, 2022, "/output/dir")
    """
    # CRITICAL: Preserve exact filename generation logic for compatibility
    # Get tile bounds for filename generation
    bbox = get_tile_bounding_box(tile)
    left, bottom, right, top = bbox
    
    # Generate filename using exact format from original implementation
    filename = f"S2_summer_mosaic_{year}_{left:.6f}_{bottom:.6f}_{right:.6f}_{top:.6f}.tif"
    return str(Path(base_dir) / filename)
