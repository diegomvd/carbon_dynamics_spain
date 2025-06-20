"""
Utility functions for Sentinel-2 satellite imagery processing.
Contains common functions for configuration, logging, data processing,
geospatial operations, and cluster management.

Author: Diego Bengochea
"""

import yaml
import logging
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


def setup_logging():
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
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


def create_scl_mask(scl, valid_classes):
    """
    Create mask from Scene Classification Layer (SCL).
    
    Args:
        scl (xarray.DataArray): SCL data array
        valid_classes (list): List of valid SCL class codes
        
    Returns:
        xarray.DataArray: Boolean mask array
    """
    mask = scl.isin(valid_classes)
    return mask


def mask_scene(dataset, valid_classes):
    """
    Apply SCL-based masking to a dataset.
    
    Args:
        dataset (xarray.Dataset): Input dataset with SCL band
        valid_classes (list): List of valid SCL class codes
        
    Returns:
        tuple: (masked_dataset, mask)
    """
    mask = create_scl_mask(dataset.scl, valid_classes)
    masked_dataset = dataset.where(mask)
    return masked_dataset, mask


def select_best_scenes(dataset, n_scenes, valid_classes, bands_to_drop):
    """
    Select n best scenes based on percentage of valid pixels obtained after masking.
    
    Args:
        dataset (xarray.Dataset): Input dataset
        n_scenes (int): Number of scenes to select
        valid_classes (list): List of valid SCL class codes
        bands_to_drop (list): Bands to drop from output
        
    Returns:
        tuple: (selected_dataset, time_span_string)
    """
    masked_dataset, mask = mask_scene(dataset, valid_classes)

    valid_percentage = (
        xr.where(mask, 1, np.nan)
        .count(dim=["x", "y"]) / (mask.shape[-1] * mask.shape[-2]) * 100
    )

    # Select top n dates
    best_dates = (
        valid_percentage
        .sortby(valid_percentage, ascending=False)
        .head(n_scenes)
        .time
    )

    time_span = str(
        (best_dates.values.max() - best_dates.values.min()).astype('timedelta64[D]')
    )

    return masked_dataset.drop_vars(bands_to_drop).sel(time=best_dates), time_span


def create_processing_tiles(spain_polygon_path, tile_size):
    """
    Create the series of tiles to process Iberian territory.
    
    Args:
        spain_polygon_path (str): Path to Spain polygon shapefile
        tile_size (int): Size of processing tiles
        
    Returns:
        list: List of GeoBox tiles
    """
    spain = (
        gpd.read_file(spain_polygon_path)
        .to_crs(epsg='25830')
        .iloc[0]
        .geometry
    )
    spain = odc.geo.geom.Geometry(spain, crs='EPSG:25830')
    spain = odc.geo.geobox.GeoBox.from_geopolygon(spain, resolution=10)

    # Divide the full geobox in Geotiles for processing
    geotiles_spain = odc.geo.geobox.GeoboxTiles(spain, (tile_size, tile_size))
    geotiles_spain = [
        geotiles_spain.__getitem__(tile) for tile in geotiles_spain._all_tiles()
    ]
    return geotiles_spain


def create_processing_list(spain_polygon_path, tile_size, years):
    """
    Create list of all tile-year combinations for processing.
    
    Args:
        spain_polygon_path (str): Path to Spain polygon shapefile
        tile_size (int): Size of processing tiles
        years (list): List of years to process
        
    Returns:
        list: List of (tile, year) tuples
    """
    tiles = create_processing_tiles(spain_polygon_path, tile_size)
    return list(itertools.product(tiles, years))


def get_tile_bounding_box(tile):
    """
    Get bounding box of a tile in EPSG:4326.
    
    Args:
        tile (odc.geo.geobox.GeoBox): Processing tile
        
    Returns:
        tuple: (left, bottom, right, top) coordinates
    """
    bbox = tile.boundingbox.to_crs('EPSG:4326')
    return (bbox.left, bbox.bottom, bbox.right, bbox.top)


def search_catalog(catalog, bbox, time_range, cloud_threshold, min_n_items, max_cloud_threshold):
    """
    Search STAC catalog for Sentinel-2 scenes with recursive cloud threshold increase.
    
    Args:
        catalog: STAC catalog client
        bbox (tuple): Bounding box coordinates
        time_range (str): Time range string
        cloud_threshold (int): Initial cloud cover threshold
        min_n_items (int): Minimum number of items required
        max_cloud_threshold (int): Maximum cloud threshold to try
        
    Returns:
        pystac.ItemCollection: Collection of STAC items
    """
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        bbox=bbox,
        datetime=time_range,
        query=[f'eo:cloud_cover<{cloud_threshold}']
    )

    item_collection = search.item_collection()

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
        item_collection: Collection of STAC items
        tile: Processing tile (GeoBox)
        bands (list): List of bands to load
        chunk_size (int): Chunk size for dask processing
        
    Returns:
        xarray.Dataset: Loaded dataset
    """
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
    
    Args:
        catalog: STAC catalog client
        tile: Processing tile (GeoBox)
        time_range (str): Time range for data search
        config (dict): Configuration dictionary
        
    Returns:
        xarray.Dataset: Complete dataset for processing
    """
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
    Setup Dask cluster optimized for processing.
    
    Args:
        n_workers (int): Number of workers
        threads_per_worker (int): Threads per worker
        memory_per_worker (str): Memory limit per worker
        
    Yields:
        dask.distributed.Client: Dask client instance
    """
    logger = logging.getLogger(__name__)
    cluster = None
    client = None
    
    try:
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
        # Cleanup sequence
        try:
            if client is not None:
                logger.info("Closing client...")
                client.close()

            if cluster is not None:
                logger.info("Closing cluster...")
                cluster.close()

            # Force garbage collection
            gc.collect()
            time.sleep(2)

            # Log memory usage after cleanup
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")

        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")


def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Path to output directory
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        logging.info(f"Created output directory: {output_dir}")


def generate_output_filename(tile, year, base_dir):
    """
    Generate standardized output filename for a tile and year.
    
    Args:
        tile (odc.geo.geobox.GeoBox): Processing tile
        year (int): Processing year
        base_dir (str): Base output directory
        
    Returns:
        str: Full path to output file
    """
    bounding_box = get_tile_bounding_box(tile)
    resolution = abs(int(tile.resolution.x))
    north = int(np.round(bounding_box[3]))

    if bounding_box[0] > 0:
        east = int(np.round(bounding_box[0]))
        filename = f'S2_summer_mosaic_{year}_N{north}_E{east}_{resolution}m.tif'
    else:
        west = -int(np.round(bounding_box[0]))
        filename = f'S2_summer_mosaic_{year}_N{north}_W{west}_{resolution}m.tif'

    return f"{base_dir}/{filename}"
