"""
Harmonized processing pipeline for Sentinel-2 satellite imagery mosaics.

This script processes Sentinel-2 L2A data to create summer mosaics over Spain.
Applies Scene Classification Layer (SCL) masking, selects optimal scenes,
and creates median composites with metadata.

Author: Diego Bengochea
"""

import rioxarray as rio
import dask.distributed
from odc.stac import configure_rio
from pystac_client import Client
from pathlib import Path
import numpy as np
import gc
import psutil
import time

# Import utilities
from s2_utils import (
    setup_logging, load_config, select_best_scenes, create_processing_list,
    get_tile_bounding_box, create_dataset, setup_optimized_cluster,
    create_output_directory, generate_output_filename
)


def process_mosaic(dataset, year, sampling_period, config):
    """
    Process a dataset to create a median mosaic.
    
    Args:
        dataset (xarray.Dataset): Input dataset with multiple time steps
        year (int): Processing year
        sampling_period (str): Time range string for sampling period
        config (dict): Configuration dictionary
        
    Returns:
        xarray.Dataset: Processed mosaic with metadata
    """
    logger = setup_logging()
    logger.info(f"Processing mosaic with {len(dataset.time)} scenes")
    
    # Select best scenes
    dataset, time_span = select_best_scenes(
        dataset,
        config['processing']['n_scenes'],
        config['scl']['valid_classes'],
        config['data']['bands_drop']
    )
    
    # Apply correction factor for scenes using processing baseline 04.00
    if year > 2021:
        dataset = dataset + 1000
    
    # Create median mosaic
    mosaic = dataset.median(dim="time", skipna=True)

    # Add metadata attributes
    mosaic.attrs['valid_pixel_percentage'] = mosaic.count()/(len(mosaic.x)*len(mosaic.y))*100
    mosaic.attrs['time_span'] = time_span
    mosaic.attrs['year'] = year
    mosaic.attrs['sampling_period'] = sampling_period
    
    # Fill NaN values and convert to uint16
    mosaic = mosaic.fillna(0).astype('uint16')

    # Set nodata values for all bands
    for band in mosaic.variables:
        mosaic[band] = mosaic[band].rio.write_nodata(0, inplace=False)
        
    return mosaic


def save_mosaic(mosaic, path, client):
    """
    Save mosaic to disk with compression and metadata.
    
    Args:
        mosaic (xarray.Dataset): Mosaic to save
        path (str): Output file path
        client: Dask client for distributed lock
    """
    mosaic.rio.to_raster(
        path,
        tags=mosaic.attrs,
        **{'compress': 'lzw'},
        tiled=True,
        lock=dask.distributed.Lock('rio', client=client)
    )


def main():
    """Main processing pipeline."""
    logger = setup_logging()
    logger.info("Starting Sentinel-2 mosaic processing pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create output directory
        create_output_directory(config['paths']['output_dir'])
        
        # Initialize STAC catalog
        catalog = Client.open(config['data']['stac_url'])
        
        # Create processing list
        processing_list = create_processing_list(
            config['paths']['spain_polygon'],
            config['processing']['tile_size'],
            config['processing']['years']
        )
        
        logger.info(f"Processing {len(processing_list)} tile-year combinations")
        
        # Process each tile-year combination
        for i, (tile, year) in enumerate(processing_list):
            
            # Generate output path
            savepath = generate_output_filename(
                tile, year, config['paths']['output_dir']
            )
            
            # Skip if already processed
            if Path(savepath).exists():
                logger.info(f'Already computed {savepath}, skipping...')
                continue

            # Get tile information for logging
            bounding_box = get_tile_bounding_box(tile)
            logger.info(f'Processing tile for year {year} in region {bounding_box}')
            
            try:
                # Create time range string
                time_range = (f'{year}-{config["processing"]["min_month"]:02d}-01/'
                             f'{year}-{config["processing"]["max_month"]:02d}-01')
                
                # Setup cluster and process
                with setup_optimized_cluster(
                    config['compute']['n_workers'],
                    config['compute']['threads_per_worker'],
                    config['compute']['memory_per_worker']
                ) as client:
                    
                    logger.info(f"Cluster dashboard: {client.dashboard_link}")
                    configure_rio(cloud_defaults=True, client=client)
                    logger.info(f'Starting processing of mosaic {i+1}, {len(processing_list)-i} remaining')
                    
                    try:
                        logger.info('Creating dataset and processing mosaic')
                        dataset = create_dataset(catalog, tile, time_range, config)
                        mosaic = process_mosaic(dataset, year, time_range, config).compute()

                        logger.info(f'Mosaic {i+1} processed, {len(processing_list)-i-1} remaining')
                    
                        save_mosaic(mosaic, savepath, client)
                        logger.info(f'Successfully saved mosaic to {savepath}')
                        mosaic.close()

                    except Exception as process_error:
                        logger.error(f"Error processing year {year}: {str(process_error)}")
                        continue
                
                    # Restart client for next iteration
                    client.restart()
            
            except Exception as cluster_error:
                logger.error(f"Cluster error for year {year} and tile {bounding_box}: {str(cluster_error)}")
                continue

            # Cleanup and memory logging
            gc.collect()
            memory_usage = psutil.virtual_memory()
            logger.info(f"System memory usage: {memory_usage.percent}%")
            time.sleep(5)

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise
    finally:
        # Final cleanup
        gc.collect()
        logger.info("Pipeline finished")


if __name__ == '__main__':
    main()
