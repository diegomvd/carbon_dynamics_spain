"""
Sentinel-2 Mosaic Processing Pipeline

Class-based pipeline for creating Sentinel-2 summer mosaics over Spain using distributed
computing. Processes L2A satellite imagery with Scene Classification Layer (SCL) masking
and creates median composites with optimized memory management.

This module implements the core processing pipeline that handles:
- Large-scale distributed processing across Spain's territory
- STAC catalog integration for data discovery and access
- Performance-critical scene selection algorithms
- Memory-optimized cluster management and garbage collection
- Comprehensive error handling and recovery mechanisms

Key processing steps preserve exact algorithmic logic including baseline correction
factors, median calculation parameters, and distributed computing patterns.

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
from typing import Optional, Dict, Any, Tuple

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory

# Component utilities - using refactored imports
from .s2_utils import (
    select_best_scenes, create_processing_list, get_tile_bounding_box,
    create_dataset, setup_optimized_cluster, create_output_directory,
    generate_output_filename
)


class MosaicProcessingPipeline:
    """
    Pipeline for Sentinel-2 mosaic processing with distributed computing.
    
    Handles the complete workflow from STAC catalog search to final mosaic creation
    with optimized memory management and error recovery. The pipeline processes
    tile-year combinations across Spain using configurable distributed clusters.
    
    Processing preserves exact algorithmic logic for scene selection, baseline
    corrections, and median calculation to ensure consistency with original results.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the mosaic processing pipeline.
        
        Args:
            config_path: Path to configuration file, uses default if None
        """
        # Load configuration using shared utilities
        self.config = load_config(config_path) if config_path else load_config()
        
        # Setup logging using shared utilities  
        self.logger = get_logger('sentinel2_processing')
        
        # Initialize STAC catalog
        self.catalog = None
        self.processing_list = None
        
        # Pipeline state
        self.start_time = None
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        
        self.logger.info("MosaicProcessingPipeline initialized")
    
    def initialize_catalog(self):
        """
        Initialize STAC catalog connection.
        
        Examples:
            >>> pipeline = MosaicProcessingPipeline()
            >>> pipeline.initialize_catalog()
        """
        try:
            self.catalog = Client.open(self.config['data']['stac_url'])
            self.logger.info(f"Connected to STAC catalog: {self.config['data']['stac_url']}")
        except Exception as e:
            self.logger.error(f"Failed to connect to STAC catalog: {str(e)}")
            raise
    
    def create_processing_plan(self):
        """
        Create the complete processing plan for all tile-year combinations.
        
        Examples:
            >>> pipeline.create_processing_plan()
            >>> print(f"Processing {len(pipeline.processing_list)} combinations")
        """
        self.processing_list = create_processing_list(
            SPAIN_BOUNDARIES,
            self.config['processing']['tile_size'],
            self.config['processing']['years']
        )
        self.logger.info(f"Created processing plan: {len(self.processing_list)} tile-year combinations")
    
    def process_mosaic(self, dataset, year: int, sampling_period: str) -> Any:
        """
        Process a dataset to create a median mosaic.
        
        CRITICAL: This function preserves the exact algorithmic logic from the original
        implementation including baseline correction factors, median calculation, and
        metadata handling to ensure consistent results.
        
        Args:
            dataset: Input xarray.Dataset with multiple time steps
            year: Processing year for baseline correction logic
            sampling_period: Time range string for metadata
            
        Returns:
            Processed xarray.Dataset mosaic with metadata
            
        Examples:
            >>> mosaic = pipeline.process_mosaic(dataset, 2022, "06-01/09-01")
        """
        self.logger.info(f"Processing mosaic with {len(dataset.time)} scenes")
        
        # CRITICAL: Preserve exact scene selection algorithm
        dataset, time_span = select_best_scenes(
            dataset,
            self.config['processing']['n_scenes'],
            self.config['scl']['valid_classes'],
            self.config['data']['bands_drop']
        )
        
        # CRITICAL: Preserve exact correction factor for processing baseline 04.00
        if year > 2021:
            dataset = dataset + 1000
        
        # CRITICAL: Preserve exact median mosaic calculation
        mosaic = dataset.median(dim="time", skipna=True)

        # CRITICAL: Preserve exact metadata attributes calculation
        mosaic.attrs['valid_pixel_percentage'] = mosaic.count()/(len(mosaic.x)*len(mosaic.y))*100
        mosaic.attrs['time_span'] = time_span
        mosaic.attrs['year'] = year
        mosaic.attrs['sampling_period'] = sampling_period
        
        # CRITICAL: Preserve exact data type conversion and nodata handling
        mosaic = mosaic.fillna(0).astype('uint16')

        # CRITICAL: Preserve exact nodata value setting for all bands
        for band in mosaic.variables:
            mosaic[band] = mosaic[band].rio.write_nodata(0, inplace=False)
            
        return mosaic
    
    def save_mosaic(self, mosaic, path: str, client):
        """
        Save mosaic to disk with compression and metadata.
        
        CRITICAL: Preserves exact saving parameters including compression settings,
        tiling options, and distributed lock handling for consistent output format.
        
        Args:
            mosaic: Processed mosaic dataset to save
            path: Output file path
            client: Dask client for distributed lock management
            
        Examples:
            >>> pipeline.save_mosaic(mosaic, "/path/to/output.tif", client)
        """
        # CRITICAL: Preserve exact raster writing parameters and distributed lock
        mosaic.rio.to_raster(
            path,
            tags=mosaic.attrs,
            **{'compress': 'lzw'},
            tiled=True,
            lock=dask.distributed.Lock('rio', client=client)
        )
    
    def process_single_combination(self, tile, year: int, i: int, total: int) -> bool:
        """
        Process a single tile-year combination.
        
        Args:
            tile: Processing tile (GeoBox)
            year: Processing year
            i: Current iteration index
            total: Total number of combinations
            
        Returns:
            bool: True if processing succeeded, False otherwise
            
        Examples:
            >>> success = pipeline.process_single_combination(tile, 2022, 0, 100)
        """
        # Generate output path using exact filename convention
        savepath = generate_output_filename(
            tile, year, SENTINEL2_MOSAICS_DIR
        )
        
        # Skip if already processed - PRESERVE exact file existence check
        if Path(savepath).exists():
            self.logger.info(f'Already computed {savepath}, skipping...')
            self.skipped_count += 1
            return True

        # Get tile information for logging - PRESERVE exact logging format
        bounding_box = get_tile_bounding_box(tile)
        self.logger.info(f'Processing tile for year {year} in region {bounding_box}')
        
        try:
            # CRITICAL: Preserve exact time range string format
            time_range = (f'{year}-{self.config["processing"]["min_month"]:02d}-01/'
                         f'{year}-{self.config["processing"]["max_month"]:02d}-01')
            
            # CRITICAL: Preserve exact cluster setup and processing workflow
            with setup_optimized_cluster(
                self.config['compute']['n_workers'],
                self.config['compute']['threads_per_worker'],
                self.config['compute']['memory_per_worker']
            ) as client:
                
                # CRITICAL: Preserve exact client configuration and logging
                self.logger.info(f"Cluster dashboard: {client.dashboard_link}")
                configure_rio(cloud_defaults=True, client=client)
                self.logger.info(f'Starting processing of mosaic {i+1}, {total-i} remaining')
                
                try:
                    self.logger.info('Creating dataset and processing mosaic')
                    # CRITICAL: Preserve exact dataset creation workflow
                    dataset = create_dataset(self.catalog, tile, time_range, self.config)
                    mosaic = self.process_mosaic(dataset, year, time_range).compute()

                    self.logger.info(f'Mosaic {i+1} processed, {total-i-1} remaining')
                
                    # CRITICAL: Preserve exact saving and cleanup sequence
                    self.save_mosaic(mosaic, savepath, client)
                    self.logger.info(f'Successfully saved mosaic to {savepath}')
                    mosaic.close()
                    
                    self.processed_count += 1
                    return True

                except Exception as process_error:
                    self.logger.error(f"Error processing year {year}: {str(process_error)}")
                    self.error_count += 1
                    return False
            
                # CRITICAL: Preserve exact client restart for memory management
                client.restart()
        
        except Exception as cluster_error:
            self.logger.error(f"Cluster error for year {year} and tile {bounding_box}: {str(cluster_error)}")
            self.error_count += 1
            return False
    
    def run_processing(self) -> Dict[str, Any]:
        """
        Execute the complete mosaic processing pipeline.
        
        CRITICAL: Preserves the exact distributed processing loop structure,
        error handling patterns, memory management, and cleanup sequences
        from the original implementation.
        
        Returns:
            dict: Processing summary with statistics
            
        Examples:
            >>> pipeline = MosaicProcessingPipeline()
            >>> results = pipeline.run_processing()
            >>> print(f"Processed: {results['processed']}, Errors: {results['errors']}")
        """
        self.start_time = time.time()
        self.logger.info("Starting Sentinel-2 mosaic processing pipeline...")
        
        try:
            # Initialize components
            if self.catalog is None:
                self.initialize_catalog()
            
            if self.processing_list is None:
                self.create_processing_plan()
            
            # Create output directory using shared utilities
            ensure_directory(SENTINEL2_MOSAICS_DIR)
            self.logger.info(f"Output directory ready: {str(SENTINEL2_MOSAICS_DIR)}")
            
            total_combinations = len(self.processing_list)
            self.logger.info(f"Processing {total_combinations} tile-year combinations")
            
            # CRITICAL: Preserve exact processing loop structure and enumeration
            for i, (tile, year) in enumerate(self.processing_list):
                
                # Process single combination
                self.process_single_combination(tile, year, i, total_combinations)
                
                # CRITICAL: Preserve exact cleanup and memory monitoring sequence
                gc.collect()
                memory_usage = psutil.virtual_memory()
                self.logger.info(f"System memory usage: {memory_usage.percent}%")
                time.sleep(5)

            self.logger.info("Processing completed successfully!")
            
            # Generate processing summary
            return self.get_processing_summary()

        except Exception as e:
            self.logger.error(f"Main process error: {str(e)}")
            raise
        finally:
            # CRITICAL: Preserve exact final cleanup sequence
            gc.collect()
            self.logger.info("Pipeline finished")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive processing summary and statistics.
        
        Returns:
            dict: Processing summary with timing and statistics
            
        Examples:
            >>> summary = pipeline.get_processing_summary()
            >>> print(f"Duration: {summary['duration_minutes']:.1f} minutes")
        """
        end_time = time.time()
        duration = end_time - (self.start_time or end_time)
        
        return {
            'processed_count': self.processed_count,
            'skipped_count': self.skipped_count,
            'error_count': self.error_count,
            'total_combinations': len(self.processing_list) if self.processing_list else 0,
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'success_rate': (self.processed_count / max(1, self.processed_count + self.error_count)) * 100,
            'config_summary': {
                'years': self.config['processing']['years'],
                'n_scenes': self.config['processing']['n_scenes'],
                'tile_size': self.config['processing']['tile_size'],
                'n_workers': self.config['compute']['n_workers']
            }
        }
    
    def validate_configuration(self) -> bool:
        """
        Validate pipeline configuration and required inputs.
        
        Returns:
            bool: True if configuration is valid, False otherwise
            
        Examples:
            >>> if pipeline.validate_configuration():
            ...     pipeline.run_processing()
        """
        required_paths = [
            'spain_polygon',
            'output_dir'
        ]
        
        # Check required path configurations
        for path_key in required_paths:
            if path_key not in self.config.get('paths', {}):
                self.logger.error(f"Missing required path configuration: {path_key}")
                return False
        
        # Validate Spain polygon file exists
        spain_polygon_path = SPAIN_BOUNDARIES
        if not spain_polygon_path.exists():
            self.logger.error(f"Spain polygon file not found: {spain_polygon_path}")
            return False
        
        # Validate required processing parameters
        required_params = ['n_scenes', 'tile_size', 'years']
        for param in required_params:
            if param not in self.config.get('processing', {}):
                self.logger.error(f"Missing required processing parameter: {param}")
                return False
        
        # Validate years is non-empty list
        if not isinstance(self.config['processing']['years'], list) or len(self.config['processing']['years']) == 0:
            self.logger.error("Processing years must be a non-empty list")
            return False
        
        self.logger.info("Configuration validation successful")
        return True
