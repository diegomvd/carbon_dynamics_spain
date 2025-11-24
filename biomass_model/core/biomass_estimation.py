"""
Main execution pipeline for biomass estimation from canopy height data.

This module orchestrates the complete biomass estimation workflow, including:
- Loading allometric relationships and forest type hierarchies
- Processing tiles with Monte Carlo simulations
- Applying forest type-specific allometries for AGB and BGB estimation
- Writing optimized GeoTIFF outputs with uncertainty quantification

The pipeline processes multiple forest types per tile using distributed computing
with Dask for memory-efficient handling of large raster datasets.


Author: Diego Bengochea
"""

from pathlib import Path
import random
import gc
import time
import re
import glob
import os
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
import pandas as pd
import dask.array as da
from typing import Optional, Union, Any, Dict

# Current code structure imports
from shared_utils import setup_logging, get_logger, load_config
from shared_utils.central_data_paths_constants import *


from biomass_model.core.allometry import AllometryManager
from biomass_model.core.monte_carlo import MonteCarloEstimator
from biomass_model.core.biomass_utils import BiomassUtils
from biomass_model.core.dask_utils import DaskClusterManager


class BiomassEstimationPipeline:
    """
    Main pipeline for biomass estimation with Monte Carlo uncertainty quantification.
    
    This pipeline processes canopy height rasters through allometric relationships
    to generate biomass maps (AGBD, BGBD, Total) with uncertainty estimates.
    Supports forest type specific processing and distributed computing.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, urban: bool = False):
        """
        Initialize the biomass estimation pipeline.
        
        Args:
            config: Configuration dictionary (processing parameters only)
        """
        # Store configuration and data paths
        self.config = load_config(config_path, component_name='biomass_model')
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_estimation',
            log_file=self.config['logging'].get('log_file')
        )

        self.allometry_manager = AllometryManager(urban=urban)
        self.monte_carlo = MonteCarloEstimator(self.config)
        self.biomass_utils = BiomassUtils(self.config)
        self.dask_manager = DaskClusterManager(self.config)

        # TODO: fix config to accept this
        self.target_resolution = self.config['processing']['target_resolution']
        
        # Pipeline state
        self.client = None
        self.start_time = None
        
        self.logger.info(f"Initialized BiomassEstimationPipeline")


    def process_forest_type(self, height_file, mask_file, code, forest_type_name):
        """
        Process a single forest type with current components for biomass estimation.
        
        Args:
            height_file (str): Path to canopy height raster
            mask_file (str): Path to forest type mask raster
            code (str): Forest type code identifier
            forest_type_name (str): Human-readable forest type name

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        logger = self.logger
        
        try:
            logger.info(f"Processing forest type: {forest_type_name} (code: {code})")

            with self.dask_manager.create_cluster() as client:

                logger.info(f"Dask dashboard: {client.dashboard_link}")
                
                # Get allometry parameters using current AllometryManager
                try:
                    agb_params, bgb_params = self.allometry_manager.get_allometry_parameters(
                        forest_type_name
                    )
                except Exception as e:
                    logger.error(f"Failed to get allometry parameters for {forest_type_name}: {e}")
                    return False
                
                try:
                    height_data, mask_data, metadata = self.biomass_utils.read_height_and_mask_xarray(
                        height_file, mask_file, 
                    )
                except Exception as e:
                    logger.error(f"Failed to load raster data: {e}")
                    return False

                # Apply forest type mask to height data
                masked_heights = height_data.where(mask_data)
                
                # Execute Monte Carlo biomass simulation using current MonteCarloEstimator
                with ProgressBar():
                    logger.info("Running Monte Carlo biomass estimation...")
                    try:
                        results = self.monte_carlo.run(
                            masked_heights, agb_params, bgb_params
                        )
                    except Exception as e:
                        logger.error(f"Monte Carlo estimation failed: {e}")
                        return False
                
                try:
                    success = self.biomass_utils.write_xarray_results(
                        results, height_file, code, forest_type_name, mask_data, metadata
                    )
                    
                    if success:
                        logger.info(f"Completed processing for {forest_type_name}")
                        return True
                    else:
                        logger.error(f"Failed to write results for {forest_type_name}")
                        return False
                except Exception as e:
                    logger.error(f"Failed to write results: {e}")
                    return False
                
        except Exception as e:
            logger.error(f"Error processing forest type {forest_type_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False


    def process_tile(self, height_file, forest_types_table, code_to_name):
        """
        Process a single tile with all associated forest types.
        
        Args:
            height_file (str): Path to height raster
            forest_types_table (pd.DataFrame): Forest type hierarchy table
            code_to_name (dict): Mapping from forest type codes to names
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        logger = get_logger('biomass_estimation')
        tile_name = Path(height_file).stem
        logger.info(f"Processing tile: {tile_name}")
        
        # Extract tile information from filename
        tile_info = self.biomass_utils.extract_tile_info(height_file)
        if not tile_info:
            logger.error(f"Failed to extract tile information from {height_file}")
            return False
            
        # Find all forest type masks for this tile using current path structure
        try:
            masks_dir = str(FOREST_TYPE_MASKS_DIR)  # Use current path constants
            masks = self.biomass_utils.find_masks_for_tile(tile_info, masks_dir)
        except OSError as e:
            logger.error(f"Error finding masks for {tile_name}: {e}")
            return False
            
        if not masks:
            logger.warning(f"No masks found for {tile_name}")
            return False
            
        logger.info(f"Found {len(masks)} masks for {tile_name}")
        
        # Process each forest type independently
        results = []
        for mask_path, code in tqdm(masks, desc=f"Processing forest types for {tile_name}"):
            forest_type = code_to_name.get(int(code), f"Unknown_{code}")
            
            # Process this specific forest type
            success = self.process_forest_type(
                height_file, mask_path, code, forest_type
            )
            
            results.append(success)
            
            # Force garbage collection between forest types
            gc.collect()
        
        # Calculate success rate for this tile
        success_count = sum(1 for r in results if r)
        logger.info(f"Processed {success_count}/{len(masks)} forest types for {tile_name}")
        
        return success_count > 0


    def run_full_pipeline(self) -> bool:
        """
        Main orchestration function to process all tiles using tile-based approach.
        
        Loads reference data once and processes tiles sequentially with detailed
        progress tracking and memory management.
        """
        # Load configuration using current system
        config = self.config
        
        overall_start_time = time.time()
        processed_tiles = 0
        failed_tiles = 0
        
            
        try:
            # Load forest type hierarchy using current path structure
            forest_types = pd.read_csv(FOREST_TYPES_TIERS_FILE)
            forest_types['Dummy'] = 'General'  # Add dummy tier for hierarchy traversal
            self.logger.info(f"Loaded forest types hierarchy with {len(forest_types)} entries")
            
            # Build forest type code-to-name mapping using current paths
            code_to_name = self.biomass_utils.build_forest_type_mapping(
                FOREST_TYPE_MAPS_DIR,
                cache_path=FOREST_TYPE_MFE_CODE_TO_NAME_FILE, 
                use_cache=config.get('forest_types', {}).get('use_cache', True)
            )
          
            code_to_name = dict(zip(code_to_name.code, code_to_name.name))
            self.logger.info(f"Built forest type mapping with {len(code_to_name)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load reference data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
        # Discover input files using current path structure
        if self.target_resolution == 100:
            input_dir = HEIGHT_MAPS_100M_DIR   # Use current path constants
        elif self.target_resolution == 10:
            input_dir = HEIGHT_MAPS_10M_DIR 
        else: 
            self.logger.error(f'Height maps at {self.target_resolution}m are not available. Ending execution.')
            return False

        file_pattern = self.config.get('processing', {}).get('file_pattern', '*.tif')
        self.logger.info(f"Looking for canopy height files in {input_dir}")
        
        try:
            files = list(Path(input_dir).rglob(file_pattern))
        except Exception as e:
            self.logger.error(f"Failed to list input files: {str(e)}")
            return False
            
        # Randomize processing order for better load balancing
        random.shuffle(files)

        # Filter and sort files by year (newest first) using target years from config
        files_with_years = []
        target_years = self.config.get('processing', {}).get('target_years', [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        
        for f in files:
            tile_info = self.biomass_utils.extract_tile_info(f)
            if tile_info:
                if int(tile_info['year']) in target_years:
                    files_with_years.append((f, tile_info['year']))
            else:
                files_with_years.append((f, '0000'))  # Default for failed extraction
                
        # Sort by year (newest first)
        files_with_years.sort(key=lambda x: x[1], reverse=True)
        files = [f[0] for f in files_with_years]

        total_files = len(files)
        self.logger.info(f"Found {total_files} input tiles to process")
        
        # Process tiles sequentially with detailed progress tracking
        for i, fname in enumerate(files):
            self.logger.info(f"\n{'='*80}\nProcessing tile {i+1}/{total_files}: {fname}\n{'='*80}")
            
            try:
                # Process this individual tile using current components
                success = self.process_tile(
                    fname, 
                    forest_types,
                    code_to_name,
                )
                
                # Update counters based on processing result
                if success:
                    processed_tiles += 1
                else:
                    failed_tiles += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process tile {fname}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                failed_tiles += 1
            
            # Calculate and log progress statistics
            elapsed = time.time() - overall_start_time
            remaining = total_files - (i + 1)
            est_time_per_tile = elapsed / (i + 1)
            est_remaining = est_time_per_tile * remaining
            
            self.logger.info(f"Progress: {(i+1)}/{total_files} ({(i+1)/total_files*100:.1f}%)")
            self.logger.info(f"Elapsed: {elapsed/3600:.2f} hours, Estimated remaining: {est_remaining/3600:.2f} hours")
            self.logger.info(f"Success rate: {processed_tiles}/{i+1} ({processed_tiles/(i+1)*100:.1f}%)")
            
            # Force garbage collection between tiles
            gc.collect()
        
        # Generate final processing report
        overall_elapsed = time.time() - overall_start_time
        self.logger.info("\n" + "="*80)
        self.logger.info("Processing completed!")
        self.logger.info(f"Total elapsed time: {overall_elapsed/3600:.2f} hours")
        self.logger.info(f"Tiles processed: {processed_tiles}/{total_files}")
        self.logger.info(f"Tiles failed: {failed_tiles}/{total_files}")

        return True
