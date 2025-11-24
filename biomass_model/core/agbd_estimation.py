"""
AGBD Estimation Pipeline

Pipeline for AGBD-only estimation using direct quantile allometries.
Parallel to BiomassEstimationPipeline but produces only AGBD outputs
(mean, lower, upper bounds) without Monte Carlo sampling.

This pipeline is optimized for producing AGBD maps that will be used for
post-processing to distinguish stress-induced apparent loss from structural loss.

Author: Diego Bengochea
"""

from pathlib import Path
import random
import gc
import time
from tqdm import tqdm
import pandas as pd
from typing import Optional, Union

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config
from shared_utils.central_data_paths_constants import *

# Component imports
from biomass_model.core.allometry import AllometryManager
from biomass_model.core.agbd_direct_estimator import AGBDEstimator
from biomass_model.core.biomass_utils import BiomassUtils
from biomass_model.core.dask_utils import DaskClusterManager


class AGBDEstimationPipeline:
    """
    Pipeline for AGBD-only estimation with direct quantile allometries.
    
    Produces three outputs per forest type:
    - AGBD_mean (from median allometry)
    - AGBD_lower (from 15th percentile allometry, 70% CI)
    - AGBD_upper (from 85th percentile allometry, 70% CI)
    
    Parallel to BiomassEstimationPipeline but simplified for AGBD-only.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the AGBD estimation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name='biomass_model')
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='agbd_estimation',
            log_file=self.config['logging'].get('log_file')
        )

        # Initialize components
        self.allometry_manager = AllometryManager()
        self.agbd_estimator = AGBDDirectEstimator(self.config)
        self.biomass_utils = BiomassUtils(self.config)
        self.dask_manager = DaskClusterManager(self.config)

        # Pipeline parameters
        self.target_resolution = self.config['processing']['target_resolution']
        
        # Pipeline state
        self.client = None
        self.start_time = None
        
        self.logger.info("Initialized AGBDEstimationPipeline")

    def process_forest_type(self, height_file, mask_file, code, forest_type_name):
        """
        Process a single forest type for AGBD estimation.
        
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
                
                # Get allometry parameters (only need AGB params)
                try:
                    agb_params, _ = self.allometry_manager.get_allometry_parameters(
                        forest_type_name
                    )
                except Exception as e:
                    logger.error(f"Failed to get allometry parameters for {forest_type_name}: {e}")
                    return False
                
                # Load height and mask data
                try:
                    height_data, mask_data, metadata = self.biomass_utils.read_height_and_mask_xarray(
                        height_file, mask_file
                    )
                except Exception as e:
                    logger.error(f"Failed to load raster data: {e}")
                    return False

                # Apply forest type mask to height data
                masked_heights = height_data.where(mask_data)
                
                # Run AGBD direct estimation (no Monte Carlo)
                logger.info("Running direct AGBD estimation...")
                try:
                    results = self.agbd_estimator.run(masked_heights, agb_params)
                except Exception as e:
                    logger.error(f"AGBD estimation failed: {e}")
                    return False
                
                # Write AGBD results (3 outputs: mean, lower, upper)
                try:
                    success = self.biomass_utils.write_agbd_results(
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
            
        # Find all forest type masks for this tile
        try:
            masks_dir = str(FOREST_TYPE_MASKS_DIR)
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
        Main orchestration function to process all tiles.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        config = self.config
        
        overall_start_time = time.time()
        processed_tiles = 0
        failed_tiles = 0
        
        try:
            # Load forest type hierarchy
            forest_types = pd.read_csv(FOREST_TYPES_TIERS_FILE)
            forest_types['Dummy'] = 'General'
            self.logger.info(f"Loaded forest types hierarchy with {len(forest_types)} entries")
            
            # Build forest type code-to-name mapping
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
        
        # Discover input files
        if self.target_resolution == 100:
            input_dir = HEIGHT_MAPS_100M_DIR
        elif self.target_resolution == 10:
            input_dir = HEIGHT_MAPS_10M_DIR 
        else: 
            self.logger.error(f'Height maps at {self.target_resolution}m are not available.')
            return False

        file_pattern = self.config.get('processing', {}).get('file_pattern', '*.tif')
        self.logger.info(f"Looking for canopy height files in {input_dir}")
        
        try:
            files = list(Path(input_dir).rglob(file_pattern))
        except Exception as e:
            self.logger.error(f"Failed to list input files: {str(e)}")
            return False
            
        # Randomize processing order
        random.shuffle(files)

        # Filter and sort files by year (newest first)
        files_with_years = []
        target_years = self.config.get('processing', {}).get('target_years', 
                                                              [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        
        for f in files:
            tile_info = self.biomass_utils.extract_tile_info(f)
            if tile_info:
                if int(tile_info['year']) in target_years:
                    files_with_years.append((f, tile_info['year']))
            else:
                files_with_years.append((f, '0000'))
                
        # Sort by year (newest first)
        files_with_years.sort(key=lambda x: x[1], reverse=True)
        files = [f[0] for f in files_with_years]

        total_files = len(files)
        self.logger.info(f"Found {total_files} input tiles to process")
        
        # Process tiles sequentially
        for i, fname in enumerate(files):
            self.logger.info(f"\n{'='*80}\nProcessing tile {i+1}/{total_files}: {fname}\n{'='*80}")
            
            try:
                success = self.process_tile(fname, forest_types, code_to_name)
                
                if success:
                    processed_tiles += 1
                else:
                    failed_tiles += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process tile {fname}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                failed_tiles += 1
            
            # Log progress
            elapsed = time.time() - overall_start_time
            remaining = total_files - (i + 1)
            est_time_per_tile = elapsed / (i + 1)
            est_remaining = est_time_per_tile * remaining
            
            self.logger.info(f"Progress: {(i+1)}/{total_files} ({(i+1)/total_files*100:.1f}%)")
            self.logger.info(f"Elapsed: {elapsed/3600:.2f} hours, Estimated remaining: {est_remaining/3600:.2f} hours")
            self.logger.info(f"Success rate: {processed_tiles}/{i+1} ({processed_tiles/(i+1)*100:.1f}%)")
            
            # Force garbage collection
            gc.collect()
        
        # Generate final report
        overall_elapsed = time.time() - overall_start_time
        self.logger.info("\n" + "="*80)
        self.logger.info("AGBD pipeline completed!")
        self.logger.info(f"Total elapsed time: {overall_elapsed/3600:.2f} hours")
        self.logger.info(f"Tiles processed: {processed_tiles}/{total_files}")
        self.logger.info(f"Tiles failed: {failed_tiles}/{total_files}")

        return True