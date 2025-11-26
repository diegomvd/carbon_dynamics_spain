"""
BGBD Estimation Pipeline

Pipeline for BGBD and Total biomass estimation using analytical error propagation.
Follows the AGBD estimation structure but applies BGB/AGB ratios to existing AGBD maps.

Author: Diego Bengochea
"""

from pathlib import Path
import random
import gc
import time
import pandas as pd
from typing import Optional, Union
import xarray as xr
import rasterio

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config
from shared_utils.central_data_paths_constants import *

# Component imports
from biomass_model.core.bgbd_estimator import BGBDDirectEstimator, TotalBiomassEstimator
from biomass_model.core.biomass_utils import BiomassUtils
from biomass_model.core.dask_utils import DaskClusterManager


class BGBDEstimationPipeline:
    """
    Pipeline for BGBD and Total biomass estimation with analytical error propagation.
    
    Produces outputs per tile:
    - BGBD_mean and BGBD_uncertainty
    - Total_mean and Total_uncertainty
    
    Follows AGBD estimation structure.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BGBD estimation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name='biomass_model')
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='bgbd_estimation',
            log_file=self.config['logging'].get('log_file')
        )

        # Initialize components
        self.bgbd_estimator = BGBDDirectEstimator(self.config)
        self.total_estimator = TotalBiomassEstimator(self.config)
        self.biomass_utils = BiomassUtils(self.config)
        self.dask_manager = DaskClusterManager(self.config)

        # Load BGB ratios
        self.ratios_df = pd.read_csv(BGB_RATIOS_FILE)
        self.logger.info(f"Loaded {len(self.ratios_df)} BGB/AGB ratios")
        
        # Pipeline state
        self.client = None
        self.start_time = None
        
        self.logger.info("Initialized BGBDEstimationPipeline")

    def get_ratio_parameters(self, forest_type_code: str) -> tuple:
        """Get ratio mean and uncertainty for a forest type code."""
        if forest_type_code in self.ratios_df['forest_type'].values:
            row = self.ratios_df[self.ratios_df['forest_type'] == forest_type_code].iloc[0]
            ratio_mean = row['mean']
            ratio_uncertainty = (row['q90'] - row['q10']) / 2
            return ratio_mean, ratio_uncertainty
        else:
            # Fallback to General
            return self.get_ratio_parameters('General')

    def process_tile(self, agbd_mean_file: Path, agbd_unc_file: Path, 
                     forest_types_table: pd.DataFrame, code_to_name: dict):
        """Process all forest types and merge into single tile arrays."""
        logger = get_logger('biomass_estimation')
        tile_name = Path(agbd_mean_file).stem.replace('AGBD_mean_', '')
        logger.info(f"Processing tile: {tile_name}")

        # Extract tile info for mask finding
        tile_info = self.biomass_utils.extract_tile_info(agbd_mean_file)
        if not tile_info:
            return False

        masks_dir = str(FOREST_TYPE_MASKS_DIR)
        masks = self.biomass_utils.find_masks_for_tile(tile_info, masks_dir)

        if not masks:
            return False

        logger.info(f"Found {len(masks)} masks for {tile_name}")

        # Initialize merged arrays (will be None until first result)
        bgbd_merged_mean = None
        bgbd_merged_unc = None
        total_merged_mean = None
        total_merged_unc = None
        metadata = None
        first_successful = True
        
        with self.dask_manager.create_cluster() as client:
            logger.info(f"Dask dashboard: {client.dashboard_link}")

            for mask_path, code in masks:
                forest_type = code_to_name.get(int(code), f"Unknown_{code}")

                try:
                    # Get ratio parameters for this forest type
                    ratio_mean, ratio_unc = self.get_ratio_parameters(code)
                    
                    # Load AGBD mean data with mask
                    agbd_mean_data, mask_data, meta = self.biomass_utils.read_height_and_mask_xarray(
                        agbd_mean_file, mask_path
                    )
                    
                    # Load AGBD uncertainty data 
                    agbd_unc_data, _, _ = self.biomass_utils.read_height_and_mask_xarray(
                        agbd_unc_file, mask_path
                    )

                    if first_successful:
                        metadata = meta
                        # Initialize with nodata
                        bgbd_merged_mean = xr.full_like(agbd_mean_data, meta['nodata'], dtype='float32')
                        bgbd_merged_unc = xr.full_like(agbd_mean_data, meta['nodata'], dtype='float32')
                        total_merged_mean = xr.full_like(agbd_mean_data, meta['nodata'], dtype='float32')
                        total_merged_unc = xr.full_like(agbd_mean_data, meta['nodata'], dtype='float32')
                        first_successful = False
                        
                    # Apply mask to AGBD data
                    masked_agbd_mean = agbd_mean_data.where(mask_data)
                    masked_agbd_unc = agbd_unc_data.where(mask_data)
                    
                    # Run BGBD estimation
                    bgbd_results = self.bgbd_estimator.run(
                        masked_agbd_mean, masked_agbd_unc, ratio_mean, ratio_unc
                    )
                    
                    # Run Total estimation  
                    total_results = self.total_estimator.run(
                        masked_agbd_mean, masked_agbd_unc, ratio_mean, ratio_unc
                    )

                    # Merge: fill in pixels where mask is True
                    bgbd_merged_mean = bgbd_merged_mean.where(~mask_data, bgbd_results['bgbd_mean'])
                    bgbd_merged_unc = bgbd_merged_unc.where(~mask_data, bgbd_results['bgbd_uncertainty'])
                    total_merged_mean = total_merged_mean.where(~mask_data, total_results['total_mean'])
                    total_merged_unc = total_merged_unc.where(~mask_data, total_results['total_uncertainty'])

                    logger.info(f"✓ Merged {forest_type}")

                except Exception as e:
                    logger.error(f"Error processing {forest_type}: {e}")
                    continue
                
            if bgbd_merged_mean is None:
                logger.error(f"No forest types processed successfully for {tile_name}")
                return False
            
            # Compute final merged arrays
            logger.info("Computing merged results...")
            bgbd_merged_mean = bgbd_merged_mean.compute()
            bgbd_merged_unc = bgbd_merged_unc.compute()
            total_merged_mean = total_merged_mean.compute()
            total_merged_unc = total_merged_unc.compute()

        # Write 4 files for entire tile
        logger.info(f"Writing merged tile results...")
        success = self.write_tile_outputs(
            bgbd_merged_mean, bgbd_merged_unc,
            total_merged_mean, total_merged_unc,
            agbd_mean_file, metadata
        )

        gc.collect()
        return success
    
    def write_tile_outputs(self, bgbd_mean, bgbd_unc, total_mean, total_unc,
                           agbd_file, metadata):
        """Write BGBD and Total outputs for a tile."""
        try:
            # Extract tile info
            tile_info = self.biomass_utils.extract_tile_info(agbd_file)
            year = tile_info['year']
            tile_name = agbd_file.stem.replace('AGBD_mean_', '')
            
            # Construct output directories
            bgbd_mean_dir = BIOMASS_MAPS_TILED_DIR / "BGBD_mean" / str(year)
            bgbd_unc_dir = BIOMASS_MAPS_TILED_DIR / "BGBD_uncertainty" / str(year)
            total_mean_dir = BIOMASS_MAPS_TILED_DIR / "Total_mean" / str(year)
            total_unc_dir = BIOMASS_MAPS_TILED_DIR / "Total_uncertainty" / str(year)
            
            # Create directories
            bgbd_mean_dir.mkdir(parents=True, exist_ok=True)
            bgbd_unc_dir.mkdir(parents=True, exist_ok=True)
            total_mean_dir.mkdir(parents=True, exist_ok=True)
            total_unc_dir.mkdir(parents=True, exist_ok=True)
            
            # Write files
            with rasterio.open(bgbd_mean_dir / f"BGBD_mean_{tile_name}.tif", 'w', **metadata) as dst:
                dst.write(bgbd_mean.astype('float32'), 1)
            
            with rasterio.open(bgbd_unc_dir / f"BGBD_uncertainty_{tile_name}.tif", 'w', **metadata) as dst:
                dst.write(bgbd_unc.astype('float32'), 1)
            
            with rasterio.open(total_mean_dir / f"Total_mean_{tile_name}.tif", 'w', **metadata) as dst:
                dst.write(total_mean.astype('float32'), 1)
            
            with rasterio.open(total_unc_dir / f"Total_uncertainty_{tile_name}.tif", 'w', **metadata) as dst:
                dst.write(total_unc.astype('float32'), 1)
            
            self.logger.info(f"✓ Wrote outputs for {tile_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write outputs: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        Main orchestration function to process all tiles.
        
        Returns:
            bool: True if pipeline succeeded, False otherwise
        """
        overall_start_time = time.time()
        processed_tiles = 0
        failed_tiles = 0

        config = self.config
        
        self.logger.info("Starting BGBD estimation pipeline")
        

        # Get forest types
        forest_types = pd.read_csv(FOREST_TYPES_TIERS_FILE)
        forest_types['Dummy'] = 'General'
        code_to_name = self.biomass_utils.build_forest_type_mapping(
                FOREST_TYPE_MAPS_DIR,
                cache_path=FOREST_TYPE_MFE_CODE_TO_NAME_FILE, 
                use_cache=config.get('forest_types', {}).get('use_cache', True)
            )
        code_to_name = dict(zip(code_to_name.code, code_to_name.name))
        self.logger.info(f"Built forest type mapping with {len(code_to_name)} entries")
        
        # Find AGBD files
        agbd_mean_dir = BIOMASS_MAPS_TILED_DIR / "AGBD_mean"
        
        if not agbd_mean_dir.exists():
            self.logger.error(f"AGBD directory not found: {agbd_mean_dir}")
            return False

        years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        
        # Collect all tiles
        all_tiles = []
        for year in years:
            year_dir = agbd_mean_dir / str(year)
            if year_dir.exists():
                tiles = list(year_dir.glob('AGBD_mean_*.tif'))
                all_tiles.extend(tiles)
        
        random.shuffle(all_tiles)
        total_files = len(all_tiles)
        self.logger.info(f"Found {total_files} AGBD tiles to process")
        
        # Process tiles sequentially
        for i, agbd_mean_file in enumerate(all_tiles):
            self.logger.info(f"\n{'='*80}\nProcessing tile {i+1}/{total_files}: {agbd_mean_file.name}\n{'='*80}")
            
            # Find corresponding uncertainty file
            agbd_unc_file = agbd_mean_file.parent.parent.parent / "AGBD_uncertainty" / str(agbd_mean_file.parent.name) / agbd_mean_file.name.replace('AGBD_mean', 'AGBD_uncertainty')
            
            if not agbd_unc_file.exists():
                self.logger.error(f"Uncertainty file not found: {agbd_unc_file}")
                failed_tiles += 1
                continue
            
            try:
                success = self.process_tile(agbd_mean_file, agbd_unc_file, forest_types, code_to_name)
                
                if success:
                    processed_tiles += 1
                else:
                    failed_tiles += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process tile {agbd_mean_file.name}: {str(e)}")
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
            
            gc.collect()
        
        # Generate final report
        overall_elapsed = time.time() - overall_start_time
        self.logger.info("\n" + "="*80)
        self.logger.info("BGBD pipeline completed!")
        self.logger.info(f"Total time: {overall_elapsed/3600:.2f} hours")
        self.logger.info(f"Processed: {processed_tiles} tiles")
        self.logger.info(f"Failed: {failed_tiles} tiles")
        self.logger.info("="*80)
        
        return failed_tiles == 0