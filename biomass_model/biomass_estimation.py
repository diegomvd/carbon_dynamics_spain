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
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
import pandas as pd
import dask.array as da
import os

from dask_utils import dask_cluster
from logging_utils import log_memory_usage, logger, timer
from config import get_config
from allometry import build_forest_type_mapping, get_allometry_parameters
from io_utils import extract_tile_info, build_savepath, check_existing_outputs, find_masks_for_tile, read_height_and_mask_xarray, write_xarray_results
from monte_carlo import monte_carlo_biomass_optimized


def process_forest_type(height_file, mask_file, code, forest_type_name, agb_allometries, 
                       bgb_allometries, forest_types, output_dir):
    """
    Process a single forest type with xarray and dask for biomass estimation.
    
    Args:
        height_file (str): Path to canopy height raster
        mask_file (str): Path to forest type mask raster
        code (str): Forest type code identifier
        forest_type_name (str): Human-readable forest type name
        agb_allometries (pd.DataFrame): Above-ground biomass allometry parameters
        bgb_allometries (pd.DataFrame): Below-ground biomass ratio parameters
        forest_types (pd.DataFrame): Forest type hierarchy table
        output_dir (str): Directory for output files
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    config = get_config()
    
    try:
        logger.info(f"Processing forest type: {forest_type_name} (code: {code})")
        
        # Check if outputs already exist to avoid reprocessing
        if check_existing_outputs(Path(height_file), code):
            logger.info(f"Outputs already exist for {forest_type_name}, skipping")
            return True
        
        # Handle non-forest areas based on configuration
        if not config['processing']['compute_no_arbolado']:
            if forest_type_name == 'No arbolado':
                logger.info(f"Non-forest area ({forest_type_name}), skipping")
                return True

        # Create fresh Dask cluster for this forest type to manage memory
        with dask_cluster(
            config['compute']['num_workers'], 
            config['compute']['memory_limit'], 
            threads_per_worker=config['compute']['threads_per_worker']
        ) as client:
            # Log dashboard link for monitoring
            logger.info(f"Dask dashboard available at: {client.dashboard_link}")
            
            # Get allometry parameters for this specific forest type
            agb_params, bgb_params = get_allometry_parameters(
                forest_type_name, forest_types, agb_allometries, bgb_allometries, 
                config['forest_types']['tier_names']
            )
            
            # Read height and mask data with chunking for memory efficiency
            height_xr, mask_xr, out_meta = read_height_and_mask_xarray(
                height_file, mask_file, chunk_size=config['compute']['chunk_size']
            )
            
            # Apply forest type mask to height data
            masked_heights = height_xr.where(mask_xr)
            
            # Execute Monte Carlo biomass simulation with progress tracking
            with ProgressBar():
                with timer("Monte Carlo simulation"):
                    results = monte_carlo_biomass_optimized(
                        masked_heights, agb_params, bgb_params, 
                        num_samples=config['monte_carlo']['num_samples'], 
                        seed=config['monte_carlo']['seed']
                    )
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Write results to optimized GeoTIFF files
            success = write_xarray_results(
                results, height_file, code, forest_type_name, mask_xr, out_meta
            )
            
            if success:
                logger.info(f"Completed processing for {forest_type_name}")
                return True
            else:
                logger.error(f"Failed to write results for {forest_type_name}")
                return False
            
    except Exception as e:
        logger.error(f"Error processing forest type {forest_type_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def process_tile(height_file, agb_allometries, bgb_allometries, forest_types_table, code_to_name):
    """
    Process a single tile with all associated forest types.
    
    Args:
        height_file (str): Path to height raster
        agb_allometries (pd.DataFrame): Above-ground biomass allometry parameters
        bgb_allometries (pd.DataFrame): Below-ground biomass ratio parameters
        forest_types_table (pd.DataFrame): Forest type hierarchy table
        code_to_name (dict): Mapping from forest type codes to names
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    config = get_config()
    tile_name = Path(height_file).stem
    logger.info(f"Processing tile: {tile_name}")
    
    # Extract tile information from filename
    tile_info = extract_tile_info(height_file)
    if not tile_info:
        logger.error(f"Failed to extract tile information from {height_file}")
        return False
        
    # Find all forest type masks for this tile
    try:
        masks = find_masks_for_tile(tile_info, config['data']['masks_dir'])
    except OSError as e:
        logger.error(f"Error finding masks for {tile_name}: {e}")
        return False
        
    if not masks:
        logger.warning(f"No masks found for {tile_name}")
        return False
        
    logger.info(f"Found {len(masks)} masks for {tile_name}")
    
    # Define output directory structure
    output_dir = os.path.join(os.path.dirname(height_file), 'outputs')
    
    # Process each forest type independently
    results = []
    for mask_path, code in tqdm(masks, desc=f"Processing forest types for {tile_name}"):
        print(code)
        forest_type = code_to_name.get(code, f"Unknown_{code}")
        
        # Process this specific forest type
        success = process_forest_type(
            height_file, mask_path, code, forest_type,
            agb_allometries, bgb_allometries, forest_types_table,
            output_dir
        )
        
        results.append(success)
        
        # Force garbage collection between forest types
        gc.collect()
    
    # Calculate success rate for this tile
    success_count = sum(1 for r in results if r)
    logger.info(f"Processed {success_count}/{len(masks)} forest types for {tile_name}")
    
    return success_count > 0


def process_all_tiles():
    """
    Main orchestration function to process all tiles using optimized approach.
    
    Loads reference data once and processes tiles sequentially with detailed
    progress tracking and memory management.
    """
    config = get_config()
    overall_start_time = time.time()
    processed_tiles = 0
    failed_tiles = 0
    
    try:
        # Log initial system state
        log_memory_usage("Initial")
        
        # Load reference data once for reuse across all tiles
        logger.info('Loading forest types hierarchy and coefficients')
        
        try:
            # Load forest type hierarchy from centralized config
            forest_types = pd.read_csv(config['data']['forest_type_dir'])
            forest_types['Dummy'] = 'General'  # Add dummy tier for hierarchy traversal
            logger.info(f"Loaded forest types hierarchy with {len(forest_types)} entries")

            # Load above-ground biomass allometries
            agb_allometries = pd.read_csv(config['data']['allometries_dir']).set_index('forest_type')
            logger.info(f"Loaded AGB allometries for {len(agb_allometries)} forest types")
            
            # Load below-ground biomass coefficients
            bgb_allometries = pd.read_csv(config['data']['bgb_coeffs_dir']).set_index('forest_type')
            logger.info(f"Loaded BGB coefficients for {len(bgb_allometries)} forest types")
            
            # Build forest type code-to-name mapping
            code_to_name = build_forest_type_mapping(
                config['data']['mfe_dir'],
                cache_path=config['forest_types']['cache_file'],
                use_cache=config['forest_types']['use_cache']
            )
            logger.info(f"Built forest type mapping with {len(code_to_name)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        log_memory_usage("After loading reference data")
        
        # Discover input files using configuration
        input_dir = config['data']['input_data_dir']
        file_pattern = config['processing']['file_pattern']
        logger.info(f"Looking for canopy height files in {input_dir}")
        
        try:
            files = list(Path(input_dir).glob(file_pattern))
        except Exception as e:
            logger.error(f"Failed to list input files: {str(e)}")
            return
            
        # Randomize processing order for better load balancing
        random.shuffle(files)

        # Filter and sort files by year (newest first) using target years from config
        files_with_years = []
        target_years = config['processing']['target_years']
        
        for f in files:
            tile_info = extract_tile_info(f)
            if tile_info:
                if tile_info['year'] in target_years:
                    files_with_years.append((f, tile_info['year']))
            else:
                files_with_years.append((f, '0000'))  # Default for failed extraction
                
        # Sort by year (newest first)
        files_with_years.sort(key=lambda x: x[1], reverse=True)
        files = [f[0] for f in files_with_years]

        total_files = len(files)
        logger.info(f"Found {total_files} input tiles to process")
        
        # Process tiles sequentially with detailed progress tracking
        for i, fname in enumerate(files):
            logger.info(f"\n{'='*80}\nProcessing tile {i+1}/{total_files}: {fname}\n{'='*80}")
            
            try:
                # Process this individual tile
                success = process_tile(
                    fname, 
                    agb_allometries,
                    bgb_allometries,
                    forest_types,
                    code_to_name,
                )
                
                # Update counters based on processing result
                if success:
                    processed_tiles += 1
                else:
                    failed_tiles += 1
                    
            except Exception as e:
                logger.error(f"Failed to process tile {fname}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                failed_tiles += 1
            
            # Calculate and log progress statistics
            elapsed = time.time() - overall_start_time
            remaining = total_files - (i + 1)
            est_time_per_tile = elapsed / (i + 1)
            est_remaining = est_time_per_tile * remaining
            
            logger.info(f"Progress: {(i+1)}/{total_files} ({(i+1)/total_files*100:.1f}%)")
            logger.info(f"Elapsed: {elapsed/3600:.2f} hours, Estimated remaining: {est_remaining/3600:.2f} hours")
            logger.info(f"Success rate: {processed_tiles}/{i+1} ({processed_tiles/(i+1)*100:.1f}%)")
            
            # Monitor memory usage
            log_memory_usage(f"After tile {i+1}/{total_files}")
            
            # Force garbage collection between tiles
            gc.collect()
        
        # Generate final processing report
        overall_elapsed = time.time() - overall_start_time
        logger.info("\n" + "="*80)
        logger.info("Processing completed!")
        logger.info(f"Total elapsed time: {overall_elapsed/3600:.2f} hours")
        logger.info(f"Tiles processed: {processed_tiles}/{total_files}")
        logger.info(f"Tiles failed: {failed_tiles}/{total_files}")
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Final cleanup and memory reporting
        gc.collect()
        log_memory_usage("Final")
        logger.info("Processing completed")


if __name__ == '__main__':
    process_all_tiles()