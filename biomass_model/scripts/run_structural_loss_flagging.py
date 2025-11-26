#!/usr/bin/env python3
"""
Parallelized Structural Loss Flagging Script

Computes probability maps distinguishing structural biomass loss from stress
with multiprocessing support.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_model.core.structural_losses import (
    load_biomass_maps,
    compute_structural_loss_probability,
    compute_loss_probability,
    save_probability_map
)
from shared_utils import setup_logging, get_logger
from shared_utils.central_data_paths_constants import (
    BIOMASS_MAPS_TILED_DIR,
    BIOMASS_MAPS_STRUCTURAL_LOSS_DIR
)


def extract_tile_id(filepath):
    """Extract spatial identifier from filename."""
    stem = filepath.stem
    parts = stem.split('_')
    tile_parts = [p for p in parts if p.startswith('N') or p.startswith('W') or p.startswith('E')]
    return '_'.join(tile_parts) if tile_parts else stem


def process_single_tile(args):
    """
    Process a single tile for structural loss probability.
    
    Args:
        args: tuple of (tile_id, mean_files_dict, unc_files_dict, output_dir, year, is_2024_edge)
    
    Returns:
        tuple: (success, tile_id, year)
    """
    tile_id, mean_files, unc_files, output_dir, year, is_2024_edge = args
    
    logger = get_logger('biomass_estimation.structural_loss_flagging')
    
    try:
        if is_2024_edge:
            # 2024 edge case: only t-1 and t
            biomass_t_minus1, unc_t_minus1, _ = load_biomass_maps(
                mean_files['t_minus1'], unc_files['t_minus1']
            )
            biomass_t, unc_t, meta = load_biomass_maps(
                mean_files['t'], unc_files['t']
            )
            
            # Check shapes
            if biomass_t.shape != biomass_t_minus1.shape:
                logger.warning(f"Shape mismatch for tile {tile_id}")
                return (False, tile_id, year)
            
            # Compute loss probability only
            prob_loss = compute_loss_probability(
                biomass_t_minus1, biomass_t,
                unc_t_minus1, unc_t
            )
            
            tile_name = mean_files['t'].stem
            save_probability_map(
                prob_loss,
                output_dir / f"{tile_name}_prob_structural_loss.tif",
                meta
            )
            
        else:
            # Regular case: t-1, t, t+1
            biomass_t_minus1, unc_t_minus1, _ = load_biomass_maps(
                mean_files['t_minus1'], unc_files['t_minus1']
            )
            biomass_t, unc_t, meta = load_biomass_maps(
                mean_files['t'], unc_files['t']
            )
            biomass_t_plus1, unc_t_plus1, _ = load_biomass_maps(
                mean_files['t_plus1'], unc_files['t_plus1']
            )
            
            # Check shapes
            if not (biomass_t.shape == biomass_t_minus1.shape == biomass_t_plus1.shape):
                logger.error(f"Shape mismatch for tile {tile_id}")
                return (False, tile_id, year)
            
            # Compute structural loss probability
            _, _, prob_structural = compute_structural_loss_probability(
                biomass_t_minus1, biomass_t, biomass_t_plus1,
                unc_t_minus1, unc_t, unc_t_plus1
            )
            
            tile_name = mean_files['t'].stem
            save_probability_map(
                prob_structural,
                output_dir / f"{tile_name}_prob_structural_loss.tif",
                meta
            )
        
        return (True, tile_id, year)
        
    except Exception as e:
        logger.error(f"Failed processing tile {tile_id} year {year}: {e}")
        return (False, tile_id, year)


def collect_tile_tasks(years, tile_pattern, input_mean_dir, input_uncertainty_dir, output_base_dir):
    """
    Collect all tile processing tasks across years.
    
    Returns:
        list of task tuples for process_single_tile
    """
    tasks = []
    
    for year in years:
        # Find files for t-1, t, t+1
        mean_files_lists = {
            't_minus1': list((input_mean_dir / str(year-1)).glob(tile_pattern.format(year=year-1))),
            't': list((input_mean_dir / str(year)).glob(tile_pattern.format(year=year))),
            't_plus1': list((input_mean_dir / str(year+1)).glob(tile_pattern.format(year=year+1)))
        }
        
        uncertainty_files_lists = {
            't_minus1': list((input_uncertainty_dir / str(year-1)).glob(tile_pattern.format(year=year-1))),
            't': list((input_uncertainty_dir / str(year)).glob(tile_pattern.format(year=year))),
            't_plus1': list((input_uncertainty_dir / str(year+1)).glob(tile_pattern.format(year=year+1)))
        }
        
        # Check files exist
        if not all(mean_files_lists.values()) or not all(uncertainty_files_lists.values()):
            continue
        
        # Build tile dicts
        tiles_t = {extract_tile_id(f): f for f in mean_files_lists['t']}
        tiles_t_minus1 = {extract_tile_id(f): f for f in mean_files_lists['t_minus1']}
        tiles_t_plus1 = {extract_tile_id(f): f for f in mean_files_lists['t_plus1']}
        
        unc_tiles_t = {extract_tile_id(f): f for f in uncertainty_files_lists['t']}
        unc_tiles_t_minus1 = {extract_tile_id(f): f for f in uncertainty_files_lists['t_minus1']}
        unc_tiles_t_plus1 = {extract_tile_id(f): f for f in uncertainty_files_lists['t_plus1']}
        
        output_dir = output_base_dir / str(year)
        
        # Create task for each matching tile
        for tile_id, mean_t in tiles_t.items():
            if (tile_id in tiles_t_minus1 and tile_id in tiles_t_plus1 and
                tile_id in unc_tiles_t and tile_id in unc_tiles_t_minus1 and tile_id in unc_tiles_t_plus1):
                
                mean_files_dict = {
                    't_minus1': tiles_t_minus1[tile_id],
                    't': mean_t,
                    't_plus1': tiles_t_plus1[tile_id]
                }
                
                unc_files_dict = {
                    't_minus1': unc_tiles_t_minus1[tile_id],
                    't': unc_tiles_t[tile_id],
                    't_plus1': unc_tiles_t_plus1[tile_id]
                }
                
                tasks.append((tile_id, mean_files_dict, unc_files_dict, output_dir, year, False))
    
    return tasks


def collect_2024_tasks(tile_pattern, input_mean_dir, input_uncertainty_dir, output_dir):
    """Collect 2024 edge case tasks."""
    mean_files_lists = {
        't_minus1': list((input_mean_dir / '2023').glob(tile_pattern.format(year=2023))),
        't': list((input_mean_dir / '2024').glob(tile_pattern.format(year=2024)))
    }
    
    uncertainty_files_lists = {
        't_minus1': list((input_uncertainty_dir / '2023').glob(tile_pattern.format(year=2023))),
        't': list((input_uncertainty_dir / '2024').glob(tile_pattern.format(year=2024)))
    }
    
    if not all(mean_files_lists.values()) or not all(uncertainty_files_lists.values()):
        return []
    
    tiles_t = {extract_tile_id(f): f for f in mean_files_lists['t']}
    tiles_t_minus1 = {extract_tile_id(f): f for f in mean_files_lists['t_minus1']}
    unc_tiles_t = {extract_tile_id(f): f for f in uncertainty_files_lists['t']}
    unc_tiles_t_minus1 = {extract_tile_id(f): f for f in uncertainty_files_lists['t_minus1']}
    
    tasks = []
    for tile_id, mean_t in tiles_t.items():
        if (tile_id in tiles_t_minus1 and tile_id in unc_tiles_t and tile_id in unc_tiles_t_minus1):
            mean_files_dict = {
                't_minus1': tiles_t_minus1[tile_id],
                't': mean_t
            }
            
            unc_files_dict = {
                't_minus1': unc_tiles_t_minus1[tile_id],
                't': unc_tiles_t[tile_id]
            }
            
            tasks.append((tile_id, mean_files_dict, unc_files_dict, output_dir, 2024, True))
    
    return tasks


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Parallel structural loss flagging')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers (default: 16)')
    args = parser.parse_args()
    
    logger = setup_logging(level='INFO', component_name='structural_loss_flagging')
    logger.info(f"Using {args.workers} parallel workers")
    
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    tile_pattern = '*{year}*.tif'
    
    input_mean_dir = BIOMASS_MAPS_TILED_DIR / "AGBD_mean"
    input_uncertainty_dir = BIOMASS_MAPS_TILED_DIR / "AGBD_uncertainty"
    output_base_dir = BIOMASS_MAPS_STRUCTURAL_LOSS_DIR
    
    # Collect all tasks
    logger.info("Collecting tasks for regular years...")
    tasks = collect_tile_tasks(years, tile_pattern, input_mean_dir, input_uncertainty_dir, output_base_dir)
    logger.info(f"Found {len(tasks)} regular tile tasks")
    
    logger.info("Collecting tasks for 2024 edge case...")
    tasks_2024 = collect_2024_tasks(tile_pattern, input_mean_dir, input_uncertainty_dir, output_base_dir / "2024")
    logger.info(f"Found {len(tasks_2024)} 2024 tile tasks")
    
    all_tasks = tasks + tasks_2024
    logger.info(f"Total tasks: {len(all_tasks)}")
    
    if len(all_tasks) == 0:
        logger.warning("No tasks found!")
        return False
    
    # Process in parallel
    logger.info("Processing tiles in parallel...")
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_tile, task) for task in all_tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
            success, tile_id, year = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    logger.info(f"\nCompleted!")
    logger.info(f"  Successfully processed: {success_count} tiles")
    logger.info(f"  Failed: {fail_count} tiles")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)