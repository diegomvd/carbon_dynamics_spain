"""
Structural Loss Flagging

Distinguishes structural biomass loss from stress-induced apparent loss
using uncertainty-aware probability framework.
"""

import numpy as np
from scipy.stats import norm
import rasterio
from pathlib import Path
from typing import Tuple, Optional

from shared_utils import get_logger


def convert_ci_to_std(uncertainty_half_width: np.ndarray, ci_level: float = 0.80) -> np.ndarray:
    """
    Convert confidence interval half-width to standard deviation.
    
    For 80% CI: U = 1.28 * σ
    """
    z_score = norm.ppf((1 + ci_level) / 2)
    return uncertainty_half_width / z_score


def compute_loss_probability(
    biomass_t_minus1: np.ndarray,
    biomass_t: np.ndarray,
    uncertainty_t_minus1: np.ndarray,
    uncertainty_t: np.ndarray
) -> np.ndarray:
    """
    Compute probability that biomass decreased from t-1 to t.
    
    P(loss) = P(B_t < B_{t-1}) given measurement uncertainties
    """
    # Convert CI half-widths to standard deviations
    sigma_t_minus1 = convert_ci_to_std(uncertainty_t_minus1)
    sigma_t = convert_ci_to_std(uncertainty_t)
    
    # Mean change
    mean_change = biomass_t_minus1 - biomass_t
    
    # Combined uncertainty of the change
    sigma_change = np.sqrt(sigma_t_minus1**2 + sigma_t**2)
    
    # Avoid division by zero
    valid_mask = sigma_change > 0
    z_score = np.zeros_like(mean_change)
    z_score[valid_mask] = mean_change[valid_mask] / sigma_change[valid_mask]
    
    # For zero uncertainty, if mean_change > 0 then prob=1, else prob=0
    z_score[~valid_mask] = np.where(mean_change[~valid_mask] > 0, 10.0, -10.0)
    
    # Standardize and compute probability
    prob_loss = norm.cdf(z_score)
    
    return prob_loss


def compute_structural_loss_probability(
    biomass_t_minus1: np.ndarray,
    biomass_t: np.ndarray,
    biomass_t_plus1: np.ndarray,
    uncertainty_t_minus1: np.ndarray,
    uncertainty_t: np.ndarray,
    uncertainty_t_plus1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute probability of structural loss in year t.
    
    Returns:
        prob_loss: P(biomass decreased at t)
        prob_no_recovery: P(biomass hasn't recovered by t+1)
        prob_structural_loss: Combined probability
    """
    # Step 1: Probability of loss at t
    prob_loss = compute_loss_probability(
        biomass_t_minus1, biomass_t,
        uncertainty_t_minus1, uncertainty_t
    )
    
    # Step 2: Probability of no recovery by t+1
    prob_no_recovery = compute_loss_probability(
        biomass_t_minus1, biomass_t_plus1,
        uncertainty_t_minus1, uncertainty_t_plus1
    )
    
    # Step 3: Combined probability
    prob_structural_loss = prob_loss * prob_no_recovery
    
    return prob_loss, prob_no_recovery, prob_structural_loss


def load_biomass_maps(
    mean_file: Path,
    uncertainty_file: Path
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load biomass mean and uncertainty maps."""
    with rasterio.open(mean_file) as src:
        biomass_mean = src.read(1)
        meta = src.meta.copy()
    
    with rasterio.open(uncertainty_file) as src:
        uncertainty = src.read(1)
    
    return biomass_mean, uncertainty, meta


def save_probability_map(
    probability: np.ndarray,
    output_file: Path,
    meta: dict
) -> None:
    """Save probability map as GeoTIFF."""
    meta.update(dtype='float32', nodata=-9999.0)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(probability.astype('float32'), 1)


def process_tile_structural_loss(
    tile_pattern: str,
    year: int,
    input_mean_dir: Path,
    input_uncertainty_dir: Path,
    output_dir: Path
) -> bool:
    """
    Process structural loss probability for a single tile and year.
    """
    logger = get_logger('biomass_estimation.structural_loss_flagging')
    
    # Find files for t-1, t, t+1 in year subdirectories
    mean_files = {
        't_minus1': list((input_mean_dir / str(year-1)).glob(tile_pattern.format(year=year-1))),
        't': list((input_mean_dir / str(year)).glob(tile_pattern.format(year=year))),
        't_plus1': list((input_mean_dir / str(year+1)).glob(tile_pattern.format(year=year+1)))
    }
    
    uncertainty_files = {
        't_minus1': list((input_uncertainty_dir / str(year-1)).glob(tile_pattern.format(year=year-1))),
        't': list((input_uncertainty_dir / str(year)).glob(tile_pattern.format(year=year))),
        't_plus1': list((input_uncertainty_dir / str(year+1)).glob(tile_pattern.format(year=year+1)))
    }
    
    # Check all files exist
    for key in ['t_minus1', 't', 't_plus1']:
        if not mean_files[key] or not uncertainty_files[key]:
            logger.warning(f"Missing files for {key} at year {year}")
            return False
    
    # Match tiles across years by extracting tile identifier
    def extract_tile_id(filepath):
        # Extract spatial identifier from filename (e.g., N44.0_W8.0 or N42.0_E2.0)
        stem = filepath.stem
        parts = stem.split('_')
        # Find N and W/E coordinates
        tile_parts = [p for p in parts if p.startswith('N') or p.startswith('W') or p.startswith('E')]
        tile_id = '_'.join(tile_parts) if tile_parts else stem
        return tile_id
    
    # Build dict of tiles for each time period
    tiles_t = {extract_tile_id(f): f for f in mean_files['t']}
    tiles_t_minus1 = {extract_tile_id(f): f for f in mean_files['t_minus1']}
    tiles_t_plus1 = {extract_tile_id(f): f for f in mean_files['t_plus1']}
    
    unc_tiles_t = {extract_tile_id(f): f for f in uncertainty_files['t']}
    unc_tiles_t_minus1 = {extract_tile_id(f): f for f in uncertainty_files['t_minus1']}
    unc_tiles_t_plus1 = {extract_tile_id(f): f for f in uncertainty_files['t_plus1']}
    
    logger.info(f"Found tiles: t-1={len(tiles_t_minus1)}, t={len(tiles_t)}, t+1={len(tiles_t_plus1)}")
    logger.info(f"Example tile IDs at t: {list(tiles_t.keys())[:3]}")
    
    # Process each tile
    success_count = 0
    for tile_id, mean_t in tiles_t.items():
        # Check if this tile exists in all time periods
        if tile_id not in tiles_t_minus1 or tile_id not in tiles_t_plus1:
            logger.debug(f"Tile {tile_id} missing in some years, skipping")
            continue
        
        if tile_id not in unc_tiles_t or tile_id not in unc_tiles_t_minus1 or tile_id not in unc_tiles_t_plus1:
            logger.debug(f"Uncertainty for tile {tile_id} missing in some years, skipping")
            continue
        
        logger.debug(f"Processing tile {tile_id}:")
        logger.debug(f"  t-1: {tiles_t_minus1[tile_id].name}")
        logger.debug(f"  t:   {mean_t.name}")
        logger.debug(f"  t+1: {tiles_t_plus1[tile_id].name}")
        try:
            # Load data for matched tiles
            biomass_t_minus1, unc_t_minus1, _ = load_biomass_maps(
                tiles_t_minus1[tile_id], unc_tiles_t_minus1[tile_id]
            )
            biomass_t, unc_t, meta = load_biomass_maps(mean_t, unc_tiles_t[tile_id])
            biomass_t_plus1, unc_t_plus1, _ = load_biomass_maps(
                tiles_t_plus1[tile_id], unc_tiles_t_plus1[tile_id]
            )
            
            # Check shapes match
            if not (biomass_t.shape == biomass_t_minus1.shape == biomass_t_plus1.shape):
                logger.error(f"SHAPE MISMATCH for tile {tile_id}:")
                logger.error(f"  t-1 ({tiles_t_minus1[tile_id].name}): {biomass_t_minus1.shape}")
                logger.error(f"  t   ({mean_t.name}): {biomass_t.shape}")
                logger.error(f"  t+1 ({tiles_t_plus1[tile_id].name}): {biomass_t_plus1.shape}")
                continue
            
            # Compute probabilities
            _, _, prob_structural = compute_structural_loss_probability(
                biomass_t_minus1, biomass_t, biomass_t_plus1,
                unc_t_minus1, unc_t, unc_t_plus1
            )
            
            # Save single output
            tile_name = mean_t.stem
            save_probability_map(
                prob_structural,
                output_dir / f"{tile_name}_prob_structural_loss.tif",
                meta
            )
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed processing tile {mean_t.name}: {e}")
            continue
    
    return success_count > 0


def process_edge_case_2024(
    tile_pattern: str,
    input_mean_dir: Path,
    input_uncertainty_dir: Path,
    output_dir: Path
) -> bool:
    """
    Process 2024 edge case - only P(loss) without recovery assessment.
    """
    logger = get_logger('biomass_estimation.structural_loss_flagging')
    
    year = 2024
    mean_files = {
        't_minus1': list((input_mean_dir / '2023').glob(tile_pattern.format(year=2023))),
        't': list((input_mean_dir / '2024').glob(tile_pattern.format(year=2024)))
    }
    
    uncertainty_files = {
        't_minus1': list((input_uncertainty_dir / '2023').glob(tile_pattern.format(year=2023))),
        't': list((input_uncertainty_dir / '2024').glob(tile_pattern.format(year=2024)))
    }
    
    if not all(mean_files.values()) or not all(uncertainty_files.values()):
        logger.warning("Missing files for 2024 edge case")
        return False
    
    # Match tiles by spatial identifier
    def extract_tile_id(filepath):
        stem = filepath.stem
        parts = stem.split('_')
        tile_parts = [p for p in parts if p.startswith('N') or p.startswith('W') or p.startswith('E')]
        return '_'.join(tile_parts) if tile_parts else stem
    
    tiles_t = {extract_tile_id(f): f for f in mean_files['t']}
    tiles_t_minus1 = {extract_tile_id(f): f for f in mean_files['t_minus1']}
    unc_tiles_t = {extract_tile_id(f): f for f in uncertainty_files['t']}
    unc_tiles_t_minus1 = {extract_tile_id(f): f for f in uncertainty_files['t_minus1']}
    
    logger.info(f"2024: Found tiles: t-1={len(tiles_t_minus1)}, t={len(tiles_t)}")
    
    success_count = 0
    for tile_id, mean_t in tiles_t.items():
        if tile_id not in tiles_t_minus1 or tile_id not in unc_tiles_t or tile_id not in unc_tiles_t_minus1:
            logger.debug(f"Tile {tile_id} missing data, skipping")
            continue
        
        logger.debug(f"Processing 2024 tile {tile_id}:")
        logger.debug(f"  t-1: {tiles_t_minus1[tile_id].name}")
        logger.debug(f"  t:   {mean_t.name}")
        
        try:
            biomass_t_minus1, unc_t_minus1, _ = load_biomass_maps(
                tiles_t_minus1[tile_id], unc_tiles_t_minus1[tile_id]
            )
            biomass_t, unc_t, meta = load_biomass_maps(mean_t, unc_tiles_t[tile_id])
            
            # Check shapes match
            if biomass_t.shape != biomass_t_minus1.shape:
                logger.warning(f"Shape mismatch for tile {tile_id}: "
                             f"t-1={biomass_t_minus1.shape}, t={biomass_t.shape}")
                continue
            
            
            prob_loss = compute_loss_probability(
                biomass_t_minus1, biomass_t,
                unc_t_minus1, unc_t
            )
            
            tile_name = mean_t.stem
            save_probability_map(
                prob_loss,
                output_dir / f"{tile_name}_prob_structural_loss.tif",
                meta
            )
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed processing 2024 tile {mean_t.name}: {e}")
            continue
    
    return success_count > 0