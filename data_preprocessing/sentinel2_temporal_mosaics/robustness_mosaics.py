"""
Harmonized robustness assessment for Sentinel-2 median mosaics.

This script assesses the robustness of median mosaics with different numbers of scenes.
Tests mosaic quality with varying numbers of input scenes and provides recommendations
for optimal scene numbers.

Author: Diego Bengochea
"""

import xarray as xr
import numpy as np
import dask.array as da
from pystac_client import Client
from datetime import datetime
from s2_utils import setup_logging, load_config, mask_scene


def assess_median_robustness(scenes, min_scenes=5, max_scenes=30, step=5):
    """
    Assess the robustness of median mosaics with different numbers of scenes.
    
    Args:
        scenes (list): List of xarray DataArrays
        min_scenes (int): Minimum number of scenes to test
        max_scenes (int): Maximum number of scenes to test
        step (int): Step size for number of scenes
        
    Returns:
        tuple: (results_dict, reference_mosaic)
    """
    logger = setup_logging()
    total_scenes = len(scenes)
    results = {}
    
    logger.info(f"Assessing robustness with {total_scenes} total scenes")
    logger.info(f"Testing scene counts from {min_scenes} to {min(max_scenes, total_scenes)} (step: {step})")
    
    # Stack all scenes
    stacked_scenes = xr.concat(scenes, dim="time")
    
    # Reference mosaic using all scenes
    reference_mosaic = stacked_scenes.median(dim="time")
    logger.info("Created reference mosaic using all available scenes")
    
    for n_scenes in range(min_scenes, min(max_scenes, total_scenes), step):
        logger.info(f"Testing with {n_scenes} scenes...")
        
        # Perform multiple random samples
        differences = []
        valid_pixel_counts = []
        
        for _ in range(10):  # 10 random samples for each n_scenes
            # Randomly sample n scenes
            sample_indices = np.random.choice(total_scenes, n_scenes, replace=False)
            sampled_scenes = stacked_scenes.isel(time=sample_indices)
            
            # Create median mosaic
            sample_mosaic = sampled_scenes.median(dim="time")
            
            # Calculate difference from reference
            diff = abs(sample_mosaic - reference_mosaic)
            differences.append(float(diff.mean().values))
            
            # Calculate valid pixel percentage
            valid_pixels = (~sample_mosaic.isnull()).sum() / sample_mosaic.size
            valid_pixel_counts.append(float(valid_pixels))
        
        results[n_scenes] = {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'mean_valid_pixels': np.mean(valid_pixel_counts),
            'std_valid_pixels': np.std(valid_pixel_counts)
        }
        
        logger.info(f"  Mean difference: {results[n_scenes]['mean_difference']:.4f}")
        logger.info(f"  Valid pixels: {results[n_scenes]['mean_valid_pixels']*100:.1f}%")
    
    return results, reference_mosaic


def load_and_mask_scene(item, config):
    """
    Load and mask a single scene using configuration parameters.
    
    Args:
        item: STAC item representing a single scene
        config (dict): Configuration dictionary
        
    Returns:
        xarray.Dataset: Loaded and masked scene
    """
    from odc.stac import load
    
    # Load scene data
    scene = load(
        [item],
        bands=config['data']['bands'],
        chunks={'x': config['processing']['chunk_size'], 
                'y': config['processing']['chunk_size']},
        resampling='bilinear'
    )
    
    # Apply masking using shared function from s2_utils
    masked_scene, _ = mask_scene(scene, config['scl']['valid_classes'])
    
    # Drop SCL band
    masked_scene = masked_scene.drop_vars(config['data']['bands_drop'])
    
    return masked_scene


def print_results(results):
    """
    Print formatted results of robustness assessment.
    
    Args:
        results (dict): Results dictionary from assess_median_robustness
    """
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("ROBUSTNESS ASSESSMENT RESULTS")
    logger.info("="*60)
    
    for n_scenes, stats in results.items():
        logger.info(f"\nNumber of scenes: {n_scenes}")
        logger.info(f"  Mean difference from reference: {stats['mean_difference']:.4f} ± {stats['std_difference']:.4f}")
        logger.info(f"  Valid pixel percentage: {stats['mean_valid_pixels']*100:.1f}% ± {stats['std_valid_pixels']*100:.1f}%")
    
    # Find optimal number of scenes (where improvement plateaus)
    scene_counts = sorted(results.keys())
    differences = [results[n]['mean_difference'] for n in scene_counts]
    
    # Simple heuristic: find where improvement becomes marginal (< 5% reduction)
    optimal_scenes = scene_counts[0]
    for i in range(1, len(scene_counts)):
        improvement = (differences[i-1] - differences[i]) / differences[i-1]
        if improvement < 0.05:  # Less than 5% improvement
            optimal_scenes = scene_counts[i-1]
            break
        optimal_scenes = scene_counts[i]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RECOMMENDATION: Use at least {optimal_scenes} scenes for optimal mosaic quality")
    logger.info(f"{'='*60}")


def main():
    """Main robustness assessment pipeline."""
    logger = setup_logging()
    logger.info("Starting Sentinel-2 mosaic robustness assessment...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Example usage - modify these parameters as needed
        bbox = [-122.34, 37.74, -122.26, 37.80]  # San Francisco Bay Area
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        
        logger.info("Using example area: San Francisco Bay Area")
        logger.info("Using example time period: 2022")
        logger.info("Modify these parameters in the script for different areas/periods")
        
        # Initialize STAC catalog
        stac_url = config['data']['stac_url']
        client = Client.open(stac_url)
        
        # Search for scenes
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        )
        
        items = list(search.get_items())
        logger.info(f"Found {len(items)} scenes matching criteria")
        
        if len(items) < config.get('robustness', {}).get('min_scenes', 5):
            raise ValueError(f"Not enough scenes found ({len(items)}). Minimum required: 5")
        
        # Load and mask all scenes
        scenes = []
        logger.info("Loading and masking scenes...")
        
        for i, item in enumerate(items):
            try:
                scene = load_and_mask_scene(item, config)
                scenes.append(scene)
                logger.info(f"Loaded scene {i+1}/{len(items)}")
            except Exception as e:
                logger.warning(f"Failed to load scene {i+1}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(scenes)} scenes")
        
        if len(scenes) < 5:
            raise ValueError(f"Not enough valid scenes loaded ({len(scenes)}). Minimum required: 5")
        
        # Run robustness assessment
        results, reference = assess_median_robustness(scenes)
        
        # Print results
        print_results(results)
        
        logger.info("Robustness assessment completed successfully!")
        
    except Exception as e:
        logger.error(f"Assessment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
