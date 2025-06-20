#!/usr/bin/env python3
"""
Calculate spatial autocorrelation of interannual biomass changes.

This script computes semivariograms for interannual difference rasters to
analyze spatial patterns and autocorrelation lengths. Outputs data suitable
for creating variograms and analyzing spatial structure.

Author: Diego Bengochea
"""

import numpy as np
import rasterio
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
import time
import argparse
import yaml
import logging
import gc
import warnings
from glob import glob
from datetime import datetime

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='analysis_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class MemoryEfficientSemivariogram:
    def __init__(self, config):
        """
        Memory-efficient semivariogram computation
        
        Args:
            config: Configuration dictionary
        """
        autocorr_config = config['interannual']['spatial_autocorr']
        self.max_points_per_bin = autocorr_config['max_points_per_bin']
        self.n_reference_points = autocorr_config['n_reference_points']
        self.sample_fraction = autocorr_config['sample_fraction']
        
    def load_and_sample_raster(self, raster_path):
        """
        Load raster and create strategic sample.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            tuple: (coordinates, values) as numpy arrays
        """
        logger.info(f"Loading {os.path.basename(raster_path)}...")
        
        with rasterio.open(raster_path) as src:
            # Check raster size first
            height, width = src.height, src.width
            total_pixels = height * width
            logger.info(f"  Raster size: {height} x {width} = {total_pixels:,} pixels")
            
            # Safety check for very large rasters
            if total_pixels > 200_000_000:  # 200M pixels
                logger.warning(f"  Very large raster. Consider reducing sample_fraction.")
            
            data = src.read(1)
            transform = src.transform
            
            # Get valid pixels (not NaN and not nodata)
            if src.nodata is not None:
                valid_mask = ~np.isnan(data) & (data != src.nodata)
            else:
                valid_mask = ~np.isnan(data)
            
            valid_indices = np.where(valid_mask)
            n_valid = len(valid_indices[0])
            
            # Calculate sample size
            n_sample = min(int(n_valid * self.sample_fraction), 100000)  # Cap at 100K
            
            logger.info(f"  Valid pixels: {n_valid:,}, Sampling: {n_sample:,} ({n_sample/n_valid*100:.3f}%)")
            
            # Random sample
            if n_sample < n_valid:
                sample_idx = np.random.choice(n_valid, n_sample, replace=False)
                sample_r = valid_indices[0][sample_idx]
                sample_c = valid_indices[1][sample_idx]
            else:
                sample_r = valid_indices[0]
                sample_c = valid_indices[1]
            
            # Get values and coordinates
            values = data[sample_r, sample_c].astype(np.float32)
            x_coords = (transform[2] + sample_c * transform[0]).astype(np.float32)
            y_coords = (transform[5] + sample_r * transform[4]).astype(np.float32)
            
            coords = np.column_stack([x_coords, y_coords])
            
            # Clean up
            del data, valid_mask, valid_indices
            gc.collect()
            
            return coords, values
    
    def compute_semivariogram_efficient(self, coords, values, distance_bins):
        """
        Compute semivariogram using spatial indexing for memory efficiency.
        
        Args:
            coords: Array of coordinates
            values: Array of values
            distance_bins: Array of distance bin edges
            
        Returns:
            tuple: (bin_centers, semivariances, counts)
        """
        n_points = len(coords)
        logger.info(f"  Computing semivariogram for {n_points:,} points using spatial indexing...")
        
        # Use subset of points as reference points to avoid memory explosion
        n_ref = min(self.n_reference_points, n_points)
        ref_indices = np.random.choice(n_points, n_ref, replace=False)
        ref_coords = coords[ref_indices]
        ref_values = values[ref_indices]
        
        logger.info(f"  Using {n_ref:,} reference points")
        
        # Build spatial index for all points
        nbrs = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
        nbrs.fit(coords)
        
        bin_centers = []
        bin_semivariances = []
        bin_counts = []
        
        # Process each distance bin
        for i in range(len(distance_bins) - 1):
            bin_start = distance_bins[i]
            bin_end = distance_bins[i + 1]
            bin_center = (bin_start + bin_end) / 2
            
            logger.info(f"    Processing bin {i+1}/{len(distance_bins)-1}: {bin_start/1000:.1f}-{bin_end/1000:.1f} km")
            
            semivariances_bin = []
            
            # For each reference point, find neighbors in this distance range
            for j, (ref_coord, ref_val) in enumerate(zip(ref_coords, ref_values)):
                if j % 1000 == 0 and j > 0:
                    logger.info(f"      Processed {j:,}/{n_ref:,} reference points")
                
                # Find all neighbors within the outer radius
                neighbor_indices = nbrs.radius_neighbors([ref_coord], radius=bin_end, return_distance=False)[0]
                
                if len(neighbor_indices) > 1:  # Need at least one neighbor besides itself
                    neighbor_coords = coords[neighbor_indices]
                    neighbor_values = values[neighbor_indices]
                    
                    # Calculate distances to all neighbors
                    distances = np.sqrt(np.sum((neighbor_coords - ref_coord)**2, axis=1))
                    
                    # Filter to current distance bin
                    in_bin_mask = (distances >= bin_start) & (distances < bin_end)
                    
                    if np.sum(in_bin_mask) > 0:
                        bin_neighbor_values = neighbor_values[in_bin_mask]
                        
                        # Calculate semivariances
                        for neighbor_val in bin_neighbor_values:
                            semivar = 0.5 * (ref_val - neighbor_val)**2
                            semivariances_bin.append(semivar)
                        
                        # Limit pairs per bin to control memory
                        if len(semivariances_bin) > self.max_points_per_bin:
                            break
            
            # Store results if we have enough pairs
            if len(semivariances_bin) >= 50:
                bin_centers.append(bin_center)
                bin_semivariances.append(np.mean(semivariances_bin))
                bin_counts.append(len(semivariances_bin))
                logger.info(f"      Bin complete: {len(semivariances_bin):,} pairs, semivariance: {np.mean(semivariances_bin):.4f}")
            else:
                logger.info(f"      Skipping bin (only {len(semivariances_bin)} pairs)")
        
        return np.array(bin_centers), np.array(bin_semivariances), np.array(bin_counts)
    
    def find_autocorr_length(self, distances, semivariances):
        """
        Find autocorrelation length using elbow detection.
        
        Args:
            distances: Array of distances
            semivariances: Array of semivariances
            
        Returns:
            float: Autocorrelation length or NaN if not found
        """
        if len(distances) < 4:
            return np.nan
        
        try:
            # Try to use kneed for elbow detection
            from kneed import KneeLocator
            
            # Smooth the data using exponential model fit
            smooth_dist, smooth_semivar, _ = self.fit_exponential_model(distances, semivariances)
            
            kl = KneeLocator(smooth_dist, smooth_semivar, curve='concave', direction='increasing')
            elbow_distance = kl.elbow
            
            if elbow_distance is None:
                # Fallback: find 95% of maximum semivariance
                max_semivar = np.max(semivariances)
                threshold = 0.95 * max_semivar
                idx = np.where(semivariances >= threshold)[0]
                if len(idx) > 0:
                    elbow_distance = distances[idx[0]]
                else:
                    elbow_distance = np.nan
            
            return elbow_distance
        
        except ImportError:
            logger.warning("kneed package not available, using fallback method")
            # Fallback: find 95% of maximum semivariance
            max_semivar = np.max(semivariances)
            threshold = 0.95 * max_semivar
            idx = np.where(semivariances >= threshold)[0]
            if len(idx) > 0:
                return distances[idx[0]]
            else:
                return np.nan
        
        except Exception as e:
            logger.warning(f"Elbow detection failed: {e}")
            return np.nan

    def fit_exponential_model(self, distances, semivariances):
        """
        Fit exponential model: γ(h) = nugget + sill*(1 - exp(-h/range)).
        
        Args:
            distances: Array of distances
            semivariances: Array of semivariances
            
        Returns:
            tuple: (smooth_distances, smooth_semivariances, parameters)
        """
        try:
            from scipy.optimize import curve_fit
            
            def exp_model(h, nugget, sill, range_param):
                return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))
            
            # Initial estimates
            nugget_init = np.min(semivariances)
            sill_init = np.max(semivariances)  
            range_init = distances[len(distances)//2]  # Middle distance as start
            
            popt, _ = curve_fit(exp_model, distances, semivariances, 
                            p0=[nugget_init, sill_init, range_init],
                            bounds=([0, nugget_init, 1000], [np.inf, np.inf, 200000]))
            
            # Generate smooth curve for elbow detection
            smooth_distances = np.linspace(distances.min(), distances.max(), 100)
            smooth_semivariances = exp_model(smooth_distances, *popt)
            
            return smooth_distances, smooth_semivariances, popt
        except:
            return distances, semivariances, None


def find_interannual_difference_files(config):
    """
    Find interannual difference raster files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        list: List of raster file paths
    """
    base_dir = config['data']['base_dir']
    
    # Look for relative difference files (they have more interesting spatial patterns)
    relative_dir = os.path.join(base_dir, config['interannual']['differences']['output_relative_dir'])
    pattern = os.path.join(relative_dir, "TBD_S2_relative_change_symmetric_*_100m.tif")
    
    files = glob(pattern)
    
    if not files:
        logger.warning(f"No relative difference files found in {relative_dir}")
        # Fallback to raw difference files
        raw_dir = os.path.join(base_dir, config['interannual']['differences']['output_raw_dir'])
        pattern = os.path.join(raw_dir, "TBD_S2_raw_change_*_100m.tif")
        files = glob(pattern)
        
        if not files:
            logger.error(f"No difference files found in either {relative_dir} or {raw_dir}")
            return []
    
    logger.info(f"Found {len(files)} interannual difference files")
    return sorted(files)


def analyze_raster(raster_path, config):
    """
    Analyze single raster with memory management.
    
    Args:
        raster_path: Path to raster file
        config: Configuration dictionary
        
    Returns:
        dict: Analysis results or None if failed
    """
    start_time = time.time()
    
    # Create distance bins from config
    autocorr_config = config['interannual']['spatial_autocorr']
    min_km, max_km, n_bins = autocorr_config['distance_bins_km']
    distance_bins = np.logspace(np.log10(min_km * 1000), np.log10(max_km * 1000), n_bins)
    
    try:
        # Initialize analyzer
        analyzer = MemoryEfficientSemivariogram(config)
        
        # Load and sample
        coords, values = analyzer.load_and_sample_raster(raster_path)
        
        logger.info(f"  Memory usage: {coords.nbytes + values.nbytes} bytes for coordinates and values")
        
        # Compute semivariogram
        distances, semivariances, counts = analyzer.compute_semivariogram_efficient(coords, values, distance_bins)
        
        # Find autocorrelation length
        autocorr_length = analyzer.find_autocorr_length(distances, semivariances)
        
        # Extract period information from filename
        filename = os.path.basename(raster_path)
        period = None
        
        # Try to extract period from relative change filename
        import re
        match = re.search(r'(\d{4})Sep-(\d{4})Aug', filename)
        if match:
            period = f"{match.group(1)}-{match.group(2)}"
        
        result = {
            'file': filename,
            'period': period,
            'n_points': len(values),
            'distances_m': distances,
            'semivariances': semivariances,
            'counts': counts,
            'autocorr_length_m': autocorr_length,
            'autocorr_length_km': autocorr_length / 1000 if not np.isnan(autocorr_length) else np.nan,
            'processing_time': time.time() - start_time
        }
        
        logger.info(f"  Completed: {autocorr_length/1000:.1f} km autocorr length ({time.time() - start_time:.1f}s)")
        
        # Clean up
        del coords, values, distances, semivariances, counts
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"  Error processing {os.path.basename(raster_path)}: {e}")
        return None


def save_autocorr_results(results, config):
    """
    Save autocorrelation analysis results.
    
    Args:
        results: List of analysis results
        config: Configuration dictionary
        
    Returns:
        tuple: (summary_file, detailed_file) - paths to saved files
    """
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary table
    summary_data = []
    all_semivariogram_data = []
    
    for r in results:
        if r is None:
            continue
            
        summary_data.append({
            'file': r['file'],
            'period': r['period'],
            'n_points': r['n_points'],
            'autocorr_length_km': r['autocorr_length_km'],
            'processing_time': r['processing_time']
        })
        
        # Detailed semivariogram data
        for i, (dist, semivar, count) in enumerate(zip(r['distances_m'], r['semivariances'], r['counts'])):
            all_semivariogram_data.append({
                'file': r['file'],
                'period': r['period'],
                'distance_km': dist / 1000,
                'semivariance': semivar,
                'n_pairs': count,
                'bin_number': i
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    if 'period' in summary_df.columns and summary_df['period'].notna().any():
        summary_df = summary_df.sort_values('period')
    
    summary_file = os.path.join(output_dir, f'spatial_autocorr_summary_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Save detailed semivariogram data
    semivar_df = pd.DataFrame(all_semivariogram_data)
    detailed_file = os.path.join(output_dir, f'semivariogram_data_{timestamp}.csv')
    semivar_df.to_csv(detailed_file, index=False)
    
    logger.info(f"Results saved to {output_dir}/")
    return summary_file, detailed_file


def main():
    """
    Main function for spatial autocorrelation analysis.
    """
    parser = argparse.ArgumentParser(description="Calculate spatial autocorrelation of interannual changes")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--input-files', nargs='+', default=None, help='Specific files to process')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    logger.info("Starting spatial autocorrelation analysis...")
    
    # Find raster files
    if args.input_files:
        raster_files = args.input_files
        logger.info(f"Processing specified files: {raster_files}")
    else:
        raster_files = find_interannual_difference_files(config)
    
    if not raster_files:
        logger.error("No raster files found to process. Run calculate_interannual_differences.py first.")
        return
    
    logger.info(f"Found {len(raster_files)} files to process:")
    for f in raster_files:
        logger.info(f"  - {os.path.basename(f)}")
    
    # Process files serially to manage memory
    logger.info("Processing rasters serially for memory management...")
    start_time = time.time()
    
    results = []
    for i, raster_path in enumerate(raster_files):
        logger.info(f"\n--- Processing {i+1}/{len(raster_files)}: {os.path.basename(raster_path)} ---")
        result = analyze_raster(raster_path, config)
        results.append(result)
        
        # Force garbage collection between rasters
        gc.collect()
    
    total_time = time.time() - start_time
    
    # Save results
    summary_file, detailed_file = save_autocorr_results(results, config)
    
    # Print summary
    valid_results = [r for r in results if r is not None]
    lengths = [r['autocorr_length_km'] for r in valid_results if not np.isnan(r['autocorr_length_km'])]
    
    logger.info(f"\n{'='*60}")
    logger.info("SPATIAL AUTOCORRELATION ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Processed: {len(valid_results)}/{len(raster_files)} files")
    
    if lengths:
        logger.info(f"Autocorrelation lengths: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} km")
        logger.info(f"Range: {np.min(lengths):.1f} - {np.max(lengths):.1f} km")
        
        logger.info(f"\nBy period:")
        for r in sorted(valid_results, key=lambda x: x['period'] or ''):
            if r['period'] and not np.isnan(r['autocorr_length_km']):
                logger.info(f"  {r['period']}: {r['autocorr_length_km']:.1f} km")
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"Detailed semivariogram data saved to: {detailed_file}")


if __name__ == "__main__":
    main()
