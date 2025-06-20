"""
Spatial analysis pipeline for autocorrelation and clustering.

Calculates spatial autocorrelation using semivariograms, determines autocorrelation 
lengths, and creates spatial clusters for cross-validation. Validates that cluster 
spatial extent exceeds autocorrelation thresholds to ensure spatial independence.

Author: Diego Bengochea
"""

import numpy as np
import rasterio
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import time
from kneed import KneeLocator
import warnings
import gc
import yaml
import logging
warnings.filterwarnings('ignore')


def setup_logging():
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path="climate_biomass_config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise


class MemoryEfficientSemivariogram:
    """Memory-efficient semivariogram computation for large raster datasets."""
    
    def __init__(self, max_points_per_bin=10000, n_reference_points=10000):
        """
        Initialize semivariogram calculator.
        
        Args:
            max_points_per_bin (int): Maximum point pairs per distance bin 
            n_reference_points (int): Number of reference points to sample from
        """
        self.max_points_per_bin = max_points_per_bin
        self.n_reference_points = n_reference_points
        
    def load_and_sample_raster(self, raster_path, sample_fraction=0.1):
        """
        Load raster and create strategic sample.
        
        Args:
            raster_path (str): Path to raster file
            sample_fraction (float): Fraction of pixels to sample
            
        Returns:
            tuple: (coordinates, values) arrays
        """
        logger = setup_logging()
        logger.info(f"Loading {os.path.basename(raster_path)}...")
        
        with rasterio.open(raster_path) as src:
            # Check raster size first
            height, width = src.height, src.width
            total_pixels = height * width
            logger.info(f"  Raster size: {height} x {width} = {total_pixels:,} pixels")
            
            # Safety check
            if total_pixels > 200_000_000:  # 200M pixels
                logger.warning(f"  Very large raster. Consider reducing sample_fraction.")
            
            data = src.read(1)
            transform = src.transform
            
            # Get valid pixels
            valid_mask = ~np.isnan(data) & (data != src.nodata)
            valid_indices = np.where(valid_mask)
            
            n_valid = len(valid_indices[0])
            n_sample = min(int(n_valid * sample_fraction), 100000)  # Cap at 100K
            
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
        Compute semivariogram using spatial indexing.
        
        Args:
            coords (np.array): Coordinate array
            values (np.array): Value array
            distance_bins (np.array): Distance bin edges
            
        Returns:
            tuple: (bin_centers, bin_semivariances, bin_counts)
        """
        logger = setup_logging()
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
            distances (np.array): Distance array
            semivariances (np.array): Semivariance array
            
        Returns:
            float: Autocorrelation length in meters
        """
        if len(distances) < 4:
            return np.nan
        
        try:
            # Find elbow point
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
        
        except Exception as e:
            logger = setup_logging()
            logger.warning(f"    Elbow detection failed: {e}")
            return np.nan

    def fit_exponential_model(self, distances, semivariances):
        """
        Fit exponential model: γ(h) = nugget + sill*(1 - exp(-h/range)).
        
        Args:
            distances (np.array): Distance array
            semivariances (np.array): Semivariance array
            
        Returns:
            tuple: (smooth_distances, smooth_semivariances, parameters)
        """
        from scipy.optimize import curve_fit
    
        def exp_model(h, nugget, sill, range_param):
            return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))
        
        try:
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


def analyze_raster(raster_path, config):
    """
    Analyze single raster with memory management.
    
    Args:
        raster_path (str): Path to raster file
        config (dict): Configuration dictionary
        
    Returns:
        dict: Analysis results
    """
    logger = setup_logging()
    start_time = time.time()
    
    spatial_config = config['spatial']['autocorr']
    distance_bins = np.logspace(
        np.log10(spatial_config['distance_bins']['start']), 
        np.log10(spatial_config['distance_bins']['end']), 
        spatial_config['distance_bins']['n_bins']
    )
    
    try:
        # Initialize analyzer
        analyzer = MemoryEfficientSemivariogram(
            max_points_per_bin=spatial_config['max_points_per_bin'], 
            n_reference_points=spatial_config['n_reference_points']
        )
        
        # Load and sample
        coords, values = analyzer.load_and_sample_raster(raster_path, spatial_config['sample_fraction'])
        
        logger.info(f"  Memory usage check: {coords.nbytes + values.nbytes} bytes for coordinates and values")
        
        # Compute semivariogram
        distances, semivariances, counts = analyzer.compute_semivariogram_efficient(coords, values, distance_bins)
        
        # Find autocorrelation length
        autocorr_length = analyzer.find_autocorr_length(distances, semivariances)
        
        # Extract year from filename pattern: TBD_S2_mean_YYYY_100m_TBD_merged.tif
        filename = os.path.basename(raster_path)
        year = None
        parts = filename.split('_')
        if len(parts) >= 4 and parts[1] == 'S2' and parts[2] == 'mean':
            if parts[3].isdigit() and len(parts[3]) == 4:
                year = int(parts[3])
        
        result = {
            'file': filename,
            'year': year,
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
        logger.error(f"  Error: {e}")
        return None


def calculate_cluster_radii(coords, labels, k):
    """
    Calculate 90th percentile radius for each cluster.
    
    Args:
        coords (np.array): Coordinate array (x, y)
        labels (np.array): Cluster labels
        k (int): Number of clusters
        
    Returns:
        tuple: (cluster_radii, mean_radius, max_radius)
    """
    logger = setup_logging()
    cluster_radii = []
    
    for cluster_id in range(k):
        # Get all points in this cluster
        cluster_points = coords[labels == cluster_id]
        
        if len(cluster_points) == 0:
            cluster_radii.append(0)
            continue
            
        # Calculate cluster centroid
        cluster_center = np.mean(cluster_points, axis=0)
        
        # Calculate distances from centroid to all points
        distances = np.sqrt(np.sum((cluster_points - cluster_center)**2, axis=1))
        
        # Radius = 90th percentile distance (robust measure of cluster extent)
        cluster_radius = np.percentile(distances, 90) if len(distances) > 0 else 0
        cluster_radii.append(cluster_radius)
        
        logger.info(f"  Cluster {cluster_id}: {len(cluster_points)} points, 90th percentile radius: {cluster_radius:.1f}m")
    
    cluster_radii = np.array(cluster_radii)
    mean_radius = np.mean(cluster_radii)
    max_radius = np.max(cluster_radii)
    
    return cluster_radii, mean_radius, max_radius


def create_spatial_clusters(df, k, autocorr_lengths):
    """
    Create spatial clusters and validate spatial independence.
    
    Args:
        df (pd.DataFrame): Dataset with x, y coordinates
        k (int): Number of clusters to create
        autocorr_lengths (list): List of autocorrelation lengths across years
        
    Returns:
        pd.DataFrame: Dataset with cluster_id column added
    """
    logger = setup_logging()
    logger.info(f"Creating {k} spatial clusters...")
    
    # Get spatial coordinates
    coords = df[['x', 'y']].values
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    # Calculate cluster radii
    cluster_radii, mean_radius, max_radius = calculate_cluster_radii(coords, cluster_labels, k)
    
    # Validate spatial independence
    valid_autocorr_lengths = [length for length in autocorr_lengths if not np.isnan(length)]
    if valid_autocorr_lengths:
        median_autocorr = np.median(valid_autocorr_lengths)
        autocorr_threshold = 2 * median_autocorr
        
        logger.info(f"Spatial independence validation:")
        logger.info(f"  Median autocorrelation length: {median_autocorr:.1f}m")
        logger.info(f"  2× autocorr threshold: {autocorr_threshold:.1f}m")
        logger.info(f"  Mean cluster 90th percentile radius: {mean_radius:.1f}m")
        logger.info(f"  Max cluster 90th percentile radius: {max_radius:.1f}m")
        
        if mean_radius > autocorr_threshold:
            logger.info(f"✅ Spatial independence: Mean radius ({mean_radius:.1f}m) > 2×autocorr ({autocorr_threshold:.1f}m)")
        else:
            logger.warning(f"⚠️  Consider fewer clusters: Mean radius ({mean_radius:.1f}m) < 2×autocorr ({autocorr_threshold:.1f}m)")
            
        if max_radius > autocorr_threshold:
            logger.info(f"✅ All clusters meet independence: Max radius ({max_radius:.1f}m) > 2×autocorr ({autocorr_threshold:.1f}m)")
        else:
            logger.warning(f"⚠️  Some clusters may have spatial leakage: Max radius ({max_radius:.1f}m) < 2×autocorr ({autocorr_threshold:.1f}m)")
    else:
        logger.warning("No valid autocorrelation lengths found for validation")
    
    # Add cluster_id to dataframe
    df['cluster_id'] = cluster_labels
    
    logger.info(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    return df


def save_results(results, output_dir='autocorr_results'):
    """
    Save all results.
    
    Args:
        results (list): List of analysis results
        output_dir (str): Output directory path
        
    Returns:
        tuple: (summary_df, semivar_df)
    """
    logger = setup_logging()
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary table
    summary_data = []
    all_semivariogram_data = []
    
    for r in results:
        if r is None:
            continue
            
        summary_data.append({
            'file': r['file'],
            'year': r['year'],
            'n_points': r['n_points'],
            'autocorr_length_km': r['autocorr_length_km'],
            'processing_time': r['processing_time']
        })
        
        # Detailed semivariogram data
        for i, (dist, semivar, count) in enumerate(zip(r['distances_m'], r['semivariances'], r['counts'])):
            all_semivariogram_data.append({
                'file': r['file'],
                'year': r['year'],
                'distance_km': dist / 1000,
                'semivariance': semivar,
                'n_pairs': count,
                'bin_number': i
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    if 'year' in summary_df.columns and summary_df['year'].notna().any():
        summary_df = summary_df.sort_values('year')
    summary_df.to_csv(os.path.join(output_dir, 'autocorr_summary.csv'), index=False)
    
    # Save detailed semivariogram data
    semivar_df = pd.DataFrame(all_semivariogram_data)
    semivar_df.to_csv(os.path.join(output_dir, 'semivariogram_data.csv'), index=False)
    
    logger.info(f"Results saved to {output_dir}/")
    return summary_df, semivar_df


def plot_results(results, output_dir='autocorr_results'):
    """
    Create plots.
    
    Args:
        results (list): Analysis results
        output_dir (str): Output directory
    """
    logger = setup_logging()
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return
    
    # 1. Plot all semivariograms
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for r in valid_results:
        distances_km = r['distances_m'] / 1000
        label = f"{r['year']}" if r['year'] else r['file']
        ax.plot(distances_km, r['semivariances'], 'o-', label=label, alpha=0.7, linewidth=2, markersize=6)
        
        # Mark autocorrelation length
        if not np.isnan(r['autocorr_length_km']):
            ax.axvline(r['autocorr_length_km'], linestyle='--', alpha=0.5, 
                      color=ax.lines[-1].get_color(), linewidth=1)
            ax.text(r['autocorr_length_km'], ax.get_ylim()[1]*0.9, 
                   f"{r['autocorr_length_km']:.0f}km", 
                   rotation=90, fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Semivariance', fontsize=12)
    ax.set_title('Semivariograms by Year\n(Dashed lines show autocorrelation lengths)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semivariograms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time series of autocorrelation length
    years = [r['year'] for r in valid_results if r['year']]
    lengths = [r['autocorr_length_km'] for r in valid_results if r['year']]
    
    if len(years) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, lengths, 'o-', linewidth=3, markersize=10, color='darkblue')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Autocorrelation Length (km)', fontsize=12)
        ax.set_title('Spatial Autocorrelation Length Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(years, lengths):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'autocorr_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Plots saved successfully")


def run_spatial_analysis_pipeline(config):
    """
    Execute the complete spatial analysis workflow.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Dataset with spatial clusters
    """
    logger = setup_logging()
    logger.info("Starting spatial analysis pipeline...")
    
    # Look for rasters in specific directory with specific pattern
    raster_dir = config['paths']['biomass_diff_dir']
    
    if not os.path.exists(raster_dir):
        logger.error(f"Directory {raster_dir} not found!")
        return None
    
    # Find files matching pattern: TBD_S2_*
    all_files = os.listdir(raster_dir)
    raster_files = []
    
    for f in all_files:
        if f.endswith('.tif'):
            parts = f.split('_')
            # Check if pattern matches
            if (len(parts) >= 7 and 
                parts[0] == 'TBD' and
                parts[1] == 'S2' and 
                f.endswith('.tif')):
                raster_files.append(f)
    
    if not raster_files:
        logger.warning(f"No files matching pattern 'TBD_S2_*' found in {raster_dir}")
        logger.info("Available files:")
        for f in all_files:
            if f.endswith('.tif'):
                logger.info(f"  - {f}")
        return None
    
    # Create full paths
    raster_paths = [os.path.join(raster_dir, f) for f in raster_files]
    
    logger.info(f"Found {len(raster_files)} matching raster files:")
    for f in sorted(raster_files):
        logger.info(f"  - {f}")
    
    # Process rasters serially to manage memory
    logger.info(f"Processing rasters serially to manage memory...")
    start_time = time.time()
    
    results = []
    for i, raster_path in enumerate(raster_paths):
        logger.info(f"\n--- Processing {i+1}/{len(raster_paths)}: {os.path.basename(raster_path)} ---")
        result = analyze_raster(raster_path, config)
        results.append(result)
        
        # Force garbage collection between rasters
        gc.collect()
    
    total_time = time.time() - start_time
    
    # Save results
    summary_df, semivar_df = save_results(results, output_dir='autocorr_results')
    
    # Create plots
    plot_results(results, output_dir='autocorr_results')
    
    # Print summary
    valid_results = [r for r in results if r is not None]
    lengths = [r['autocorr_length_km'] for r in valid_results if not np.isnan(r['autocorr_length_km'])]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"AUTOCORRELATION ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Processed: {len(valid_results)}/{len(raster_paths)} files")
    
    if lengths:
        logger.info(f"Autocorrelation lengths: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} km")
        logger.info(f"Range: {np.min(lengths):.1f} - {np.max(lengths):.1f} km")
        
        logger.info(f"By year:")
        for r in sorted(valid_results, key=lambda x: x['year'] or 0):
            if r['year'] and not np.isnan(r['autocorr_length_km']):
                logger.info(f"  {r['year']}: {r['autocorr_length_km']:.1f} km")
    
    # Now create spatial clusters using the ML dataset
    training_dataset_path = config['paths']['training_dataset']
    
    if not os.path.exists(training_dataset_path):
        logger.error(f"Training dataset not found: {training_dataset_path}")
        logger.error("Please run biomass integration pipeline first")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SPATIAL CLUSTERING")
    logger.info(f"{'='*60}")
    
    # Load ML dataset
    logger.info(f"Loading ML dataset from {training_dataset_path}")
    df = pd.read_csv(training_dataset_path)
    logger.info(f"Loaded {len(df)} data points with columns: {', '.join(df.columns)}")
    
    # Create spatial clusters
    k = config['spatial']['clustering']['default_k']
    autocorr_lengths_m = [r['autocorr_length_m'] for r in valid_results if not np.isnan(r['autocorr_length_m'])]
    
    df_clustered = create_spatial_clusters(df, k, autocorr_lengths_m)
    
    # Save clustered dataset
    clustered_dataset_path = config['paths']['clustered_dataset']
    df_clustered.to_csv(clustered_dataset_path, index=False)
    logger.info(f"Saved clustered dataset to {clustered_dataset_path}")
    
    logger.info("Spatial analysis pipeline completed successfully!")
    
    return df_clustered


def main():
    """Main function to run spatial analysis."""
    logger = setup_logging()
    logger.info("Starting spatial analysis pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Run spatial analysis pipeline
        results = run_spatial_analysis_pipeline(config)
        
        if results is not None:
            logger.info("Spatial analysis completed successfully!")
        else:
            logger.error("Spatial analysis failed!")
            
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
