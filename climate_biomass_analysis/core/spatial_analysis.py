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
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pickle

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory

warnings.filterwarnings('ignore')


class MemoryEfficientSemivariogram:
    """Memory-efficient semivariogram computation for large raster datasets."""
    
    def __init__(self, max_points_per_bin: int = 10000, n_reference_points: int = 10000):
        """
        Initialize semivariogram calculator.
        
        Args:
            max_points_per_bin: Maximum point pairs per distance bin 
            n_reference_points: Number of reference points to sample from
        """
        self.max_points_per_bin = max_points_per_bin
        self.n_reference_points = n_reference_points
        
    def load_and_sample_raster(self, raster_path: Union[str, Path], sample_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raster and create strategic sample.
        
        Args:
            raster_path: Path to raster file
            sample_fraction: Fraction of pixels to sample
            
        Returns:
            Tuple of (coordinates, values) arrays
        """
        logger = setup_logging()
        logger.info(f"Loading {Path(raster_path).name}...")
        
        with rasterio.open(raster_path) as src:
            # Check raster size first
            height, width = src.height, src.width
            total_pixels = height * width
            logger.info(f"  Raster size: {height} x {width} = {total_pixels:,} pixels")
            
            # Safety check
            if total_pixels > 200_000_000:  # 200M pixels
                logger.warning(f"  Very large raster. Consider reducing sample_fraction.")
            
            # Read data
            data = src.read(1)
            transform = src.transform
            
            # Create coordinate grids
            rows, cols = np.mgrid[0:height, 0:width]
            
            # Convert to geographic coordinates
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            coords = np.column_stack([np.array(xs).flatten(), np.array(ys).flatten()])
            values = data.flatten()
            
            # Filter valid data
            valid_mask = np.isfinite(values) & (values != src.nodata)
            coords = coords[valid_mask]
            values = values[valid_mask]
            
            logger.info(f"  Valid pixels: {len(values):,}")
            
            # Sample data
            n_sample = int(len(values) * sample_fraction)
            if n_sample > len(values):
                n_sample = len(values)
            
            if n_sample < len(values):
                indices = np.random.choice(len(values), n_sample, replace=False)
                coords = coords[indices]
                values = values[indices]
                logger.info(f"  Sampled to: {len(values):,} pixels")
            
            return coords, values
    
    def calculate_distances(self, coords: np.ndarray, max_distance_km: float = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate pairwise distances and semi-variances efficiently.
        
        Args:
            coords: Coordinate array (N, 2)
            max_distance_km: Maximum distance in kilometers
            
        Returns:
            Tuple of (distances, semivariances)
        """
        logger = setup_logging()
        n_points = len(coords)
        
        # Sample reference points if dataset is large
        if n_points > self.n_reference_points:
            ref_indices = np.random.choice(n_points, self.n_reference_points, replace=False)
            ref_coords = coords[ref_indices]
            logger.info(f"Using {len(ref_indices)} reference points")
        else:
            ref_coords = coords
            ref_indices = np.arange(n_points)
        
        # Convert max distance to meters
        max_distance_m = max_distance_km * 1000
        
        # Use NearestNeighbors for efficient distance computation
        nbrs = NearestNeighbors(radius=max_distance_m, algorithm='ball_tree').fit(coords)
        
        distances = []
        semivariances = []
        
        for i, ref_coord in enumerate(ref_coords):
            if i % 1000 == 0:
                logger.info(f"  Processing reference point {i+1}/{len(ref_coords)}")
            
            # Find neighbors within max distance
            neighbor_indices = nbrs.radius_neighbors([ref_coord], return_distance=False)[0]
            
            if len(neighbor_indices) > 1:  # Exclude self
                # Calculate distances and semivariances
                neighbor_coords = coords[neighbor_indices]
                point_distances = np.linalg.norm(neighbor_coords - ref_coord, axis=1)
                
                # Remove self-distance (should be 0)
                non_zero_mask = point_distances > 0
                point_distances = point_distances[non_zero_mask]
                neighbor_indices = neighbor_indices[non_zero_mask]
                
                if len(point_distances) > 0:
                    distances.extend(point_distances)
                    
                    # For semivariance, we need values - this is simplified version
                    # In practice, you'd calculate 0.5 * (value[i] - value[j])^2
                    # Here we'll use distance as a proxy for demonstration
                    semivariances.extend(point_distances)  # Placeholder
        
        return np.array(distances), np.array(semivariances)
    
    def bin_semivariogram(
        self, 
        distances: np.ndarray, 
        semivariances: np.ndarray, 
        n_bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin distances and calculate average semivariances.
        
        Args:
            distances: Distance array
            semivariances: Semivariance array
            n_bins: Number of distance bins
            
        Returns:
            Tuple of (bin_centers, mean_semivariances, bin_counts)
        """
        # Create distance bins
        max_distance = np.max(distances)
        bin_edges = np.linspace(0, max_distance, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign points to bins
        bin_indices = np.digitize(distances, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        mean_semivariances = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            
            if np.any(bin_mask):
                bin_semivariances = semivariances[bin_mask]
                
                # Limit points per bin for memory efficiency
                if len(bin_semivariances) > self.max_points_per_bin:
                    sampled_indices = np.random.choice(
                        len(bin_semivariances), 
                        self.max_points_per_bin, 
                        replace=False
                    )
                    bin_semivariances = bin_semivariances[sampled_indices]
                
                mean_semivariances.append(np.mean(bin_semivariances))
                bin_counts.append(len(bin_semivariances))
            else:
                mean_semivariances.append(np.nan)
                bin_counts.append(0)
        
        return bin_centers, np.array(mean_semivariances), np.array(bin_counts)


class SpatialAnalyzer:
    """
    Spatial analysis pipeline for autocorrelation and clustering.
    
    This class handles spatial autocorrelation analysis using semivariograms
    and creates spatial clusters for cross-validation in machine learning.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the spatial analyzer.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='spatial_analysis',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration
        self.spatial_config = self.config['spatial_analysis']
        
        # Initialize semivariogram calculator
        self.semivariogram = MemoryEfficientSemivariogram(
            max_points_per_bin=self.spatial_config['max_points_per_bin'],
            n_reference_points=self.spatial_config['n_reference_points']
        )
        
        self.logger.info("Initialized SpatialAnalyzer")
    
    def analyze_spatial_autocorrelation(
        self, 
        raster_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Analyze spatial autocorrelation using semivariogram analysis.
        
        Args:
            raster_path: Path to raster file for analysis
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary with autocorrelation metrics
        """
        self.logger.info(f"Analyzing spatial autocorrelation for: {Path(raster_path).name}")
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Load and sample raster data
        sample_fraction = self.spatial_config['sample_fraction']
        coords, values = self.semivariogram.load_and_sample_raster(raster_path, sample_fraction)
        
        if len(coords) < 100:
            self.logger.warning("Not enough valid data points for semivariogram analysis")
            return {}
        
        # Calculate semivariogram
        self.logger.info("Calculating semivariogram...")
        start_time = time.time()
        
        # For real semivariogram, we need to calculate actual semivariances
        distances, semivariances = self._calculate_true_semivariogram(coords, values)
        
        # Bin the semivariogram
        n_bins = self.spatial_config['n_distance_bins']
        bin_centers, mean_semivariances, bin_counts = self.semivariogram.bin_semivariogram(
            distances, semivariances, n_bins
        )
        
        # Convert distances to kilometers
        bin_centers_km = bin_centers / 1000
        
        # Find autocorrelation length (range)
        autocorr_range_km = self._estimate_autocorrelation_range(bin_centers_km, mean_semivariances)
        
        # Calculate other metrics
        max_distance_km = np.max(bin_centers_km)
        sill = np.nanmax(mean_semivariances)
        nugget = mean_semivariances[0] if not np.isnan(mean_semivariances[0]) else 0
        
        results = {
            'autocorrelation_range_km': autocorr_range_km,
            'max_distance_km': max_distance_km,
            'sill': sill,
            'nugget': nugget,
            'n_points_analyzed': len(values),
            'n_distance_pairs': len(distances)
        }
        
        self.logger.info(f"Semivariogram calculation completed in {(time.time() - start_time)/60:.2f} minutes")
        self.logger.info(f"Estimated autocorrelation range: {autocorr_range_km:.2f} km")
        
        # Save semivariogram plot
        self._plot_semivariogram(bin_centers_km, mean_semivariances, bin_counts, 
                                output_dir, autocorr_range_km)
        
        # Save results
        results_file = Path(output_dir) / "spatial_autocorrelation_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def _calculate_true_semivariogram(
        self, 
        coords: np.ndarray, 
        values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate true semivariogram values.
        
        Args:
            coords: Coordinate array (N, 2)
            values: Value array (N,)
            
        Returns:
            Tuple of (distances, semivariances)
        """
        # Limit number of points for memory efficiency
        max_points = min(len(coords), self.spatial_config['n_reference_points'])
        
        if len(coords) > max_points:
            indices = np.random.choice(len(coords), max_points, replace=False)
            coords = coords[indices]
            values = values[indices]
        
        distances = []
        semivariances = []
        
        # Calculate pairwise distances and semivariances
        for i in range(len(coords)):
            if i % 1000 == 0 and i > 0:
                self.logger.debug(f"  Processed {i}/{len(coords)} points")
            
            # Calculate distances to all other points
            point_distances = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
            
            # Calculate semivariances: 0.5 * (value[i] - value[j])^2
            point_semivariances = 0.5 * (values[i+1:] - values[i])**2
            
            distances.extend(point_distances)
            semivariances.extend(point_semivariances)
        
        return np.array(distances), np.array(semivariances)
    
    def _estimate_autocorrelation_range(
        self, 
        distances_km: np.ndarray, 
        semivariances: np.ndarray
    ) -> float:
        """
        Estimate autocorrelation range from semivariogram.
        
        Args:
            distances_km: Distance array in kilometers
            semivariances: Semivariance array
            
        Returns:
            Estimated autocorrelation range in kilometers
        """
        # Remove NaN values
        valid_mask = ~np.isnan(semivariances)
        distances_clean = distances_km[valid_mask]
        semivariances_clean = semivariances[valid_mask]
        
        if len(distances_clean) < 3:
            self.logger.warning("Not enough valid points for range estimation")
            return 0.0
        
        try:
            # Find the range as 95% of the sill
            sill = np.max(semivariances_clean)
            target_semivariance = 0.95 * sill
            
            # Find distance where semivariance reaches 95% of sill
            above_target = semivariances_clean >= target_semivariance
            
            if np.any(above_target):
                range_km = distances_clean[above_target][0]
            else:
                # Use knee detection as fallback
                try:
                    knee_locator = KneeLocator(
                        distances_clean, semivariances_clean, 
                        curve="concave", direction="increasing"
                    )
                    range_km = knee_locator.knee if knee_locator.knee else distances_clean[-1] / 2
                except:
                    range_km = distances_clean[-1] / 2
            
            return float(range_km)
            
        except Exception as e:
            self.logger.warning(f"Error estimating autocorrelation range: {e}")
            return 0.0
    
    def _plot_semivariogram(
        self, 
        distances_km: np.ndarray, 
        semivariances: np.ndarray,
        counts: np.ndarray,
        output_dir: Union[str, Path],
        range_km: float
    ) -> None:
        """
        Create semivariogram plot.
        
        Args:
            distances_km: Distance array in kilometers
            semivariances: Semivariance array
            counts: Point counts per bin
            output_dir: Output directory
            range_km: Estimated autocorrelation range
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot semivariogram
            valid_mask = ~np.isnan(semivariances) & (counts > 10)
            
            ax1.plot(distances_km[valid_mask], semivariances[valid_mask], 'bo-', alpha=0.7)
            ax1.axvline(range_km, color='red', linestyle='--', 
                       label=f'Range: {range_km:.1f} km')
            ax1.set_xlabel('Distance (km)')
            ax1.set_ylabel('Semivariance')
            ax1.set_title('Empirical Semivariogram')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot point counts
            ax2.bar(distances_km, counts, width=distances_km[1]-distances_km[0], alpha=0.7)
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Number of Point Pairs')
            ax2.set_title('Point Pairs per Distance Bin')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = Path(output_dir) / "semivariogram.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Semivariogram plot saved: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"Error creating semivariogram plot: {e}")
    
    def create_spatial_clusters(
        self, 
        dataset_path: Union[str, Path],
        output_path: Union[str, Path],
        autocorr_range_km: float
    ) -> pd.DataFrame:
        """
        Create spatial clusters for cross-validation.
        
        Args:
            dataset_path: Path to training dataset CSV
            output_path: Path to save clustered dataset
            autocorr_range_km: Autocorrelation range in kilometers
            
        Returns:
            DataFrame with cluster assignments
        """
        self.logger.info("Creating spatial clusters...")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("Dataset must contain 'x' and 'y' coordinate columns")
        
        # Extract coordinates
        coords = df[['x', 'y']].values
        
        # Determine optimal number of clusters
        clustering_config = self.spatial_config['clustering']
        n_clusters_range = clustering_config['n_clusters_range']
        min_cluster_size = clustering_config['min_cluster_size']
        
        # Ensure minimum cluster size constraint
        max_clusters = len(df) // min_cluster_size
        n_clusters_max = min(n_clusters_range[1], max_clusters)
        n_clusters_min = max(n_clusters_range[0], 2)
        
        if n_clusters_max < n_clusters_min:
            n_clusters = n_clusters_min
            self.logger.warning(f"Adjusted cluster count to {n_clusters} due to size constraints")
        else:
            n_clusters = self._find_optimal_clusters(coords, n_clusters_min, n_clusters_max)
        
        # Create clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Add cluster assignments to dataset
        df['cluster_id'] = cluster_labels
        
        # Validate cluster spatial separation
        cluster_stats = self._validate_cluster_separation(df, autocorr_range_km)
        
        # Save clustered dataset
        ensure_directory(Path(output_path).parent)
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Created {n_clusters} spatial clusters")
        self.logger.info(f"Mean cluster separation: {cluster_stats['mean_separation_km']:.2f} km")
        self.logger.info(f"Clustered dataset saved: {output_path}")
        
        return df
    
    def _find_optimal_clusters(
        self, 
        coords: np.ndarray, 
        min_clusters: int, 
        max_clusters: int
    ) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            coords: Coordinate array
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            
        Returns:
            Optimal number of clusters
        """
        if max_clusters <= min_clusters:
            return min_clusters
        
        cluster_range = range(min_clusters, min(max_clusters + 1, len(coords) // 10))
        inertias = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)
        
        # Use knee detection to find optimal number
        try:
            knee_locator = KneeLocator(
                list(cluster_range), inertias, 
                curve="convex", direction="decreasing"
            )
            optimal_clusters = knee_locator.knee if knee_locator.knee else min_clusters + 2
        except:
            optimal_clusters = min_clusters + 2
        
        return optimal_clusters
    
    def _validate_cluster_separation(
        self, 
        df: pd.DataFrame, 
        autocorr_range_km: float
    ) -> Dict[str, float]:
        """
        Validate that clusters are spatially separated beyond autocorrelation range.
        
        Args:
            df: DataFrame with cluster assignments
            autocorr_range_km: Autocorrelation range in kilometers
            
        Returns:
            Dictionary with cluster separation statistics
        """
        cluster_centers = df.groupby('cluster_id')[['x', 'y']].mean()
        
        # Calculate pairwise distances between cluster centers
        separations = []
        for i, (_, center1) in enumerate(cluster_centers.iterrows()):
            for j, (_, center2) in enumerate(cluster_centers.iterrows()):
                if i < j:
                    distance = np.linalg.norm([center1['x'] - center2['x'], 
                                             center1['y'] - center2['y']])
                    separations.append(distance / 1000)  # Convert to km
        
        if separations:
            mean_separation = np.mean(separations)
            min_separation = np.min(separations)
            
            # Check if clusters are well separated
            well_separated = min_separation > autocorr_range_km
            
            stats = {
                'mean_separation_km': mean_separation,
                'min_separation_km': min_separation,
                'autocorr_range_km': autocorr_range_km,
                'well_separated': well_separated
            }
            
            if not well_separated:
                self.logger.warning(
                    f"Minimum cluster separation ({min_separation:.2f} km) is less than "
                    f"autocorrelation range ({autocorr_range_km:.2f} km)"
                )
            
            return stats
        
        return {'mean_separation_km': 0, 'min_separation_km': 0, 
                'autocorr_range_km': autocorr_range_km, 'well_separated': False}
    
    def find_interannual_difference_files(self) -> List[str]:
        """
        Find interannual difference raster files.
        
        Returns:
            List of raster file paths
        """
        import glob
        
        base_dir = self.config['data'].get('biomass_diff_dir', 'biomass_differences')
        
        # Look for relative difference files (they have more interesting spatial patterns)
        pattern = os.path.join(base_dir, "*_rel_change_*.tif")
        files = glob.glob(pattern)
        
        if not files:
            self.logger.warning(f"No relative difference files found in {base_dir}")
            # Fallback to any TIF files
            pattern = os.path.join(base_dir, "*.tif")
            files = glob.glob(pattern)
        
        if not files:
            self.logger.error(f"No raster files found in {base_dir}")
            return []
        
        self.logger.info(f"Found {len(files)} interannual difference files")
        return sorted(files)
    
    def fit_exponential_model(self, distances: np.ndarray, semivariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fit exponential model to semivariogram.
        
        Args:
            distances: Array of distances
            semivariances: Array of semivariances
            
        Returns:
            Tuple of (smooth_distances, smooth_semivariances, parameters)
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
        except Exception as e:
            self.logger.warning(f"Exponential model fitting failed: {e}")
            return distances, semivariances, None
    
    def analyze_raster(self, raster_path: Union[str, Path]) -> Optional[Dict[str, any]]:
        """
        Analyze single raster with memory management.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        import re
        
        start_time = time.time()
        self.logger.info(f"Analyzing: {Path(raster_path).name}")
        
        try:
            # Extract year from filename if possible
            filename = Path(raster_path).name
            year_match = re.search(r'(\d{4})', filename)
            year = int(year_match.group(1)) if year_match else None
            
            # Load and sample raster data
            coords, values = self.semivariogram.load_and_sample_raster(
                raster_path, self.spatial_config['sample_fraction']
            )
            
            if len(coords) < 100:
                self.logger.warning(f"Not enough valid data points ({len(coords)}) for {filename}")
                return None
            
            # Calculate true semivariogram
            distances, semivariances = self._calculate_true_semivariogram(coords, values)
            
            # Bin the semivariogram
            n_bins = self.spatial_config['n_distance_bins']
            bin_centers, mean_semivariances, bin_counts = self.semivariogram.bin_semivariogram(
                distances, semivariances, n_bins
            )
            
            # Convert to kilometers for analysis
            bin_centers_km = bin_centers / 1000
            
            # Estimate autocorrelation range
            autocorr_range_km = self._estimate_autocorrelation_range(bin_centers_km, mean_semivariances)
            
            processing_time = time.time() - start_time
            
            result = {
                'file': filename,
                'year': year,
                'n_points': len(values),
                'distances_m': bin_centers,
                'semivariances': mean_semivariances,
                'counts': bin_counts,
                'autocorr_length_km': autocorr_range_km,
                'autocorr_length_m': autocorr_range_km * 1000,
                'processing_time': processing_time
            }
            
            self.logger.info(f"  Processed {len(values):,} points, autocorr: {autocorr_range_km:.1f} km")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {raster_path}: {e}")
            return None
    
    def save_results(self, results: List[Dict], output_dir: Union[str, Path]) -> Tuple[str, str]:
        """
        Save analysis results to files.
        
        Args:
            results: List of analysis results
            output_dir: Output directory
            
        Returns:
            Tuple of (summary_file, detailed_file) paths
        """
        from datetime import datetime
        
        ensure_directory(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        summary_file = Path(output_dir) / f'spatial_autocorr_summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed semivariogram data
        semivar_df = pd.DataFrame(all_semivariogram_data)
        detailed_file = Path(output_dir) / f'semivariogram_data_{timestamp}.csv'
        semivar_df.to_csv(detailed_file, index=False)
        
        self.logger.info(f"Results saved to {output_dir}/")
        return str(summary_file), str(detailed_file)
    
    def plot_results(self, results: List[Dict], output_dir: Union[str, Path]) -> None:
        """
        Create visualization plots for analysis results.
        
        Args:
            results: Analysis results
            output_dir: Output directory
        """
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            ensure_directory(output_dir)
            
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
            plt.savefig(Path(output_dir) / 'semivariograms.png', dpi=300, bbox_inches='tight')
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
                plt.savefig(Path(output_dir) / 'autocorr_timeseries.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("Plots saved successfully")
            
        except Exception as e:
            self.logger.warning(f"Error creating plots: {e}")
    
    def calculate_cluster_radii(self, coords: np.ndarray, labels: np.ndarray, k: int) -> Tuple[np.ndarray, float, float]:
        """
        Calculate spatial extent of each cluster.
        
        Args:
            coords: Coordinate array (x, y)
            labels: Cluster labels
            k: Number of clusters
            
        Returns:
            Tuple of (cluster_radii, mean_radius, max_radius)
        """
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
            
            self.logger.debug(f"  Cluster {cluster_id}: {len(cluster_points)} points, 90th percentile radius: {cluster_radius:.1f}m")
        
        cluster_radii = np.array(cluster_radii)
        mean_radius = np.mean(cluster_radii)
        max_radius = np.max(cluster_radii)
        
        return cluster_radii, mean_radius, max_radius
    
    def create_spatial_clusters_advanced(
        self, 
        df: pd.DataFrame, 
        k: int, 
        autocorr_lengths: List[float]
    ) -> pd.DataFrame:
        """
        Create spatial clusters and validate spatial independence.
        
        Args:
            df: Dataset with x, y coordinates
            k: Number of clusters
            autocorr_lengths: List of autocorrelation lengths in meters
            
        Returns:
            DataFrame with cluster_id column added
        """
        self.logger.info(f"Creating {k} spatial clusters...")
        
        # Extract coordinates
        coords = df[['x', 'y']].values
        
        # Create clusters using KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Add cluster assignments to dataframe
        df_clustered = df.copy()
        df_clustered['cluster_id'] = cluster_labels
        
        # Analyze cluster spatial properties
        cluster_radii, mean_radius, max_radius = self.calculate_cluster_radii(coords, cluster_labels, k)
        
        # Calculate cluster separation
        cluster_centers = kmeans.cluster_centers_
        separations = []
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                separations.append(distance)
        
        min_separation = np.min(separations) if separations else 0
        mean_separation = np.mean(separations) if separations else 0
        
        # Validate against autocorrelation lengths
        mean_autocorr_m = np.mean(autocorr_lengths) if autocorr_lengths else 50000  # Default 50km
        
        self.logger.info(f"Cluster analysis:")
        self.logger.info(f"  Mean cluster radius: {mean_radius:.0f} m")
        self.logger.info(f"  Max cluster radius: {max_radius:.0f} m")
        self.logger.info(f"  Min cluster separation: {min_separation:.0f} m")
        self.logger.info(f"  Mean cluster separation: {mean_separation:.0f} m")
        self.logger.info(f"  Mean autocorrelation length: {mean_autocorr_m:.0f} m")
        
        # Check if clusters are well separated
        well_separated = min_separation > mean_autocorr_m
        if well_separated:
            self.logger.info("✅ Clusters are well separated (min separation > autocorrelation length)")
        else:
            self.logger.warning("⚠️ Clusters may not be sufficiently separated for spatial independence")
        
        return df_clustered
    
    def run_spatial_analysis_pipeline(self) -> Optional[pd.DataFrame]:
        """
        Execute the complete spatial analysis workflow.
        
        Returns:
            DataFrame with spatial clusters or None if failed
        """
        self.logger.info("Starting spatial analysis pipeline...")
        
        # Step 1: Find biomass difference raster files
        raster_files = self.find_interannual_difference_files()
        
        if not raster_files:
            self.logger.error("No raster files found for spatial analysis")
            return None
        
        # Step 2: Analyze spatial autocorrelation for each raster
        self.logger.info(f"Analyzing spatial autocorrelation for {len(raster_files)} raster files...")
        results = []
        
        start_time = time.time()
        for raster_path in raster_files:
            self.logger.info(f"Processing: {Path(raster_path).name}")
            result = self.analyze_raster(raster_path)
            results.append(result)
            
            # Force garbage collection between rasters
            gc.collect()
        
        total_time = time.time() - start_time
        
        # Step 3: Save results and create plots
        output_dir = Path("autocorr_results")
        summary_file, detailed_file = self.save_results(results, output_dir)
        self.plot_results(results, output_dir)
        
        # Print summary
        valid_results = [r for r in results if r is not None]
        lengths = [r['autocorr_length_km'] for r in valid_results if not np.isnan(r['autocorr_length_km'])]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"AUTOCORRELATION ANALYSIS COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Processed: {len(valid_results)}/{len(raster_files)} files")
        
        if lengths:
            self.logger.info(f"Autocorrelation lengths: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} km")
            self.logger.info(f"Range: {np.min(lengths):.1f} - {np.max(lengths):.1f} km")
            
            self.logger.info(f"By year:")
            for r in sorted(valid_results, key=lambda x: x['year'] or 0):
                if r['year'] and not np.isnan(r['autocorr_length_km']):
                    self.logger.info(f"  {r['year']}: {r['autocorr_length_km']:.1f} km")
        
        # Step 4: Create spatial clusters using the ML dataset
        training_dataset_path = self.config['data']['training_dataset']
        
        if not os.path.exists(training_dataset_path):
            self.logger.error(f"Training dataset not found: {training_dataset_path}")
            self.logger.error("Please run biomass integration pipeline first")
            return None
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SPATIAL CLUSTERING")
        self.logger.info(f"{'='*60}")
        
        # Load ML dataset
        self.logger.info(f"Loading ML dataset from {training_dataset_path}")
        df = pd.read_csv(training_dataset_path)
        self.logger.info(f"Loaded {len(df)} data points with columns: {', '.join(df.columns)}")
        
        # Create spatial clusters
        k = self.spatial_config['clustering'].get('n_clusters_range', [10, 50])[0]  # Use minimum as default
        autocorr_lengths_m = [r['autocorr_length_m'] for r in valid_results if not np.isnan(r['autocorr_length_m'])]
        
        df_clustered = self.create_spatial_clusters_advanced(df, k, autocorr_lengths_m)
        
        # Save clustered dataset
        clustered_dataset_path = self.config['data']['clustered_dataset']
        ensure_directory(Path(clustered_dataset_path).parent)
        df_clustered.to_csv(clustered_dataset_path, index=False)
        self.logger.info(f"Saved clustered dataset to {clustered_dataset_path}")
        
        self.logger.info("Spatial analysis pipeline completed successfully!")
        
        return df_clustered