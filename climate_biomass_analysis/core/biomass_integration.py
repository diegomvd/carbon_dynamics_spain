"""
Biomass-climate data integration pipeline.

Integrates biomass change data with climate anomalies to create machine learning
training datasets. Handles raster resampling, spatial alignment, and data point
extraction for modeling biomass response to climate variables.

Author: Diego Bengochea
"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from rasterio.enums import Resampling
import rasterio.warp
import re

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory


class BiomassIntegrator:
    """
    Biomass-climate data integration pipeline.
    
    This class handles the integration of biomass change data with climate
    anomalies to create training datasets for machine learning models.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the biomass integrator.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='biomass_integration',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration
        self.integration_config = self.config['biomass_integration']
        self.data_paths = self.config['data']
        
        self.logger.info("Initialized BiomassIntegrator")
    
    def harmonize_raster(
        self,
        input_path: Union[str, Path],
        reference_shape: Tuple[int, int],
        reference_transform: rasterio.Affine,
        reference_crs: rasterio.CRS
    ) -> np.ndarray:
        """
        Harmonize a raster to match reference grid properties.
        
        Args:
            input_path: Path to input raster
            reference_shape: Target shape (height, width)
            reference_transform: Target transform
            reference_crs: Target CRS
            
        Returns:
            Harmonized raster data array
        """
        with rasterio.open(input_path) as src:
            # Check if already aligned
            if (src.shape == reference_shape and 
                src.transform == reference_transform and 
                src.crs == reference_crs):
                return src.read(1)
            
            # Reproject and resample to match reference
            harmonized = np.empty(reference_shape, dtype=src.dtypes[0])
            
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=harmonized,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=getattr(Resampling, self.integration_config['resampling_method'].lower())
            )
            
            return harmonized
    
    def batch_resample_biomass(
        self,
        biomass_diff_dir: Union[str, Path],
        ref_raster_path: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*_rel_change_*.tif",
        resampling_method: Resampling = Resampling.bilinear
    ) -> List[str]:
        """
        Batch resample biomass difference files to match climate data resolution.
        
        Args:
            biomass_diff_dir: Directory containing biomass difference files
            ref_raster_path: Reference raster for target resolution/alignment
            output_dir: Output directory for resampled files
            pattern: File pattern to match
            resampling_method: Resampling method to use
            
        Returns:
            List of resampled file paths
        """
        biomass_diff_dir = Path(biomass_diff_dir)
        output_dir = Path(output_dir)
        
        # Find biomass difference files
        diff_files = list(biomass_diff_dir.glob(pattern))
        
        if not diff_files:
            self.logger.warning(f"No biomass difference files found matching pattern: {pattern}")
            return []
        
        self.logger.info(f"Found {len(diff_files)} biomass difference files")
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Get reference raster properties
        with rasterio.open(ref_raster_path) as ref:
            ref_shape = (ref.height, ref.width)
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_profile = ref.profile.copy()
        
        self.logger.info(f"Reference raster: {ref_shape[1]}x{ref_shape[0]}, CRS: {ref_crs}")
        
        # Resample each file
        resampled_files = []
        for diff_file in diff_files:
            try:
                # Generate output filename
                output_filename = f"resampled_{diff_file.name}"
                output_path = output_dir / output_filename
                
                # Check if already processed
                if output_path.exists():
                    self.logger.debug(f"Already resampled: {output_filename}")
                    resampled_files.append(str(output_path))
                    continue
                
                # Harmonize to reference grid
                resampled_data = self.harmonize_raster(
                    diff_file, ref_shape, ref_transform, ref_crs
                )
                
                # Save resampled file
                with rasterio.open(output_path, 'w', **ref_profile) as dst:
                    dst.write(resampled_data.astype(np.float32), 1)
                
                resampled_files.append(str(output_path))
                self.logger.debug(f"Resampled: {diff_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error resampling {diff_file}: {e}")
        
        self.logger.info(f"Resampled {len(resampled_files)}/{len(diff_files)} files")
        return resampled_files
    
    def extract_year_range_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract start and end years from biomass difference filename.
        
        Args:
            filename: Biomass difference filename
            
        Returns:
            Tuple of (start_year, end_year) or (None, None) if not found
        """
        # Look for patterns like "2017_2020" or "2017-2020"
        year_pattern = r'(\d{4})[-_](\d{4})'
        match = re.search(year_pattern, filename)
        
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            return start_year, end_year
        
        # Look for single year pattern
        single_year_pattern = r'(\d{4})'
        matches = re.findall(single_year_pattern, filename)
        
        if len(matches) >= 2:
            # Take first two years found
            return int(matches[0]), int(matches[1])
        
        return None, None
    
    def create_training_dataset(
        self,
        biomass_diff_files: List[str],
        anomaly_dir: Union[str, Path],
        output_file: Union[str, Path]
    ) -> Optional[pd.DataFrame]:
        """
        Create training dataset by integrating biomass changes with climate anomalies.
        
        Args:
            biomass_diff_files: List of biomass difference file paths
            anomaly_dir: Directory containing climate anomaly files
            output_file: Output CSV file path
            
        Returns:
            DataFrame with integrated data or None if failed
        """
        anomaly_dir = Path(anomaly_dir)
        all_data = []
        
        # Process each biomass difference file
        for diff_file in biomass_diff_files:
            self.logger.info(f"Processing: {Path(diff_file).name}")
            
            # Extract year range from filename
            start_year, end_year = self.extract_year_range_from_filename(Path(diff_file).name)
            
            if start_year is None or end_year is None:
                self.logger.warning(f"Could not extract year range from {diff_file}")
                continue
            
            # Find corresponding anomaly directory
            year_pattern = f"anomalies_{start_year}"  # or f"anomalies_{start_year}_{end_year}"
            
            anomaly_year_dirs = list(anomaly_dir.glob(f"*{start_year}*"))
            
            if not anomaly_year_dirs:
                self.logger.warning(f"No anomaly directory found for year pattern: {year_pattern}")
                continue
            
            anomaly_year_dir = anomaly_year_dirs[0]
            self.logger.debug(f"Using anomaly directory: {anomaly_year_dir.name}")
            
            # Load biomass difference data
            with rasterio.open(diff_file) as biomass_src:
                biomass_diff = biomass_src.read(1).astype(np.float32)
                reference_shape = biomass_diff.shape
                reference_transform = biomass_src.transform
                reference_crs = biomass_src.crs
                
                # Create valid data mask (non-NaN, finite values)
                valid_mask = np.isfinite(biomass_diff) & (biomass_diff != biomass_src.nodata)
            
            # Load climate anomaly data
            anomaly_files = list(anomaly_year_dir.glob("*.tif"))
            
            if not anomaly_files:
                self.logger.warning(f"No anomaly files found in {anomaly_year_dir}")
                continue
            
            # Load all anomaly variables
            anomaly_data = {}
            for anomaly_file in anomaly_files:
                try:
                    # Extract variable name from filename
                    var_name = anomaly_file.stem.replace(f"_anomaly_{start_year}", "")
                    
                    # Load and harmonize anomaly data
                    with rasterio.open(anomaly_file) as src:
                        # Check if harmonization is needed
                        if (src.shape != reference_shape or 
                            src.transform != reference_transform or 
                            src.crs != reference_crs):
                            self.logger.debug(f"Harmonizing {var_name}...")
                            try:
                                anomaly_data[var_name] = self.harmonize_raster(
                                    anomaly_file, 
                                    reference_shape, 
                                    reference_transform, 
                                    reference_crs
                                )
                            except Exception as e:
                                self.logger.error(f"Error harmonizing {anomaly_file}: {e}")
                                continue
                        else:
                            anomaly_data[var_name] = src.read(1)

                        # Update valid mask with anomaly data
                        anomaly_mask = np.isfinite(anomaly_data[var_name]) & (anomaly_data[var_name] != -9999)
                        valid_mask = valid_mask & anomaly_mask
                       
                except Exception as e:
                    self.logger.error(f"Error reading {anomaly_file}: {e}")
                    continue
            
            # Check how many variables were loaded
            if not anomaly_data:
                self.logger.warning(f"No anomaly data found for {year_pattern}. Skipping...")
                continue
            
            self.logger.info(f"Loaded {len(anomaly_data)} anomaly variables for {year_pattern}")

            # Get indices of valid pixels
            valid_indices = np.where(valid_mask)
            
            if len(valid_indices[0]) == 0:
                self.logger.warning(f"No valid data points for {year_pattern}")
                continue
            
            # Limit number of points if too many
            max_points = self.integration_config.get('max_valid_pixels', 1000000)
            if len(valid_indices[0]) > max_points:
                # Randomly sample points
                indices = np.random.choice(len(valid_indices[0]), max_points, replace=False)
                valid_indices = (valid_indices[0][indices], valid_indices[1][indices])
                self.logger.info(f"Sampled {max_points} points from {len(valid_indices[0])} valid points")
            
            # Create data points
            year_data = []
            
            for y, x in zip(valid_indices[0], valid_indices[1]):
                # Convert pixel coordinates to geographic coordinates
                geo_x, geo_y = reference_transform * (x, y)
                
                # Create data point with year info and biomass change
                data_point = {
                    'x': geo_x,
                    'y': geo_y,
                    'year_start': start_year,
                    'year_end': end_year,
                    'biomass_rel_change': biomass_diff[y, x]
                }
                
                # Add all anomaly variables
                for var_name, var_data in anomaly_data.items():
                    data_point[var_name] = var_data[y, x]
                
                year_data.append(data_point)
            
            self.logger.info(f"Added {len(year_data)} data points for {year_pattern}")
            all_data.extend(year_data)
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Remove rows with NaN values
            original_len = len(df)
            df = df.dropna()
            if len(df) < original_len:
                self.logger.info(f"Removed {original_len - len(df)} rows with NaN values")
            
            # Quality control: remove outliers if specified
            if self.integration_config.get('remove_outliers', False):
                df = self._remove_outliers(df)
            
            # Save to CSV
            ensure_directory(Path(output_file).parent)
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Dataset created with {len(df)} data points. Saved to {output_file}")
            return df
        else:
            self.logger.warning("No data points found.")
            return None
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        threshold = self.integration_config.get('outlier_threshold', 3.0)
        original_len = len(df)
        
        # Remove outliers based on biomass change
        if 'biomass_rel_change' in df.columns:
            biomass_col = df['biomass_rel_change']
            mean_val = biomass_col.mean()
            std_val = biomass_col.std()
            
            # Keep points within threshold standard deviations
            outlier_mask = np.abs(biomass_col - mean_val) <= threshold * std_val
            df = df[outlier_mask]
            
            removed = original_len - len(df)
            if removed > 0:
                self.logger.info(f"Removed {removed} outliers (>{threshold}σ from mean)")
        
        return df
    
    def run_biomass_integration_pipeline(self) -> Optional[pd.DataFrame]:
        """
        Execute the complete biomass-climate integration workflow.
        
        Returns:
            Final ML dataset as DataFrame
        """
        self.logger.info("Starting biomass-climate integration pipeline...")
        
        # Extract config parameters
        biomass_diff_dir = self.data_paths['biomass_diff_dir']
        anomaly_dir = self.data_paths['anomaly_dir']
        output_dir = self.data_paths['temp_resampled_dir']
        training_dataset_path = self.data_paths['training_dataset']
        biomass_pattern = self.integration_config['pattern']
        
        # Get reference raster from first available anomaly file
        first_anomaly_dir = None
        for item in os.listdir(anomaly_dir):
            if item.startswith("anomalies_") and os.path.isdir(os.path.join(anomaly_dir, item)):
                first_anomaly_dir = os.path.join(anomaly_dir, item)
                break
        
        if not first_anomaly_dir:
            self.logger.error("No anomaly directories found. Run bioclimatic calculation first.")
            return None
        
        # Find first anomaly file as reference
        anomaly_files = glob.glob(os.path.join(first_anomaly_dir, "*.tif"))
        if not anomaly_files:
            self.logger.error("No anomaly files found in directory. Run bioclimatic calculation first.")
            return None
        
        ref_raster_path = anomaly_files[0]
        self.logger.info(f"Using reference raster: {Path(ref_raster_path).name}")
        
        # Step 1: Batch resample biomass data to climate resolution
        self.logger.info("Resampling biomass difference files to climate resolution...")
        diff_files = self.batch_resample_biomass(
            biomass_diff_dir, 
            ref_raster_path, 
            output_dir=output_dir, 
            pattern=biomass_pattern,
            resampling_method=getattr(Resampling, self.integration_config['resampling_method'].lower())
        )
        
        if not diff_files:
            self.logger.error("No biomass difference files found.")
            return None
        
        self.logger.info(f"Found {len(diff_files)} biomass difference files.")
        
        # Step 2: Create training dataset
        self.logger.info("Creating training dataset...")
        dataset = self.create_training_dataset(
            diff_files,
            anomaly_dir,
            training_dataset_path
        )
        
        if dataset is not None:
            self.logger.info("Biomass-climate integration completed successfully!")
        else:
            self.logger.error("Failed to create training dataset")
        
        return dataset