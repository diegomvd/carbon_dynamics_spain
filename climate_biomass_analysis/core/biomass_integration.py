"""
FIXED: Biomass integration with corrected harmonization logic.

Key fixes:
1. Use np.zeros instead of np.empty for harmonization
2. Preserve all other original logic

Author: Diego Bengochea
"""

import rasterio
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from rasterio.enums import Resampling

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils.central_data_paths_constants import *


class BiomassIntegrator:
    """
    Biomass-climate data integration pipeline.
    
    This class handles the integration of biomass changes with climate anomalies,
    including spatial harmonization and dataset creation for machine learning.
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
        self.integration_config = self.config.get('biomass_integration', {})
        
        self.logger.info("Initialized BiomassIntegrator")
    
    def harmonize_raster(
        self, 
        input_path: Union[str, Path], 
        reference_shape: Tuple[int, int],
        reference_transform: rasterio.Affine,
        reference_crs: rasterio.CRS
    ) -> np.ndarray:
        """
        Harmonize a single raster to match reference grid properties.
        
        FIXED: Use np.zeros instead of np.empty
        
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
            
            # FIXED: Use np.zeros instead of np.empty to avoid uninitialized values
            harmonized = np.zeros(reference_shape, dtype=src.dtypes[0])
            
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=harmonized,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=getattr(Resampling, self.integration_config.get('resampling_method', 'bilinear').lower())
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
            biomass_diff_dir: Directory containing biomass rasters to process
            ref_raster_path: Path to reference raster for resolution and extent
            output_dir: Directory where resampled rasters will be saved
            pattern: Glob pattern to match biomass raster files
            resampling_method: Method used for resampling
            
        Returns:
            List of paths to resampled output rasters
        """
        self.logger.info("Starting batch resampling of biomass files...")
        
        biomass_dir = Path(biomass_diff_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        ensure_directory(output_dir)
        
        # Find all biomass rasters matching the pattern
        biomass_files = list(biomass_dir.glob(pattern))
        
        if not biomass_files:
            self.logger.warning(f"No files matching pattern '{pattern}' found in {biomass_dir}")
            return []
        
        self.logger.info(f"Found {len(biomass_files)} biomass rasters to process")
        
        # Get reference raster metadata
        with rasterio.open(ref_raster_path) as ref_src:
            ref_shape = (ref_src.height, ref_src.width)
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
        
        output_files = []
        
        # Process each biomass raster
        for i, biomass_file in enumerate(biomass_files):
            self.logger.info(f"Processing {i+1}/{len(biomass_files)}: {biomass_file.name}")
            
            output_path = output_dir / f"{biomass_file.stem}_resampled{biomass_file.suffix}"
            
            try:
                # Harmonize to reference grid
                harmonized_data = self.harmonize_raster(
                    biomass_file, ref_shape, ref_transform, ref_crs
                )
                
                # Save resampled raster
                with rasterio.open(biomass_file) as src:
                    profile = src.profile.copy()
                    profile.update({
                        'height': ref_shape[0],
                        'width': ref_shape[1],
                        'transform': ref_transform,
                        'crs': ref_crs
                    })
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(harmonized_data.astype(profile['dtype']), 1)
                
                output_files.append(str(output_path))
                
            except Exception as e:
                self.logger.error(f"Error processing {biomass_file.name}: {str(e)}")
                continue
        
        self.logger.info(f"Batch processing complete. {len(output_files)} files successfully resampled.")
        return output_files
    
    def create_ml_dataset(
        self, 
        diff_files: List[str], 
        anomaly_dir: Union[str, Path], 
        output_file: Union[str, Path]
    ) -> Optional[pd.DataFrame]:
        """
        Create a machine learning dataset combining biomass changes with climate anomalies.
        
        This function preserves the original logic from the old implementation,
        including support for 1-year, 2-year, and 3-year cumulative anomalies.
        
        Args:
            diff_files: List of paths to biomass difference files
            anomaly_dir: Directory containing bioclimatic anomaly files
            output_file: Path to save the output dataset (CSV)
        
        Returns:
            pandas.DataFrame: The created ML dataset
        """
        self.logger.info("Creating ML dataset from biomass differences and climate anomalies...")
        
        # Data collection for all years
        all_data = []
        
        for diff_file in diff_files:
            # Extract years from filename
            basename = os.path.basename(diff_file)
            self.logger.info(f"Processing: {basename}")
            
            # Parse filename to extract years (format may vary)
            try:
                # Try to extract year pattern from filename
                if 'biomass_rel_change' in basename:
                    # Expected format: biomass_rel_change_YYYY_YYYY.tif
                    parts = basename.split('_')
                    for i, part in enumerate(parts):
                        if len(part) == 4 and part.isdigit():
                            start_year = int(part)
                            if i + 1 < len(parts) and len(parts[i + 1]) == 4 and parts[i + 1].isdigit():
                                end_year = int(parts[i + 1])
                                break
                    else:
                        self.logger.warning(f"Could not parse years from filename: {basename}")
                        continue
                else:
                    # Try alternative parsing
                    years_str = basename.split('_')[5] if len(basename.split('_')) > 5 else basename.split('_')[-1]
                    start_year, end_year = years_str.split('-')
                    start_year = int(start_year[:4])
                    end_year = int(end_year[:4])
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error parsing filename {basename}: {e}")
                continue
            
            self.logger.info(f"Processing biomass difference for {start_year}-{end_year}...")
            
            # Load biomass difference data
            with rasterio.open(diff_file) as src:
                biomass_diff = src.read(1)
                nodata = src.nodata
                reference_shape = biomass_diff.shape
                reference_transform = src.transform
                reference_crs = src.crs
                biomass_mask = biomass_diff != nodata if nodata is not None else np.ones_like(biomass_diff, dtype=bool)

            self.logger.info(f"Reference shape: {reference_shape}, CRS: {reference_crs}")
            
            # Find all anomaly files for this year
            year_pattern = f"{start_year}Sep-{end_year}Aug"
            
            # Get 1-year anomalies
            anomaly_1yr_dir = os.path.join(anomaly_dir, f"anomalies_{year_pattern}")
            anomaly_1yr_files = glob.glob(os.path.join(anomaly_1yr_dir, "*.tif")) if os.path.exists(anomaly_1yr_dir) else []
            
            # Get 2-year anomalies if available
            anomaly_2yr_dir = os.path.join(anomaly_dir, f"anomalies_2yr_{year_pattern}")
            anomaly_2yr_files = glob.glob(os.path.join(anomaly_2yr_dir, "*.tif")) if os.path.exists(anomaly_2yr_dir) else []
            
            # Get 3-year anomalies if available
            anomaly_3yr_dir = os.path.join(anomaly_dir, f"anomalies_3yr_{year_pattern}")
            anomaly_3yr_files = glob.glob(os.path.join(anomaly_3yr_dir, "*.tif")) if os.path.exists(anomaly_3yr_dir) else []
            
            # Load all anomaly data
            anomaly_data = {}
            valid_mask = biomass_mask.copy()
            
            for anomaly_file in anomaly_1yr_files + anomaly_2yr_files + anomaly_3yr_files:
                # Extract variable name from filename
                var_name = os.path.basename(anomaly_file).split('_')[0]
                
                # Add year suffix based on directory
                if "2yr" in anomaly_file:
                    var_name = f"{var_name}_2yr"
                elif "3yr" in anomaly_file:
                    var_name = f"{var_name}_3yr"
                
                try:
                    with rasterio.open(anomaly_file) as src:
                        # Check if shapes match
                        if src.shape != biomass_diff.shape or src.crs != reference_crs:
                            self.logger.warning(f"Shape/CRS mismatch for {anomaly_file}. Harmonizing...")
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

                        # Update valid mask
                        anomaly_mask = anomaly_data[var_name] != -9999
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
        biomass_diff_dir = BIOMASS_MAPS_RELDIFF_DIR
        anomaly_dir = BIOCLIM_ANOMALIES_DIR
        output_dir = "" # TODO: I am not sure what should go here.
        training_dataset_path = CLIMATE_BIOMASS_DATASET_FILE
        biomass_pattern = self.integration_config.get('pattern', "*_rel_change_*.tif")
        
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
        
        reference_file = anomaly_files[0]
        
        # Step 1: Resample biomass files to match climate resolution
        self.logger.info("Resampling biomass files to match climate resolution...")
        resampled_files = self.batch_resample_biomass(
            biomass_diff_dir, reference_file, output_dir, biomass_pattern
        )
        
        if not resampled_files:
            self.logger.error("No biomass files were successfully resampled.")
            return None
        
        # Step 2: Create ML dataset
        self.logger.info("Creating ML training dataset...")
        df = self.create_ml_dataset(resampled_files, anomaly_dir, training_dataset_path)
        
        if df is not None:
            self.logger.info("Biomass-climate integration completed successfully!")
        else:
            self.logger.error("Failed to create ML dataset.")
        
        return df
