"""
Interannual Biomass Analysis Module

This module implements interannual biomass difference mapping and transition
distribution analysis. All algorithmic logic preserved exactly from original scripts:
- map_interannual_differences.py
- calculate_transition_distributions.py

Author: Diego Bengochea
"""

import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from glob import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import gc

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config


class InterannualAnalyzer:
    """
    Interannual biomass analysis for difference mapping and transition distributions.
    
    Implements difference calculation algorithms and transition statistics with
    exact preservation of original mathematical formulations and processing logic.
    """
    
    def __init__(self, config: Optional[Union[str, Path, Dict]] = None):
        """
        Initialize the interannual analyzer.
        
        Args:
            config: Configuration dictionary or path to config file
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config, component_name="biomass_analysis")
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = load_config(component_name="biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='interannual_analysis'
        )
        
        self.logger.info("Initialized InterannualAnalyzer")

    # ==================== SHARED UTILITY METHODS ====================
    
    def find_biomass_files(self, input_dir: str) -> Dict[int, str]:
        """
        Find and organize biomass files by year.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            input_dir: Directory containing biomass files
            
        Returns:
            dict: Dictionary with years as keys and file paths as values
        """
        # Find all raster files matching the pattern
        raster_pattern = os.path.join(input_dir, "TBD_S2_mean_*_100m_TBD_merged.tif")
        raster_files = glob(raster_pattern)
        
        if not raster_files:
            self.logger.error(f"No biomass files found with pattern: {raster_pattern}")
            return {}
        
        # Extract years from filenames
        year_pattern = re.compile(r"TBD_S2_mean_(\d{4})_100m_TBD_merged\.tif")
        year_files = {}
        
        for file in raster_files:
            basename = os.path.basename(file)
            match = year_pattern.match(basename)
            if match:
                year = int(match.group(1))
                year_files[year] = file
        
        self.logger.info(f"Found biomass files for years: {sorted(year_files.keys())}")
        return year_files

    def load_raster_data(self, file_path: str, max_biomass: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Load raster data with quality control.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            file_path: Path to raster file
            max_biomass: Maximum biomass threshold for filtering
            
        Returns:
            numpy array: Raster data with NaN for invalid values
        """
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32)
                
                # Handle nodata values
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                # Apply maximum biomass threshold if specified
                if max_biomass is not None:
                    data = np.where(data > max_biomass, np.nan, data)
                
                return data
        except Exception as e:
            self.logger.error(f"Error loading raster {file_path}: {e}")
            return None

    # ==================== DIFFERENCE MAPPING METHODS ====================
    
    def calculate_raw_difference(self, data1: np.ndarray, data2: np.ndarray, nodata_value: float) -> np.ndarray:
        """
        Calculate raw signed difference (year2 - year1).
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            data1: Biomass data for first year
            data2: Biomass data for second year  
            nodata_value: NoData value to use
            
        Returns:
            numpy array: Raw difference
        """
        # Handle nodata and negative values
        data1_clean = np.where(data1 < 0, 0, data1)
        data2_clean = np.where(data2 < 0, 0, data2)
        
        # Calculate raw difference (year2 - year1)
        diff_data = data2_clean - data1_clean
        
        # Set nodata where either input was nodata
        mask_nodata = np.isnan(data1) | np.isnan(data2)
        diff_data = np.where(mask_nodata, nodata_value, diff_data)
        
        return diff_data

    def calculate_relative_difference(self, data1: np.ndarray, data2: np.ndarray, nodata_value: float) -> np.ndarray:
        """
        Calculate relative symmetric difference: 200*(year2-year1)/(year2+year1).
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            data1: Biomass data for first year
            data2: Biomass data for second year
            nodata_value: NoData value to use
            
        Returns:
            numpy array: Relative symmetric difference (%)
        """
        # Handle nodata and negative values
        data1_clean = np.where(data1 < 0, 0, data1)
        data2_clean = np.where(data2 < 0, 0, data2)
        
        # Calculate denominator (sum)
        denominator = data1_clean + data2_clean
        
        # Calculate relative difference where denominator > 0
        # Formula: 200 * (year2 - year1) / (year2 + year1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(
                denominator > 0,
                200 * (data2_clean - data1_clean) / denominator,
                0  # Set to 0 where both years have 0 biomass
            )
        
        # Set nodata where either input was nodata
        mask_nodata = np.isnan(data1) | np.isnan(data2)
        rel_diff = np.where(mask_nodata, nodata_value, rel_diff)
        
        return rel_diff

    def create_difference_maps(self, year_files: Dict[int, str], output_raw_dir: str, output_relative_dir: str) -> bool:
        """
        Create interannual difference maps for consecutive years.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            year_files: Dictionary with years as keys and file paths as values
            output_raw_dir: Directory for raw difference outputs
            output_relative_dir: Directory for relative difference outputs
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directories
        os.makedirs(output_raw_dir, exist_ok=True)
        os.makedirs(output_relative_dir, exist_ok=True)
        
        # Get consecutive year pairs
        years = sorted(year_files.keys())
        
        if len(years) < 2:
            self.logger.error("Need at least 2 years of data to calculate differences")
            return False
        
        year_pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
        self.logger.info(f"Processing {len(year_pairs)} year pairs: {year_pairs}")
        
        # Process each year pair
        for year1, year2 in year_pairs:
            self.logger.info(f"Processing difference: {year2} - {year1}")
            
            # Load raster data
            with rasterio.open(year_files[year1]) as src1:
                data1 = src1.read(1).astype(np.float32)
                profile = src1.profile.copy()
                
                # Handle nodata
                if src1.nodata is not None:
                    data1 = np.where(data1 == src1.nodata, np.nan, data1)
            
            with rasterio.open(year_files[year2]) as src2:
                data2 = src2.read(1).astype(np.float32)
                
                # Handle nodata
                if src2.nodata is not None:
                    data2 = np.where(data2 == src2.nodata, np.nan, data2)
            
            # Check dimensions match
            if data1.shape != data2.shape:
                self.logger.error(f"Shape mismatch: {year1} {data1.shape} vs {year2} {data2.shape}")
                continue
            
            # Set nodata value for outputs
            output_nodata = -9999.0
            
            # Calculate raw difference
            raw_diff = self.calculate_raw_difference(data1, data2, output_nodata)
            
            # Calculate relative difference
            rel_diff = self.calculate_relative_difference(data1, data2, output_nodata)
            
            # Update profile for outputs
            profile.update(dtype=rasterio.float32, nodata=output_nodata)
            
            # Save raw difference
            raw_output = os.path.join(output_raw_dir, f"TBD_S2_raw_change_{year1}Sep-{year2}Aug_100m.tif")
            with rasterio.open(raw_output, 'w', **profile) as dst:
                dst.write(raw_diff, 1)
            self.logger.info(f"  Saved raw difference: {raw_output}")
            
            # Save relative difference
            rel_output = os.path.join(output_relative_dir, f"TBD_S2_relative_change_symmetric_{year1}Sep-{year2}Aug_100m.tif")
            with rasterio.open(rel_output, 'w', **profile) as dst:
                dst.write(rel_diff, 1)
            self.logger.info(f"  Saved relative difference: {rel_output}")
        
        return True

    # ==================== TRANSITION DISTRIBUTION METHODS ====================
    
    def calculate_transition_statistics(self, year1_data: np.ndarray, year2_data: np.ndarray, year1: int, year2: int) -> Dict[str, Any]:
        """
        Calculate comprehensive transition statistics for a year pair.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            year1_data: Biomass data for first year
            year2_data: Biomass data for second year
            year1: First year
            year2: Second year
            
        Returns:
            dict: Comprehensive transition statistics
        """
        # Create mask for valid pixels (both years have data)
        valid_mask = ~np.isnan(year1_data) & ~np.isnan(year2_data)
        
        if np.sum(valid_mask) == 0:
            self.logger.warning(f"No valid pixels for transition {year1}-{year2}")
            return None
        
        # Extract valid data
        y1_valid = year1_data[valid_mask]
        y2_valid = year2_data[valid_mask]
        
        # Calculate differences
        raw_diff = y2_valid - y1_valid
        
        # Calculate relative differences (avoiding division by zero)
        denominator = y1_valid + y2_valid
        rel_diff = np.zeros_like(raw_diff)
        
        # Only calculate relative difference where denominator > 0
        nonzero_mask = denominator > 0
        rel_diff[nonzero_mask] = 200 * raw_diff[nonzero_mask] / denominator[nonzero_mask]
        
        # Count different transition types
        gain_mask = raw_diff > 0
        loss_mask = raw_diff < 0
        stable_mask = raw_diff == 0
        
        # Calculate statistics
        stats = {
            'year1': year1,
            'year2': year2,
            'period': f"{year1}-{year2}",
            'n_valid_pixels': int(np.sum(valid_mask)),
            
            # Raw difference statistics
            'mean_raw_diff': float(np.mean(raw_diff)),
            'median_raw_diff': float(np.median(raw_diff)),
            'std_raw_diff': float(np.std(raw_diff)),
            'min_raw_diff': float(np.min(raw_diff)),
            'max_raw_diff': float(np.max(raw_diff)),
            'q25_raw_diff': float(np.percentile(raw_diff, 25)),
            'q75_raw_diff': float(np.percentile(raw_diff, 75)),
            
            # Relative difference statistics  
            'mean_rel_diff': float(np.mean(rel_diff)),
            'median_rel_diff': float(np.median(rel_diff)),
            'std_rel_diff': float(np.std(rel_diff)),
            'min_rel_diff': float(np.min(rel_diff)),
            'max_rel_diff': float(np.max(rel_diff)),
            'q25_rel_diff': float(np.percentile(rel_diff, 25)),
            'q75_rel_diff': float(np.percentile(rel_diff, 75)),
            
            # Transition counts
            'n_gain_pixels': int(np.sum(gain_mask)),
            'n_loss_pixels': int(np.sum(loss_mask)),
            'n_stable_pixels': int(np.sum(stable_mask)),
            'pct_gain': float(100 * np.sum(gain_mask) / len(raw_diff)),
            'pct_loss': float(100 * np.sum(loss_mask) / len(raw_diff)),
            'pct_stable': float(100 * np.sum(stable_mask) / len(raw_diff)),
            
            # Biomass level statistics
            'mean_biomass_y1': float(np.mean(y1_valid)),
            'mean_biomass_y2': float(np.mean(y2_valid)),
            'median_biomass_y1': float(np.median(y1_valid)),
            'median_biomass_y2': float(np.median(y2_valid)),
            'std_biomass_y1': float(np.std(y1_valid)),
            'std_biomass_y2': float(np.std(y2_valid)),
        }
        
        return stats

    def save_raw_transition_data(self, year1_data: np.ndarray, year2_data: np.ndarray, year1: int, year2: int) -> None:
        """
        Save raw transition data to CSV file.
        
        Args:
            year1_data: Biomass data for first year
            year2_data: Biomass data for second year
            year1: First year
            year2: Second year
        """
        # Create mask for valid pixels
        valid_mask = ~np.isnan(year1_data) & ~np.isnan(year2_data)
        
        if np.sum(valid_mask) == 0:
            return
        
        # Extract valid data
        y1_valid = year1_data[valid_mask]
        y2_valid = year2_data[valid_mask]
        
        # Calculate differences
        raw_diff = y2_valid - y1_valid
        
        # Create DataFrame
        df = pd.DataFrame({
            'biomass_y1': y1_valid,
            'biomass_y2': y2_valid,
            'raw_diff': raw_diff
        })
        
        # Save to output directory
        output_dir = ANALYSIS_OUTPUTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"raw_transition_data_{year1}_{year2}.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"  Saved raw transition data: {output_file}")

    def calculate_transition_distributions(self, data_dir: str, target_years: Optional[List[int]] = None, save_raw_data: bool = False) -> List[Dict[str, Any]]:
        """
        Calculate biomass transition distributions between consecutive years.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            data_dir: Directory containing biomass files
            target_years: Optional list of specific years to process
            save_raw_data: Whether to save raw transition data
            
        Returns:
            List of transition statistics dictionaries
        """
        # Find biomass files
        biomass_files = self.find_biomass_files(data_dir)
        
        if len(biomass_files) < 2:
            self.logger.error("Need at least 2 years of biomass data")
            return []
        
        # Filter to target years if specified
        if target_years:
            biomass_files = {year: path for year, path in biomass_files.items() if year in target_years}
        
        # Get consecutive year pairs
        years = sorted(biomass_files.keys())
        year_pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
        
        self.logger.info(f"Processing {len(year_pairs)} year pairs for transition analysis")
        
        # Quality control threshold from config
        max_biomass = self.config['quality_control']['max_biomass_threshold']
        
        results = []
        
        # Process each year pair
        for year1, year2 in year_pairs:
            self.logger.info(f"Processing transition: {year1} -> {year2}")
            
            # Load data for both years
            year1_data = self.load_raster_data(biomass_files[year1], max_biomass)
            year2_data = self.load_raster_data(biomass_files[year2], max_biomass)
            
            if year1_data is None or year2_data is None:
                self.logger.error(f"Failed to load data for {year1}-{year2}")
                continue
            
            # Check dimensions match
            if year1_data.shape != year2_data.shape:
                self.logger.error(f"Shape mismatch: {year1} {year1_data.shape} vs {year2} {year2_data.shape}")
                continue
            
            # Calculate transition statistics
            stats = self.calculate_transition_statistics(year1_data, year2_data, year1, year2)
            
            if stats:
                results.append(stats)
                self.logger.info(f"  Completed: {stats['n_valid_pixels']:,} valid pixels, "
                               f"mean change: {stats['mean_raw_diff']:.2f} Mg/ha")
            
            # Save raw transition data if requested
            if save_raw_data and stats:
                self.save_raw_transition_data(year1_data, year2_data, year1, year2)
            
            # Force garbage collection
            del year1_data, year2_data
            gc.collect()
        
        return results

    # ==================== MAIN ANALYSIS METHODS ====================
    
    def run_difference_mapping(self, target_years: Optional[List[int]] = None) -> bool:
        """
        Run complete interannual difference mapping analysis.
        
        Args:
            target_years: Optional list of specific years to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Starting interannual difference mapping...")
        
        # Get input and output directories from config
        input_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR / "mean"
        output_raw_dir = ANALYSIS_OUTPUTS_DIR
        output_relative_dir = ANALYSIS_OUTPUTS_DIR
        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Raw output directory: {output_raw_dir}")
        self.logger.info(f"Relative output directory: {output_relative_dir}")
        
        # Find biomass files
        year_files = self.find_biomass_files(input_dir)
        
        if len(year_files) < 2:
            self.logger.error("Need at least 2 years of biomass data")
            return False
        
        # Filter to target years if specified
        if target_years:
            year_files = {year: path for year, path in year_files.items() if year in target_years}
        
        # Create difference maps
        return self.create_difference_maps(year_files, output_raw_dir, output_relative_dir)

    def run_transition_analysis(self, target_years: Optional[List[int]] = None, save_raw_data: bool = False) -> List[Dict[str, Any]]:
        """
        Run complete transition distribution analysis.
        
        Args:
            target_years: Optional list of specific years to process
            save_raw_data: Whether to save raw transition data
            
        Returns:
            List of transition statistics dictionaries
        """
        self.logger.info("Starting biomass transition distribution analysis...")
        
        # Get input directory from config
        input_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR / "mean"
        
        self.logger.info(f"Input directory: {input_dir}")
        
        # Calculate transition distributions
        return self.calculate_transition_distributions(input_dir, target_years, save_raw_data)

    # ==================== RESULTS SAVING ====================
    
    def save_results(self, results: List[Dict[str, Any]], analysis_type: str) -> Optional[str]:
        """
        Save interannual analysis results to files.
        
        Args:
            results: List of analysis results
            analysis_type: Type of analysis ('differences' or 'transitions')
            
        Returns:
            Path to main output file or None if no results
        """
        if not results:
            self.logger.warning("No results to save")
            return None
        
        output_dir = ANALYSIS_OUTPUTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if analysis_type == 'transitions':
            # Save transition statistics
            df = pd.DataFrame(results)
            output_file = os.path.join(output_dir, f"biomass_transition_statistics_{timestamp}.csv")
            df.to_csv(output_file, index=False)
            self.logger.info(f"Transition statistics saved to: {output_file}")
            return output_file
        else:
            # For difference mapping, results are already saved as raster files
            self.logger.info("Difference maps saved to output directories")
            return None
