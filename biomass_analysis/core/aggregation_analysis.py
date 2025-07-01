"""
Biomass Aggregation Analysis Module

This module implements hierarchical biomass aggregation by forest type, landcover,
and height ranges. 

Author: Diego Bengochea
"""

import os
import csv
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config
from shared_utils.central_data_paths_constants import *


class BiomassAggregationPipeline:
    """
    Hierarchical biomass aggregation for forest types, landcover, and height ranges.
    
    Implements aggregation analysis with exact preservation of original algorithms
    for forest type hierarchies, landcover grouping, and height bin processing.
    """
    
    def __init__(self, config: Optional[Union[str, Path]] = None):
        """
        Initialize the biomass aggregator.
        
        Args:
            config:  Path to config file
        """
        
        self.config = load_config(config, component_name="biomass_analysis")

        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='aggregation_analysis'
        )
        
    def run_full_pipeline(self) -> bool:
        """
        Run complete biomass aggregation analysis pipeline.
        
        Executes all analysis types and saves results in coherent format.
        
        Returns:
            bool: True if all analyses completed successfully
        """
        try:
            self.logger.info("Starting complete biomass aggregation analysis pipeline...")
            
            target_years = self.config['analysis']['target_years']
            success_flags = []
            
            # 1. Forest Type Analysis
            self.logger.info("Running forest type analysis...")
            try:
                combined_forest_type_df, genus_df, clade_df = self.run_forest_type_analysis(target_years)
                
                if combined_forest_type_df is not None:
                    # Save forest type results with coherent format
                    forest_type_results = {
                        'forest_type': combined_forest_type_df,
                        'genus': genus_df, 
                        'clade': clade_df
                    }
                    output_file = self.save_results(forest_type_results, 'forest_type')
                    self.logger.info(f"Forest type analysis completed successfully. Results saved to: {output_file}")
                    success_flags.append(True)
                else:
                    self.logger.error("Forest type analysis returned None results")
                    success_flags.append(False)
                    
            except Exception as e:
                self.logger.error(f"Forest type analysis failed: {str(e)}")
                success_flags.append(False)
            
            # 2. Landcover Analysis  
            self.logger.info("Running landcover analysis...")
            try:
                landcover_results_list = self.run_landcover_analysis(target_years)
                
                if landcover_results_list:
                    # Convert List[Dict] to coherent DataFrame format
                    landcover_df = pd.DataFrame(landcover_results_list)
                    landcover_results = {'landcover': landcover_df}
                    output_file = self.save_results(landcover_results, 'landcover')
                    self.logger.info(f"Landcover analysis completed successfully. Results saved to: {output_file}")
                    success_flags.append(True)
                else:
                    self.logger.error("Landcover analysis returned empty results")
                    success_flags.append(False)
                    
            except Exception as e:
                self.logger.error(f"Landcover analysis failed: {str(e)}")
                success_flags.append(False)
            
            # 3. Height Bin Analysis
            self.logger.info("Running height bin analysis...")
            try:
                height_bin_results_list = self.run_height_bin_analysis(target_years)
                
                if height_bin_results_list:
                    # Convert List[Dict] to coherent DataFrame format
                    height_bin_df = pd.DataFrame(height_bin_results_list)
                    height_bin_results = {'height_bin': height_bin_df}
                    output_file = self.save_results(height_bin_results, 'height_bin')
                    self.logger.info(f"Height bin analysis completed successfully. Results saved to: {output_file}")
                    success_flags.append(True)
                else:
                    self.logger.error("Height bin analysis returned empty results")
                    success_flags.append(False)
                    
            except Exception as e:
                self.logger.error(f"Height bin analysis failed: {str(e)}")
                success_flags.append(False)
            
            # Overall success assessment
            overall_success = all(success_flags)
            
            if overall_success:
                self.logger.info("✅ Complete biomass aggregation analysis pipeline completed successfully")
            else:
                successful_analyses = sum(success_flags)
                total_analyses = len(success_flags)
                self.logger.warning(f"⚠️ Pipeline completed with partial success: {successful_analyses}/{total_analyses} analyses succeeded")
            
            return overall_success
        
        except Exception as e:
            self.logger.error(f"Biomass aggregation pipeline failed with unexpected error: {str(e)}")
            return False

    
    def load_and_merge_forest_type_mappings(self) -> pd.DataFrame:
        """
        Load and merge the two CSV files containing forest type mappings.
        Returns a DataFrame with code, name, genus, and clade.
        
        Returns:
            DataFrame with forest type mappings
        """
        self.logger.info("Loading forest type mappings...")
                
        # Load CSV files
        # TODO: VERIFIY EXACTLY WHAT THESE FILES ARE
        forest_types_path = FOREST_TYPES_TIERS_FILE
        forest_codes_path = FOREST_TYPE_CODES
        
        forest_types = pd.read_csv(forest_types_path)
        forest_codes = pd.read_csv(forest_codes_path)
        
        # Merge on the forest type name
        merged = pd.merge(
            forest_codes, 
            forest_types, 
            left_on='name', 
            right_on='ForestTypeMFE', 
            how='left'
        )
        
        # Handle non-forest code (0)
        if 0 not in merged['code'].values:
            non_forest_row = pd.DataFrame({
                'code': [0],
                'name': ['Non forest'],
                'ForestTypeMFE': ['Non forest'],
                'Species': ['Non forest'],
                'Genus': ['Non forest'],
                'Family': ['Non forest'],
                'Clade': ['Non forest']
            })
            merged = pd.concat([merged, non_forest_row], ignore_index=True)
        
        # Replace "Angiosperm" with "Broadleaved" in Clade column
        merged['Clade'] = merged['Clade'].replace('Angiosperm', 'Broadleaved')
        
        # Check for missing values and report
        missing_genus = merged[merged['Genus'].isna()]
        missing_clade = merged[merged['Clade'].isna()]
        
        if not missing_genus.empty or not missing_clade.empty:
            self.logger.warning(f"Found {len(missing_genus)} forest types without genus mapping")
            self.logger.warning(f"Found {len(missing_clade)} forest types without clade mapping")
            self.logger.warning(f"Forest types with missing mappings: {missing_genus['name'].tolist() if not missing_genus.empty else []}")
            
            # Fill missing values with the forest type name
            merged['Genus'] = merged['Genus'].fillna(merged['name'])
            merged['Clade'] = merged['Clade'].fillna('Unknown')
            merged['Clade'] = merged['Clade'].replace('Angiosperm', 'Broadleaved')  # Apply replacement again in case of filled values
        
        self.logger.info(f"Loaded and merged mapping for {len(merged)} forest types")
        return merged[['code', 'name', 'Genus', 'Clade']]

    def get_forest_type_masks(self) -> List[str]:
        """
        Get list of forest type mask files from configuration.
        
        Returns:
            List of forest type mask file paths
        """
        mask_dir = FOREST_TYPE_MASKS_DIR
        
        mask_files = glob(os.path.join(mask_dir, "forest_type_mask_100m_code*.tif"))
        
        if not mask_files:
            self.logger.error(f"No forest type mask files found in {mask_dir}")
            return []
        
        self.logger.info(f"Found {len(mask_files)} forest type mask files")
        return mask_files

    def calculate_biomass_by_forest_type(self, year: int, forest_type_masks: List[str], forest_type_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate total biomass by forest type for a specific year.
                
        Args:
            year: Year to process
            forest_type_masks: List of forest type mask file paths
            forest_type_df: DataFrame with forest type mappings
            
        Returns:
            DataFrame with biomass by forest type with genus and clade info
        """
        self.logger.info(f"Processing biomass for year {year}...")
        
        # Build biomass file path
        biomass_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR / "mean" 
        biomass_file = os.path.join(biomass_dir, f"TBD_S2_mean_{year}_100m_TBD_merged.tif")
        
        if not os.path.exists(biomass_file):
            self.logger.error(f"Biomass file not found for year {year}: {biomass_file}")
            return None
        
        # Load biomass raster
        with rasterio.open(biomass_file) as src:
            biomass = src.read(1)
            # Check for no-data values and replace with 0
            nodata = src.nodata
            if nodata is not None:
                biomass = np.where(biomass == nodata, 0, biomass)
            
            # Convert to float if needed
            if biomass.dtype != np.float32 and biomass.dtype != np.float64:
                biomass = biomass.astype(np.float32)
                
            self.logger.info(f"Loaded biomass raster for {year}, shape: {biomass.shape}, dtype: {biomass.dtype}")
            
            # Initialize results container
            results = []
            
            # Process each forest type mask
            for mask_file in forest_type_masks:
                # Extract forest type code from filename
                try:
                    code = int(os.path.basename(mask_file).split('_code')[1].split('.')[0])
                except (ValueError, IndexError):
                    self.logger.warning(f"Could not extract forest type code from {mask_file}")
                    continue
                
                # Find mapping for this code
                mapping = forest_type_df[forest_type_df['code'] == code]
                if mapping.empty:
                    self.logger.warning(f"No mapping found for forest type code {code}")
                    continue
                
                mapping_row = mapping.iloc[0]
                
                # Load forest type mask
                with rasterio.open(mask_file) as mask_src:
                    mask = mask_src.read(1)
                    
                    # Check dimensions match
                    if mask.shape != biomass.shape:
                        self.logger.error(f"Shape mismatch: biomass {biomass.shape} vs mask {mask.shape} for code {code}")
                        continue
                    
                    # Apply mask (mask should be 1 where forest type exists)
                    masked_biomass = np.where(mask == 1, biomass, 0)
                    
                    # Calculate total biomass for this forest type (in tonnes)
                    pixel_area_ha = self.config['analysis']['pixel_area_ha']
                    total_biomass = np.sum(masked_biomass) * pixel_area_ha
                    
                    # Store result
                    results.append({
                        'Year': year,
                        'ForestType_Code': code,
                        'ForestType_Name': mapping_row['name'],
                        'Genus': mapping_row['Genus'],
                        'Clade': mapping_row['Clade'],
                        'Total_Biomass_Tonnes': total_biomass
                    })
                    
                    self.logger.info(f"  Forest type {code} ({mapping_row['name']}): {total_biomass:.2f} tonnes")
            
            # Convert to DataFrame
            if results:
                return pd.DataFrame(results)
            else:
                self.logger.warning(f"No forest type results for year {year}")
                return None

    def aggregate_results(self, all_forest_type_results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create aggregated results by genus and clade.
                
        Args:
            all_forest_type_results: DataFrame with all forest type results
            
        Returns:
            Tuple of (genus_df, clade_df) with aggregated results
        """
        if all_forest_type_results.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        self.logger.info("Creating aggregated results by genus and clade...")
        
        # Aggregate by genus (keeping clade info)
        genus_df = all_forest_type_results.groupby(['Year', 'Genus', 'Clade'])['Total_Biomass_Tonnes'].sum().reset_index()
        genus_df = genus_df.sort_values(['Year', 'Genus'])
        
        # Aggregate by clade only
        clade_df = all_forest_type_results.groupby(['Year', 'Clade'])['Total_Biomass_Tonnes'].sum().reset_index()
        clade_df = clade_df.sort_values(['Year', 'Clade'])
        
        self.logger.info(f"Created genus aggregation: {len(genus_df)} rows")
        self.logger.info(f"Created clade aggregation: {len(clade_df)} rows")
        
        return genus_df, clade_df

    
    def calculate_biomass_by_landcover(self, target_years: List[int]) -> List[Dict[str, Any]]:
        """
        Calculate total biomass for each landcover group and year.
                
        Args:
            target_years: List of years to process
            
        Returns:
            List of result dictionaries
        """
        biomass_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR / "mean" 
        corine_raster_path = CORINE_LAND_COVER_FILE
        
        # Get landcover groups from config
        landcover_groups = self.config['landcover']['groups']
        pixel_area_ha = self.config['analysis']['pixel_area_ha']
        
        self.logger.info(f"Processing biomass by landcover for years: {target_years}")
        self.logger.info(f"Landcover groups: {list(landcover_groups.keys())}")
        self.logger.info(f"Using Corine raster: {corine_raster_path}")
        
        # Check if Corine raster exists
        if not os.path.exists(corine_raster_path):
            self.logger.error(f"Corine raster file not found: {corine_raster_path}")
            return []
        
        results = []
        
        # Open Corine landcover raster
        with rasterio.open(corine_raster_path) as corine_src:
            corine_crs = corine_src.crs
            
            # Process each year
            for year in target_years:
                self.logger.info(f"Processing year: {year}")
                
                # Load biomass data
                biomass_file = f"TBD_S2_mean_{year}_100m_TBD_merged.tif"
                biomass_path = os.path.join(biomass_dir, biomass_file)
                
                if not os.path.exists(biomass_path):
                    self.logger.error(f"Biomass file not found: {biomass_file}")
                    continue
                
                with rasterio.open(biomass_path) as biomass_src:
                    biomass_data = biomass_src.read(1)
                    biomass_nodata = biomass_src.nodata
                    
                    # Create mask for NoData in biomass
                    if biomass_nodata is not None:
                        biomass_nodata_mask = (biomass_data == biomass_nodata)
                    else:
                        biomass_nodata_mask = np.zeros_like(biomass_data, dtype=bool)
                    
                    # Check if CRS match
                    if corine_src.crs != biomass_src.crs:
                        self.logger.info(f"Reprojecting Corine data from {corine_src.crs} to {biomass_src.crs}")
                        
                        # Reproject Corine to match biomass CRS and shape
                        corine_resized = np.zeros(biomass_data.shape, dtype=corine_src.dtypes[0])
                        
                        reproject(
                            source=rasterio.band(corine_src, 1),
                            destination=corine_resized,
                            src_transform=corine_src.transform,
                            src_crs=corine_src.crs,
                            dst_transform=biomass_src.transform,
                            dst_crs=biomass_src.crs,
                            dst_nodata = 255,
                            resampling=Resampling.nearest
                        )
                        
                        corine_data = corine_resized
                    else:
                        logger.info("  Aligning Corine data with biomass grid...")
                        target_bounds = biomass_src.bounds
                        window = rasterio.windows.from_bounds(*target_bounds, corine_src.transform)
                        corine_data = corine_src.read(1, window=window, boundless=True, fill_value=0)
                        
                        if corine_data.shape != biomass_data.shape:
                            corine_resized = np.zeros(biomass_data.shape, dtype=rasterio.uint8)
                            window_transform = rasterio.windows.transform(window, corine_src.transform)
                            
                            reproject(
                                source=corine_data,
                                destination=corine_resized,
                                src_transform=window_transform,
                                src_crs=corine_crs,
                                dst_transform=biomass_transform,
                                dst_crs=biomass_crs,
                                dst_nodata=0,
                                resampling=Resampling.nearest
                            )

                            corine_data = corine_resized
                    
                    corine_nodata = corine_src.nodata
                    corine_nodata_mask = (corine_data == corine_nodata) if corine_nodata is not None else np.zeros_like(corine_data, dtype=bool)
                
                # Process each landcover group
                for group_name, class_values in landcover_groups.items():
                    self.logger.info(f"  Processing landcover group: {group_name}")
                    
                    # Create mask for this landcover group
                    group_mask = np.isin(corine_data, class_values)
                    
                    # Apply both masks to get valid biomass values for this landcover group
                    combined_mask = group_mask & ~biomass_nodata_mask & ~corine_nodata_mask
                    
                    # Calculate total biomass (tonnes per hectare * hectares = tonnes)
                    # Since each pixel is 1 hectare (100m x 100m), we can sum directly
                    total_biomass = np.sum(biomass_data[combined_mask]) * pixel_area_ha
                    
                    # Store result
                    results.append({
                        'landcover_group': group_name,
                        'year': year,
                        'biomass': total_biomass
                    })
                    
                    self.logger.info(f"    Landcover group {group_name}: {total_biomass:.2f} tonnes")
        
        return results

    
    def create_height_masks_for_year(self, year: int) -> bool:
        """
        Creates binary masks for different vegetation height ranges for a specific year.
        
        Args:
            year: Year to process
            
        Returns:
            bool: True if masks were created successfully, False otherwise
        """
        height_dir = HEIGHT_MAPS_100M_DIR
        mask_dir = HEIGHT_MASK_BINS_DIR
        
        # Get height ranges from config
        height_bins = self.config['height_ranges']['bins']
        height_labels = self.config['height_ranges']['labels']
        
        # Build height file path
        height_file_pattern = self.config['height_ranges']['height_file_pattern']
        height_file = height_file_pattern.format(year=year)
        height_path = os.path.join(height_dir, height_file)
        
        # Check if height file exists
        if not os.path.exists(height_path):
            self.logger.error(f"Height file not found for year {year}: {height_path}")
            return False
        
        # Ensure mask output directory exists
        os.makedirs(mask_dir, exist_ok=True)
        
        self.logger.info(f"Creating height masks for year {year}")
        
        # Read input height raster
        with rasterio.open(height_path) as src:
            height_data = src.read(1)
            profile = src.profile.copy()
            
            # Handle NoData values
            if src.nodata is not None:
                height_data = np.where(height_data == src.nodata, np.nan, height_data)
            
            # Create masks for each height range
            for i, label in enumerate(height_labels):
                if i < len(height_bins) - 1:
                    # Regular range
                    min_height = height_bins[i]
                    max_height = height_bins[i + 1]
                    mask = (height_data >= min_height) & (height_data < max_height)
                    self.logger.info(f"  Height range {label}: {min_height} <= height < {max_height}")
                else:
                    # Last range (20m+)
                    min_height = height_bins[i]
                    mask = (height_data >= min_height)
                    self.logger.info(f"  Height range {label}: height >= {min_height}")
                
                # Convert boolean mask to integer (1 for True, 0 for False)
                mask_data = mask.astype(np.uint8)
                
                # Set NoData pixels to 0
                mask_data = np.where(np.isnan(height_data), 255, mask_data)
                
                # Update profile for output
                profile.update(dtype=rasterio.uint8, nodata=255)
                
                # Save mask
                output_file = os.path.join(mask_dir, f"height_{label}_{year}.tif")
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(mask_data, 1)
                
                self.logger.info(f"    Saved mask: {output_file}")
        
        return True

    def calculate_biomass_by_height_ranges(self, target_years: List[int]) -> List[Dict[str, Any]]:
        """
        Calculate total biomass for each height range and year.
                
        Args:
            target_years: List of years to process
            
        Returns:
            List of result dictionaries
        """
        biomass_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR
        mask_dir = HEIGHT_MAPS_BIN_MASKS_DIR
        height_labels = self.config['height_ranges']['labels']
        pixel_area_ha = self.config['analysis']['pixel_area_ha']
        
        results = []
        
        # Process each year
        for year in target_years:
            self.logger.info(f"Processing biomass by height ranges for year {year}")
            
            # Load biomass data
            biomass_file = f"TBD_S2_mean_{year}_100m_TBD_merged.tif"
            biomass_path = os.path.join(biomass_dir, biomass_file)
            
            if not os.path.exists(biomass_path):
                self.logger.error(f"Biomass file not found: {biomass_file}")
                continue
            
            with rasterio.open(biomass_path) as biomass_src:
                biomass_data = biomass_src.read(1)
                biomass_nodata = biomass_src.nodata
                
                # Create mask for NoData
                if biomass_nodata is not None:
                    biomass_nodata_mask = (biomass_data == biomass_nodata)
                else:
                    biomass_nodata_mask = np.zeros_like(biomass_data, dtype=bool)
                
                # Process each height range
                for height_range in height_labels:
                    height_mask_file = f"height_{height_range}_{year}.tif"
                    height_mask_path = os.path.join(mask_dir, height_mask_file)
                    
                    if not os.path.exists(height_mask_path):
                        self.logger.error(f"Height mask file not found: {height_mask_file}")
                        continue
                    
                    with rasterio.open(height_mask_path) as mask_src:
                        mask_data = mask_src.read(1)
                        mask_nodata = mask_src.nodata
                        
                        # Create mask for valid height data
                        if mask_nodata is not None:
                            height_mask = (mask_data == 1) & (mask_data != mask_nodata)
                        else:
                            height_mask = (mask_data == 1)
                        
                        # Apply both masks to get valid biomass values for this height range
                        combined_mask = height_mask & ~biomass_nodata_mask
                        
                        # Calculate total biomass (tonnes per hectare * hectares = tonnes)
                        # Since each pixel is 1 hectare, we can sum directly
                        total_biomass = np.sum(biomass_data[combined_mask]) * pixel_area_ha
                        
                        # Store result
                        results.append({
                            'height_bin': height_range,
                            'year': year,
                            'biomass': total_biomass
                        })
                        
                        self.logger.info(f"  Height range {height_range}: {total_biomass:.2f} tonnes")
        
        return results

    def check_and_create_height_masks(self, target_years: List[int]) -> bool:
        """
        Check if height range masks exist for target years and create them if needed.
        
        Args:
            target_years: List of years to process
            
        Returns:
            bool: True if all masks are available, False otherwise
        """
        mask_dir = HEIGHT_MAPS_BIN_MASKS_DIR
        height_labels = self.config['height_ranges']['labels']
        
        all_masks_available = True
        
        for year in target_years:
            year_masks_exist = True
            
            # Check if all height range masks exist for this year
            for label in height_labels:
                mask_file = f"height_{label}_{year}.tif"
                mask_path = os.path.join(mask_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    year_masks_exist = False
                    break
            
            # If masks don't exist for this year, create them
            if not year_masks_exist:
                self.logger.info(f"Height masks missing for year {year}, creating them...")
                success = self.create_height_masks_for_year(year)
                if not success:
                    self.logger.error(f"Failed to create height masks for year {year}")
                    all_masks_available = False
            else:
                self.logger.info(f"All height masks exist for year {year}")
        
        return all_masks_available

    
    def run_forest_type_analysis(self, target_years: Optional[List[int]] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Run complete forest type analysis with hierarchical aggregation.
        
        Args:
            target_years: List of years to process (default from config)
            
        Returns:
            Tuple of (forest_type_df, genus_df, clade_df)
        """
        if target_years is None:
            target_years = self.config['analysis']['target_years']
        
        self.logger.info("Starting biomass by forest type calculation...")
        
        # Load and merge forest type mappings
        forest_type_df = self.load_and_merge_forest_type_mappings()
        
        # Get list of forest type masks
        mask_files = self.get_forest_type_masks()
        if not mask_files:
            self.logger.error("No forest type mask files found. Exiting.")
            return None, None, None
        
        # Initialize DataFrame to store all years' results
        all_forest_type_results = []
        
        # Process each year
        for year in target_years:
            forest_type_results = self.calculate_biomass_by_forest_type(year, mask_files, forest_type_df)
            
            if forest_type_results is not None:
                all_forest_type_results.append(forest_type_results)
        
        # Combine results from all years
        if all_forest_type_results:
            # Combine all forest type results
            combined_forest_type_df = pd.concat(all_forest_type_results, ignore_index=True)
            combined_forest_type_df = combined_forest_type_df.sort_values(['Year', 'ForestType_Code'])
            
            # Create aggregated results by genus and clade
            genus_df, clade_df = self.aggregate_results(combined_forest_type_df)
            
            return combined_forest_type_df, genus_df, clade_df
        else:
            self.logger.error("No results were calculated.")
            return None, None, None

    def run_landcover_analysis(self, target_years: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Run complete landcover analysis.
        
        Args:
            target_years: List of years to process (default from config)
            
        Returns:
            List of result dictionaries
        """
        if target_years is None:
            target_years = self.config['analysis']['target_years']
        
        self.logger.info("Starting landcover biomass analysis...")
        return self.calculate_biomass_by_landcover(target_years)

    def run_height_bin_analysis(self, target_years: Optional[List[int]] = None, skip_mask_creation: bool = False) -> List[Dict[str, Any]]:
        """
        Run complete height bin analysis with optional mask creation.
        
        Args:
            target_years: List of years to process (default from config)
            skip_mask_creation: Skip mask creation and use existing masks only
            
        Returns:
            List of result dictionaries
        """
        if target_years is None:
            target_years = self.config['analysis']['target_years']
        
        self.logger.info("Starting height range biomass analysis...")
        self.logger.info(f"Height ranges: {self.config['height_ranges']['labels']}")
        
        # Check and create height masks if needed
        if not skip_mask_creation:
            masks_available = self.check_and_create_height_masks(target_years)
            if not masks_available:
                self.logger.error("Some height masks could not be created. Continuing with available masks...")
        else:
            self.logger.info("Skipping mask creation - using existing masks only")
        
        # Calculate biomass by height ranges
        return self.calculate_biomass_by_height_ranges(target_years)

    
    def save_results(self, results_data: Dict[str, Any], analysis_type: str) -> str:
        """
        Save aggregation analysis results to CSV files.
        
        Args:
            results_data: Dictionary containing results data
            analysis_type: Type of analysis ('forest_type', 'landcover', 'height_bin')
            
        Returns:
            Path to main output file
        """

        output_dir = BIOMASS_STOCKS_DIR
        os.makedirs(output_dir, exist_ok=True)

        if analysis_type == 'forest_type':
            
            output_dir = BIOMASS_PER_FOREST_TYPE_DIR
            os.makedirs(output_dir, exist_ok=True)

            forest_type_df = results_data['forest_type']
            genus_df = results_data['genus']
            clade_df = results_data['clade']
            
            # Save forest type level results
            forest_type_file = BIOMASS_PER_FOREST_TYPE_FILE
            forest_type_df.to_csv(forest_type_file, index=False)
            self.logger.info(f"Forest type results exported to {forest_type_file}")
            
            # Save genus level results (matching original naming)
            genus_file = BIOMASS_PER_GENUS_FILE
            genus_df.to_csv(genus_file, index=False)
            self.logger.info(f"Genus results exported to {genus_file}")
            
            # Save clade level results (matching original naming)
            clade_file = BIOMASS_PER_CLADE_FILE
            clade_df.to_csv(clade_file, index=False)
            self.logger.info(f"Clade results exported to {clade_file}")
            
            return forest_type_file
            
        elif analysis_type == 'landcover':

            output_dir = BIOMASS_PER_LAND_COVER_DIR
            os.makedirs(output_dir, exist_ok=True)

            results = results_data['results']
            output_csv = BIOMASS_PER_LAND_COVER_FILE
            
            # Write results to CSV
            with open(output_csv, 'w', newline='') as csvfile:
                if results:
                    fieldnames = ['landcover_group', 'year', 'biomass']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
            
            self.logger.info(f"Results saved to: {output_csv}")
            return output_csv
            
        elif analysis_type == 'height_bin':

            output_dir = BIOMASS_PER_HEIGHT_BIN_DIR
            os.makedirs(output_dir, exist_ok=True)

            results = results_data['results']
            output_csv = BIOMASS_PER_HEIGHT_BIN_FILE
            
            # Write results to CSV
            with open(output_csv, 'w', newline='') as csvfile:
                if results:
                    fieldnames = ['height_bin', 'year', 'biomass']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
            
            self.logger.info(f"Results saved to: {output_csv}")
            return output_csv
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
