"""
Integrated Post-processing Pipeline for Sentinel-2 Mosaics

This module provides comprehensive post-processing capabilities for Sentinel-2 mosaics including:
- Spatial downsampling and merging operations for analysis-ready products
- Quality assurance through missing tile detection and gap analysis
- Robustness assessment for optimal scene selection parameters
- Interannual consistency analysis for temporal stability validation

Each specialized class preserves exact algorithmic logic from original implementations
while integrating with shared utilities and configuration management. All performance-
critical operations including statistical sampling, resampling methods, and pattern
matching are maintained exactly to ensure consistent results.

Author: Diego Bengochea
"""

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import dask.array as da
from pystac_client import Client

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory

# Component utilities - using refactored imports
from .s2_utils import mask_scene


class DownsamplingMergingProcessor:
    """
    Processor for spatial downsampling and merging operations.
    
    Handles the conversion of raw mosaic tiles into analysis-ready products through
    two sequential operations: downsampling (reduces spatial resolution) and merging
    (combines tiles into yearly country-wide mosaics). Preserves exact rasterio
    parameters and compression settings from original implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize downsampling and merging processor.
        
        Args:
            config: Configuration dictionary with postprocessing parameters
        """
        self.config = config
        self.logger = get_logger('sentinel2_processing')
        
        # Extract processing parameters with defaults
        self.postprocessing_config = self.config.get('postprocessing', {})
        self.downsample_config = self.postprocessing_config.get('downsample', {})
        self.merge_config = self.postprocessing_config.get('merge', {})
        
        # Processing statistics
        self.stats = {
            'downsample': {'successful': 0, 'failed': 0},
            'merge': {'successful': 0, 'failed': 0}
        }
        
        self.logger.info("DownsamplingMergingProcessor initialized")
    
    def downsample_rasters(self, input_dir: str, output_dir: str, scale_factor: Optional[int] = None) -> Dict[str, int]:
        """
        Downsample raster files by a given scale factor.
        
        CRITICAL: Preserves exact rasterio resampling parameters, compression settings,
        and file pattern matching from original implementation to ensure consistent
        spatial processing results.
        
        Args:
            input_dir: Path to directory containing input raster files
            output_dir: Path to directory where downsampled files will be saved
            scale_factor: Factor by which to downsample (uses config default if None)
            
        Returns:
            dict: Processing statistics with success/failure counts
            
        Examples:
            >>> processor = DownsamplingMergingProcessor(config)
            >>> stats = processor.downsample_rasters("/input", "/output", 10)
        """
        self.logger.info("Starting raster downsampling...")
        
        # Use provided scale factor or config default
        if scale_factor is None:
            scale_factor = self.downsample_config.get('scale_factor', 10)
        
        # Create output directory using shared utilities
        ensure_directory(output_dir)
        
        # CRITICAL: Preserve exact file pattern matching from original
        pattern = re.compile(r'S2_summer_mosaic_\d{4}_.*\.tif$', re.IGNORECASE)
        
        # Get all files matching the pattern
        raster_files = [
            f for f in os.listdir(input_dir) 
            if pattern.match(f)
        ]
        
        if not raster_files:
            self.logger.warning("No files found matching the pattern 'S2_summer_mosaic_YYYY_*.tif'")
            return {'successful': 0, 'failed': 0}
        
        self.logger.info(f"Found {len(raster_files)} raster files to downsample")
        
        # CRITICAL: Preserve exact progress bar setup
        pbar = tqdm(raster_files, desc="Processing rasters", unit="file")
        
        successful = 0
        failed = 0
        
        for raster_file in pbar:
            # Update progress bar description - PRESERVE exact format
            pbar.set_description(f"Processing {raster_file}")
            
            input_path = os.path.join(input_dir, raster_file)
            
            # CRITICAL: Preserve exact output filename generation
            filename = Path(raster_file).stem
            output_filename = f"{filename}_downsampled.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            if not os.path.exists(output_path):
                try:
                    with rasterio.open(input_path) as dataset:
                        # CRITICAL: Preserve exact dimension calculations
                        new_height = dataset.height // scale_factor
                        new_width = dataset.width // scale_factor
                        
                        # CRITICAL: Preserve exact transform calculation
                        transform = dataset.transform * dataset.transform.scale(
                            (dataset.width / new_width),
                            (dataset.height / new_height)
                        )
                        
                        # CRITICAL: Preserve exact profile setup
                        profile = dataset.profile.copy()
                        profile.update({
                            'height': new_height,
                            'width': new_width,
                            'transform': transform
                        })
                        
                        # CRITICAL: Preserve exact resampling method and data reading
                        data = dataset.read(
                            out_shape=(dataset.count, new_height, new_width),
                            resampling=Resampling.average
                        )
                        
                        # Write output file with preserved settings
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            dst.write(data)
                    
                    successful += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {raster_file}: {str(e)}")
                    failed += 1
                    continue
            else:
                self.logger.info(f"Skipping {raster_file} (already processed)")
                successful += 1
        
        pbar.close()
        self.logger.info(f"Downsampling completed! Successful: {successful}, Failed: {failed}")
        
        # Update internal statistics
        self.stats['downsample'] = {'successful': successful, 'failed': failed}
        
        return {'successful': successful, 'failed': failed}
    
    def group_rasters_by_year(self, input_dir: str) -> Dict[str, List[str]]:
        """
        Group raster files by year based on filename pattern.
        
        CRITICAL: Preserves exact regex pattern and grouping logic from original.
        
        Args:
            input_dir: Directory containing the downsampled raster files
            
        Returns:
            dict: Dictionary with years as keys and lists of file paths as values
            
        Examples:
            >>> groups = processor.group_rasters_by_year("/downsampled/dir")
        """
        # CRITICAL: Preserve exact regex pattern for downsampled files
        pattern = re.compile(r'S2_summer_mosaic_(\d{4})_.*_downsampled\.tif$', re.IGNORECASE)
        raster_groups = defaultdict(list)
        
        for file in os.listdir(input_dir):
            match = pattern.match(file)
            if match:
                year = match.group(1)
                raster_groups[year].append(os.path.join(input_dir, file))
                
        return dict(raster_groups)
    
    def merge_rasters_by_year(self, input_dir: str, output_dir: str) -> Dict[str, int]:
        """
        Merge raster files grouped by year into single country-wide mosaics.
        
        CRITICAL: Preserves exact merging parameters, compression settings, and
        metadata handling from original implementation to ensure consistent
        spatial products.
        
        Args:
            input_dir: Directory containing the downsampled raster files
            output_dir: Directory where merged rasters will be saved
            
        Returns:
            dict: Processing statistics with success/failure counts
            
        Examples:
            >>> stats = processor.merge_rasters_by_year("/downsampled", "/merged")
        """
        self.logger.info("Starting raster merging by year...")
        
        # Create output directory using shared utilities
        ensure_directory(output_dir)
        
        # Group rasters by year using exact logic
        raster_groups = self.group_rasters_by_year(input_dir)
        
        if not raster_groups:
            self.logger.warning("No raster files found matching the pattern 'S2_summer_mosaic_YYYY_*_downsampled.tif'")
            return {'successful': 0, 'failed': 0}
        
        self.logger.info(f"Found raster groups for years: {list(raster_groups.keys())}")
        
        successful = 0
        failed = 0
        
        # CRITICAL: Preserve exact year processing loop with tqdm
        for year, raster_files in tqdm(raster_groups.items(), desc="Processing years"):
            output_path = os.path.join(output_dir, f'S2_summer_mosaic_{year}_merged.tif')
            
            if not os.path.exists(output_path):
                try:
                    self.logger.info(f"Merging {len(raster_files)} rasters for year {year}")
                    
                    # CRITICAL: Preserve exact file opening sequence
                    src_files = []
                    for raster_path in raster_files:
                        src = rasterio.open(raster_path)
                        src_files.append(src)
                    
                    # CRITICAL: Preserve exact merge operation
                    mosaic, out_transform = merge(src_files)
                    
                    # CRITICAL: Preserve exact metadata handling
                    out_meta = src_files[0].meta.copy()
                    
                    # CRITICAL: Preserve exact metadata updates including compression
                    out_meta.update({
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_transform,
                        "compress": "LZW"  # Preserve exact compression setting
                    })
                    
                    # Write merged raster with preserved settings
                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(mosaic)
                    
                    self.logger.info(f"Successfully merged {len(raster_files)} rasters for year {year}")
                    successful += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing year {year}: {str(e)}")
                    failed += 1
                    continue
                    
                finally:
                    # CRITICAL: Preserve exact cleanup sequence
                    for src in src_files:
                        src.close()
            else:
                self.logger.info(f"Skipping year {year} (already processed)")
                successful += 1
        
        self.logger.info(f"Merging completed! Successful: {successful}, Failed: {failed}")
        
        # Update internal statistics
        self.stats['merge'] = {'successful': successful, 'failed': failed}
        
        return {'successful': successful, 'failed': failed}
    
    def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete downsampling and merging workflow.
        
        Returns:
            dict: Complete processing statistics and summary
            
        Examples:
            >>> processor = DownsamplingMergingProcessor(config)
            >>> results = processor.run_complete_workflow()
        """
        self.logger.info("Starting complete downsampling and merging workflow...")
        
        try:
            # Get directories from configuration
            input_directory = self.config['paths']['output_dir']
            downsampled_directory = self.config['paths']['downsampled_dir']
            merged_directory = self.config['paths']['merged_dir']
            
            # Step 1: Downsample rasters
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 1: DOWNSAMPLING RASTERS")
            self.logger.info("="*50)
            
            downsample_stats = self.downsample_rasters(input_directory, downsampled_directory)
            
            # Step 2: Merge rasters by year
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 2: MERGING RASTERS BY YEAR")
            self.logger.info("="*50)
            
            merge_stats = self.merge_rasters_by_year(downsampled_directory, merged_directory)
            
            # Final summary
            self.logger.info("\n" + "="*50)
            self.logger.info("POST-PROCESSING SUMMARY")
            self.logger.info("="*50)
            self.logger.info(f"Downsampling: {downsample_stats['successful']} successful, {downsample_stats['failed']} failed")
            self.logger.info(f"Merging: {merge_stats['successful']} successful, {merge_stats['failed']} failed")
            
            self.logger.info("Downsampling and merging workflow completed successfully!")
            
            return {
                'downsampling': downsample_stats,
                'merging': merge_stats,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            return {
                'downsampling': {'successful': 0, 'failed': 0},
                'merging': {'successful': 0, 'failed': 0},
                'success': False,
                'error': str(e)
            }


class MissingTilesAnalyzer:
    """
    Analyzer for detecting missing tile-year combinations and processing gaps.
    
    Provides comprehensive quality assurance by scanning output directories to identify
    missing spatial-temporal combinations, calculating completeness statistics, and
    generating detailed reports for gap analysis. Preserves exact pattern matching
    and statistical calculation logic from original implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize missing tiles analyzer.
        
        Args:
            config: Configuration dictionary with path information
        """
        self.config = config
        self.logger = get_logger('sentinel2_processing')
        self.logger.info("MissingTilesAnalyzer initialized")
    
    def analyze_missing_files(self, directory_path: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """
        Analyze directory to identify missing location-year combinations.
        
        CRITICAL: Preserves exact regex patterns, year filtering, and location
        extraction logic from original implementation to ensure consistent
        gap detection and analysis.
        
        Args:
            directory_path: Path to the directory containing raster files
            
        Returns:
            tuple: (missing_files_dict, all_years_list, missing_file_paths_list)
            
        Examples:
            >>> analyzer = MissingTilesAnalyzer(config)
            >>> missing, years, paths = analyzer.analyze_missing_files("/output/dir")
        """
        self.logger.info(f"Analyzing missing files in directory: {directory_path}")
        
        # Get all files in the directory
        files = os.listdir(directory_path)
        
        # CRITICAL: Preserve exact year pattern from original
        year_pattern = r'S2_summer_mosaic_(\d{4})'
        location_to_years = defaultdict(set)
        all_years = set()
        
        # CRITICAL: Preserve exact file storage structure
        location_files = defaultdict(dict)
        
        for file in files:
            # CRITICAL: Preserve exact year matching and filtering
            year_match = re.search(year_pattern, file)
            if year_match:
                year = year_match.group(1)
                # CRITICAL: Preserve exact 2024 exclusion logic
                if year == '2024':
                    continue
                all_years.add(year)
                
                # CRITICAL: Preserve exact location identifier extraction
                location = re.sub(year_pattern, '', file)
                location_to_years[location].add(year)
                location_files[location][year] = file

        self.logger.info(f"Found {len(files)} total files")
        self.logger.info(f"Valid years found: {sorted(all_years)}")
        self.logger.info(f"Unique locations: {len(location_to_years)}")

        # CRITICAL: Preserve exact missing file detection and path generation
        missing_files = {}
        missing_file_paths = []
        
        for location, years in location_to_years.items():
            missing_years = all_years - years
            if missing_years:
                # CRITICAL: Preserve exact template file selection
                example_year = next(iter(years))
                example_filename = location_files[location][example_year]
                
                # CRITICAL: Preserve exact missing years storage structure
                missing_files[location] = {
                    'missing_years': sorted(list(missing_years)),
                    'existing_files': {
                        year: location_files[location][year]
                        for year in years
                    }
                }
                
                # CRITICAL: Preserve exact hypothetical path generation
                for missing_year in missing_years:
                    missing_filename = re.sub(
                        example_year,
                        missing_year,
                        example_filename
                    )
                    missing_file_paths.append(os.path.join(directory_path, missing_filename))
        
        return missing_files, sorted(list(all_years)), sorted(missing_file_paths)
    
    def calculate_statistics(self, missing_files: Dict[str, Any], all_years: List[str], location_to_years: Dict[str, set]) -> Dict[str, Any]:
        """
        Calculate processing completeness statistics.
        
        CRITICAL: Preserves exact statistical calculation formulas from original.
        
        Args:
            missing_files: Missing files information
            all_years: All years found
            location_to_years: Mapping of locations to years
            
        Returns:
            dict: Statistics dictionary with completeness metrics
            
        Examples:
            >>> stats = analyzer.calculate_statistics(missing_files, years, location_map)
        """
        # CRITICAL: Preserve exact statistical calculations
        total_locations = len(location_to_years)
        total_expected = total_locations * len(all_years)
        total_existing = sum(len(years) for years in location_to_years.values())
        total_missing = len([path for data in missing_files.values() for _ in data['missing_years']])
        completeness_rate = (total_existing / total_expected) * 100 if total_expected > 0 else 0
        
        return {
            'total_locations': total_locations,
            'total_years': len(all_years),
            'total_expected': total_expected,
            'total_existing': total_existing,
            'total_missing': total_missing,
            'completeness_rate': completeness_rate
        }
    
    def print_report(self, missing_files: Dict[str, Any], all_years: List[str], missing_file_paths: List[str]) -> None:
        """
        Print formatted report of missing files.
        
        CRITICAL: Preserves exact report formatting and content from original.
        
        Args:
            missing_files: Missing files information
            all_years: All years found in directory
            missing_file_paths: List of missing file paths
        """
        self.logger.info(f"\nAnalysis Report")
        self.logger.info(f"Years found in directory: {', '.join(all_years)}")
        self.logger.info(f"\nLocations with missing files:")
        self.logger.info("-" * 50)
        
        if not missing_files:
            self.logger.info("No missing files found. All locations present in all years.")
            return
            
        for location, data in missing_files.items():
            self.logger.info(f"\nLocation pattern: {location}")
            self.logger.info(f"Missing in years: {', '.join(data['missing_years'])}")
            self.logger.info("Existing files:")
            for year, filename in data['existing_files'].items():
                self.logger.info(f"  {year}: {filename}")
        
        self.logger.info("\nFirst 10 missing file paths:")
        self.logger.info("-" * 50)
        for i, path in enumerate(missing_file_paths[:10]):
            self.logger.info(f"{i+1}. {path}")
        
        if len(missing_file_paths) > 10:
            self.logger.info(f"... and {len(missing_file_paths) - 10} more missing files")
    
    def save_missing_file_paths(self, missing_file_paths: List[str], output_file: str = "missing_file_paths.txt") -> None:
        """
        Save missing file paths to a text file.
        
        Args:
            missing_file_paths: List of missing file paths
            output_file: Output file name
        """
        with open(output_file, 'w') as f:
            for path in missing_file_paths:
                f.write(f"{path}\n")
        
        self.logger.info(f"Missing file paths saved to: {output_file}")
    
    def run_missing_analysis(self) -> Dict[str, Any]:
        """
        Execute complete missing tiles analysis workflow.
        
        Returns:
            dict: Analysis results with statistics and file information
            
        Examples:
            >>> analyzer = MissingTilesAnalyzer(config)
            >>> results = analyzer.run_missing_analysis()
        """
        self.logger.info("Starting Sentinel-2 missing mosaics analysis...")
        
        try:
            # Get directory path from config
            directory_path = self.config['paths']['output_dir']
            
            self.logger.info(f"Analyzing directory: {directory_path}")
            
            if not os.path.exists(directory_path):
                self.logger.error(f"Directory does not exist: {directory_path}")
                return {'success': False, 'error': 'Directory not found'}
            
            # Analyze missing files
            missing_files, all_years, missing_file_paths = self.analyze_missing_files(directory_path)
            
            # Calculate statistics - CRITICAL: Preserve exact calculation logic
            location_to_years = defaultdict(set)
            files = os.listdir(directory_path)
            year_pattern = r'S2_summer_mosaic_(\d{4})'
            
            for file in files:
                year_match = re.search(year_pattern, file)
                if year_match:
                    year = year_match.group(1)
                    if year != '2024':
                        location = re.sub(year_pattern, '', file)
                        location_to_years[location].add(year)
            
            stats = self.calculate_statistics(missing_files, all_years, location_to_years)
            
            # Print comprehensive report
            self.logger.info("\n" + "="*60)
            self.logger.info("PROCESSING COMPLETENESS ANALYSIS")
            self.logger.info("="*60)
            self.logger.info(f"Total locations: {stats['total_locations']}")
            self.logger.info(f"Years processed: {', '.join(all_years)}")
            self.logger.info(f"Expected files: {stats['total_expected']}")
            self.logger.info(f"Existing files: {stats['total_existing']}")
            self.logger.info(f"Missing files: {stats['total_missing']}")
            self.logger.info(f"Completeness rate: {stats['completeness_rate']:.1f}%")
            
            # Print detailed report
            self.print_report(missing_files, all_years, missing_file_paths)
            
            # Save missing file paths if any exist
            if missing_file_paths:
                self.save_missing_file_paths(missing_file_paths)
            
            self.logger.info("\nMissing files analysis completed successfully!")
            
            return {
                'success': True,
                'statistics': stats,
                'missing_files': missing_files,
                'all_years': all_years,
                'missing_file_paths': missing_file_paths
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}


class RobustnessAssessor:
    """
    Assessor for mosaic robustness with varying numbers of input scenes.
    
    Evaluates how mosaic quality changes with different numbers of input scenes
    to determine optimal scene selection parameters. Integrates with STAC catalog
    for scene loading and applies the same masking logic as main processing.
    Preserves exact statistical sampling and analysis algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize robustness assessor.
        
        Args:
            config: Configuration dictionary with robustness parameters
        """
        self.config = config
        self.logger = get_logger('sentinel2_processing')
        
        # Extract robustness configuration
        self.robustness_config = self.config.get('robustness', {})
        self.min_scenes = self.robustness_config.get('min_scenes', 5)
        self.max_scenes = self.robustness_config.get('max_scenes', 30)
        self.step = self.robustness_config.get('step', 5)
        self.n_samples = self.robustness_config.get('n_samples', 10)
        
        self.logger.info("RobustnessAssessor initialized")
    
    def assess_median_robustness(self, scenes: List, min_scenes: Optional[int] = None, 
                                max_scenes: Optional[int] = None, step: Optional[int] = None) -> Tuple[Dict[int, Dict[str, float]], Any]:
        """
        Assess the robustness of median mosaics with different numbers of scenes.
        
        CRITICAL: Preserves exact statistical sampling, median calculation, and
        analysis algorithms from original implementation to ensure consistent
        robustness assessment results.
        
        Args:
            scenes: List of xarray DataArrays representing masked scenes
            min_scenes: Minimum number of scenes to test (uses config if None)
            max_scenes: Maximum number of scenes to test (uses config if None)
            step: Step size for number of scenes (uses config if None)
            
        Returns:
            tuple: (results_dict, reference_mosaic) with statistical analysis
            
        Examples:
            >>> assessor = RobustnessAssessor(config)
            >>> results, reference = assessor.assess_median_robustness(scenes)
        """
        # Use provided parameters or config defaults
        min_scenes = min_scenes or self.min_scenes
        max_scenes = max_scenes or self.max_scenes
        step = step or self.step
        
        total_scenes = len(scenes)
        results = {}
        
        self.logger.info(f"Assessing robustness with {total_scenes} total scenes")
        self.logger.info(f"Testing scene counts from {min_scenes} to {min(max_scenes, total_scenes)} (step: {step})")
        
        # CRITICAL: Preserve exact scene stacking and reference creation
        stacked_scenes = xr.concat(scenes, dim="time")
        
        # Reference mosaic using all scenes - PRESERVE exact median calculation
        reference_mosaic = stacked_scenes.median(dim="time")
        self.logger.info("Created reference mosaic using all available scenes")
        
        # CRITICAL: Preserve exact scene count iteration and sampling logic
        for n_scenes in range(min_scenes, min(max_scenes, total_scenes), step):
            self.logger.info(f"Testing with {n_scenes} scenes...")
            
            # CRITICAL: Preserve exact multiple random sampling approach
            differences = []
            valid_pixel_counts = []
            
            # CRITICAL: Preserve exact number of random samples (10)
            for _ in range(10):
                # CRITICAL: Preserve exact random sampling without replacement
                sample_indices = np.random.choice(total_scenes, n_scenes, replace=False)
                sampled_scenes = stacked_scenes.isel(time=sample_indices)
                
                # CRITICAL: Preserve exact median mosaic creation
                sample_mosaic = sampled_scenes.median(dim="time")
                
                # CRITICAL: Preserve exact difference calculation from reference
                diff = abs(sample_mosaic - reference_mosaic)
                differences.append(float(diff.mean().values))
                
                # CRITICAL: Preserve exact valid pixel percentage calculation
                valid_pixels = (~sample_mosaic.isnull()).sum() / sample_mosaic.size
                valid_pixel_counts.append(float(valid_pixels))
            
            # CRITICAL: Preserve exact results structure and statistics
            results[n_scenes] = {
                'mean_difference': np.mean(differences),
                'std_difference': np.std(differences),
                'mean_valid_pixels': np.mean(valid_pixel_counts),
                'std_valid_pixels': np.std(valid_pixel_counts)
            }
            
            self.logger.info(f"  Mean difference: {results[n_scenes]['mean_difference']:.4f}")
            self.logger.info(f"  Valid pixels: {results[n_scenes]['mean_valid_pixels']*100:.1f}%")
        
        return results, reference_mosaic
    
    def load_and_mask_scene(self, item, config: Dict[str, Any]):
        """
        Load and mask a single scene using configuration parameters.
        
        CRITICAL: Preserves exact data loading parameters and masking integration
        with refactored s2_utils to ensure consistent scene processing.
        
        Args:
            item: STAC item representing a single scene
            config: Configuration dictionary with processing parameters
            
        Returns:
            Loaded and masked xarray.Dataset scene
            
        Examples:
            >>> scene = assessor.load_and_mask_scene(stac_item, config)
        """
        from odc.stac import load
        
        # CRITICAL: Preserve exact data loading parameters
        scene = load(
            [item],
            bands=config['data']['bands'],
            chunks={'x': config['processing']['chunk_size'], 
                    'y': config['processing']['chunk_size']},
            resampling='bilinear'
        )
        
        # CRITICAL: Apply masking using refactored s2_utils function
        masked_scene, _ = mask_scene(scene, config['scl']['valid_classes'])
        
        # CRITICAL: Preserve exact band dropping
        masked_scene = masked_scene.drop_vars(config['data']['bands_drop'])
        
        return masked_scene
    
    def print_results(self, results: Dict[int, Dict[str, float]]) -> int:
        """
        Print formatted results of robustness assessment.
        
        CRITICAL: Preserves exact formatting and optimal scene detection logic.
        
        Args:
            results: Results dictionary from assess_median_robustness
            
        Returns:
            int: Recommended optimal number of scenes
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("ROBUSTNESS ASSESSMENT RESULTS")
        self.logger.info("="*60)
        
        for n_scenes, stats in results.items():
            self.logger.info(f"\nNumber of scenes: {n_scenes}")
            self.logger.info(f"  Mean difference from reference: {stats['mean_difference']:.4f} ± {stats['std_difference']:.4f}")
            self.logger.info(f"  Valid pixel percentage: {stats['mean_valid_pixels']*100:.1f}% ± {stats['std_valid_pixels']*100:.1f}%")
        
        # CRITICAL: Preserve exact optimal scene detection algorithm
        scene_counts = sorted(results.keys())
        differences = [results[n]['mean_difference'] for n in scene_counts]
        
        # CRITICAL: Preserve exact heuristic (< 5% improvement threshold)
        optimal_scenes = scene_counts[0]
        for i in range(1, len(scene_counts)):
            improvement = (differences[i-1] - differences[i]) / differences[i-1]
            if improvement < 0.05:  # Less than 5% improvement
                optimal_scenes = scene_counts[i-1]
                break
            optimal_scenes = scene_counts[i]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"RECOMMENDATION: Use at least {optimal_scenes} scenes for optimal mosaic quality")
        self.logger.info(f"{'='*60}")
        
        return optimal_scenes
    
    def run_robustness_assessment(self, bbox: Optional[List[float]] = None, 
                                 start_date: Optional[datetime] = None, 
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute complete robustness assessment workflow.
        
        Args:
            bbox: Bounding box for analysis area (uses example if None)
            start_date: Start date for scene search (uses example if None)
            end_date: End date for scene search (uses example if None)
            
        Returns:
            dict: Assessment results with optimal scene recommendation
            
        Examples:
            >>> assessor = RobustnessAssessor(config)
            >>> results = assessor.run_robustness_assessment()
        """
        self.logger.info("Starting Sentinel-2 mosaic robustness assessment...")
        
        try:
            # Use provided parameters or example defaults
            if bbox is None:
                bbox = [-122.34, 37.74, -122.26, 37.80]  # San Francisco Bay Area
                self.logger.info("Using example area: San Francisco Bay Area")
            
            if start_date is None or end_date is None:
                start_date = datetime(2022, 1, 1)
                end_date = datetime(2022, 12, 31)
                self.logger.info("Using example time period: 2022")
            
            self.logger.info("Modify these parameters for different areas/periods")
            
            # Initialize STAC catalog
            stac_url = self.config['data']['stac_url']
            client = Client.open(stac_url)
            
            # Search for scenes
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
            )
            
            items = list(search.get_items())
            self.logger.info(f"Found {len(items)} scenes matching criteria")
            
            if len(items) < self.min_scenes:
                raise ValueError(f"Not enough scenes found ({len(items)}). Minimum required: {self.min_scenes}")
            
            # Load and mask all scenes
            scenes = []
            self.logger.info("Loading and masking scenes...")
            
            for i, item in enumerate(items):
                try:
                    scene = self.load_and_mask_scene(item, self.config)
                    scenes.append(scene)
                    self.logger.info(f"Loaded scene {i+1}/{len(items)}")
                except Exception as e:
                    self.logger.warning(f"Failed to load scene {i+1}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully loaded {len(scenes)} scenes")
            
            if len(scenes) < self.min_scenes:
                raise ValueError(f"Not enough valid scenes loaded ({len(scenes)}). Minimum required: {self.min_scenes}")
            
            # Run robustness assessment
            results, reference = self.assess_median_robustness(scenes)
            
            # Print results and get recommendation
            optimal_scenes = self.print_results(results)
            
            self.logger.info("Robustness assessment completed successfully!")
            
            return {
                'success': True,
                'results': results,
                'optimal_scenes': optimal_scenes,
                'total_scenes_loaded': len(scenes),
                'bbox': bbox,
                'time_period': f"{start_date.isoformat()}/{end_date.isoformat()}"
            }
            
        except Exception as e:
            self.logger.error(f"Assessment failed: {str(e)}")
            return {'success': False, 'error': str(e)}


class InterannualConsistencyAnalyzer:
    """
    Analyzer for interannual spectral consistency across multiple years.
    
    Evaluates temporal stability of spectral band distributions by performing
    statistical analyses between consecutive years. Creates comprehensive 
    visualizations and reports to detect potential processing artifacts or
    systematic changes. Preserves exact statistical testing and visualization
    parameters from original implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize interannual consistency analyzer.
        
        Args:
            config: Configuration dictionary with consistency parameters
        """
        self.config = config
        self.logger = get_logger('sentinel2_processing')
        
        # Extract consistency configuration
        self.consistency_config = self.config.get('consistency', {})
        self.sample_size = self.consistency_config.get('sample_size', 10000)
        
        self.logger.info("InterannualConsistencyAnalyzer initialized")
    
    def load_raster_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load raster data and return as a dictionary of band arrays.
        
        CRITICAL: Preserves exact band indexing and dictionary structure.
        
        Args:
            filepath: Path to raster file
            
        Returns:
            dict: Dictionary with band names as keys and arrays as values
            
        Examples:
            >>> data = analyzer.load_raster_data("/path/to/raster.tif")
        """
        with rasterio.open(filepath) as src:
            # CRITICAL: Preserve exact band naming convention
            return {f"band_{i+1}": src.read(i+1) for i in range(src.count)}
    
    def calculate_statistics(self, data_array: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistics for a single band.
        
        CRITICAL: Preserves exact statistical measures and calculations.
        
        Args:
            data_array: Band data array
            
        Returns:
            dict: Dictionary with statistical measures
            
        Examples:
            >>> stats = analyzer.calculate_statistics(band_data)
        """
        # CRITICAL: Preserve exact statistical calculations
        return {
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'median': np.median(data_array),
            'q25': np.percentile(data_array, 25),
            'q75': np.percentile(data_array, 75),
            'min': np.min(data_array),
            'max': np.max(data_array)
        }
    
    def perform_ks_test(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test on samples from two distributions.
        
        CRITICAL: Preserves exact sampling size and KS test implementation
        from original to ensure consistent statistical analysis.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            tuple: (statistic, p_value) from KS test
            
        Examples:
            >>> statistic, p_value = analyzer.perform_ks_test(data1, data2)
        """
        # CRITICAL: Preserve exact sample size from original (10,000)
        sample_size = self.sample_size
        sample1 = np.random.choice(data1.flatten(), sample_size)
        sample2 = np.random.choice(data2.flatten(), sample_size)
        
        # CRITICAL: Preserve exact KS test implementation
        statistic, p_value = stats.ks_2samp(sample1, sample2)
        return statistic, p_value
    
    def analyze_band_distributions(self, input_dir: str, output_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict[str, Any]]]]:
        """
        Analyze band value distributions across years.
        
        CRITICAL: Preserves exact file pattern matching, statistical calculations,
        visualization parameters, and report generation from original implementation.
        
        Args:
            input_dir: Directory containing the merged raster files
            output_dir: Directory where analysis outputs will be saved
            
        Returns:
            tuple: (statistics_dataframe, statistical_tests_list)
            
        Examples:
            >>> df_stats, tests = analyzer.analyze_band_distributions("/merged", "/output")
        """
        self.logger.info("Starting band distribution analysis...")
        
        # Create output directory using shared utilities
        ensure_directory(output_dir)
        
        # CRITICAL: Preserve exact file pattern for merged rasters
        pattern = re.compile(r'S2_summer_mosaic_(\d{4})_merged\.tif$', re.IGNORECASE)
        raster_files = {
            pattern.match(f).group(1): os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if pattern.match(f)
        }
        
        if not raster_files:
            self.logger.error("No merged raster files found")
            return None, None
        
        # Initialize results storage
        years = sorted(raster_files.keys())
        stats_data = []
        
        self.logger.info(f"Found merged raster files for years: {years}")
        
        # CRITICAL: Preserve exact processing loop with tqdm
        for year in tqdm(years, desc="Processing years"):
            data = self.load_raster_data(raster_files[year])
            
            # Calculate statistics for each band
            for band_name, band_data in data.items():
                stats = self.calculate_statistics(band_data)
                stats['year'] = year
                stats['band'] = band_name
                stats_data.append(stats)
        
        # Convert to DataFrame
        df_stats = pd.DataFrame(stats_data)
        
        # CRITICAL: Preserve exact visualization creation and parameters
        self.logger.info("Creating distribution visualizations...")
        
        # 1. Box plots for each band across years - PRESERVE exact plot parameters
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=df_stats, x='band', y='mean', hue='year')
        plt.title('Band Mean Values Distribution by Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'band_means_boxplot.png'))
        plt.close()
        
        # 2. Violin plots for distribution comparison - PRESERVE exact parameters
        plt.figure(figsize=(15, 10))
        sns.violinplot(data=df_stats, x='band', y='mean', hue='year')
        plt.title('Band Mean Values Distribution Density by Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'band_means_violin.png'))
        plt.close()
        
        # CRITICAL: Preserve exact statistical tests between consecutive years
        self.logger.info("Performing statistical tests between consecutive years...")
        statistical_tests = []
        
        # CRITICAL: Preserve exact band iteration (bands 1-10)
        for band_num in range(1, 11):
            band_name = f"band_{band_num}"
            self.logger.info(f"Analyzing {band_name}:")
            
            for i in range(len(years)-1):
                year1, year2 = years[i], years[i+1]
                
                # Load data for both years
                data1 = self.load_raster_data(raster_files[year1])[band_name]
                data2 = self.load_raster_data(raster_files[year2])[band_name]
                
                # Perform KS test with preserved parameters
                statistic, p_value = self.perform_ks_test(data1, data2)
                
                # CRITICAL: Preserve exact test results structure
                statistical_tests.append({
                    'band': band_name,
                    'year1': year1,
                    'year2': year2,
                    'ks_statistic': statistic,
                    'p_value': p_value
                })
        
        # Save results with preserved filenames
        df_tests = pd.DataFrame(statistical_tests)
        df_tests.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
        df_stats.to_csv(os.path.join(output_dir, 'band_statistics.csv'), index=False)
        
        # CRITICAL: Preserve exact analysis report format and content
        self.logger.info("Creating analysis report...")
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("Band Distribution Analysis Report\n")
            f.write("===============================\n\n")
            
            for band in df_stats['band'].unique():
                f.write(f"\n{band.upper()} Analysis:\n")
                band_data = df_stats[df_stats['band'] == band]
                
                f.write("\nYearly Means:\n")
                for _, row in band_data.iterrows():
                    f.write(f"{row['year']}: {row['mean']:.2f}\n")
                
                f.write("\nYearly Medians:\n")
                for _, row in band_data.iterrows():
                    f.write(f"{row['year']}: {row['median']:.2f}\n")
                    
                test_results = df_tests[df_tests['band'] == band]
                f.write("\nKS Test Results (between consecutive years):\n")
                for _, row in test_results.iterrows():
                    f.write(f"{row['year1']} vs {row['year2']}: ")
                    f.write(f"statistic={row['ks_statistic']:.3f}, ")
                    f.write(f"p-value={row['p_value']:.3e}\n")
                
                f.write("\n" + "="*50 + "\n")
        
        self.logger.info("Analysis completed successfully!")
        return df_stats, statistical_tests
    
    def run_consistency_analysis(self) -> Dict[str, Any]:
        """
        Execute complete interannual consistency analysis workflow.
        
        Returns:
            dict: Analysis results with statistics and file paths
            
        Examples:
            >>> analyzer = InterannualConsistencyAnalyzer(config)
            >>> results = analyzer.run_consistency_analysis()
        """
        self.logger.info("Starting Sentinel-2 interannual consistency analysis...")
        
        try:
            # Get directories from config
            input_directory = self.config['paths']['merged_dir']
            output_directory = os.path.join(input_directory, 'interannual_consistency_results')
            
            self.logger.info(f"Input directory: {input_directory}")
            self.logger.info(f"Output directory: {output_directory}")
            
            # Run analysis
            df_stats, statistical_tests = self.analyze_band_distributions(input_directory, output_directory)
            
            if df_stats is not None and statistical_tests is not None:
                self.logger.info("Interannual consistency analysis completed successfully!")
                self.logger.info(f"Results saved to: {output_directory}")
                
                return {
                    'success': True,
                    'output_directory': output_directory,
                    'years_analyzed': sorted(df_stats['year'].unique().tolist()) if df_stats is not None else [],
                    'bands_analyzed': sorted(df_stats['band'].unique().tolist()) if df_stats is not None else [],
                    'n_statistical_tests': len(statistical_tests) if statistical_tests else 0
                }
            else:
                self.logger.error("Analysis failed - no data to process")
                return {'success': False, 'error': 'No data to process'}
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
