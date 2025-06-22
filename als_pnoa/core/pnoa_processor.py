#!/usr/bin/env python3
"""
ALS PNOA Core Processing Module

Processes PNOA LiDAR tiles by selecting those that intersect with Sentinel-2 tiles
and standardizes them for training data preparation.

Author: Diego Bengochea
"""

import rasterio
from pathlib import Path
from shapely import geometry
import geopandas as gpd
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging


class PNOAProcessor:
    """
    PNOA LiDAR tile processor for training data preparation.
    
    Selects and processes PNOA tiles that intersect with Sentinel-2 tiles,
    standardizing naming and format for machine learning training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PNOA processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract paths from config
        self.sentinel_path = config['paths']['sentinel2_dir']
        self.pnoa_coverage_paths = config['paths']['pnoa_coverage_dirs']
        self.pnoa_data_dir = config['paths']['pnoa_data_dir']
        self.target_output_dir = config['paths']['output_dir']
        
        # Processing parameters
        self.target_years = config['processing']['target_years']
        self.target_crs = config['processing']['target_crs']
        
        self.logger.info("Initialized PNOAProcessor")
    
    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        output_path = Path(self.target_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {output_path}")
    
    def find_intersecting_pnoa_tiles(self, sentinel_tile_path: Path, year: int) -> List[str]:
        """
        Find PNOA tiles that intersect with a Sentinel-2 tile for a given year.
        
        Args:
            sentinel_tile_path: Path to Sentinel-2 tile
            year: Target year
            
        Returns:
            List of PNOA file paths that intersect
        """
        intersecting_files = []
        fileids = set()
        
        try:
            with rasterio.open(sentinel_tile_path) as src_sentinel:
                crs = src_sentinel.crs
                bounds_sentinel = src_sentinel.bounds
                polygon_sentinel = geometry.box(
                    minx=bounds_sentinel.left, 
                    maxx=bounds_sentinel.right, 
                    miny=bounds_sentinel.bottom, 
                    maxy=bounds_sentinel.top
                )
                
                # Process each UTM zone coverage directory
                for pnoa_coverage_path in self.pnoa_coverage_paths:
                    coverage_dir = Path(pnoa_coverage_path)
                    
                    for pnoa_polygons in coverage_dir.glob('*.shp'):
                        # Extract UTM zone from filename
                        utm_matches = re.findall('_HU(..)_', str(pnoa_polygons))
                        if not utm_matches:
                            self.logger.warning(f"Could not extract UTM zone from {pnoa_polygons}")
                            continue
                        
                        utm = utm_matches[0]
                        
                        # Read polygon coverage file
                        pnoa_gdf = gpd.read_file(pnoa_polygons)
                        pnoa_gdf['geometry'] = pnoa_gdf['geometry'].to_crs(crs)
                        
                        # Find intersecting tiles
                        intersecting_tiles = pnoa_gdf.intersects(polygon_sentinel)
                        pnoa_intersecting = pnoa_gdf[intersecting_tiles]
                        
                        # Filter by year
                        pnoa_intersecting = pnoa_intersecting[pnoa_intersecting.FECHA == str(year)]
                        
                        # Get file paths
                        selected_files = pnoa_intersecting['PATH'].apply(
                            lambda x: f"{self.pnoa_data_dir}{x.split('/')[-1]}"
                        ).to_list()
                        
                        # Remove redundant files based on file ID
                        selected_dict = {}
                        for f in selected_files:
                            try:
                                fid = re.findall('NDSM-VEGETACION-(?:...)-(.*)-COB2.tif', f)[0]
                                if fid not in fileids:
                                    selected_dict[fid] = f
                            except IndexError:
                                self.logger.warning(f"Could not extract file ID from {f}")
                                continue
                        
                        # Update file IDs and add to results
                        fileids.update(selected_dict.keys())
                        intersecting_files.extend(selected_dict.values())
                        
        except Exception as e:
            self.logger.error(f"Error processing {sentinel_tile_path}: {str(e)}")
            return []
        
        return intersecting_files
    
    def generate_output_filename(self, input_path: str, year: int) -> str:
        """
        Generate standardized output filename.
        
        Args:
            input_path: Input PNOA file path
            year: Processing year
            
        Returns:
            Standardized output filename
        """
        try:
            # Extract the NDSM part from the original filename (matches original logic exactly)
            ndsm_part = re.findall('.*/(NDSM-.*)', input_path)[0]
            
            # Create standardized filename: PNOA_2021_NDSM-VEGETACION-H30-0177-COB2_epsg_25830.tif
            output_filename = f"PNOA_{year}_{ndsm_part}_epsg_{self.target_crs.split(':')[1]}.tif"
            
            return output_filename
            
        except IndexError:
            # Final fallback naming if regex fails
            original_name = Path(input_path).stem
            return f"PNOA_{year}_{original_name}_epsg_{self.target_crs.split(':')[1]}.tif"
    
    def process_pnoa_tile(self, input_path: str, output_path: str) -> bool:
        """
        Process a single PNOA tile (copy with standardized naming).
        
        Args:
            input_path: Input PNOA file path
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            input_file = Path(input_path)
            output_file = Path(output_path)
            
            if not input_file.exists():
                self.logger.error(f"Input file not found: {input_file}")
                return False
            
            if output_file.exists():
                self.logger.debug(f"Output file already exists, skipping: {output_file}")
                return True
            
            # Read and write with standardized parameters
            with rasterio.open(input_file) as src:
                image = src.read(1)
                
                with rasterio.open(
                    output_file,
                    mode="w",
                    driver="GTiff",
                    height=image.shape[-2],
                    width=image.shape[-1],
                    dtype=np.float32,
                    count=1,
                    nodata=float(src.nodata) if src.nodata is not None else -9999,
                    crs=src.crs,
                    transform=src.transform,
                    compress='lzw'
                ) as new_dataset:
                    new_dataset.write(image, 1)
            
            self.logger.debug(f"Processed: {input_file.name} -> {output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            return False
    
    def process_all_tiles(self) -> Dict[str, any]:
        """
        Process all PNOA tiles that intersect with Sentinel-2 tiles.
        
        Returns:
            Processing results summary
        """
        self.logger.info("Starting PNOA tile processing...")
        
        # Create output directory
        self.create_output_directory()
        
        # Track processing results
        selected_tiles = []
        processed_count = 0
        error_count = 0
        
        sentinel_dir = Path(self.sentinel_path)
        
        for year in self.target_years:
            self.logger.info(f"Processing year {year}...")
            
            # Find Sentinel-2 tiles for this year
            sentinel_pattern = f'sentinel2_mosaic_{year}_*.tif'
            sentinel_tiles = list(sentinel_dir.rglob(sentinel_pattern))
            
            if not sentinel_tiles:
                self.logger.warning(f"No Sentinel-2 tiles found for year {year}")
                continue
            
            self.logger.info(f"Found {len(sentinel_tiles)} Sentinel-2 tiles for {year}")
            
            for sentinel_tile in sentinel_tiles:
                self.logger.debug(f"Processing Sentinel tile: {sentinel_tile.name}")
                
                # Find intersecting PNOA tiles
                intersecting_pnoa = self.find_intersecting_pnoa_tiles(sentinel_tile, year)
                
                self.logger.debug(f"Found {len(intersecting_pnoa)} intersecting PNOA tiles")
                
                # Process each intersecting PNOA tile
                for pnoa_file in intersecting_pnoa:
                    output_filename = self.generate_output_filename(pnoa_file, year)
                    output_path = Path(self.target_output_dir) / output_filename
                    
                    # Track tile selection
                    selected_tiles.append({
                        'input_path': pnoa_file,
                        'output_path': str(output_path),
                        'year': year,
                        'sentinel_tile': str(sentinel_tile)
                    })
                    
                    # Process the tile
                    success = self.process_pnoa_tile(pnoa_file, str(output_path))
                    
                    if success:
                        processed_count += 1
                    else:
                        error_count += 1
        
        # Summary results
        results = {
            'total_selected': len(selected_tiles),
            'successfully_processed': processed_count,
            'errors': error_count,
            'selected_tiles': selected_tiles,
            'output_directory': self.target_output_dir
        }
        
        self.logger.info(f"Processing complete: {processed_count} tiles processed, {error_count} errors")
        
        return results
    
    def validate_inputs(self) -> bool:
        """
        Validate that required input directories and files exist.
        
        Returns:
            True if validation passes
        """
        self.logger.info("Validating inputs...")
        
        # Check Sentinel-2 directory
        sentinel_dir = Path(self.sentinel_path)
        if not sentinel_dir.exists():
            self.logger.error(f"Sentinel-2 directory not found: {sentinel_dir}")
            return False
        
        # Check for Sentinel-2 files
        sentinel_files = list(sentinel_dir.rglob('sentinel2_mosaic_*.tif'))
        if not sentinel_files:
            self.logger.error(f"No Sentinel-2 mosaic files found in {sentinel_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(sentinel_files)} Sentinel-2 mosaic files")
        
        # Check PNOA coverage directories
        missing_coverage = []
        for coverage_path in self.pnoa_coverage_paths:
            coverage_dir = Path(coverage_path)
            if not coverage_dir.exists():
                missing_coverage.append(coverage_path)
            else:
                shp_files = list(coverage_dir.glob('*.shp'))
                if not shp_files:
                    self.logger.warning(f"No shapefile coverage found in {coverage_dir}")
                else:
                    self.logger.info(f"✅ Found {len(shp_files)} coverage files in {coverage_dir}")
        
        if missing_coverage:
            self.logger.error(f"PNOA coverage directories not found: {missing_coverage}")
            return False
        
        # Check PNOA data directory
        pnoa_data_dir = Path(self.pnoa_data_dir)
        if not pnoa_data_dir.exists():
            self.logger.error(f"PNOA data directory not found: {pnoa_data_dir}")
            return False
        
        # Check for PNOA data files
        pnoa_files = list(pnoa_data_dir.rglob('NDSM-VEGETACION-*.tif'))
        if not pnoa_files:
            self.logger.error(f"No PNOA NDSM files found in {pnoa_data_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(pnoa_files)} PNOA NDSM files")
        
        self.logger.info("Input validation passed")
        return True
    
    def get_processing_summary(self) -> Dict[str, any]:
        """
        Get summary of what would be processed without actually processing.
        
        Returns:
            Summary of processing scope
        """
        summary = {
            'target_years': self.target_years,
            'sentinel_tiles_by_year': {},
            'total_pnoa_files': 0,
            'coverage_areas': len(self.pnoa_coverage_paths)
        }
        
        sentinel_dir = Path(self.sentinel_path)
        
        for year in self.target_years:
            sentinel_pattern = f'sentinel2_mosaic_{year}_*.tif'
            sentinel_tiles = list(sentinel_dir.rglob(sentinel_pattern))
            summary['sentinel_tiles_by_year'][year] = len(sentinel_tiles)
        
        # Count total PNOA files
        pnoa_data_dir = Path(self.pnoa_data_dir)
        if pnoa_data_dir.exists():
            pnoa_files = list(pnoa_data_dir.rglob('NDSM-VEGETACION-*.tif'))
            summary['total_pnoa_files'] = len(pnoa_files)
        
        return summary