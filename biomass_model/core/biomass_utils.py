"""
I/O Utilities for Biomass Estimation

This module provides efficient raster I/O operations with support for
large-scale processing, chunking, and distributed computing.
Updated to use CentralDataPaths instead of config file paths.

Author: Diego Bengochea
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
import dask.array as da
from rasterio.windows import Window
from rasterio.enums import Resampling
import pandas as pd
import time 
from contextlib import contextmanager

# Shared utilities
from shared_utils import get_logger, find_files # validate_file_exists
from shared_utils.central_data_paths_constants import *


class BiomassUtils:
    """
    Manager for raster I/O operations in biomass estimation pipeline.
    
    Handles efficient loading, processing, and saving of raster data
    with support for chunking and distributed processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the raster manager.
        
        Args:
            config: Configuration dictionary (processing parameters only)
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.io')
        
        # Processing parameters
        self.chunk_size = config['compute']['chunk_size']
        self.nodata_value = config['output']['geotiff']['nodata_value']
        
        self.logger.info("RasterManager initialized")
    

    def read_height_and_mask_xarray(self, height_file, mask_file):
        """
        Read height and mask rasters into xarray with dask chunking.
        
        Args:
            height_file (str): Path to height raster
            mask_file (str): Path to forest type mask
            
        Returns:
            tuple: (height_xr, mask_xr, out_meta) where:
                - height_xr is an xarray DataArray of height values
                - mask_xr is an xarray DataArray of boolean mask values
                - out_meta is a dictionary with raster metadata for output
                
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If CRS mismatch between height and mask
            rasterio.errors.RasterioIOError: If files can't be read
        """
        # Validate input files exist
        if not os.path.exists(height_file):
            raise FileNotFoundError(f"Height file not found: {height_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
        
        chunk_size = self.chunk_size
        try:
            # Read rasters with rioxarray (preserves geospatial metadata)
            height_xr = rxr.open_rasterio(height_file, chunks={'x': chunk_size, 'y': chunk_size})
            mask_xr = rxr.open_rasterio(mask_file, chunks={'x': chunk_size, 'y': chunk_size})
        except Exception as e:
            print('here')
            self.logger.error(f"Error reading raster files: {e}")
            raise
        
        # Verify CRS match
        if height_xr.rio.crs != mask_xr.rio.crs:
            raise ValueError(f"CRS mismatch between height ({height_xr.rio.crs}) and mask ({mask_xr.rio.crs})")
        
        # Convert to appropriate data types and squeeze to remove band dimension
        height_xr = height_xr.astype(np.float32).squeeze()
        mask_xr = (mask_xr > 0).astype(bool).squeeze()
    
        # Prepare metadata for output files
        out_meta = {
        'driver': 'GTiff',
        'height': height_xr.rio.height,
        'width': height_xr.rio.width,
        'count': 1,
        'dtype': 'float32',
        'crs': height_xr.rio.crs,
        'transform': height_xr.rio.transform(),
        'nodata': height_xr.rio.nodata
        }
        
        return height_xr, mask_xr, out_meta

    @contextmanager
    def timer(self,description):
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            print(f"{description}: {elapsed:.2f}s")


    def write_agbd_tile(self, median, uncertainty, height_file, output_meta):
        """Write merged AGBD results for entire tile."""

        geotiff_options = {
            'tiled': self.config['output']['geotiff']['tiled'],
            'blockxsize': self.config['output']['geotiff']['blockxsize'],
            'blockysize': self.config['output']['geotiff']['blockysize'],
            'compress': self.config['output']['geotiff'].get('compress'),
            'dtype': 'float32'
        }

        stem = Path(height_file).stem
        try:
            specs = re.findall(r'canopy_height_(.*)', stem)[0]
        except IndexError:
            self.logger.error(f"Invalid filename: {stem}")
            return False

        year = specs.split('_')[0]
        output_base = BIOMASS_MAPS_TILED_DIR

        for data, measure in [(median, 'mean'), (uncertainty, 'uncertainty')]:
            output_dir = output_base / f"AGBD_{measure}" / year
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"AGBD_{measure}_S2_{specs}.tif"
            savepath = output_dir / filename

            # Set spatial metadata
            if hasattr(data, 'rio'):
                if 'crs' in output_meta:
                    data.rio.write_crs(output_meta['crs'], inplace=True)
                data.rio.write_nodata(output_meta['nodata'], inplace=True)

            try:
                data.rio.to_raster(savepath, driver='GTiff', **geotiff_options)
                self.logger.info(f"Saved {savepath}")
            except Exception as e:
                self.logger.error(f"Write failed: {e}")
                return False

        return True
            
    def write_agbd_results(self, results, height_file, code, forest_type_name, mask_data, output_meta):
        """
        Write AGBD results (mean + uncertainty).

        Args:
            results: xarray.Dataset with agbd_median and agbd_uncertainty
            height_file: Path to input height file
            code: Forest type code
            forest_type_name: Forest type name
            mask_data: Boolean mask array
            output_meta: Raster metadata

        Returns:
            bool: Success status
        """
        geotiff_options = {
            'tiled': self.config['output']['geotiff']['tiled'],
            'blockxsize': self.config['output']['geotiff']['blockxsize'],
            'blockysize': self.config['output']['geotiff']['blockysize'],
            'compress': self.config['output']['geotiff']['compress'],
            'dtype': output_meta.get('dtype', 'float32')
        }

        try:
            for variable in results.data_vars:  # agbd_mean, agbd_uncertainty
                measure = variable.split('_')[1]  # 'mean' or 'uncertainty'

                savepath = self.build_agbd_savepath(Path(height_file), measure, code)
                os.makedirs(os.path.dirname(savepath), exist_ok=True)

                self.logger.info(f"Writing AGBD {measure} for {forest_type_name}")

                result = results[variable]
                masked_result = result.where(mask_data, output_meta['nodata'])
                masked_result = masked_result.fillna(output_meta['nodata'])

                if hasattr(mask_data, 'rio') and hasattr(mask_data.rio, 'crs'):
                    masked_result.rio.write_crs(mask_data.rio.crs, inplace=True)
                elif 'crs' in output_meta:
                    masked_result.rio.write_crs(output_meta['crs'], inplace=True)

                masked_result.rio.write_nodata(output_meta['nodata'], inplace=True)

                try:
                    masked_result.rio.to_raster(savepath, driver='GTiff', **geotiff_options)
                    self.logger.info(f"Saved AGBD {measure} to: {savepath}")
                except Exception as write_error:
                    self.logger.error(f"Error writing {savepath}: {write_error}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error writing AGBD results for {forest_type_name}: {str(e)}")
            return False


    def build_agbd_savepath(self, fname: Path, measure: str, code: str) -> str:
        """
        Build output path for AGBD results.

        Args:
            fname: Input file path
            measure: 'mean' or 'uncertainty'
            code: Forest type code

        Returns:
            Full path to output file
        """
        stem = fname.stem

        try:
            specs = re.findall(r'canopy_height_(.*)', stem)[0]
        except IndexError:
            self.logger.error(f"Could not extract specifications from filename: {stem}")
            raise ValueError(f"Invalid filename format: {stem}")

        output_base_dir = BIOMASS_MAPS_PER_FOREST_TYPE_RAW_DIR
        year = specs.split('_')[0]
        subdir = f"AGBD_{measure}"
        output_dir = output_base_dir / subdir / year
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = f"AGBD_{measure}_S2_{specs}_code{code}.tif"
        return os.path.join(output_dir, filename) 

    def write_xarray_results(self, results, height_file, code, forest_type_name, mask_data, output_meta):
        """
        Write xarray Monte Carlo results using rioxarray with optimized GeoTIFF settings.
        
        Args:
            results (xarray.Dataset): Results from monte_carlo_biomass_optimized
            height_file (str): Path to input height file
            code (str): Forest type code
            forest_type_name (str): Name of forest type
            mask_data (xarray.DataArray): Boolean mask array
            output_meta (dict): Raster metadata for outputs
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        # Get GeoTIFF optimization settings from config
        geotiff_options = self.config['output']['geotiff'].copy()
        geotiff_options['dtype'] = output_meta.get('dtype', 'float32')
        
        geotiff_options = {
            'tiled': self.config['output']['geotiff']['tiled'],
            'blockxsize': self.config['output']['geotiff']['blockxsize'], 
            'blockysize': self.config['output']['geotiff']['blockysize'],
            'compress': self.config['output']['geotiff']['compress'],
            'dtype': output_meta.get('dtype', 'float32')
        }
        try:
            # Process each variable in the results dataset
            for variable in results.data_vars:
                # Parse variable name to determine output type and measure
                parts = variable.split('_')
                if len(parts) < 2:
                    self.logger.warning(f"Unexpected variable name format: {variable}")
                    continue
                    
                output_type = parts[0]
                measure = parts[1]
                
                # Build save path using centralized configuration
                try:
                    savepath = self.build_savepath(Path(height_file), output_type, measure, code)
                except ValueError as e:
                    self.logger.error(f"Error building save path for {variable}: {e}")
                    continue
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                
                self.logger.info(f"Writing {output_type} {measure} for {forest_type_name}")
                with self.timer(f"Writing {output_type} {measure}"):
                    # Get this result variable
                    result = results[variable]
                    
                    # Apply mask to the result and handle nodata values
                    masked_result = result.where(mask_data, output_meta['nodata'])
                    masked_result = masked_result.fillna(output_meta['nodata'])
                    
                    # Set CRS information for proper georeferencing
                    if hasattr(mask_data, 'rio') and hasattr(mask_data.rio, 'crs'):
                        masked_result.rio.write_crs(mask_data.rio.crs, inplace=True)
                    elif 'crs' in output_meta:
                        masked_result.rio.write_crs(output_meta['crs'], inplace=True)
                    
                    # Set nodata value
                    masked_result.rio.write_nodata(output_meta['nodata'], inplace=True)
                    
                    # Write to optimized GeoTIFF
                    try:
                        masked_result.rio.to_raster(
                            savepath,
                            driver='GTiff',
                            **geotiff_options
                        )
                        self.logger.info(f"Saved {output_type} {measure} to: {savepath}")
                    except Exception as write_error:
                        self.logger.error(f"Error writing {savepath}: {write_error}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing results for {forest_type_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def build_savepath(self, fname, output_type, kind, code):
        """
        Build output file path based on input filename, output type and statistical measure.
        
        Args:
            fname (Path): Input file path
            output_type (str): Type of output ('agbd', 'bgbd', 'tbd')
            kind (str): Statistical measure ('mean', 'uncertainty')
            code (str): Forest type code
            
        Returns:
            str: Full path to output file
            
        Raises:
            ValueError: If output_type is not recognized
        """
        stem = fname.stem

        # Extract specifications from canopy height filename
        try:
            specs = re.findall(r'canopy_height_(.*)', stem)[0]
        except IndexError:
            logger.error(f"Could not extract specifications from filename: {stem}")
            raise ValueError(f"Invalid filename format: {stem}")

        # Get base output directory for this output type
        output_base_dir = BIOMASS_MAPS_PER_FOREST_TYPE_RAW_DIR
        
        year = specs.split('_')[0]
        # Build filename components
        subdir = f"{output_type.upper()}_MonteCarlo_{self.config['processing']['target_resolution']}"

        output_dir = output_base_dir / year / subdir

        Path(output_dir).mkdir(parents=True,exist_ok=True)

        filename = f"{output_type.upper()}_S2_{kind}_{specs}_code{code}.tif"
        
        return os.path.join(output_dir, filename)


    def extract_tile_info(self, filepath):
        """
        Extract year and pattern information from a canopy height filename.
        Args:
            filepath (str): Path to canopy height file
            
        Returns:
            dict: Dictionary with year and base pattern, or None if pattern doesn't match
            
        Example:
            >>> 
            {'year': '2020', 'base_pattern': 'canopy_height_2020_100m'}
        """
        stem = Path(filepath).stem

        # Match pattern for canopy height files
        match_year = re.search(r'canopy_height_(\d{4})_(N[\d.]+_[WE][\d.]+)', stem)
        if match_year:
            year = match_year.group(1)
            coords = match_year.group(2)
            return {
                'year': year,
                'base_pattern': f"canopy_height_{year}_{coords}"
            }
        else:
            match_year2 = re.search(r'AGBD_mean_S2_(\d{4})_(N[\d.]+_[WE][\d.]+)', stem)
            if match_year2:
                year = match_year2.group(1)
                coords = match_year2.group(2)
                return {
                    'year': year,
                    'base_pattern': f"canopy_height_{year}_{coords}"
                }

        logger = get_logger('biomass_estimation')
        logger.error(f"Failed to extract tile info from: {stem}")
        return None

    def find_masks_for_tile(self, tile_info, masks_dir):
        """Find all mask files for a given tile."""
        logger = get_logger('biomass_estimation')
        
        if not os.path.exists(masks_dir):
            raise OSError(f"Masks directory not found: {masks_dir}")
        
        # Build pattern: canopy_height_2019_N43.0_W3.0_code*.tif
        base_pattern = tile_info['base_pattern']
        mask_pattern = f"{base_pattern}_code*.tif"
        
        try:
            mask_paths = glob.glob(os.path.join(masks_dir, mask_pattern))
        except OSError as e:
            logger.error(f"Error searching for masks: {e}")
            return []

        results = []
        for mask_path in mask_paths:
            match = re.search(r'_code(\d+)\.tif$', mask_path)
            if match:
                forest_type_code = match.group(1)
                results.append((mask_path, forest_type_code))
            else:
                logger.warning(f"Could not extract code from: {mask_path}")
        
        return results


    def build_forest_type_mapping(self, mfe_dir, cache_path=None, use_cache=True):
        """
        Build forest type code-to-name mapping from MFE data.
        
        Args:
            mfe_dir (Path): Directory containing MFE forest type data
            cache_path (Path): Optional cache file path
            use_cache (bool): Whether to use cached mapping
            
        Returns:
            dict: Mapping from forest type codes to names
        """
        logger = get_logger('biomass_estimation')
        
        # Try to load from cache first
        if use_cache and cache_path and Path(cache_path).exists():
            try:
                mapping = pd.read_csv(cache_path)
                # import pickle
                # with open(cache_path, 'rb') as f:
                #     mapping = pickle.load(f)
                logger.info(f"Loaded forest type mapping from cache: {len(mapping)} entries")
                return mapping
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding mapping")
        
        # Build mapping from MFE files
        mapping = {}
        mfe_files = list(Path(mfe_dir).glob("*.shp"))
        
        for mfe_file in mfe_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(mfe_file)
                
                # Extract forest type information (adapt to actual MFE schema)
                if 'TIPO_MASA' in gdf.columns and 'CODIGO' in gdf.columns:
                    for _, row in gdf.iterrows():
                        code = str(row['CODIGO'])
                        name = row['TIPO_MASA']
                        mapping[code] = name
            except Exception as e:
                logger.warning(f"Error processing {mfe_file}: {e}")
        
        # Save cache if specified
        if cache_path and mapping:
            try:
                import pickle
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(mapping, f)
                logger.info(f"Saved forest type mapping cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        logger.info(f"Built forest type mapping: {len(mapping)} entries")
        return mapping    

    def read_agbd_and_mask_xarray(self, agbd_mean_file, agbd_unc_file, mask_file):
        """
        Read AGBD mean, uncertainty, and mask rasters into xarray with dask chunking.
        
        Args:
            agbd_mean_file (str): Path to AGBD mean raster
            agbd_unc_file (str): Path to AGBD uncertainty raster
            mask_file (str): Path to forest type mask
            
        Returns:
            tuple: ((agbd_mean_xr, agbd_unc_xr), mask_xr, out_meta)
        """
        import os
        import rioxarray as rxr
        import numpy as np
        
        # Validate input files exist
        if not os.path.exists(agbd_mean_file):
            raise FileNotFoundError(f"AGBD mean file not found: {agbd_mean_file}")
        if not os.path.exists(agbd_unc_file):
            raise FileNotFoundError(f"AGBD uncertainty file not found: {agbd_unc_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
        
        chunk_size = self.chunk_size
        try:
            # Read rasters with rioxarray
            agbd_mean_xr = rxr.open_rasterio(agbd_mean_file, chunks={'x': chunk_size, 'y': chunk_size})
            agbd_unc_xr = rxr.open_rasterio(agbd_unc_file, chunks={'x': chunk_size, 'y': chunk_size})
            mask_xr = rxr.open_rasterio(mask_file, chunks={'x': chunk_size, 'y': chunk_size})
        except Exception as e:
            self.logger.error(f"Error reading raster files: {e}")
            raise
        
        # Verify CRS match
        if agbd_mean_xr.rio.crs != mask_xr.rio.crs:
            raise ValueError(f"CRS mismatch between AGBD and mask")
        
        # Convert to appropriate data types and squeeze
        agbd_mean_xr = agbd_mean_xr.astype(np.float32).squeeze()
        agbd_unc_xr = agbd_unc_xr.astype(np.float32).squeeze()
        mask_xr = (mask_xr > 0).astype(bool).squeeze()

        # Prepare metadata for output files
        out_meta = {
            'driver': 'GTiff',
            'height': agbd_mean_xr.rio.height,
            'width': agbd_mean_xr.rio.width,
            'count': 1,
            'dtype': 'float32',
            'crs': agbd_mean_xr.rio.crs,
            'transform': agbd_mean_xr.rio.transform(),
            'nodata': agbd_mean_xr.rio.nodata
        }
        
        return (agbd_mean_xr, agbd_unc_xr), mask_xr, out_meta