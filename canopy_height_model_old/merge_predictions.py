"""
Parallel merging of canopy height prediction tiles into larger geographic tiles.

This script merges individual prediction patches into larger 120km x 120km tiles
for efficient storage and processing. Implements parallel processing with geographic
masking to Spain boundaries and handles overlapping predictions through averaging.

The merging process creates a regular grid of tiles covering continental Spain,
buffering tiles to prevent stitching artifacts, and applies geographic masks
to exclude areas outside Spanish territory.

Author: Diego Bengochea
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import itertools
import multiprocessing

# Third-party imports
import numpy as np
import rasterio
import rasterio.features
from rasterio.merge import merge
import geopandas as gpd
import shapely
import odc.geo
import odc.geo.geobox
from tqdm import tqdm

# Local imports
from config import load_config, setup_logging, create_output_directory


class PredictionMerger:
    """
    Parallel processor for merging canopy height prediction tiles.
    
    This class handles the conversion of small prediction patches into larger
    geographic tiles suitable for further processing and analysis.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the prediction merger with configuration.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging']['level'])
        
        # Get merge configuration
        self.merge_config = self.config['post_processing']['merge']
        
        # Create output directory
        create_output_directory(Path(self.merge_config['output_dir']))
        
        # Load Spain boundaries for masking
        self._load_spain_boundaries()
        
        # Create geographic grid
        self._create_geographic_grid()
        
        self.logger.info("Prediction merger initialized")
        self.logger.info(f"Input directory: {self.merge_config['input_dir']}")
        self.logger.info(f"Output directory: {self.merge_config['output_dir']}")
        self.logger.info(f"Tile size: {self.merge_config['tile_size_km']}km")
    
    def _load_spain_boundaries(self) -> None:
        """Load and prepare Spain boundaries for geographic masking."""
        try:
            self.spain = gpd.read_file(self.merge_config['spain_shapefile'])
            self.spain = self.spain[['geometry', 'COUNTRY']].to_crs(
                epsg=self.merge_config['target_crs'].split(':')[1]
            )
            
            # Create geometry for odc.geo
            self.spain_geometry = odc.geo.geom.Geometry(
                self.spain.geometry[0], 
                crs=self.merge_config['target_crs']
            )
            
            self.logger.info(f"Spain boundaries loaded from: {self.merge_config['spain_shapefile']}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Spain boundaries: {str(e)}")
            raise
    
    def _create_geographic_grid(self) -> None:
        """Create a regular geographic grid covering Spain."""
        try:
            # Create a GeoBox for all continental Spain
            geobox_spain = odc.geo.geobox.GeoBox.from_geopolygon(
                self.spain_geometry, 
                resolution=self.merge_config['resolution_meters']
            )
            
            # Calculate tile size in grid units
            tile_size_meters = self.merge_config['tile_size_km'] * 1000
            tile_size_grid = tile_size_meters // self.merge_config['resolution_meters']
            
            # Divide the full geobox into tiles
            self.geotiles = odc.geo.geobox.GeoboxTiles(
                geobox_spain, 
                (tile_size_grid, tile_size_grid)
            )
            
            # Extract all tiles
            self.geotiles_list = [
                self.geotiles.__getitem__(tile) 
                for tile in self.geotiles._all_tiles()
            ]
            
            self.logger.info(f"Created {len(self.geotiles_list)} geographic tiles")
            
        except Exception as e:
            self.logger.error(f"Failed to create geographic grid: {str(e)}")
            raise
    
    def _convert_timestamp_to_year(self, timestamp: int) -> int:
        """
        Convert timestamp to year using configuration mapping.
        
        Args:
            timestamp (int): Unix timestamp
            
        Returns:
            int: Corresponding year
        """
        return self.merge_config['year_timestamps'].get(timestamp, timestamp)
    
    def _generate_output_filename(
        self, 
        year: int, 
        lat: float, 
        lon: float
    ) -> str:
        """
        Generate standardized output filename.
        
        Args:
            year (int): Year of the data
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
            
        Returns:
            str: Generated filename
        """
        lat_rounded = np.round(lat, 0)
        
        if lon > 0:
            lon_rounded = np.round(lon, 0)
            filename = f"canopy_height_{year}_N{lat_rounded}_E{lon_rounded}.tif"
        else:
            lon_rounded = np.round(-lon, 0)
            filename = f"canopy_height_{year}_N{lat_rounded}_W{lon_rounded}.tif"
        
        return os.path.join(self.merge_config['output_dir'], filename)
    
    def _find_intersecting_files(
        self, 
        tile_shapely_box: shapely.geometry.box, 
        year: int
    ) -> List[Path]:
        """
        Find prediction files that intersect with the given tile.
        
        Args:
            tile_shapely_box: Shapely box representing the tile boundaries
            year (int): Year to search for files
            
        Returns:
            List[Path]: List of intersecting file paths
        """
        predictions_dir = Path(self.merge_config['input_dir']) / f'{year}.0'
        
        if not predictions_dir.exists():
            return []
        
        all_files = list(predictions_dir.glob(f'*{year}.0.tif'))
        files_to_merge = []
        
        for fname in all_files:
            try:
                with rasterio.open(fname) as src:
                    bounds = src.bounds
                    prediction_shapely_box = shapely.box(
                        bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                    
                    if shapely.intersects(tile_shapely_box, prediction_shapely_box):
                        files_to_merge.append(fname)
                        
            except Exception as e:
                self.logger.warning(f"Error reading {fname}: {str(e)}")
                continue
        
        return files_to_merge
    
    def _merge_tile_files(
        self, 
        files_to_merge: List[Path], 
        original_bounds: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Merge multiple raster files into a single array.
        
        Args:
            files_to_merge: List of file paths to merge
            original_bounds: Bounds for the output raster
            
        Returns:
            Tuple[np.ndarray, rasterio.Affine]: Merged array and transform
        """
        try:
            # Merge using sum and count to handle overlaps
            image_sum, transform_sum = merge(
                files_to_merge, 
                bounds=original_bounds, 
                method='sum'
            )
            image_count, transform_count = merge(
                files_to_merge, 
                bounds=original_bounds, 
                method='count'
            )
            
            # Average overlapping areas
            image = image_sum / image_count
            image = image[0, :, :]
            
            return image, transform_count
            
        except Exception as e:
            self.logger.error(f"Error merging files: {str(e)}")
            raise
    
    def _apply_spain_mask(
        self, 
        image: np.ndarray, 
        transform: rasterio.Affine
    ) -> np.ndarray:
        """
        Apply Spain geographic mask to exclude areas outside Spain.
        
        Args:
            image: Input image array
            transform: Raster transform
            
        Returns:
            np.ndarray: Masked image array
        """
        try:
            land_mask = rasterio.features.geometry_mask(
                self.spain.geometry, 
                image.shape, 
                transform
            )
            
            # Set areas outside Spain to nodata
            masked_image = np.where(
                land_mask, 
                self.merge_config['nodata_value'], 
                image
            )
            
            return masked_image
            
        except Exception as e:
            self.logger.error(f"Error applying Spain mask: {str(e)}")
            raise
    
    def process_tile_year(self, args: Tuple[int, Tuple[Any, int]]) -> str:
        """
        Process a single tile and year combination.
        
        Args:
            args: Tuple containing (index, (tile, year))
            
        Returns:
            str: Processing status message
        """
        i, (tile, year) = args
        
        try:
            # Buffer the tile to prevent stitching artifacts
            original_tile = tile
            tile = original_tile.buffered(self.merge_config['buffer_meters'])
            
            # Get tile coordinates
            tile_bbox = tile.boundingbox
            tile_shapely_box = shapely.box(
                tile_bbox.left, tile_bbox.bottom, 
                tile_bbox.right, tile_bbox.top
            )
            
            # Convert to lat/lon for filename
            tile_bbox_latlon = tile_bbox.to_crs('EPSG:4326')
            lon = tile_bbox_latlon.left
            lat = tile_bbox_latlon.top
            
            # Convert year and generate filename
            year_converted = self._convert_timestamp_to_year(year)
            savepath = self._generate_output_filename(year_converted, lat, lon)
            
            # Skip if file already exists
            if Path(savepath).exists():
                return f"Skipped {i} (file already exists): {savepath}"
            
            # Find intersecting prediction files
            files_to_merge = self._find_intersecting_files(tile_shapely_box, year)
            
            if not files_to_merge:
                return f"No files in tile {tile_bbox} for year {year_converted}"
            
            # Get original tile bounds (without buffer)
            original_bbox = original_tile.boundingbox
            original_bounds = (
                original_bbox.left, original_bbox.bottom,
                original_bbox.right, original_bbox.top
            )
            
            # Merge files
            image, transform = self._merge_tile_files(files_to_merge, original_bounds)
            
            # Apply Spain mask
            masked_image = self._apply_spain_mask(image, transform)
            
            # Write output raster
            self._write_output_raster(
                masked_image, transform, savepath, year_converted
            )
            
            return f"Processed {i}: {savepath}"
            
        except Exception as e:
            return f"Error processing task {i} (year={year}): {str(e)}"
    
    def _write_output_raster(
        self, 
        image: np.ndarray, 
        transform: rasterio.Affine, 
        savepath: str, 
        year: int
    ) -> None:
        """
        Write the processed image to a raster file.
        
        Args:
            image: Image array to write
            transform: Raster transform
            savepath: Output file path
            year: Year for metadata
        """
        try:
            with rasterio.open(
                savepath,
                mode="w",
                driver="GTiff",
                height=image.shape[-2],
                width=image.shape[-1],
                count=1,
                dtype='float32',
                crs=self.merge_config['target_crs'],
                transform=transform,
                nodata=self.merge_config['nodata_value'],
                compress=self.merge_config['compression'],
                tiled=True
            ) as new_dataset:
                new_dataset.write(image, 1)
                new_dataset.update_tags(DATE=year)
                
        except Exception as e:
            self.logger.error(f"Error writing raster {savepath}: {str(e)}")
            raise
    
    def run_parallel_merge(self) -> None:
        """Run the parallel merging process."""
        self.logger.info("Starting parallel prediction merging...")
        
        # Create task list
        years = list(self.merge_config['year_timestamps'].keys())
        args_list = list(itertools.product(self.geotiles_list, years))
        
        # Prepare indexed arguments for progress tracking
        indexed_args = [(i, args) for i, args in enumerate(args_list)]
        
        # Use multiprocessing
        num_cores = min(
            self.merge_config['num_workers'], 
            multiprocessing.cpu_count()
        )
        
        self.logger.info(f"Using {num_cores} cores for processing {len(args_list)} tasks")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = list(tqdm(
                pool.imap(self.process_tile_year, indexed_args),
                total=len(indexed_args),
                desc="Merging tiles"
            ))
        
        # Print summary statistics
        successful = sum(1 for r in results if r.startswith("Processed"))
        skipped = sum(1 for r in results if r.startswith("Skipped"))
        no_files = sum(1 for r in results if r.startswith("No files"))
        errors = sum(1 for r in results if r.startswith("Error"))
        
        self.logger.info(f"\nMerging summary:")
        self.logger.info(f"Total tasks: {len(results)}")
        self.logger.info(f"Successfully processed: {successful}")
        self.logger.info(f"Skipped (already exist): {skipped}")
        self.logger.info(f"No files found: {no_files}")
        self.logger.info(f"Errors: {errors}")
        
        # Log detailed errors if any
        if errors > 0:
            self.logger.error("Detailed errors:")
            for r in results:
                if r.startswith("Error"):
                    self.logger.error(r)


def main():
    """Main entry point for the merge predictions pipeline."""
    try:
        # Initialize and run merger
        merger = PredictionMerger()
        merger.run_parallel_merge()
        
        merger.logger.info("Prediction merging completed successfully!")
        
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Prediction merging failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()