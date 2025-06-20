"""
Downsampling and merging of canopy height predictions into country-wide mosaics.

This script provides the final post-processing step for canopy height predictions,
downsampling high-resolution tiles to a target resolution and merging them into
seamless country-wide mosaics for each year. Implements efficient resampling
strategies and memory management for large-scale raster processing.

The process groups raster files by year, applies configurable downsampling,
and creates compressed, tiled outputs optimized for analysis and distribution.

Author: Diego Bengochea
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Third-party imports
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

# Local imports
from config import load_config, setup_logging, create_output_directory


class FinalMerger:
    """
    Final merger for creating country-wide canopy height mosaics.
    
    This class handles the downsampling and merging of processed height tiles
    into final country-wide products at the target resolution.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the final merger with configuration.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config['logging']['level'])
        
        # Get final merge configuration
        self.merge_config = self.config['post_processing']['final_merge']
        
        # Create output directory
        create_output_directory(Path(self.merge_config['output_dir']))
        
        # Create temporary directory for processing
        self.temp_dir = Path(self.merge_config['output_dir']) / 'temp'
        create_output_directory(self.temp_dir)
        
        self.logger.info("Final merger initialized")
        self.logger.info(f"Input directory: {self.merge_config['input_dir']}")
        self.logger.info(f"Output directory: {self.merge_config['output_dir']}")
        self.logger.info(f"Target resolution: {self.merge_config['target_resolution']}m")
        
        # Map resampling method string to enum
        self.resampling_method = getattr(
            Resampling, 
            self.merge_config['resampling_method'].upper()
        )
    
    def find_rasters_by_year(self) -> Dict[str, List[str]]:
        """
        Group raster files by year.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping years to file lists
        """
        rasters_by_year = defaultdict(list)
        
        input_path = Path(self.merge_config['input_dir'])
        file_pattern = self.merge_config['file_pattern']
        
        for file_path in input_path.glob(file_pattern):
            try:
                # Extract year from filename
                # Assuming format: canopy_height_YYYY_*.tif
                filename_parts = file_path.name.split('_')
                if len(filename_parts) >= 3:
                    year = filename_parts[2]
                    self.logger.debug(f'Found file for year: {year}')
                    rasters_by_year[year].append(str(file_path))
                else:
                    self.logger.warning(f"Could not extract year from filename: {file_path.name}")
            except Exception as e:
                self.logger.warning(f"Error processing filename {file_path.name}: {str(e)}")
                continue
        
        self.logger.info(f"Found {len(rasters_by_year)} years with data")
        for year, files in rasters_by_year.items():
            self.logger.info(f"Year {year}: {len(files)} files")
        
        return rasters_by_year
    
    def downsample_raster(
        self, 
        input_path: str, 
        output_path: str
    ) -> None:
        """
        Downsample a single raster to the target resolution.
        
        Args:
            input_path (str): Path to input raster
            output_path (str): Path to output raster
        """
        with rasterio.open(input_path) as src:
            # Calculate scaling factor
            scale_factor = self.merge_config['target_resolution'] / src.res[0]
            
            # Calculate new dimensions
            new_height = int(src.height / scale_factor)
            new_width = int(src.width / scale_factor)
            
            # Create output transform
            transform = rasterio.Affine(
                self.merge_config['target_resolution'], 0.0, src.bounds.left,
                0.0, -self.merge_config['target_resolution'], src.bounds.top
            )
            
            # Update profile for output
            profile = src.profile.copy()
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': transform,
                'compress': self.merge_config['compression']
            })
            
            # Perform the resampling using WarpedVRT
            with WarpedVRT(
                src,
                height=new_height,
                width=new_width,
                resampling=self.resampling_method,
                transform=transform
            ) as vrt:
                # Read all bands
                data = vrt.read()
                
                # Write output
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)
    
    def merge_rasters(self, input_files: List[str], output_path: str) -> None:
        """
        Merge multiple rasters into one.
        
        Args:
            input_files (List[str]): List of input file paths
            output_path (str): Path to output merged raster
        """
        # Open all input files
        src_files = [rasterio.open(f) for f in input_files]
        
        try:
            # Merge rasters
            mosaic, out_transform = merge(src_files)
            
            # Get output metadata from first input file
            out_meta = src_files[0].meta.copy()
            
            # Update metadata for merged file
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": self.merge_config['compression'],
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256
            })
            
            # Write merged raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
                
            self.logger.info(f"Successfully merged {len(input_files)} files into {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error merging rasters for {output_path}: {str(e)}")
            raise
        finally:
            # Make sure to close all input files
            for src in src_files:
                try:
                    src.close()
                except:
                    pass
    
    def cleanup_temporary_files(self, temp_files: List[str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            temp_files (List[str]): List of temporary file paths to remove
        """
        for file_path in temp_files:
            try:
                os.remove(file_path)
                self.logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Error removing temporary file {file_path}: {str(e)}")
    
    def process_year(self, year: str, raster_files: List[str]) -> None:
        """
        Process all rasters for a single year.
        
        Args:
            year (str): Year to process
            raster_files (List[str]): List of raster file paths for this year
        """
        self.logger.info(f"Processing year {year} with {len(raster_files)} files")
        
        # Check if final output already exists
        final_output_name = f"canopy_height_{year}_100m.tif"
        final_output_path = Path(self.merge_config['output_dir']) / final_output_name
        
        if final_output_path.exists():
            self.logger.info(f"Final output already exists for year {year}: {final_output_path}")
            return
        
        # Downsample each raster
        downsampled_files = []
        
        for i, raster_file in enumerate(raster_files):
            temp_output_name = f"downsampled_{year}_{i}.tif"
            temp_output_path = self.temp_dir / temp_output_name
            
            self.logger.debug(f"Downsampling {Path(raster_file).name}")
            
            try:
                self.downsample_raster(str(raster_file), str(temp_output_path))
                downsampled_files.append(str(temp_output_path))
                
            except Exception as e:
                self.logger.error(f"Error processing {raster_file}: {str(e)}")
                continue
        
        if not downsampled_files:
            self.logger.warning(f"No files successfully processed for year {year}")
            return
        
        # Merge downsampled rasters
        self.logger.info(f"Merging {len(downsampled_files)} downsampled rasters for year {year}")
        
        try:
            self.merge_rasters(downsampled_files, str(final_output_path))
            self.logger.info(f"Successfully created final mosaic for year {year}: {final_output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating final mosaic for year {year}: {str(e)}")
        
        # Clean up temporary files
        self.cleanup_temporary_files(downsampled_files)
    
    def run_final_merge_pipeline(self) -> None:
        """Run the complete final merge pipeline."""
        self.logger.info("Starting final merge pipeline...")
        
        try:
            # Find all rasters grouped by year
            rasters_by_year = self.find_rasters_by_year()
            
            if not rasters_by_year:
                self.logger.error("No raster files found to process")
                return
            
            # Process each year
            for year, raster_files in rasters_by_year.items():
                try:
                    self.process_year(year, raster_files)
                except Exception as e:
                    self.logger.error(f"Error processing year {year}: {str(e)}")
                    continue
            
            # Remove temporary directory if empty
            try:
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    self.logger.info("Removed empty temporary directory")
            except Exception as e:
                self.logger.warning(f"Error removing temporary directory: {str(e)}")
            
            self.logger.info("Final merge pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Final merge pipeline failed: {str(e)}")
            raise
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of processing results.
        
        Returns:
            Dict[str, Any]: Processing summary statistics
        """
        output_dir = Path(self.merge_config['output_dir'])
        output_files = list(output_dir.glob("canopy_height_*_100m.tif"))
        
        years_processed = []
        total_size_mb = 0
        
        for output_file in output_files:
            # Extract year from filename
            try:
                year = output_file.name.split('_')[2]
                years_processed.append(year)
                
                # Get file size
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                total_size_mb += file_size
                
            except Exception as e:
                self.logger.warning(f"Error processing summary for {output_file.name}: {str(e)}")
        
        summary = {
            "years_processed": sorted(years_processed),
            "total_files": len(output_files),
            "total_size_mb": round(total_size_mb, 2),
            "output_directory": str(output_dir),
            "target_resolution": self.merge_config['target_resolution']
        }
        
        return summary


def main():
    """Main entry point for the final merge pipeline."""
    try:
        # Initialize and run final merger
        merger = FinalMerger()
        merger.run_final_merge_pipeline()
        
        # Print processing summary
        summary = merger.get_processing_summary()
        merger.logger.info("\n" + "="*50)
        merger.logger.info("FINAL PROCESSING SUMMARY")
        merger.logger.info("="*50)
        merger.logger.info(f"Years processed: {', '.join(summary['years_processed'])}")
        merger.logger.info(f"Total files created: {summary['total_files']}")
        merger.logger.info(f"Total output size: {summary['total_size_mb']} MB")
        merger.logger.info(f"Output resolution: {summary['target_resolution']}m")
        merger.logger.info(f"Output directory: {summary['output_directory']}")
        merger.logger.info("="*50)
        
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Final merge pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()