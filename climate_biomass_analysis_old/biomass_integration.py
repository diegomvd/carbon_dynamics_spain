"""
Biomass-climate data integration pipeline.

Resamples high-resolution biomass change data to climate resolution and integrates
with bioclimatic anomalies to create machine learning training datasets. Handles
spatial data preservation and dataset construction for model training.

Author: Diego Bengochea
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import os
import glob
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pathlib import Path
import yaml
import logging


def setup_logging():
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path="climate_biomass_config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise


def resample_biomass_to_climate(biomass_raster_path, climate_raster_path, output_path=None,
                               resampling_method=Resampling.average):
    """
    Resample a high-resolution biomass raster to match the resolution and extent of a climate raster.
    
    Args:
        biomass_raster_path (str or Path): Path to the high-resolution (100m) biomass raster
        climate_raster_path (str or Path): Path to the lower-resolution (10km) climate raster 
                                         that will be used as reference
        output_path (str or Path, optional): Path where the resampled biomass raster will be saved
                                           If None, will use the biomass filename with '_resampled' suffix
        resampling_method (rasterio.warp.Resampling, optional): Method used for resampling. 
                                                              Default is average, which is often good for continuous data
                                                              like biomass when downsampling. Other options include:
                                                              - Resampling.nearest: nearest neighbor (good for categorical data)
                                                              - Resampling.bilinear: bilinear interpolation
                                                              - Resampling.cubic: cubic interpolation
                                                              - Resampling.sum: sum of all values (useful for keeping mass/count consistent)
    
    Returns:
        str: Path to resampled output raster
    """
    logger = setup_logging()
    
    # Create output path if not provided
    if output_path is None:
        biomass_path = Path(biomass_raster_path)
        output_path = biomass_path.parent / f"{biomass_path.stem}_resampled{biomass_path.suffix}"
    
    # Open the climate raster to get its properties
    with rasterio.open(climate_raster_path) as climate_src:
        # Get metadata from the climate raster
        dst_crs = climate_src.crs
        dst_transform = climate_src.transform
        dst_height = climate_src.height
        dst_width = climate_src.width
        dst_bounds = climate_src.bounds
        
        # Get the nodata value from climate raster or use a default
        dst_nodata = climate_src.nodata if climate_src.nodata is not None else -9999
        
        # Read the metadata only - we'll use this to create the output profile
        profile = climate_src.profile
    
    # Open the biomass raster to get its properties
    with rasterio.open(biomass_raster_path) as biomass_src:
        # Get the source CRS
        src_crs = biomass_src.crs
        
        # Check if projections match, warn if they don't
        if src_crs != dst_crs:
            logger.warning(f"Input rasters have different projections: {src_crs} vs {dst_crs}")
            logger.info("Reprojection will be performed, but verify results carefully.")
        
        # Get the nodata value from biomass raster
        src_nodata = biomass_src.nodata
        
        # Read the biomass data, assuming it's a single band
        biomass_data = biomass_src.read(1)
        
        # Prepare the output array (same dimensions as climate raster)
        dst_array = np.zeros((dst_height, dst_width), dtype=profile['dtype'])
        
        # Apply the mask to handle nodata values properly
        if src_nodata is not None:
            mask = biomass_data == src_nodata
            biomass_data = np.ma.masked_array(biomass_data, mask=mask)
        
        # Update the output profile for the new raster
        profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': dst_nodata,
            'count': 1  # Assuming the biomass data is single-band
        })
        
        # Reproject and resample
        reproject(
            source=biomass_data,
            destination=dst_array,
            src_transform=biomass_src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling_method,
            num_threads=4  # Use multiple threads for faster processing
        )
        
        # Write the output raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_array, 1)
            
        logger.info(f"Successfully resampled biomass raster to match climate raster.")
        logger.info(f"Output saved to: {output_path}")
        
        return str(output_path)


def batch_resample_biomass(biomass_dir, climate_raster_path, output_dir=None, 
                          pattern="TBD*.tif", resampling_method=Resampling.average):
    """
    Batch process multiple biomass rasters to resample them to a climate raster resolution.
    
    Args:
        biomass_dir (str or Path): Directory containing biomass rasters to process
        climate_raster_path (str or Path): Path to the climate raster to use as reference 
                                         for resolution and extent
        output_dir (str or Path, optional): Directory where resampled rasters will be saved
                                          If None, will create a 'resampled' subdirectory in biomass_dir
        pattern (str, optional): Glob pattern to match biomass raster files
        resampling_method (rasterio.warp.Resampling, optional): Method used for resampling
        
    Returns:
        list: Paths to all resampled output rasters
    """
    logger = setup_logging()
    biomass_dir = Path(biomass_dir)
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = biomass_dir / "resampled"
    else:
        output_dir = Path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all biomass rasters matching the pattern
    biomass_files = list(biomass_dir.glob(pattern))
    
    if not biomass_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {biomass_dir}")
        return []
    
    logger.info(f"Found {len(biomass_files)} biomass rasters to process")
    
    output_files = []
    
    # Process each biomass raster
    for i, biomass_file in enumerate(biomass_files):
        logger.info(f"Processing {i+1}/{len(biomass_files)}: {biomass_file.name}")
        
        output_path = output_dir / f"{biomass_file.stem}_resampled{biomass_file.suffix}"
        
        try:
            result = resample_biomass_to_climate(
                biomass_file, 
                climate_raster_path, 
                output_path, 
                resampling_method
            )
            output_files.append(result)
        except Exception as e:
            logger.error(f"Error processing {biomass_file.name}: {str(e)}")
    
    logger.info(f"Batch processing complete. {len(output_files)} files successfully resampled.")
    return output_files


def harmonize_raster(raster_path, reference_shape, reference_transform, reference_crs):
    """
    Harmonize a raster to match reference parameters.
    
    Args:
        raster_path (str): Path to raster to harmonize
        reference_shape (tuple): Reference raster shape (height, width)
        reference_transform (Affine): Reference transform
        reference_crs: Reference CRS
        
    Returns:
        numpy.ndarray: Harmonized raster data
    """
    with rasterio.open(raster_path) as src:
        if src.shape != reference_shape or src.crs != reference_crs:
            # Prepare destination array
            dst_array = np.zeros(reference_shape, dtype=src.dtypes[0])
            
            # Reproject
            reproject(
                source=src.read(1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear
            )
            
            return dst_array
        else:
            return src.read(1)


def create_ml_dataset(diff_files, anomaly_dir, output_file):
    """
    Create a machine learning dataset combining biomass symmetric differences with climate anomalies.
    
    Args:
        diff_files (list): List of paths to biomass symmetric difference files
        anomaly_dir (str): Directory containing bioclimatic anomaly files
        output_file (str): Path to save the output dataset (CSV)
    
    Returns:
        pandas.DataFrame: The created ML dataset
    """
    logger = setup_logging()
    logger.info("Creating ML dataset from biomass differences and climate anomalies...")
    
    # Data collection for all years
    all_data = []
    
    for diff_file in diff_files:
        # Extract years from filename
        basename = os.path.basename(diff_file)
        logger.info(f"Processing: {basename}")
        years_str = basename.split('_')[5]
        logger.info(f"Years string: {years_str}")
        start_year, end_year = years_str.split('-')

        start_year = int(start_year[:4])
        end_year = int(end_year[:4])
        logger.info(f"Processing biomass difference for {start_year}-{end_year}...")
        
        # Load biomass difference data
        with rasterio.open(diff_file) as src:
            biomass_diff = src.read(1)
            nodata = src.nodata
            reference_shape = biomass_diff.shape
            reference_transform = src.transform
            reference_crs = src.crs
            biomass_mask = biomass_diff != nodata if nodata is not None else np.ones_like(biomass_diff, dtype=bool)

        logger.info(f"Reference shape: {reference_shape}, CRS: {reference_crs}")
        
        # Find all anomaly files for this year
        year_pattern = f"{start_year}Sep-{end_year}Aug"
        
        # Get 1-year anomalies
        anomaly_1yr_dir = os.path.join(anomaly_dir, f"anomalies_{year_pattern}")
        anomaly_1yr_files = glob.glob(os.path.join(anomaly_1yr_dir, "*.tif"))
        
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
                        logger.warning(f"Shape/CRS mismatch for {anomaly_file}. Harmonizing...")
                        try:
                            anomaly_data[var_name] = harmonize_raster(
                                anomaly_file, 
                                reference_shape, 
                                reference_transform, 
                                reference_crs
                            )
                        except Exception as e:
                            logger.error(f"Error harmonizing {anomaly_file}: {e}")
                            continue
                    else:
                        anomaly_data[var_name] = src.read(1)

                    # Update valid mask
                    anomaly_mask = anomaly_data[var_name] != -9999
                    valid_mask = valid_mask & anomaly_mask
                   
            except Exception as e:
                logger.error(f"Error reading {anomaly_file}: {e}")
                continue
        
        # Check how many variables were loaded
        if not anomaly_data:
            logger.warning(f"No anomaly data found for {year_pattern}. Skipping...")
            continue
        
        logger.info(f"Loaded {len(anomaly_data)} anomaly variables for {year_pattern}")

        # Get indices of valid pixels
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) == 0:
            logger.warning(f"No valid data points for {year_pattern}")
            continue
        
        # Create data points
        year_data = []
        
        for y, x in zip(valid_indices[0], valid_indices[1]):
            geo_x, geo_y = reference_transform * (x, y)
            # Create data point with just year info and biomass diff
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
        
        logger.info(f"Added {len(year_data)} data points for {year_pattern}")
        all_data.extend(year_data)
    
    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Remove rows with NaN values
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} rows with NaN values")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"Dataset created with {len(df)} data points. Saved to {output_file}")
        return df
    else:
        logger.warning("No data points found.")
        return None


def run_biomass_integration_pipeline(config):
    """
    Execute the complete biomass-climate integration workflow.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pandas.DataFrame: Final ML dataset
    """
    logger = setup_logging()
    logger.info("Starting biomass-climate integration pipeline...")
    
    # Extract config parameters
    biomass_diff_dir = config['paths']['biomass_diff_dir']
    anomaly_dir = config['paths']['anomaly_dir']
    output_dir = config['paths']['temp_resampled_dir']
    training_dataset_path = config['paths']['training_dataset']
    biomass_pattern = config['biomass']['pattern']
    resampling_method = getattr(Resampling, config['biomass']['resampling_method'])
    
    # Get reference raster from first available anomaly file
    first_anomaly_dir = None
    for item in os.listdir(anomaly_dir):
        if item.startswith("anomalies_") and os.path.isdir(os.path.join(anomaly_dir, item)):
            first_anomaly_dir = os.path.join(anomaly_dir, item)
            break
    
    if not first_anomaly_dir:
        logger.error("No anomaly directories found. Run bioclimatic calculation first.")
        return None
    
    # Find first anomaly file as reference
    anomaly_files = glob.glob(os.path.join(first_anomaly_dir, "*.tif"))
    if not anomaly_files:
        logger.error("No anomaly files found in directory. Run bioclimatic calculation first.")
        return None
    
    ref_raster_path = anomaly_files[0]
    logger.info(f"Using reference raster: {ref_raster_path}")
    
    # Step 1: Batch resample biomass data to climate resolution
    logger.info("Resampling biomass difference files to climate resolution...")
    diff_files = batch_resample_biomass(
        biomass_diff_dir, 
        ref_raster_path, 
        output_dir=output_dir, 
        pattern=biomass_pattern, 
        resampling_method=resampling_method
    )
    
    if not diff_files:
        logger.error("No biomass difference files found.")
        return None
    
    logger.info(f"Found {len(diff_files)} biomass difference files.")
    
    # Step 2: Create ML dataset
    logger.info("Creating ML dataset...")
    ml_data = create_ml_dataset(diff_files, anomaly_dir, training_dataset_path)
    
    # Print dataset summary
    if ml_data is not None:
        logger.info("Dataset Summary:")
        logger.info(f"Total samples: {len(ml_data)}")
        logger.info("Feature statistics:")
        logger.info(ml_data.describe())
        
        # Count samples per year transition
        year_counts = ml_data.groupby(['year_start', 'year_end']).size()
        logger.info("Samples per year transition:")
        logger.info(year_counts)
        
        # Print column names for reference
        logger.info("Features available in the dataset:")
        logger.info(ml_data.columns.tolist())
        
        logger.info("Biomass-climate integration pipeline completed successfully!")
    else:
        logger.error("Failed to create ML dataset.")
    
    return ml_data


def main():
    """Main function to run biomass-climate integration."""
    logger = setup_logging()
    logger.info("Starting biomass-climate integration pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Run integration pipeline
        results = run_biomass_integration_pipeline(config)
        
        if results is not None:
            logger.info("Biomass-climate integration completed successfully!")
        else:
            logger.error("Biomass-climate integration failed!")
            
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
