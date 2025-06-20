"""
Bioclimatic variables calculation and anomaly analysis pipeline.

Calculates bioclimatic variables (bio1-bio19) from monthly temperature and precipitation
data, harmonizes rasters to common grids, and computes anomalies for analysis periods.
Supports both calendar and rolling year periods (Sep-Aug).

Author: Diego Bengochea
"""

import rasterio
import numpy as np
import os
import glob
from datetime import datetime
import pandas as pd
from pathlib import Path
import rasterio.warp
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


def extract_date(filename):
    """
    Extract date from filename with pattern {var}_{YYYY}-{MM}.tif
    
    Args:
        filename (str): Input filename to parse
        
    Returns:
        datetime: Parsed datetime object
    """
    basename = os.path.basename(filename)
    date_part = basename.split('_')  # Gets "YYYY-MM"
    year = date_part[1]
    month = date_part[2].split('.')[0]
    return datetime(int(year), int(month), 1)


def harmonize_rasters(raster_files, output_dir, reference_file=None):
    """
    Harmonize all raster files to the same dimensions and coordinate system.
    
    Args:
        raster_files (list): List of paths to raster files to harmonize
        output_dir (str): Directory to save harmonized rasters
        reference_file (str, optional): File to use as reference for dimensions and CRS.
                                      If None, the first file is used.
    
    Returns:
        list: Paths to the harmonized raster files
    """
    logger = setup_logging()
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine reference file if not provided
    if not reference_file and raster_files:
        reference_file = raster_files[0]
    
    # Get reference metadata
    with rasterio.open(reference_file) as src:
        reference_crs = src.crs
        reference_transform = src.transform
        reference_height = src.height
        reference_width = src.width
        reference_bounds = src.bounds
    
    logger.info(f"Reference dimensions: {reference_height} x {reference_width}")
    logger.info(f"Reference bounds: {reference_bounds}")
    
    harmonized_files = []
    
    for file in raster_files:
        # Get original filename for output
        basename = os.path.basename(file)
        output_file = os.path.join(output_dir, basename)
        
        # Check if file already exists
        if os.path.exists(output_file):
            harmonized_files.append(output_file)
            continue
        
        with rasterio.open(file) as src:
            # Check if resampling is needed
            if (src.height == reference_height and 
                src.width == reference_width and 
                src.crs == reference_crs and
                src.bounds == reference_bounds):
                # No resampling needed, just copy the file
                harmonized_files.append(file)  # Use original file
                continue
            
            # Resampling is needed
            logger.info(f"Resampling {basename}")
            
            # Prepare output profile
            profile = src.profile.copy()
            profile.update({
                'height': reference_height,
                'width': reference_width,
                'transform': reference_transform,
                'crs': reference_crs
            })
            
            # Resample the data
            with rasterio.open(output_file, 'w', **profile) as dst:
                # Read source data
                source_data = src.read(1)
                
                # Prepare destination data
                destination = np.zeros((reference_height, reference_width), dtype=profile['dtype'])
                
                # Perform the resampling
                resampled_data = rasterio.warp.reproject(
                    source=source_data,
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=reference_transform,
                    dst_crs=reference_crs,
                    resampling=rasterio.warp.Resampling.bilinear
                )[0]
                
                dst.write(resampled_data, 1)
            
            harmonized_files.append(output_file)
    
    return harmonized_files


def calculate_bioclim_variables(temp_files, precip_files, output_dir, start_year=1985, 
                              end_year=2015, rolling=True):
    """
    Calculate bioclimatic variables from monthly temperature and precipitation files.
    
    Args:
        temp_files (list): List of paths to temperature raster files
        precip_files (list): List of paths to precipitation raster files
        output_dir (str): Directory to save output bioclim rasters
        start_year (int): Start year for climatology period
        end_year (int): End year for climatology period
        rolling (bool): If True, use rolling years (Sep-Aug) instead of calendar years
        
    Returns:
        dict: Dictionary containing calculated bioclimatic variables
    """
    logger = setup_logging()
    logger.info(f"Calculating bioclimatic variables for period {start_year}-{end_year}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dates and sort files
    temp_files_with_dates = [(f, extract_date(f)) for f in temp_files]
    precip_files_with_dates = [(f, extract_date(f)) for f in precip_files]
    
    temp_files_with_dates.sort(key=lambda x: x[1])
    precip_files_with_dates.sort(key=lambda x: x[1])
    
    # Filter files to the reference period
    if rolling:
        # For rolling years (Sep to Aug)
        temp_files_filtered = []
        precip_files_filtered = []
        
        for file, date in temp_files_with_dates:
            # If month is Sep-Dec, use the year as is
            # If month is Jan-Aug, use the previous year
            effective_year = date.year if date.month >= 9 else date.year - 1
            if start_year <= effective_year <= end_year - 1:  # -1 because end_year refers to the Aug end
                temp_files_filtered.append((file, date))
        
        for file, date in precip_files_with_dates:
            effective_year = date.year if date.month >= 9 else date.year - 1
            if start_year <= effective_year <= end_year - 1:
                precip_files_filtered.append((file, date))
    else:
        # For calendar years
        temp_files_filtered = [(f, d) for f, d in temp_files_with_dates if start_year <= d.year <= end_year]
        precip_files_filtered = [(f, d) for f, d in precip_files_with_dates if start_year <= d.year <= end_year]
    
    # Group files by month
    temp_by_month = {month: [] for month in range(1, 13)}
    precip_by_month = {month: [] for month in range(1, 13)}
    
    for file, date in temp_files_filtered:
        temp_by_month[date.month].append(file)
    
    for file, date in precip_files_filtered:
        precip_by_month[date.month].append(file)
    
    # Get metadata from first file for output rasters
    with rasterio.open(temp_files_filtered[0][0]) as src:
        profile = src.profile
    
    # Calculate monthly climatologies
    temp_climatology = {}
    precip_climatology = {}
    
    for month in range(1, 13):
        # Temperature
        temp_data = []
        for file in temp_by_month[month]:
            with rasterio.open(file) as src:
                temp_data.append(src.read(1))
        
        # Now we can safely stack and average since all files have been harmonized
        temp_climatology[month] = np.mean(np.stack(temp_data), axis=0) if temp_data else None
        
        # Precipitation
        precip_data = []
        for file in precip_by_month[month]:
            with rasterio.open(file) as src:
                precip_data.append(src.read(1))
        
        precip_climatology[month] = np.mean(np.stack(precip_data), axis=0) if precip_data else None
    
    # Check if we have data for all months
    missing_temp_months = [m for m in range(1, 13) if temp_climatology[m] is None]
    missing_precip_months = [m for m in range(1, 13) if precip_climatology[m] is None]
    
    if missing_temp_months or missing_precip_months:
        logger.warning(f"Missing temperature data for months: {missing_temp_months}")
        logger.warning(f"Missing precipitation data for months: {missing_precip_months}")
        logger.error("Cannot calculate all bioclim variables without complete monthly data.")
        return None
    
    # Calculate bioclim variables
    
    # BIO1 - Annual Mean Temperature
    annual_mean_temp = np.mean([temp_climatology[m] for m in range(1, 13)], axis=0)
    
    # BIO4 - Temperature Seasonality (standard deviation * 100)
    temp_seasonality = np.std([temp_climatology[m] for m in range(1, 13)], axis=0) * 100
    
    # BIO5 - Max Temperature of Warmest Month
    max_temp_warmest = np.max([temp_climatology[m] for m in range(1, 13)], axis=0)
    
    # BIO6 - Min Temperature of Coldest Month
    min_temp_coldest = np.min([temp_climatology[m] for m in range(1, 13)], axis=0)
    
    # BIO7 - Temperature Annual Range (BIO5-BIO6)
    temp_annual_range = max_temp_warmest - min_temp_coldest
    
    # BIO12 - Annual Precipitation
    annual_precip = np.sum([precip_climatology[m] for m in range(1, 13)], axis=0)
    
    # BIO13 - Precipitation of Wettest Month
    precip_wettest = np.max([precip_climatology[m] for m in range(1, 13)], axis=0)
    
    # BIO14 - Precipitation of Driest Month
    precip_driest = np.min([precip_climatology[m] for m in range(1, 13)], axis=0)
    
    # BIO15 - Precipitation Seasonality (Coefficient of Variation)
    precip_mean = np.mean([precip_climatology[m] for m in range(1, 13)], axis=0)
    precip_std = np.std([precip_climatology[m] for m in range(1, 13)], axis=0)
    # Avoid division by zero
    precip_seasonality = np.zeros_like(precip_mean)
    mask = precip_mean > 0
    precip_seasonality[mask] = (precip_std[mask] / precip_mean[mask]) * 100
    
    # Define quarters differently for rolling years if needed
    if rolling:
        quarters = [
            [9, 10, 11],  # Fall (Sep, Oct, Nov)
            [12, 1, 2],   # Winter
            [3, 4, 5],    # Spring
            [6, 7, 8]     # Summer
        ]
    else:
        quarters = [
            [12, 1, 2],  # Winter (Dec, Jan, Feb)
            [3, 4, 5],   # Spring
            [6, 7, 8],   # Summer
            [9, 10, 11]  # Fall
        ]
    
    quarter_temp = {}
    quarter_precip = {}
    
    for i, q in enumerate(quarters):
        temp_q = np.mean([temp_climatology[m] for m in q], axis=0)
        precip_q = np.sum([precip_climatology[m] for m in q], axis=0)
        quarter_temp[i] = temp_q
        quarter_precip[i] = precip_q
    
    # Find warmest, coldest, wettest, driest quarters on a pixel-by-pixel basis
    quarter_temp_array = np.array([quarter_temp[i] for i in range(4)])
    quarter_precip_array = np.array([quarter_precip[i] for i in range(4)])
    
    warmest_q_pixels = np.argmax(quarter_temp_array, axis=0)
    coldest_q_pixels = np.argmin(quarter_temp_array, axis=0)
    wettest_q_pixels = np.argmax(quarter_precip_array, axis=0)
    driest_q_pixels = np.argmin(quarter_precip_array, axis=0)
    
    # Calculate quarter-based bioclim variables on a pixel basis
    # BIO8 - Mean Temperature of Wettest Quarter
    temp_wettest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        temp_wettest_quarter[wettest_q_pixels == i] = quarter_temp[i][wettest_q_pixels == i]
    
    # BIO9 - Mean Temperature of Driest Quarter
    temp_driest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        temp_driest_quarter[driest_q_pixels == i] = quarter_temp[i][driest_q_pixels == i]
    
    # BIO10 - Mean Temperature of Warmest Quarter
    temp_warmest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        temp_warmest_quarter[warmest_q_pixels == i] = quarter_temp[i][warmest_q_pixels == i]
    
    # BIO11 - Mean Temperature of Coldest Quarter
    temp_coldest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        temp_coldest_quarter[coldest_q_pixels == i] = quarter_temp[i][coldest_q_pixels == i]
    
    # BIO16 - Precipitation of Wettest Quarter
    precip_wettest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        precip_wettest_quarter[wettest_q_pixels == i] = quarter_precip[i][wettest_q_pixels == i]
    
    # BIO17 - Precipitation of Driest Quarter
    precip_driest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        precip_driest_quarter[driest_q_pixels == i] = quarter_precip[i][driest_q_pixels == i]
    
    # BIO18 - Precipitation of Warmest Quarter
    precip_warmest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        precip_warmest_quarter[warmest_q_pixels == i] = quarter_precip[i][warmest_q_pixels == i]
    
    # BIO19 - Precipitation of Coldest Quarter
    precip_coldest_quarter = np.zeros_like(annual_mean_temp)
    for i in range(4):
        precip_coldest_quarter[coldest_q_pixels == i] = quarter_precip[i][coldest_q_pixels == i]
    
    # Save bioclim variables
    year_label = f"{start_year}-{end_year}" if not rolling else f"{start_year}Sep-{end_year}Aug"
    
    bioclim_vars = {
        'bio1': annual_mean_temp,
        'bio4': temp_seasonality,
        'bio5': max_temp_warmest,
        'bio6': min_temp_coldest,
        'bio7': temp_annual_range,
        'bio12': annual_precip,
        'bio13': precip_wettest,
        'bio14': precip_driest,
        'bio15': precip_seasonality,
        'bio8': temp_wettest_quarter,
        'bio9': temp_driest_quarter,
        'bio10': temp_warmest_quarter,
        'bio11': temp_coldest_quarter,
        'bio16': precip_wettest_quarter,
        'bio17': precip_driest_quarter,
        'bio18': precip_warmest_quarter,
        'bio19': precip_coldest_quarter
    }
    
    # Write output rasters
    for var_name, data in bioclim_vars.items():
        output_path = os.path.join(output_dir, f"{var_name}_{year_label}.tif")
        
        # Update profile for output
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=-9999
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)
    
    logger.info(f"Bioclimatic variables calculated and saved to {output_dir}")
    return bioclim_vars


def calculate_bioclim_anomalies(temp_files, precip_files, bioclim_dir, output_dir, 
                              start_year=2015, end_year=2024, rolling=True):
    """
    Calculate annual and cumulative anomalies for bioclimatic variables.
    
    Args:
        temp_files (list): List of paths to temperature raster files
        precip_files (list): List of paths to precipitation raster files
        bioclim_dir (str): Directory containing bioclim reference files (30-year climatology)
        output_dir (str): Directory to save output anomaly rasters
        start_year (int): Start year for anomaly calculation (for rolling years, this is the year of September)
        end_year (int): End year for anomaly calculation (for rolling years, this is the year of August)
        rolling (bool): If True, use rolling years (Sep-Aug) instead of calendar years
        
    Returns:
        dict: Dictionary containing yearly anomalies
    """
    logger = setup_logging()
    logger.info(f"Calculating bioclimatic anomalies for period {start_year}-{end_year}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the reference bioclim variables (30-year climatology)
    reference_pattern = "1985Sep-2015Aug" if rolling else "1985_2015"
    reference_bioclim = {}
    
    # List of bioclim variables to process
    bioclim_vars = [
        'bio1', 'bio4', 'bio5', 'bio6', 'bio7', 'bio12', 
        'bio13', 'bio14', 'bio15', 'bio8', 'bio9', 'bio10', 
        'bio11', 'bio16', 'bio17', 'bio18', 'bio19'
    ]
    
    # Load reference bioclim rasters
    for var in bioclim_vars:
        ref_file = os.path.join(bioclim_dir, f"{var}_{reference_pattern}.tif")
        if os.path.exists(ref_file):
            with rasterio.open(ref_file) as src:
                reference_bioclim[var] = src.read(1)
                # Save profile from first file for output
                if var == bioclim_vars[0]:
                    profile = src.profile
        else:
            logger.warning(f"Reference file not found for {var}: {ref_file}")
    
    # Function to calculate bioclim variables for a specific year
    def calculate_year_bioclim(year):
        year_temp_files = []
        year_precip_files = []
        
        # Gather files for this year
        for file in temp_files:
            date = extract_date(file)
            if rolling:
                # For rolling years (Sep to Aug)
                effective_year = date.year if date.month >= 9 else date.year - 1
            else:
                effective_year = date.year
            
            if effective_year == year:
                year_temp_files.append(file)
        
        for file in precip_files:
            date = extract_date(file)
            if rolling:
                effective_year = date.year if date.month >= 9 else date.year - 1
            else:
                effective_year = date.year
            
            if effective_year == year:
                year_precip_files.append(file)
        
        # Check if we have complete data
        if len(year_temp_files) < 12 or len(year_precip_files) < 12:
            logger.warning(f"Incomplete data for year {year}. Found {len(year_temp_files)} temperature files and {len(year_precip_files)} precipitation files.")
            return None
        
        # Create temporary directory for this year's bioclim
        year_dir = os.path.join(output_dir, f"temp_bioclim_{year}")
        os.makedirs(year_dir, exist_ok=True)
        
        # Calculate bioclim variables for this year
        year_bioclim = calculate_bioclim_variables(
            year_temp_files,
            year_precip_files,
            year_dir,
            start_year=year,
            end_year=year+1,
            rolling=rolling
        )
        
        return year_bioclim
    
    # Calculate bioclim variables for each year in the analysis period
    yearly_bioclim = {}
    for year in range(start_year, end_year + 1):
        logger.info(f"Calculating bioclim variables for year {year}")
        yearly_bioclim[year] = calculate_year_bioclim(year)
    
    # Calculate anomalies for each year
    yearly_anomalies = {}
    for year in range(start_year, end_year + 1):
        if yearly_bioclim[year] is None:
            continue
        
        # Create directory for this year's anomalies
        year_label = f"{year}Sep-{year+1}Aug" if rolling else f"{year}"
        year_anomaly_dir = os.path.join(output_dir, f"anomalies_{year_label}")
        os.makedirs(year_anomaly_dir, exist_ok=True)
        
        # Calculate anomalies for each bioclim variable
        year_anomalies = {}
        for var in bioclim_vars:
            if var not in reference_bioclim or var not in yearly_bioclim[year]:
                continue
            
            # Calculate anomaly
            anomaly = yearly_bioclim[year][var] - reference_bioclim[var]
            year_anomalies[var] = anomaly
            
            # Save anomaly raster
            anomaly_path = os.path.join(year_anomaly_dir, f"{var}_anomaly_{year_label}.tif")
            with rasterio.open(anomaly_path, 'w', **profile) as dst:
                dst.write(anomaly.astype(rasterio.float32), 1)
        
        yearly_anomalies[year] = year_anomalies
    
    # Calculate cumulative anomalies (2-year and 3-year)
    for year in range(start_year + 2, end_year + 1):  # Start from year that has 2 years before it
        # Check if we have all required data
        if not all(yearly_anomalies.get(y) for y in range(year-2, year+1)):
            continue
        
        # Calculate 2-year and 3-year cumulative anomalies
        for cum_years in [2, 3]:
            if year < start_year + cum_years - 1:
                continue
                
            prior_years = list(range(year - cum_years + 1, year + 1))
            
            year_label = f"{year}Sep-{year+1}Aug" if rolling else f"{year}"
            cum_dir = os.path.join(output_dir, f"anomalies_{cum_years}yr_{year_label}")
            os.makedirs(cum_dir, exist_ok=True)
            
            for var in bioclim_vars:
                if not all(var in yearly_anomalies[y] for y in prior_years):
                    continue
                
                # For all variables, use the average of anomalies (not sum)
                cum_anomaly = np.mean([yearly_anomalies[y][var] for y in prior_years], axis=0)
                
                # Save cumulative anomaly raster
                cum_path = os.path.join(cum_dir, f"{var}_anomaly_{cum_years}yr_{year_label}.tif")
                with rasterio.open(cum_path, 'w', **profile) as dst:
                    dst.write(cum_anomaly.astype(rasterio.float32), 1)
    
    logger.info(f"Bioclim anomalies calculated successfully and saved to {output_dir}")
    return yearly_anomalies


def run_bioclim_pipeline(config):
    """
    Execute the full bioclimatic workflow with harmonization.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Results from anomaly calculations
    """
    logger = setup_logging()
    logger.info("Starting bioclimatic variables calculation pipeline...")
    
    # Extract config parameters
    data_dir = config['paths']['climate_outputs']
    harmonized_dir = config['paths']['harmonized_dir']
    bioclim_dir = config['paths']['bioclim_dir']
    anomaly_dir = config['paths']['anomaly_dir']
    
    reference_period = config['time_periods']['reference']
    analysis_period = config['time_periods']['analysis']
    
    # Find all temperature and precipitation files
    temp_files = glob.glob(os.path.join(data_dir, "2t_*.tif"))
    precip_files = glob.glob(os.path.join(data_dir, "tp_*.tif"))
    
    # Create directories for harmonized files
    harmonized_temp_dir = os.path.join(harmonized_dir, "temperature")
    harmonized_precip_dir = os.path.join(harmonized_dir, "precipitation")
    
    logger.info(f"Found {len(temp_files)} temperature files and {len(precip_files)} precipitation files")
    
    # Step 0: Harmonize all raster files to a common grid
    logger.info("Harmonizing temperature files...")
    harmonized_temp_files = harmonize_rasters(temp_files, harmonized_temp_dir)
    
    logger.info("Harmonizing precipitation files...")
    harmonized_precip_files = harmonize_rasters(precip_files, harmonized_precip_dir)
    
    # Step 1: Calculate bioclimatic variables for the reference period
    logger.info(f"Calculating bioclimatic variables for reference period ({reference_period['start_year']}-{reference_period['end_year']})...")
    reference_bioclim = calculate_bioclim_variables(
        harmonized_temp_files, 
        harmonized_precip_files, 
        bioclim_dir, 
        start_year=reference_period['start_year'], 
        end_year=reference_period['end_year'],
        rolling=reference_period['rolling']
    )
    
    if reference_bioclim is None:
        logger.error("Error calculating reference bioclimatic variables. Exiting.")
        return None
    
    logger.info(f"Reference bioclimatic variables calculated successfully and saved to {bioclim_dir}")
    
    # Step 2: Calculate bioclimatic anomalies for the analysis period
    logger.info(f"Calculating bioclimatic anomalies for analysis period ({analysis_period['start_year']}-{analysis_period['end_year']})...")
    yearly_anomalies = calculate_bioclim_anomalies(
        harmonized_temp_files,
        harmonized_precip_files,
        bioclim_dir,
        anomaly_dir,
        start_year=analysis_period['start_year'],
        end_year=analysis_period['end_year'],
        rolling=analysis_period['rolling']
    )
    
    logger.info("Bioclimatic pipeline completed successfully!")
    logger.info(f"Harmonized files: {harmonized_dir}")
    logger.info(f"Reference climatology: {bioclim_dir}")
    logger.info(f"Anomalies: {anomaly_dir}")
    
    return yearly_anomalies


def main():
    """Main function to run bioclimatic variables calculation."""
    logger = setup_logging()
    logger.info("Starting bioclimatic variables calculation pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Run bioclimatic pipeline
        results = run_bioclim_pipeline(config)
        
        if results is not None:
            logger.info("Bioclimatic variables calculation completed successfully!")
        else:
            logger.error("Bioclimatic variables calculation failed!")
            
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
