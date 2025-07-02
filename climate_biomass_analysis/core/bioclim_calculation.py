"""
Bioclimatic variables calculation with restored quarter logic.


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
from typing import Dict, List, Optional, Union, Tuple
from rasterio.enums import Resampling

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory, find_files
from shared_utils.central_data_paths_constants import *

class BioclimCalculationPipeline:
    """
    Bioclimatic variables calculation and anomaly analysis pipeline.
    
    This class handles the calculation of bioclimatic variables (bio1-bio19)
    from monthly temperature and precipitation data, and computes climate
    anomalies for specified analysis periods.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the bioclimatic calculator.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='bioclim_calculation',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration
        self.bioclim_config = self.config['bioclim']
        self.climate_config = self.config['climate_processing']
        self.time_periods = self.config['time_periods']
        
        available_vars = ['bio1', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 
                         'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']
        self.bio_variables = [v for v in self.bioclim_config['variables'] if v in available_vars]
        
        # Unit conversions
        self.temp_conversion = self.bioclim_config['temp_conversion']
        self.precip_scaling = self.bioclim_config['precip_scaling']
        
        self.data_dir = CLIMATE_RASTERS_RAW_DIR
        self.harmonized_dir = CLIMATE_HARMONIZED_DIR
        self.bioclim_dir = BIOCLIM_VARIABLES_DIR
        self.anomalies_dir = BIOCLIM_ANOMALIES_DIR


    def run_full_pipeline(self):
        data_dir = self.data_dir

        temp_files = glob.glob(os.path.join(data_dir, self.climate_config['variable_patterns']['temperature']))
        precip_files = glob.glob(os.path.join(data_dir, self.climate_config['variable_patterns']['precipitation']))
        
        self.logger.info(f"Found {len(temp_files)} temperature files and {len(precip_files)} precipitation files")
        
        if not temp_files or not precip_files:
            raise ValueError("No temperature or precipitation files found")

        harmonized_dir = self.harmonized_dir   

        harmonized_temp_dir = harmonized_dir / "temperature"
        harmonized_precip_dir = harmonized_dir / "precipitation"
        
        self.logger.info("Harmonizing raster files...")
        harmonized_temp_files = self.harmonize_rasters(temp_files, harmonized_temp_dir)
        harmonized_precip_files = self.harmonize_rasters(precip_files, harmonized_precip_dir)
 
        bioclim_dir = self.bioclim_dir
        reference_bioclim = self.calculate_bioclim_variables(
            harmonized_temp_files,
            harmonized_precip_files,
            self.bioclim_dir,
            start_year=self.time_periods['reference']['start_year'],
            end_year=self.time_periods['reference']['end_year'],
            rolling=self.time_periods['reference']['rolling']
        )
        
        if reference_bioclim:
            self.logger.info(f"Reference bioclimatic variables calculated: {len(reference_bioclim)} variables")
        else:
            self.logger.error("Failed to calculate reference bioclimatic variables")
            return False

        anomalies_dir = self.anomalies_dir
        yearly_anomalies = self.calculate_bioclim_anomalies(
            harmonized_temp_files,
            harmonized_precip_files,
            self.bioclim_dir,
            self.anomalies_dir,
            start_year=self.time_periods['analysis']['start_year'],
            end_year=self.time_periods['analysis']['end_year'],
            rolling=self.time_periods['analysis']['rolling']
        )

        if yearly_anomalies:
            total_anomalies = sum(len(year_data) for year_data in yearly_anomalies.values())
            self.logger.info(f" Bioclimatic anomalies calculated: {len(yearly_anomalies)} years, "
                        f"{total_anomalies} total anomaly files")
        else:
            self.logger.error("Failed to calculate bioclimatic anomalies")
            return False

        self.logger.info("Bioclimatic calculation completed successfully!")

        return True    

    def extract_date(self, filename: str) -> datetime:
        """
        Extract date from filename with pattern {var}_{YYYY}-{MM}.tif
        
        Args:
            filename: Input filename to parse
            
        Returns:
            Parsed datetime object
        """
        basename = os.path.basename(filename)
        date_part = basename.split('_')  # Gets "YYYY-MM"
        year = date_part[1].split('-')[0]
        month = date_part[1].split('-')[1].split('.')[0]
        return datetime(int(year), int(month), 1)

    def harmonize_rasters(self, raster_files: List[str], output_dir: Union[str, Path], reference_file: Optional[str] = None) -> List[str]:
        """
        Harmonize all raster files to the same dimensions and coordinate system.
        
        Args:
            raster_files: List of paths to raster files to harmonize
            output_dir: Directory to save harmonized rasters
            reference_file: File to use as reference. If None, the first file is used.
        
        Returns:
            Paths to the harmonized raster files
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not raster_files:
            self.logger.warning("No raster files provided for harmonization")
            return []
        
        # Determine reference file if not provided
        if not reference_file:
            reference_file = raster_files[0]
        
        # Get reference metadata
        with rasterio.open(reference_file) as src:
            reference_crs = src.crs
            reference_transform = src.transform
            reference_height = src.height
            reference_width = src.width
            reference_bounds = src.bounds
        
        self.logger.info(f"Reference dimensions: {reference_height} x {reference_width}")
        self.logger.debug(f"Reference bounds: {reference_bounds}")
        
        harmonized_files = []
        
        for file in raster_files:
            basename = os.path.basename(file)
            output_file = output_dir / basename
            
            # Check if file already exists
            if output_file.exists():
                harmonized_files.append(str(output_file))
                continue
            
            with rasterio.open(file) as src:
                # Check if resampling is needed
                if (src.height == reference_height and 
                    src.width == reference_width and 
                    src.crs == reference_crs and
                    src.bounds == reference_bounds):
                    # No resampling needed
                    harmonized_files.append(file)
                    continue
                
                # Resampling is needed
                self.logger.debug(f"Resampling {basename}")
                
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
                    source_data = src.read(1)
                    destination = np.zeros((reference_height, reference_width), dtype=profile['dtype'])
                    
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
                
                harmonized_files.append(str(output_file))
        
        self.logger.info(f"Harmonized {len(harmonized_files)}/{len(raster_files)} files")
        return harmonized_files

    def calculate_bioclim_variables(self, temp_files: List[str], precip_files: List[str], 
                               output_dir: Union[str, Path], start_year: int = 1985, 
                               end_year: int = 2015, rolling: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculate bioclimatic variables from monthly temperature and precipitation files.
        
        Args:
            temp_files: List of paths to temperature raster files
            precip_files: List of paths to precipitation raster files
            output_dir: Directory to save output bioclim rasters
            start_year: Start year for climatology period
            end_year: End year for climatology period
            rolling: If True, use rolling years (Sep-Aug) instead of calendar years
            
        Returns:
            Dictionary of bioclimatic variable arrays, or None if failed
        """
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Calculating bioclimatic variables for {start_year}-{end_year}")
        
        # Extract dates and sort files
        temp_files_with_dates = [(f, self.extract_date(f)) for f in temp_files]
        precip_files_with_dates = [(f, self.extract_date(f)) for f in precip_files]
        
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
        
        self.logger.info(f"Found {len(temp_files_filtered)} temperature files and {len(precip_files_filtered)} precipitation files for reference period")
        
        # Group files by month
        temp_by_month = {month: [] for month in range(1, 13)}
        precip_by_month = {month: [] for month in range(1, 13)}
        
        for file, date in temp_files_filtered:
            temp_by_month[date.month].append(file)
        
        for file, date in precip_files_filtered:
            precip_by_month[date.month].append(file)
        
        # Get metadata from first file for output rasters
        if not temp_files_filtered:
            self.logger.error("No temperature files found for the specified period")
            return None
            
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
            self.logger.error(f"Missing temperature data for months: {missing_temp_months}")
            self.logger.error(f"Missing precipitation data for months: {missing_precip_months}")
            self.logger.error("Cannot calculate all bioclim variables without complete monthly data.")
            return None
        
        # Calculate bioclim variables
        self.logger.info("Computing bioclimatic variables...")
        
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
        
        # Write output rasters (only for variables in self.bio_variables)
        saved_count = 0
        for var_name, data in bioclim_vars.items():
            if var_name in self.bio_variables:
                output_path = output_dir / f"{var_name}_{year_label}.tif"
                
                # Update profile for output
                output_profile = profile.copy()
                output_profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress='lzw',
                    nodata=-9999
                )
                
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    dst.write(data.astype(rasterio.float32), 1)
                
                saved_count += 1
                self.logger.debug(f"Saved: {var_name}_{year_label}.tif")
        
        self.logger.info(f"Successfully calculated and saved {saved_count} bioclimatic variables to {output_dir}")
        return bioclim_vars

    def calculate_bioclim_anomalies(self, temp_files: List[str], precip_files: List[str], 
                               bioclim_dir: Union[str, Path], output_dir: Union[str, Path],
                               start_year: int = 2015, end_year: int = 2024, rolling: bool = True) -> Optional[Dict[int, Dict[str, str]]]:
        """
        Calculate annual and cumulative anomalies for bioclimatic variables.
        
        Args:
            temp_files: List of paths to temperature raster files
            precip_files: List of paths to precipitation raster files
            bioclim_dir: Directory containing bioclim reference files (30-year climatology)
            output_dir: Directory to save output anomaly rasters
            start_year: Start year for anomaly calculation
            end_year: End year for anomaly calculation
            rolling: If True, use rolling years (Sep-Aug) instead of calendar years
            
        Returns:
            Dictionary mapping years to anomaly file paths
        """
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Calculating bioclimatic anomalies for period {start_year}-{end_year}")
        
        # Load the reference bioclim variables (30-year climatology)
        reference_pattern = "1985Sep-2015Aug" if rolling else "1985_2015"
        reference_bioclim = {}
        
        # Load reference bioclim rasters
        for var in self.bio_variables:
            ref_file = Path(bioclim_dir) / f"{var}_{reference_pattern}.tif"
            if ref_file.exists():
                with rasterio.open(ref_file) as src:
                    reference_bioclim[var] = src.read(1)
                    # Save profile from first file for output
                    if var == self.bio_variables[0]:
                        profile = src.profile
            else:
                self.logger.warning(f"Reference file not found for {var}: {ref_file}")
        
        if not reference_bioclim:
            self.logger.error(f"No reference bioclimatic variables found in {bioclim_dir}")
            return None
        
        # Function to calculate bioclim variables for a specific year
        def calculate_year_bioclim(year):
            year_temp_files = []
            year_precip_files = []
            
            # Gather files for this year
            for file in temp_files:
                date = self.extract_date(file)
                if rolling:
                    # For rolling years (Sep to Aug)
                    effective_year = date.year if date.month >= 9 else date.year - 1
                else:
                    effective_year = date.year
                
                if effective_year == year:
                    year_temp_files.append(file)
            
            for file in precip_files:
                date = self.extract_date(file)
                if rolling:
                    effective_year = date.year if date.month >= 9 else date.year - 1
                else:
                    effective_year = date.year
                
                if effective_year == year:
                    year_precip_files.append(file)
            
            # Check if we have complete data
            if len(year_temp_files) < 12 or len(year_precip_files) < 12:
                self.logger.warning(f"Incomplete data for year {year}. Found {len(year_temp_files)} temperature files and {len(year_precip_files)} precipitation files.")
                return None
            
            # Create temporary directory for this year's bioclim
            year_dir = output_dir / 'tmp' /f"bioclim_{year}"
            year_dir.mkdir(exist_ok=True, parents=True)
            
            # Calculate bioclim variables for this year
            year_bioclim = self.calculate_bioclim_variables(
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
            self.logger.info(f"Calculating bioclim variables for year {year}")
            yearly_bioclim[year] = calculate_year_bioclim(year)
        
        # Calculate anomalies for each year
        yearly_anomalies = {}
        for year in range(start_year, end_year + 1):
            if yearly_bioclim[year] is None:
                continue
            
            # Create directory for this year's anomalies
            year_label = f"{year}Sep-{year+1}Aug" if rolling else f"{year}"
            year_anomaly_dir = output_dir / f"anomalies_{year_label}"
            year_anomaly_dir.mkdir(exist_ok=True)
            
            # Calculate anomalies for each bioclim variable
            year_anomalies = {}
            for var in self.bio_variables:
                if var not in reference_bioclim or var not in yearly_bioclim[year]:
                    continue
                
                # Calculate anomaly
                anomaly = yearly_bioclim[year][var] - reference_bioclim[var]
                year_anomalies[var] = anomaly
                
                # Save anomaly raster
                anomaly_path = year_anomaly_dir / f"{var}_anomaly_{year_label}.tif"
                with rasterio.open(anomaly_path, 'w', **profile) as dst:
                    dst.write(anomaly.astype(rasterio.float32), 1)
            
            yearly_anomalies[year] = year_anomalies
        
        # Calculate cumulative anomalies (2-year and 3-year)
        self.logger.info("Calculating cumulative anomalies...")
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
                cum_dir = output_dir / f"anomalies_{cum_years}yr_{year_label}"
                cum_dir.mkdir(exist_ok=True)
                
                for var in self.bio_variables:
                    if not all(var in yearly_anomalies[y] for y in prior_years):
                        continue
                    
                    # For all variables, use the average of anomalies (not sum)
                    cum_anomaly = np.mean([yearly_anomalies[y][var] for y in prior_years], axis=0)
                    
                    # Save cumulative anomaly raster
                    cum_path = cum_dir / f"{var}_anomaly_{cum_years}yr_{year_label}.tif"
                    with rasterio.open(cum_path, 'w', **profile) as dst:
                        dst.write(cum_anomaly.astype(rasterio.float32), 1)
        
        self.logger.info(f"Bioclim anomalies calculated successfully and saved to {output_dir}")
        return yearly_anomalies