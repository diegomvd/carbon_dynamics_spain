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
from typing import Dict, List, Optional, Union, Tuple
from rasterio.enums import Resampling

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory, find_files


class BioclimCalculator:
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
        
        # Bioclimatic variables to calculate
        self.bio_variables = self.bioclim_config['variables']
        
        # Unit conversions
        self.temp_conversion = self.bioclim_config['temp_conversion']
        self.precip_scaling = self.bioclim_config['precip_scaling']
        
        self.logger.info(f"Initialized BioclimCalculator for {len(self.bio_variables)} variables")
    
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
        year = date_part[1]
        month = date_part[2].split('.')[0]
        return datetime(int(year), int(month), 1)
    
    def harmonize_raster(
        self, 
        input_path: Union[str, Path], 
        reference_shape: Tuple[int, int],
        reference_transform: rasterio.Affine,
        reference_crs: rasterio.CRS
    ) -> np.ndarray:
        """
        Harmonize a single raster to match reference grid properties.
        
        Args:
            input_path: Path to input raster
            reference_shape: Target shape (height, width)
            reference_transform: Target transform
            reference_crs: Target CRS
            
        Returns:
            Harmonized raster data array
        """
        with rasterio.open(input_path) as src:
            # Reproject and resample to match reference
            harmonized = np.empty(reference_shape, dtype=src.dtypes[0])
            
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=harmonized,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=getattr(Resampling, self.climate_config['resampling_method'].lower())
            )
            
            return harmonized
    
    def harmonize_rasters(
        self, 
        raster_files: List[Union[str, Path]], 
        output_dir: Union[str, Path],
        reference_file: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """
        Harmonize all raster files to the same dimensions and coordinate system.
        
        Args:
            raster_files: List of paths to raster files to harmonize
            output_dir: Directory to save harmonized rasters
            reference_file: File to use as reference. If None, uses the first file.
            
        Returns:
            List of paths to harmonized raster files
        """
        if not raster_files:
            self.logger.warning("No raster files provided for harmonization")
            return []
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Determine reference file
        if reference_file is None:
            reference_file = raster_files[0]
        
        self.logger.info(f"Using reference file: {Path(reference_file).name}")
        
        # Get reference properties
        with rasterio.open(reference_file) as ref:
            reference_shape = (ref.height, ref.width)
            reference_transform = ref.transform
            reference_crs = ref.crs
            reference_profile = ref.profile.copy()
        
        self.logger.info(f"Reference grid: {reference_shape[1]}x{reference_shape[0]}, CRS: {reference_crs}")
        
        # Process each file
        harmonized_files = []
        for raster_file in raster_files:
            try:
                # Generate output filename
                output_filename = Path(raster_file).name
                output_path = Path(output_dir) / output_filename
                
                # Check if already harmonized
                if output_path.exists():
                    self.logger.debug(f"Already harmonized: {output_filename}")
                    harmonized_files.append(str(output_path))
                    continue
                
                # Harmonize raster
                harmonized_data = self.harmonize_raster(
                    raster_file, reference_shape, reference_transform, reference_crs
                )
                
                # Save harmonized raster
                with rasterio.open(output_path, 'w', **reference_profile) as dst:
                    dst.write(harmonized_data, 1)
                
                harmonized_files.append(str(output_path))
                self.logger.debug(f"Harmonized: {output_filename}")
                
            except Exception as e:
                self.logger.error(f"Error harmonizing {raster_file}: {e}")
        
        self.logger.info(f"Harmonized {len(harmonized_files)}/{len(raster_files)} files")
        return harmonized_files
    
    def filter_files_by_period(
        self, 
        files: List[str], 
        start_year: int, 
        end_year: int,
        rolling: bool = False
    ) -> List[str]:
        """
        Filter files by time period.
        
        Args:
            files: List of file paths
            start_year: Start year for filtering
            end_year: End year for filtering  
            rolling: If True, use Sep-Aug rolling years
            
        Returns:
            Filtered list of file paths
        """
        filtered_files = []
        
        for file_path in files:
            try:
                date = self.extract_date(file_path)
                year = date.year
                month = date.month
                
                if rolling:
                    # For rolling years (Sep-Aug), assign months Sep-Dec to current year,
                    # Jan-Aug to previous year
                    if month >= 9:  # Sep-Dec
                        rolling_year = year
                    else:  # Jan-Aug
                        rolling_year = year - 1
                    
                    if start_year <= rolling_year <= end_year:
                        filtered_files.append(file_path)
                else:
                    # Calendar year filtering
                    if start_year <= year <= end_year:
                        filtered_files.append(file_path)
                        
            except Exception as e:
                self.logger.warning(f"Error parsing date from {file_path}: {e}")
                continue
        
        self.logger.info(f"Filtered to {len(filtered_files)} files for period {start_year}-{end_year} (rolling={rolling})")
        return filtered_files
    
    def calculate_bioclim_variables(
        self,
        temp_files: List[str],
        precip_files: List[str],
        output_dir: Union[str, Path],
        start_year: int,
        end_year: int,
        rolling: bool = False
    ) -> Optional[Dict[str, str]]:
        """
        Calculate bioclimatic variables for a given period.
        
        Args:
            temp_files: List of temperature file paths
            precip_files: List of precipitation file paths
            output_dir: Output directory for bioclimatic variables
            start_year: Start year for calculation
            end_year: End year for calculation
            rolling: Use Sep-Aug rolling years
            
        Returns:
            Dictionary mapping bioclimatic variable names to output file paths
        """
        # Filter files by period
        period_temp_files = self.filter_files_by_period(temp_files, start_year, end_year, rolling)
        period_precip_files = self.filter_files_by_period(precip_files, start_year, end_year, rolling)
        
        if not period_temp_files or not period_precip_files:
            self.logger.error(f"No files found for period {start_year}-{end_year}")
            return None
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        try:
            # Load temperature data
            temp_data = {}
            for temp_file in period_temp_files:
                date = self.extract_date(temp_file)
                month_key = date.strftime('%m')
                
                if month_key not in temp_data:
                    temp_data[month_key] = []
                
                with rasterio.open(temp_file) as src:
                    data = src.read(1).astype(np.float32)
                    # Convert temperature units if needed (e.g., from Kelvin*10 to Celsius)
                    data = data * self.temp_conversion
                    temp_data[month_key].append(data)
                    
                    # Store reference profile from first file
                    if 'profile' not in locals():
                        profile = src.profile.copy()
                        profile.update(dtype=rasterio.float32)
            
            # Load precipitation data
            precip_data = {}
            for precip_file in period_precip_files:
                date = self.extract_date(precip_file)
                month_key = date.strftime('%m')
                
                if month_key not in precip_data:
                    precip_data[month_key] = []
                
                with rasterio.open(precip_file) as src:
                    data = src.read(1).astype(np.float32)
                    # Convert precipitation units if needed (e.g., from m to mm)
                    data = data * self.precip_scaling
                    precip_data[month_key].append(data)
            
            # Calculate monthly averages
            monthly_temp = {}
            monthly_precip = {}
            
            for month in range(1, 13):
                month_key = f"{month:02d}"
                
                if month_key in temp_data:
                    monthly_temp[month] = np.mean(temp_data[month_key], axis=0)
                if month_key in precip_data:
                    monthly_precip[month] = np.mean(precip_data[month_key], axis=0)
            
            # Calculate bioclimatic variables
            bioclim_results = self._calculate_bioclim_from_monthly(monthly_temp, monthly_precip)
            
            # Save bioclimatic variables
            output_files = {}
            for bio_var, data in bioclim_results.items():
                if bio_var in self.bio_variables:
                    output_file = Path(output_dir) / f"{bio_var}_{start_year}_{end_year}.tif"
                    
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(data.astype(np.float32), 1)
                    
                    output_files[bio_var] = str(output_file)
                    self.logger.debug(f"Saved {bio_var}: {output_file.name}")
            
            self.logger.info(f"Calculated {len(output_files)} bioclimatic variables for {start_year}-{end_year}")
            return output_files
            
        except Exception as e:
            self.logger.error(f"Error calculating bioclimatic variables: {e}")
            return None
    
    def _calculate_bioclim_from_monthly(
        self, 
        monthly_temp: Dict[int, np.ndarray], 
        monthly_precip: Dict[int, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all 19 bioclimatic variables from monthly data.
        
        Args:
            monthly_temp: Dictionary of monthly temperature arrays (1-12)
            monthly_precip: Dictionary of monthly precipitation arrays (1-12)
            
        Returns:
            Dictionary of bioclimatic variable arrays
        """
        # Initialize result dictionary
        bioclim = {}
        
        # Convert to arrays for easier calculation
        temp_array = np.stack([monthly_temp[i] for i in range(1, 13)], axis=0)
        precip_array = np.stack([monthly_precip[i] for i in range(1, 13)], axis=0)
        
        # Bio1: Annual Mean Temperature
        bioclim['bio1'] = np.mean(temp_array, axis=0)
        
        # Bio2: Mean Diurnal Range (here we approximate as monthly range)
        bioclim['bio2'] = np.mean(np.max(temp_array, axis=0) - np.min(temp_array, axis=0))
        
        # Bio3: Isothermality (Bio2/Bio7) * 100
        bio7 = np.max(temp_array, axis=0) - np.min(temp_array, axis=0)
        bioclim['bio3'] = (bioclim['bio2'] / bio7) * 100
        
        # Bio4: Temperature Seasonality (standard deviation * 100)
        bioclim['bio4'] = np.std(temp_array, axis=0) * 100
        
        # Bio5: Max Temperature of Warmest Month
        bioclim['bio5'] = np.max(temp_array, axis=0)
        
        # Bio6: Min Temperature of Coldest Month
        bioclim['bio6'] = np.min(temp_array, axis=0)
        
        # Bio7: Temperature Annual Range
        bioclim['bio7'] = bio7
        
        # Bio8: Mean Temperature of Wettest Quarter
        # Find wettest quarter for each pixel
        quarterly_precip = np.array([
            np.sum(precip_array[i:i+3], axis=0) for i in range(12)
        ])
        wettest_quarter_idx = np.argmax(quarterly_precip, axis=0)
        
        # Calculate mean temperature for wettest quarter
        bio8 = np.zeros_like(bioclim['bio1'])
        for i in range(12):
            mask = wettest_quarter_idx == i
            if np.any(mask):
                quarter_temp = np.mean(temp_array[i:i+3], axis=0)
                bio8[mask] = quarter_temp[mask]
        bioclim['bio8'] = bio8
        
        # Bio9: Mean Temperature of Driest Quarter
        driest_quarter_idx = np.argmin(quarterly_precip, axis=0)
        
        bio9 = np.zeros_like(bioclim['bio1'])
        for i in range(12):
            mask = driest_quarter_idx == i
            if np.any(mask):
                quarter_temp = np.mean(temp_array[i:i+3], axis=0)
                bio9[mask] = quarter_temp[mask]
        bioclim['bio9'] = bio9
        
        # Bio10: Mean Temperature of Warmest Quarter
        quarterly_temp = np.array([
            np.mean(temp_array[i:i+3], axis=0) for i in range(12)
        ])
        bioclim['bio10'] = np.max(quarterly_temp, axis=0)
        
        # Bio11: Mean Temperature of Coldest Quarter
        bioclim['bio11'] = np.min(quarterly_temp, axis=0)
        
        # Bio12: Annual Precipitation
        bioclim['bio12'] = np.sum(precip_array, axis=0)
        
        # Bio13: Precipitation of Wettest Month
        bioclim['bio13'] = np.max(precip_array, axis=0)
        
        # Bio14: Precipitation of Driest Month
        bioclim['bio14'] = np.min(precip_array, axis=0)
        
        # Bio15: Precipitation Seasonality (CV)
        mean_precip = np.mean(precip_array, axis=0)
        std_precip = np.std(precip_array, axis=0)
        # Avoid division by zero
        cv = np.where(mean_precip > 0, (std_precip / mean_precip) * 100, 0)
        bioclim['bio15'] = cv
        
        # Bio16: Precipitation of Wettest Quarter
        bioclim['bio16'] = np.max(quarterly_precip, axis=0)
        
        # Bio17: Precipitation of Driest Quarter
        bioclim['bio17'] = np.min(quarterly_precip, axis=0)
        
        # Bio18: Precipitation of Warmest Quarter
        warmest_quarter_idx = np.argmax(quarterly_temp, axis=0)
        
        bio18 = np.zeros_like(bioclim['bio12'])
        for i in range(12):
            mask = warmest_quarter_idx == i
            if np.any(mask):
                quarter_precip = np.sum(precip_array[i:i+3], axis=0)
                bio18[mask] = quarter_precip[mask]
        bioclim['bio18'] = bio18
        
        # Bio19: Precipitation of Coldest Quarter
        coldest_quarter_idx = np.argmin(quarterly_temp, axis=0)
        
        bio19 = np.zeros_like(bioclim['bio12'])
        for i in range(12):
            mask = coldest_quarter_idx == i
            if np.any(mask):
                quarter_precip = np.sum(precip_array[i:i+3], axis=0)
                bio19[mask] = quarter_precip[mask]
        bioclim['bio19'] = bio19
        
        return bioclim
    
    def calculate_bioclim_anomalies(
        self,
        temp_files: List[str],
        precip_files: List[str],
        reference_bioclim_dir: Union[str, Path],
        output_dir: Union[str, Path],
        start_year: int,
        end_year: int,
        rolling: bool = False
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Calculate bioclimatic anomalies for each year in the analysis period.
        
        Args:
            temp_files: List of temperature file paths
            precip_files: List of precipitation file paths
            reference_bioclim_dir: Directory containing reference bioclimatic variables
            output_dir: Output directory for anomaly files
            start_year: Start year for analysis
            end_year: End year for analysis
            rolling: Use Sep-Aug rolling years
            
        Returns:
            Dictionary mapping years to anomaly file paths
        """
        # Load reference bioclimatic variables
        reference_bioclim = {}
        for bio_var in self.bio_variables:
            ref_files = list(Path(reference_bioclim_dir).glob(f"{bio_var}_*.tif"))
            if ref_files:
                ref_file = ref_files[0]  # Take first match
                with rasterio.open(ref_file) as src:
                    reference_bioclim[bio_var] = src.read(1).astype(np.float32)
                    if 'profile' not in locals():
                        profile = src.profile.copy()
                        profile.update(dtype=rasterio.float32)
        
        if not reference_bioclim:
            self.logger.error(f"No reference bioclimatic variables found in {reference_bioclim_dir}")
            return None
        
        # Calculate anomalies for each year
        yearly_anomalies = {}
        
        for year in range(start_year, end_year + 1):
            self.logger.info(f"Calculating anomalies for year {year}")
            
            # Create year-specific output directory
            year_output_dir = Path(output_dir) / f"anomalies_{year}"
            ensure_directory(year_output_dir)
            
            # Calculate bioclimatic variables for this year
            year_bioclim = self.calculate_bioclim_variables(
                temp_files, precip_files, year_output_dir / "temp",
                year, year, rolling
            )
            
            if year_bioclim is None:
                self.logger.warning(f"Could not calculate bioclimatic variables for year {year}")
                continue
            
            # Calculate and save anomalies
            year_anomalies = {}
            for bio_var in self.bio_variables:
                if bio_var in year_bioclim and bio_var in reference_bioclim:
                    try:
                        # Load year data
                        with rasterio.open(year_bioclim[bio_var]) as src:
                            year_data = src.read(1).astype(np.float32)
                        
                        # Calculate anomaly
                        anomaly = year_data - reference_bioclim[bio_var]
                        
                        # Save anomaly
                        anomaly_file = year_output_dir / f"{bio_var}_anomaly_{year}.tif"
                        with rasterio.open(anomaly_file, 'w', **profile) as dst:
                            dst.write(anomaly.astype(np.float32), 1)
                        
                        year_anomalies[bio_var] = str(anomaly_file)
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating {bio_var} anomaly for {year}: {e}")
            
            yearly_anomalies[year] = year_anomalies
            self.logger.info(f"Calculated {len(year_anomalies)} anomalies for year {year}")
        
        return yearly_anomalies
    
    def run_bioclim_pipeline(self) -> Optional[Dict[str, str]]:
        """
        Run the complete bioclimatic variables calculation pipeline.
        
        Returns:
            Dictionary with results from anomaly calculations
        """
        self.logger.info("Starting bioclimatic variables calculation pipeline...")
        
        # Extract config parameters
        data_dir = self.config['data']['climate_outputs']
        harmonized_dir = self.config['data']['harmonized_dir']
        bioclim_dir = self.config['data']['bioclim_dir']
        anomaly_dir = self.config['data']['anomaly_dir']
        
        reference_period = self.time_periods['reference']
        analysis_period = self.time_periods['analysis']
        
        # Find all temperature and precipitation files
        temp_files = glob.glob(os.path.join(data_dir, self.climate_config['temp_pattern']))
        precip_files = glob.glob(os.path.join(data_dir, self.climate_config['precip_pattern']))
        
        # Create directories for harmonized files
        harmonized_temp_dir = os.path.join(harmonized_dir, "temperature")
        harmonized_precip_dir = os.path.join(harmonized_dir, "precipitation")
        
        self.logger.info(f"Found {len(temp_files)} temperature files and {len(precip_files)} precipitation files")
        
        # Step 1: Harmonize all raster files to a common grid
        self.logger.info("Harmonizing temperature files...")
        harmonized_temp_files = self.harmonize_rasters(temp_files, harmonized_temp_dir)
        
        self.logger.info("Harmonizing precipitation files...")
        harmonized_precip_files = self.harmonize_rasters(precip_files, harmonized_precip_dir)
        
        # Step 2: Calculate bioclimatic variables for the reference period
        self.logger.info(f"Calculating bioclimatic variables for reference period ({reference_period['start_year']}-{reference_period['end_year']})...")
        reference_bioclim = self.calculate_bioclim_variables(
            harmonized_temp_files, 
            harmonized_precip_files, 
            bioclim_dir, 
            start_year=reference_period['start_year'], 
            end_year=reference_period['end_year'],
            rolling=reference_period['rolling']
        )
        
        if reference_bioclim is None:
            self.logger.error("Error calculating reference bioclimatic variables. Exiting.")
            return None
        
        self.logger.info(f"Reference bioclimatic variables calculated successfully and saved to {bioclim_dir}")
        
        # Step 3: Calculate bioclimatic anomalies for the analysis period
        self.logger.info(f"Calculating bioclimatic anomalies for analysis period ({analysis_period['start_year']}-{analysis_period['end_year']})...")
        yearly_anomalies = self.calculate_bioclim_anomalies(
            harmonized_temp_files,
            harmonized_precip_files,
            bioclim_dir,
            anomaly_dir,
            start_year=analysis_period['start_year'],
            end_year=analysis_period['end_year'],
            rolling=analysis_period['rolling']
        )
        
        self.logger.info("Bioclimatic pipeline completed successfully!")
        return yearly_anomalies