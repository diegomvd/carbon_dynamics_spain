"""
FIXED: Bioclimatic variables calculation with restored quarter logic.

Key fixes:
1. Restored original 4-quarter seasonal logic (not 12 sliding quarters)
2. Removed bio2 and bio3 completely
3. Use np.zeros for harmonization
4. Added cumulative anomaly calculations (2yr, 3yr)

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
        
        self.data_dir = CLIMATE_RAW_DIR
        self.harmonized_dir = CLIMATE_HARMONIZED_DIR
        self.bioclim_dir = BIOCLIM_VARIABLES_DIR
        self.anomalies_dir = BIOCLIM_ANOMALIES_DIR


    def run_full_pipeline(self):
        data_dir = self.data_dir

        temp_files = glob.glob(os.path.join(data_dir, calculator.climate_config['temp_pattern']))
        precip_files = glob.glob(os.path.join(data_dir, calculator.climate_config['precip_pattern']))
        
        logger.info(f"Found {len(temp_files)} temperature files and {len(precip_files)} precipitation files")
        
        if not temp_files or not precip_files:
            raise ValueError("No temperature or precipitation files found")

        harmonized_dir = self.harmonized_dir   

        harmonized_temp_dir = harmonized_dir / "temperature"
        harmonized_precip_dir = harmonized_dir / "precipitation"
        
        logger.info("Harmonizing raster files...")
        harmonized_temp_files = calculator.harmonize_rasters(temp_files, harmonized_temp_dir)
        harmonized_precip_files = calculator.harmonize_rasters(precip_files, harmonized_precip_dir)
 
        bioclim_dir = self.bioclim_dir
        reference_bioclim = calculator.calculate_bioclim_variables(
            harmonized_temp_files,
            harmonized_precip_files,
            output_dir,
            start_year=time_periods['reference']['start_year'],
            end_year=time_periods['reference']['end_year'],
            rolling=time_periods['reference']['rolling']
        )
        
        if reference_bioclim:
            logger.info(f"✅ Reference bioclimatic variables calculated: {len(reference_bioclim)} variables")
        else:
            logger.error("❌ Failed to calculate reference bioclimatic variables")
            return False

        anomalies_dir = self.anomalies_dir
        yearly_anomalies = calculator.calculate_bioclim_anomalies(
            harmonized_temp_files,
            harmonized_precip_files,
            bioclim_dir,
            anomaly_dir,
            start_year=time_periods['analysis']['start_year'],
            end_year=time_periods['analysis']['end_year'],
            rolling=time_periods['analysis']['rolling']
        )

        if yearly_anomalies:
            total_anomalies = sum(len(year_data) for year_data in yearly_anomalies.values())
            logger.info(f"✅ Bioclimatic anomalies calculated: {len(yearly_anomalies)} years, "
                        f"{total_anomalies} total anomaly files")
        else:
            logger.error("❌ Failed to calculate bioclimatic anomalies")
            return False

        logger.info("Bioclimatic calculation completed successfully!")

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
            # FIXED: Use np.zeros instead of np.empty
            harmonized = np.zeros(reference_shape, dtype=src.dtypes[0])
            
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
    
    def _calculate_bioclim_from_monthly(
        self, 
        monthly_temp: Dict[int, np.ndarray], 
        monthly_precip: Dict[int, np.ndarray],
        rolling: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all bioclimatic variables from monthly data.
                
        Args:
            monthly_temp: Dictionary of monthly temperature arrays (1-12)
            monthly_precip: Dictionary of monthly precipitation arrays (1-12)
            rolling: Whether to use rolling year definition
            
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
        
        # Bio4: Temperature Seasonality (standard deviation * 100)
        bioclim['bio4'] = np.std(temp_array, axis=0) * 100
        
        # Bio5: Max Temperature of Warmest Month
        bioclim['bio5'] = np.max(temp_array, axis=0)
        
        # Bio6: Min Temperature of Coldest Month
        bioclim['bio6'] = np.min(temp_array, axis=0)
        
        # Bio7: Temperature Annual Range
        bioclim['bio7'] = bioclim['bio5'] - bioclim['bio6']
        
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
        
        # Define quarters based on rolling vs calendar year
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
        
        # Calculate quarterly means and sums
        quarter_temp = {}
        quarter_precip = {}
        
        for i, q in enumerate(quarters):
            temp_q = np.mean([monthly_temp[m] for m in q], axis=0)
            precip_q = np.sum([monthly_precip[m] for m in q], axis=0)
            quarter_temp[i] = temp_q
            quarter_precip[i] = precip_q
        
        # Find warmest, coldest, wettest, driest quarters on a pixel-by-pixel basis
        quarter_temp_array = np.array([quarter_temp[i] for i in range(4)])
        quarter_precip_array = np.array([quarter_precip[i] for i in range(4)])
        
        warmest_q_pixels = np.argmax(quarter_temp_array, axis=0)
        coldest_q_pixels = np.argmin(quarter_temp_array, axis=0)
        wettest_q_pixels = np.argmax(quarter_precip_array, axis=0)
        driest_q_pixels = np.argmin(quarter_precip_array, axis=0)
        
        # Bio8: Mean Temperature of Wettest Quarter
        temp_wettest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            temp_wettest_quarter[wettest_q_pixels == i] = quarter_temp[i][wettest_q_pixels == i]
        bioclim['bio8'] = temp_wettest_quarter
        
        # Bio9: Mean Temperature of Driest Quarter
        temp_driest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            temp_driest_quarter[driest_q_pixels == i] = quarter_temp[i][driest_q_pixels == i]
        bioclim['bio9'] = temp_driest_quarter
        
        # Bio10: Mean Temperature of Warmest Quarter
        temp_warmest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            temp_warmest_quarter[warmest_q_pixels == i] = quarter_temp[i][warmest_q_pixels == i]
        bioclim['bio10'] = temp_warmest_quarter
        
        # Bio11: Mean Temperature of Coldest Quarter
        temp_coldest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            temp_coldest_quarter[coldest_q_pixels == i] = quarter_temp[i][coldest_q_pixels == i]
        bioclim['bio11'] = temp_coldest_quarter
        
        # Bio16: Precipitation of Wettest Quarter
        precip_wettest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            precip_wettest_quarter[wettest_q_pixels == i] = quarter_precip[i][wettest_q_pixels == i]
        bioclim['bio16'] = precip_wettest_quarter
        
        # Bio17: Precipitation of Driest Quarter
        precip_driest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            precip_driest_quarter[driest_q_pixels == i] = quarter_precip[i][driest_q_pixels == i]
        bioclim['bio17'] = precip_driest_quarter
        
        # Bio18: Precipitation of Warmest Quarter
        precip_warmest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            precip_warmest_quarter[warmest_q_pixels == i] = quarter_precip[i][warmest_q_pixels == i]
        bioclim['bio18'] = precip_warmest_quarter
        
        # Bio19: Precipitation of Coldest Quarter
        precip_coldest_quarter = np.zeros_like(bioclim['bio1'])
        for i in range(4):
            precip_coldest_quarter[coldest_q_pixels == i] = quarter_precip[i][coldest_q_pixels == i]
        bioclim['bio19'] = precip_coldest_quarter
        
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
        self.logger.info(f"Calculating bioclimatic anomalies for period {start_year}-{end_year}")
        
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
        
        # Calculate bioclim variables for each year and compute anomalies
        yearly_anomalies = {}
        
        for year in range(start_year, end_year + 1):
            self.logger.info(f"Processing year {year}...")
            
            # Calculate bioclim variables for this year
            year_bioclim = self._calculate_year_bioclim(temp_files, precip_files, year, rolling)
            
            if year_bioclim is None:
                self.logger.warning(f"Skipping year {year} due to incomplete data")
                continue
            
            # Create directory for this year's anomalies
            year_label = f"{year}Sep-{year+1}Aug" if rolling else f"{year}"
            year_anomaly_dir = os.path.join(output_dir, f"anomalies_{year_label}")
            os.makedirs(year_anomaly_dir, exist_ok=True)
            
            # Calculate anomalies for each bioclim variable
            year_anomalies = {}
            for var in self.bio_variables:
                if var not in reference_bioclim or var not in year_bioclim:
                    continue
                
                # Calculate anomaly
                anomaly = year_bioclim[var] - reference_bioclim[var]
                year_anomalies[var] = anomaly
                
                # Save anomaly raster
                anomaly_path = os.path.join(year_anomaly_dir, f"{var}_anomaly_{year_label}.tif")
                with rasterio.open(anomaly_path, 'w', **profile) as dst:
                    dst.write(anomaly.astype(rasterio.float32), 1)
            
            yearly_anomalies[year] = year_anomalies
        
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
                cum_dir = os.path.join(output_dir, f"anomalies_{cum_years}yr_{year_label}")
                os.makedirs(cum_dir, exist_ok=True)
                
                for var in self.bio_variables:
                    if not all(var in yearly_anomalies[y] for y in prior_years):
                        continue
                    
                    # For all variables, use the average of anomalies (not sum)
                    cum_anomaly = np.mean([yearly_anomalies[y][var] for y in prior_years], axis=0)
                    
                    # Save cumulative anomaly raster
                    cum_path = os.path.join(cum_dir, f"{var}_anomaly_{cum_years}yr_{year_label}.tif")
                    with rasterio.open(cum_path, 'w', **profile) as dst:
                        dst.write(cum_anomaly.astype(rasterio.float32), 1)
        
        self.logger.info(f"Bioclim anomalies calculated successfully and saved to {output_dir}")
        return yearly_anomalies
    
    def _calculate_year_bioclim(self, temp_files, precip_files, year, rolling):
        """Calculate bioclim variables for a specific year."""
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
            self.logger.warning(f"Incomplete data for year {year}. Found {len(year_temp_files)} temp files and {len(year_precip_files)} precip files.")
            return None
        
        # Load monthly data and calculate bioclim
        monthly_temp = {}
        monthly_precip = {}
        
        # Load temperature data
        for file in year_temp_files:
            date = self.extract_date(file)
            month = date.month
            with rasterio.open(file) as src:
                monthly_temp[month] = src.read(1).astype(np.float32)
        
        # Load precipitation data
        for file in year_precip_files:
            date = self.extract_date(file)
            month = date.month
            with rasterio.open(file) as src:
                monthly_precip[month] = src.read(1).astype(np.float32)
        
        # Calculate bioclimatic variables
        year_bioclim = self._calculate_bioclim_from_monthly(monthly_temp, monthly_precip, rolling)
        
        return year_bioclim
