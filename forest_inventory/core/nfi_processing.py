"""
Spanish National Forest Inventory (NFI) Processing Pipeline

This module contains the main processing pipeline for NFI4 data extraction and
biomass calculation. Preserves all original algorithmic logic while providing
a clean class-based interface with configuration file support.

The pipeline processes NFI4 data to:
1. Extract biomass stocks from IFN4 and SIG database files
2. Add forest type information from MFE (Spanish Forest Map) data
3. Calculate below-ground to above-ground biomass ratios
4. Export data in multiple formats

Author: Diego Bengochea
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union, Any

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, log_pipeline_start, log_pipeline_end, log_section

# Component imports
from .nfi_utils import (
    get_region_UTM, compute_biomass, load_reference_databases,
    calculate_bgb_ratio, create_output_directory, get_valid_utm_zones
)


class NFIProcessingPipeline:
    """
    Main processing pipeline for Spanish National Forest Inventory (NFI) data.
    
    Handles the complete workflow from Access database extraction to final
    shapefile exports with forest type information integration.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the NFI processing pipeline.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="forest_inventory")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name=self.config['logging']['component_name'],
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration values
        self.data_dir = self.config['data']['base_dir']
        self.output_dir = self.config['output']['base_dir']
        self.temp_dir = f"{self.data_dir}/{self.config['data']['temp_dir']}"
        self.target_crs = self.config['output']['target_crs']
        
        # Initialize reference databases
        self.wood_density_db = None
        self.ifn_species_codes = None
        
        self.logger.info("NFIProcessingPipeline initialized")
        self.logger.info(f"Configuration loaded from: {self.config.get('_meta', {}).get('config_file', 'default')}")

    def run_full_pipeline(self) -> bool:
        """
        Run the complete NFI processing pipeline.
        
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        try:
            log_pipeline_start(self.logger, "NFI Biomass Processing", self.config)
            
            # Load reference databases
            log_section(self.logger, "Loading Reference Databases")
            self.wood_density_db, self.ifn_species_codes = load_reference_databases(self.data_dir)
            self.logger.info(f"Loaded wood density database: {len(self.wood_density_db)} records")
            self.logger.info(f"Loaded species codes: {len(self.ifn_species_codes)} species")
            
            # Extract biomass data
            log_section(self.logger, "Extracting Biomass Data")
            utm_gdfs = self.extract_biomass_data()
            
            # Add forest type information
            log_section(self.logger, "Adding Forest Type Information")
            utm_gdfs = self.add_forest_type_data(utm_gdfs)
            
            # Export data in all formats
            log_section(self.logger, "Exporting Results")
            self.export_data(utm_gdfs)
            
            log_pipeline_end(self.logger, "NFI Biomass Processing", success=True)
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            log_pipeline_end(self.logger, "NFI Biomass Processing", success=False)
            raise

    def extract_biomass_data(self) -> Dict[int, gpd.GeoDataFrame]:
        """
        Extract biomass data from NFI database files for all provinces.
        
        CRITICAL: Preserves all original algorithmic logic exactly as-is.
        
        Returns:
            dict: Dictionary with UTM zones as keys and GeoDataFrames as values
        """
        # Create temporary directory
        if not Path(self.temp_dir).exists():
            Path(self.temp_dir).mkdir(parents=True)

        # Get file lists using configuration patterns
        data_path = Path(f"{self.data_dir}/{self.config['data']['ifn4_dir']}")
        ifn4_files = list(data_path.glob(self.config['file_patterns']['ifn4_files']))
        sig_files = list(data_path.glob(self.config['file_patterns']['sig_files']))

        # Initialize GeoDataFrames for each UTM zone
        utm_gdfs = {utm: gpd.GeoDataFrame() for utm in get_valid_utm_zones()}

        self.logger.info(f"Found {len(ifn4_files)} IFN4 files and {len(sig_files)} SIG files")

        # Process each province
        for accdb_ifn4 in ifn4_files:
            region = re.findall("_(.+?)_", str(accdb_ifn4.name))[0]
            self.logger.info(f'Processing region: {region}')
            
            # Extract data for this region
            plot_df = self._extract_region_data(
                accdb_ifn4, sig_files, region
            )
            
            if plot_df is None or len(plot_df) == 0:
                self.logger.warning(f"No data extracted for region {region}")
                continue
            
            # Convert to GeoDataFrame and assign to appropriate UTM zone
            for _, row in plot_df.iterrows():
                utm_zone = int(row['Huso'])
                if utm_zone in utm_gdfs:
                    # Create point geometry
                    current_crs, target_crs = get_region_UTM(region, str(utm_zone))
                    
                    # Create temporary GeoDataFrame for this row
                    temp_gdf = gpd.GeoDataFrame(
                        [row], 
                        geometry=gpd.points_from_xy([row['CoorX']], [row['CoorY']]),
                        crs=current_crs
                    )
                    
                    # Transform to target CRS if needed
                    if current_crs != target_crs:
                        temp_gdf = temp_gdf.to_crs(target_crs)
                    
                    # Append to UTM collection
                    if len(utm_gdfs[utm_zone]) == 0:
                        utm_gdfs[utm_zone] = temp_gdf
                    else:
                        utm_gdfs[utm_zone] = pd.concat([utm_gdfs[utm_zone], temp_gdf], ignore_index=True)
            
            self.logger.info(f'Processed {len(plot_df)} plots from region {region}')

        # Log summary statistics
        total_plots = sum(len(gdf) for gdf in utm_gdfs.values())
        self.logger.info(f"Total plots extracted: {total_plots}")
        for utm, gdf in utm_gdfs.items():
            if len(gdf) > 0:
                self.logger.info(f"  UTM {utm}: {len(gdf)} plots")
        
        return utm_gdfs

    def _extract_region_data(
        self, 
        accdb_ifn4: Path, 
        sig_files: list, 
        region: str
    ) -> Optional[pd.DataFrame]:
        """
        Extract biomass data for a specific region.
        
        CRITICAL: Preserves all original algorithmic logic exactly as-is.
        
        Args:
            accdb_ifn4: Path to IFN4 Access database file
            sig_files: List of SIG database files
            region: Region name
            
        Returns:
            pd.DataFrame or None: Processed plot data or None if extraction fails
        """
        # Export database tables to CSV
        table_config = self.config['processing']['database_tables']
        ifn4_out_file = f"{self.temp_dir}PCDatosMAP_{region}.csv"
        ifn4_pc_parcelas = f"{self.temp_dir}PCParcelas_{region}.csv"
        
        for table_name, output_file in [
            (table_config['ifn4_data'], ifn4_out_file),
            (table_config['ifn4_parcels'], ifn4_pc_parcelas)
        ]:
            if not Path(output_file).exists():
                mdb_command = f"mdb-export {accdb_ifn4} {table_name} > {output_file}"
                os.system(mdb_command)

        # Get corresponding SIG file
        accdb_sig = [file for file in sig_files if region in str(file)]
        if not accdb_sig:
            self.logger.warning(f"No SIG file found for {region}")
            return None
        accdb_sig = accdb_sig[0]

        # Extract biomass/volume data
        sig_out_file = f"{self.temp_dir}Parcelas_exs_{region}.csv"
        if not Path(sig_out_file).exists():
            mdb_command = f"mdb-export {accdb_sig} {table_config['sig_biomass']} > {sig_out_file}"
            os.system(mdb_command)
        
        plot_df = pd.read_csv(sig_out_file)
        
        if len(plot_df.index) == 0:
            self.logger.error(f'Empty stocks table for {region}')
            return None

        # Process biomass data based on available columns
        biomass_cols = self.config['processing']['column_mappings']['biomass_columns']
        volume_cols = self.config['processing']['column_mappings']['volume_columns']
        
        if biomass_cols[0] not in list(plot_df.columns):  # "BA" not in columns
            # Calculate biomass from volume data
            self.logger.info('Calculating biomass from woody volume data...')
            plot_df = plot_df[["Estadillo", volume_cols[1], volume_cols[0], "Especie", "Provincia"]]  # VLE, VCC
            plot_df["AGB"] = plot_df.apply(
                lambda x: compute_biomass(
                    x[volume_cols[0]], x[volume_cols[1]], x["Especie"], 
                    self.wood_density_db, self.ifn_species_codes
                ), axis=1
            )
            plot_df["BGB"] = np.nan
            plot_df = plot_df[["Estadillo", "AGB", "BGB", "Provincia"]]
        else:
            # Use reported biomass values
            self.logger.info('Using reported biomass data...')
            plot_df = plot_df[["Estadillo", "Provincia", biomass_cols[0], biomass_cols[1]]]  # BA, BR
            plot_df = plot_df.rename(columns={biomass_cols[0]: "AGB", biomass_cols[1]: "BGB"})

        # Create compound index and remove NaN values
        plot_df['Index'] = list(zip(plot_df["Estadillo"], plot_df["Provincia"]))
        na_indices = plot_df[plot_df["AGB"].isnull()]["Index"].unique()
        plot_df = plot_df[~plot_df["Index"].isin(na_indices)]
        
        # Aggregate by plot
        plot_df = plot_df.groupby("Index").agg({"AGB": "sum", "BGB": "sum"}).reset_index()

        # Add spatial and temporal information
        plot_df = self._add_spatial_temporal_info(plot_df, ifn4_out_file, ifn4_pc_parcelas, region)
        
        return plot_df

    def _add_spatial_temporal_info(
        self, 
        plot_df: pd.DataFrame, 
        ifn4_out_file: str, 
        ifn4_pc_parcelas: str, 
        region: str
    ) -> pd.DataFrame:
        """
        Add UTM coordinates, date, and region information to plot data.
        
        CRITICAL: Preserves all original algorithmic logic exactly as-is.
        """
        # Add coordinates and UTM info
        ifn4_df = pd.read_csv(ifn4_out_file)
        coord_cols = self.config['processing']['column_mappings']['coordinate_columns']
        ifn4_df = ifn4_df[["Estadillo", coord_cols[0], coord_cols[1], 'Huso', 'Provincia']]  # CoorX, CoorY
        ifn4_df['Index'] = list(zip(ifn4_df["Estadillo"], ifn4_df["Provincia"]))
        ifn4_df = ifn4_df.drop(columns=['Estadillo', 'Provincia'])
        plot_df = plot_df.join(ifn4_df.set_index("Index"), on="Index", how='inner')

        # Add date information
        ifn4_date_df = pd.read_csv(ifn4_pc_parcelas)
        ifn4_date_df = ifn4_date_df[["Estadillo", "FechaIni", "Provincia"]]
        ifn4_date_df['Index'] = list(zip(ifn4_date_df["Estadillo"], ifn4_date_df["Provincia"]))
        ifn4_date_df = ifn4_date_df.drop(columns=['Estadillo', 'Provincia'])
        plot_df = plot_df.join(ifn4_date_df.set_index("Index"), on="Index", how='inner')

        # Extract year and add region
        plot_df['Year'] = plot_df['FechaIni'].apply(lambda date: f"20{date[6:8]}")
        plot_df['Region'] = region
        
        return plot_df

    def add_forest_type_data(self, utm_gdfs: Dict[int, gpd.GeoDataFrame]) -> Dict[int, gpd.GeoDataFrame]:
        """
        Add forest type information from MFE (Spanish Forest Map) data.
        
        CRITICAL: Preserves all original algorithmic logic exactly as-is.
        
        Args:
            utm_gdfs: Dictionary of UTM GeoDataFrames
            
        Returns:
            dict: Updated UTM GeoDataFrames with forest type information
        """
        mfe_dir = f"{self.data_dir}/{self.config['data']['mfe_dir']}"
        enhanced_gdfs = {}
        
        for utm, plots in utm_gdfs.items():
            if len(plots) == 0:
                enhanced_gdfs[utm] = plots
                continue
                
            self.logger.info(f"Processing forest types for UTM {utm}")
            original_crs = plots.crs
            enhanced_plots = gpd.GeoDataFrame()
            
            # Iterate over MFE files to find spatial matches
            mfe_pattern = self.config['file_patterns']['mfe_files']
            for mfe_file in Path(mfe_dir).glob(mfe_pattern):
                try:
                    # Check if dissolved version exists, otherwise use original
                    dissolved_prefix = self.config['forest_types']['dissolved_prefix']
                    dissolved_file = f'{mfe_dir}{dissolved_prefix}{mfe_file.stem}.shp'
                    
                    if Path(dissolved_file).exists() and self.config['forest_types']['use_dissolved_files']:
                        mfe = gpd.read_file(dissolved_file)
                    else:
                        mfe = gpd.read_file(mfe_file)
                    
                    # Ensure same CRS
                    if mfe.crs != original_crs:
                        mfe = mfe.to_crs(original_crs)
                    
                    # Spatial join
                    joined = gpd.sjoin(plots, mfe, how='left', predicate='within')
                    
                    if 'TESELA' in joined.columns:
                        # Keep only plots that intersect with this MFE file
                        valid_plots = joined[~joined['TESELA'].isna()]
                        if len(valid_plots) > 0:
                            enhanced_plots = pd.concat([enhanced_plots, valid_plots], ignore_index=True)
                            # Remove processed plots from the original set
                            processed_indices = valid_plots.index
                            plots = plots.drop(processed_indices).reset_index(drop=True)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing MFE file {mfe_file}: {e}")
                    continue
            
            # Add remaining plots without forest type
            if len(plots) > 0:
                plots['ForestType'] = self.config['forest_types']['default_forest_type']
                enhanced_plots = pd.concat([enhanced_plots, plots], ignore_index=True)
            
            # Clean up duplicate index_right columns if present
            cols_to_drop = [col for col in enhanced_plots.columns if col.endswith('_right')]
            if cols_to_drop:
                enhanced_plots = enhanced_plots.drop(columns=cols_to_drop)
            
            enhanced_gdfs[utm] = enhanced_plots
            self.logger.info(f"UTM {utm}: {len(enhanced_plots)} plots with forest type information")
        
        return enhanced_gdfs

    def export_data(self, utm_gdfs: Dict[int, gpd.GeoDataFrame]) -> None:
        """
        Export data in multiple formats.
        
        CRITICAL: Preserves all original algorithmic logic exactly as-is.
        
        Args:
            utm_gdfs: Dictionary of UTM GeoDataFrames
        """
        create_output_directory(self.output_dir)
        
        all_data = []
        
        # Export UTM-specific files and collect data for combined export
        utm_template = self.config['output_templates']['utm_biomass']
        for utm, gdf in utm_gdfs.items():
            if len(gdf) > 0:
                # Add BGB ratio calculation
                gdf = calculate_bgb_ratio(gdf)
                
                # Convert to target CRS
                gdf = gdf.to_crs(self.target_crs)
                
                # Export UTM-specific file
                utm_filename = f"{self.output_dir}/{utm_template.format(utm=utm)}"
                gdf.to_file(utm_filename, driver='ESRI Shapefile')
                self.logger.info(f"Exported UTM {utm} data: {len(gdf)} plots -> {utm_filename}")
                
                # Collect for combined export
                all_data.append(gdf)
        
        # Export combined file with all UTMs
        if all_data:
            combined_gdf = pd.concat(all_data, ignore_index=True)
            combined_template = self.config['output_templates']['combined_biomass']
            combined_filename = f"{self.output_dir}/{combined_template}"
            combined_gdf.to_file(combined_filename, driver='ESRI Shapefile')
            self.logger.info(f"Exported combined data: {len(combined_gdf)} plots -> {combined_filename}")
            
            # Export year-stratified files
            self._export_year_stratified(combined_gdf)

    def _export_year_stratified(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Export data stratified by year.
        
        CRITICAL: Preserves all original algorithmic logic exactly as-is.
        
        Args:
            gdf: Combined GeoDataFrame
        """
        # Get all unique years in the data
        years = sorted(gdf['Year'].dropna().unique())
        self.logger.info(f"Found data for years: {years}")
        
        yearly_template = self.config['output_templates']['yearly_biomass']
        for year in years:
            year_data = gdf[gdf['Year'] == year]
            if len(year_data) > 0:
                year_filename = f"{self.output_dir}/{yearly_template.format(year=year)}"
                year_data.to_file(year_filename, driver='ESRI Shapefile')
                self.logger.info(f"Exported {year} data: {len(year_data)} plots -> {year_filename}")

    def validate_inputs(self) -> bool:
        """
        Validate that all required input files and directories exist.
        
        Returns:
            bool: True if all inputs are valid, False otherwise
        """
        base_path = Path(self.data_dir)
        
        # Check required directories
        required_dirs = [
            self.config['data']['ifn4_dir'],
            self.config['data']['mfe_dir']
        ]
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            if not dir_path.exists():
                self.logger.error(f"Required directory not found: {dir_path}")
                return False
        
        # Check required files
        required_files = [
            self.config['data']['wood_density_file'],
            self.config['data']['species_codes_file']
        ]
        
        for file_name in required_files:
            file_path = base_path / file_name
            if not file_path.exists():
                self.logger.error(f"Required file not found: {file_path}")
                return False
        
        self.logger.info("Input validation successful")
        return True

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the processing configuration.
        
        Returns:
            dict: Summary information
        """
        return {
            'data_directory': self.data_dir,
            'output_directory': self.output_dir,
            'target_crs': self.target_crs,
            'valid_utm_zones': get_valid_utm_zones(),
            'temp_directory': self.temp_dir,
            'component_version': "1.0.0"
        }
