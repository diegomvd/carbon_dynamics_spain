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
import time

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, log_pipeline_start, log_pipeline_end, log_section
from shared_utils.central_data_paths_constants import *


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
        self.data_dir = NFI4_DATABASE_DIR
        self.output_dir = FOREST_INVENTORY_PROCESSED_DIR
        self.temp_dir = NFI4_DATABASE_DIR / "tmp"
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
            self.wood_density_db, self.ifn_species_codes = load_reference_databases()
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
        
        Preserves all original algorithmic logic exactly as-is.
        
        Returns:
            dict: Dictionary with UTM zones as keys and GeoDataFrames as values
        """
        # Create temporary directory
        if not Path(self.temp_dir).exists():
            Path(self.temp_dir).mkdir(parents=True)

        # Get file lists using configuration patterns
        data_path = NFI4_DATABASE_DIR
        ifn4_files = list(data_path.glob(self.config['file_patterns']['ifn4_files']))
        sig_files = list(data_path.glob(self.config['file_patterns']['sig_files']))

        # Initialize GeoDataFrames for each UTM zone
        utm_gdfs = {utm: gpd.GeoDataFrame() for utm in get_valid_utm_zones()}

        self.logger.info(f"Found {len(ifn4_files)} IFN4 files and {len(sig_files)} SIG files")

        # Process each province
        for accdb_ifn4 in ifn4_files:
            region = re.findall("_(.+?)\.", str(accdb_ifn4.name))[0]
            self.logger.info(f'Processing region: {region}')
            
            # Extract data for this region
            plot_df = self._extract_region_data(
                accdb_ifn4, sig_files, region
            )
            
            if plot_df is None or len(plot_df) == 0:
                self.logger.warning(f"No data extracted for region {region}")
                continue
            
            for utm in plot_df['Huso'].unique():
                if utm is None or int(utm) not in get_valid_utm_zones():
                    continue
                    
                utm = int(utm)
                plot_df_filtered = plot_df[plot_df['Huso'] == utm]
                
                # Create georeferenced dataset
                plot_gdf = gpd.GeoDataFrame(
                    plot_df_filtered,
                    geometry=gpd.points_from_xy(
                        x=plot_df_filtered.CoorX, 
                        y=plot_df_filtered.CoorY
                    )
                )
                
                # Set correct CRS
                current_crs, target_crs = get_region_UTM(region, str(utm))
                plot_gdf = plot_gdf.set_crs(current_crs).to_crs(target_crs)
                
                # Clean up columns
                plot_gdf = plot_gdf.drop(columns=['CoorX', 'CoorY', 'Huso', 'FechaIni'])
                
                # Concatenate to UTM-specific dataframe
                utm_gdfs[utm] = pd.concat([utm_gdfs[utm], plot_gdf], axis='rows')
            
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
        
        Preserves all original algorithmic logic exactly as-is.
        
        Args:
            accdb_ifn4: Path to IFN4 Access database file
            sig_files: List of SIG database files
            region: Region name
            
        Returns:
            pd.DataFrame or None: Processed plot data or None if extraction fails
        """
        # Export database tables to CSV
        table_config = self.config['processing']['database_tables']
        ifn4_out_file = self.temp_dir/f"PCDatosMAP_{region}.csv"
        ifn4_pc_parcelas = self.temp_dir/f"PCParcelas_{region}.csv"
        
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
        sig_out_file = self.temp_dir / f"Parcelas_exs_{region}.csv"
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
        
        Preserves all original algorithmic logic exactly as-is.
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

    def _preprocess_mfe_data(self) -> gpd.GeoDataFrame:
        """
        Load and combine all MFE files into a single GeoDataFrame with spatial indexing.
        
        This function loads all MFE shapefile data once, combines them, and builds
        a spatial index for fast spatial queries. Only essential columns are kept
        to minimize memory usage.
        
        Returns:
            gpd.GeoDataFrame: Combined MFE data with spatial index, or empty GeoDataFrame if no data found
        """
        self.logger.info("Loading and indexing forest type data...")
        start_time = time.time()
        
        mfe_dir = FOREST_TYPE_MAPS_DIR
        mfe_files = list(Path(mfe_dir).glob(self.config['file_patterns']['mfe_files']))
        mfe_parts = []
        
        if not mfe_files:
            self.logger.warning(f"No MFE files found in {mfe_dir}")
            return gpd.GeoDataFrame()
        
        # Load each MFE file
        for mfe_file in mfe_files:
            try:
                mfe = gpd.read_file(mfe_file)
                
                if 'FormArbol' not in mfe.columns:
                    self.logger.warning(f"No FormArbol column in {mfe_file.name}, skipping")
                    continue
                
                # Keep only essential columns to reduce memory usage
                mfe_simple = mfe[['FormArbol', 'geometry']].copy()
                mfe_simple.rename(columns={'FormArbol': 'ForestType'}, inplace=True)
                
                # Validate geometries
                valid_geoms = mfe_simple.geometry.is_valid
                if not valid_geoms.all():
                    self.logger.warning(f"Found {(~valid_geoms).sum()} invalid geometries in {mfe_file.name}")
                    mfe_simple = mfe_simple[valid_geoms]
                
                mfe_parts.append(mfe_simple)
                self.logger.info(f"Loaded {mfe_file.name}: {len(mfe_simple):,} polygons")
                
            except Exception as e:
                self.logger.warning(f"Error loading {mfe_file}: {e}")
                continue
        
        if not mfe_parts:
            self.logger.error("No valid MFE files could be loaded")
            return gpd.GeoDataFrame()
        
        # Combine all MFE data
        combined_mfe = pd.concat(mfe_parts, ignore_index=True)
        
        # Force spatial index creation for fast spatial queries
        _ = combined_mfe.sindex
        
        load_time = time.time() - start_time
        self.logger.info(f"MFE preprocessing complete: {len(combined_mfe):,} polygons in {load_time:.1f}s")
        
        return combined_mfe

    def _spatial_filter_mfe(self, plots: gpd.GeoDataFrame, mfe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter MFE data to only polygons that could potentially intersect with plots.
        
        Uses bounding box filtering for fast spatial pre-filtering before expensive
        spatial join operations. This dramatically reduces the number of polygons
        that need to be checked during spatial joins.
        
        Args:
            plots: GeoDataFrame containing plot points
            mfe: GeoDataFrame containing forest type polygons
            
        Returns:
            gpd.GeoDataFrame: Filtered MFE data containing only potentially intersecting polygons
        """
        if len(plots) == 0 or len(mfe) == 0:
            return gpd.GeoDataFrame()
        
        # Ensure same CRS
        if mfe.crs != plots.crs:
            self.logger.info(f"Reprojecting MFE from {mfe.crs} to {plots.crs}")
            mfe = mfe.to_crs(plots.crs)
        
        # Get bounding box of all plots
        plots_bounds = plots.total_bounds  # [minx, miny, maxx, maxy]
        
        # Add small buffer to account for edge cases (100m buffer)
        buffer = 100  # meters
        buffered_bounds = [
            plots_bounds[0] - buffer,  # minx
            plots_bounds[1] - buffer,  # miny
            plots_bounds[2] + buffer,  # maxx
            plots_bounds[3] + buffer   # maxy
        ]
        
        # Fast bounding box filtering using cx indexer
        mfe_filtered = mfe.cx[
            buffered_bounds[0]:buffered_bounds[2], 
            buffered_bounds[1]:buffered_bounds[3]
        ]
        
        self.logger.info(f"Spatial filtering: {len(mfe):,} → {len(mfe_filtered):,} polygons "
                        f"({len(mfe_filtered)/len(mfe)*100:.1f}% retained)")
        
        return mfe_filtered

    def add_forest_type_data(self, utm_gdfs: Dict[int, gpd.GeoDataFrame]) -> Dict[int, gpd.GeoDataFrame]:
        """
        Add forest type information from MFE data using optimized spatial processing.
        
        This optimized version:
        1. Loads all MFE data once with spatial indexing
        2. Uses bounding box pre-filtering for each UTM zone
        3. Performs single spatial join per UTM zone
        4. Handles missing forest types gracefully
        
        Args:
            utm_gdfs: Dictionary of UTM GeoDataFrames containing plot data
            
        Returns:
            dict: Updated UTM GeoDataFrames with forest type information
        """
        # Preprocess all MFE data once
        combined_mfe = self._preprocess_mfe_data()
        
        if len(combined_mfe) == 0:
            self.logger.error("No MFE data available - adding default forest types")
            enhanced_gdfs = {}
            for utm, plots in utm_gdfs.items():
                if len(plots) > 0:
                    plots = plots.copy()
                    plots['ForestType'] = self.config['forest_types']['default_forest_type']
                enhanced_gdfs[utm] = plots
            return enhanced_gdfs
        
        # Process each UTM zone
        enhanced_gdfs = {}
        total_start = time.time()
        
        for utm, plots in utm_gdfs.items():
            if len(plots) == 0:
                enhanced_gdfs[utm] = plots
                continue
                
            self.logger.info(f"Processing forest types for UTM {utm}: {len(plots):,} plots")
            utm_start = time.time()
            
            # Spatial filtering to reduce MFE data size
            mfe_filtered = self._spatial_filter_mfe(plots, combined_mfe)
            
            if len(mfe_filtered) == 0:
                self.logger.info(f"No MFE polygons overlap with UTM {utm} plots")
                plots_copy = plots.copy()
                plots_copy['ForestType'] = self.config['forest_types']['default_forest_type']
                enhanced_gdfs[utm] = plots_copy
                continue
            
            # Single spatial join operation
            self.logger.info(f"Performing spatial join for UTM {utm}...")
            join_start = time.time()
            
            enhanced_plots = gpd.sjoin(
                plots, mfe_filtered, 
                how='left', 
                predicate='intersects'  # More inclusive than 'within'
            )
            
            join_time = time.time() - join_start
            self.logger.info(f"Spatial join completed in {join_time:.1f}s")
            
            # Clean up spatial join artifacts
            enhanced_plots = enhanced_plots.drop(columns=['index_right'], errors='ignore')
            
            # Handle missing forest types
            missing_count = enhanced_plots['ForestType'].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Filling {missing_count:,} missing forest types with default value")
                enhanced_plots['ForestType'] = enhanced_plots['ForestType'].fillna(
                    self.config['forest_types']['default_forest_type']
                )
            
            # Log forest type distribution
            forest_type_counts = enhanced_plots['ForestType'].value_counts()
            self.logger.info(f"Forest type distribution for UTM {utm}:")
            for forest_type, count in forest_type_counts.head(5).items():
                self.logger.info(f"  {forest_type}: {count:,} plots ({count/len(enhanced_plots)*100:.1f}%)")
            
            enhanced_gdfs[utm] = enhanced_plots
            
            utm_time = time.time() - utm_start
            self.logger.info(f"UTM {utm} completed in {utm_time:.1f}s")
        
        total_time = time.time() - total_start
        total_plots = sum(len(gdf) for gdf in enhanced_gdfs.values())
        self.logger.info(f"Forest type processing completed: {total_plots:,} plots in {total_time:.1f}s")
        
        return enhanced_gdfs

    def export_data(self, utm_gdfs: Dict[int, gpd.GeoDataFrame]) -> None:
        """
        Export data in multiple formats.
        
        Preserves all original algorithmic logic exactly as-is.
        
        Args:
            utm_gdfs: Dictionary of UTM GeoDataFrames
        """
        create_output_directory(self.output_dir)
        
        output_utm = self.output_dir / 'per_utm'
        create_output_directory(output_utm)
        
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
                utm_filename = output_utm / f"{utm_template.format(utm=utm)}"
                gdf.to_file(utm_filename, driver='ESRI Shapefile')
                self.logger.info(f"Exported UTM {utm} data: {len(gdf)} plots -> {utm_filename}")
                
                # Collect for combined export
                all_data.append(gdf)
        
        # Export combined file with all UTMs
        if all_data:
            combined_gdf = pd.concat(all_data, ignore_index=True)
            combined_template = self.config['output_templates']['combined_biomass']
            combined_filename = self.output_dir / f"{combined_template}"
            combined_gdf.to_file(combined_filename, driver='ESRI Shapefile')
            self.logger.info(f"Exported combined data: {len(combined_gdf)} plots -> {combined_filename}")
            
            # Export year-stratified files
            self._export_year_stratified(combined_gdf)

    def _export_year_stratified(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Export data stratified by year.
        
        Preserves all original algorithmic logic exactly as-is.
        
        Args:
            gdf: Combined GeoDataFrame
        """
        # Get all unique years in the data
        years = sorted(gdf['Year'].dropna().unique())
        self.logger.info(f"Found data for years: {years}")
        
        output_year = self.output_dir / 'per_year'
        create_output_directory(output_year)

        yearly_template = self.config['output_templates']['yearly_biomass']
        for year in years:
            year_data = gdf[gdf['Year'] == year]
            if len(year_data) > 0:
                year_filename = output_year/f"{yearly_template.format(year=year)}"
                year_data.to_file(year_filename, driver='ESRI Shapefile')
                self.logger.info(f"Exported {year} data: {len(year_data)} plots -> {year_filename}")
