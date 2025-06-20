"""
Harmonized processing pipeline for Spanish National Forest Inventory (NFI) biomass data.

This script processes NFI4 data to:
1. Extract biomass stocks from IFN4 and SIG database files
2. Add forest type information from MFE (Spanish Forest Map) data
3. Calculate below-ground to above-ground biomass ratios
4. Export data in multiple formats:
   - UTM-specific shapefiles (29, 30, 31)
   - Combined shapefile with all UTMs
   - Year-stratified shapefiles

Author: Diego Bengochea
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import re
from pathlib import Path
from nfi_utils import (
    setup_logging, get_region_UTM, compute_biomass, load_reference_databases,
    calculate_bgb_ratio, create_output_directory, get_valid_utm_zones
)

# =============================================================================
# CONFIGURATION MACROS
# =============================================================================
DATA_DIR = "/path/to/nfi/data/"  # Contains IFN_4_SP/, MFESpain/, databases, etc.
OUTPUT_DIR = "/path/to/output/"  # All exports will be saved here
TEMP_DIR = f"{DATA_DIR}/IFN_4_SP/tmp/"  # Temporary files directory
TARGET_CRS = "EPSG:25830"  # Final CRS for all outputs

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def extract_biomass_data(data_dir, temp_dir, wood_density_db, ifn_species_codes, logger):
    """
    Extract biomass data from NFI database files for all provinces.
    
    Returns:
        dict: Dictionary with UTM zones as keys and GeoDataFrames as values
    """
    # Create temporary directory
    if not Path(temp_dir).exists():
        Path(temp_dir).mkdir(parents=True)

    # Get file lists
    ifn4_files = [file for file in Path(f"{data_dir}/IFN_4_SP/").glob('Ifn4*')]
    sig_files = [file for file in Path(f"{data_dir}/IFN_4_SP/").glob('Sig_*')]

    # Initialize GeoDataFrames for each UTM zone
    utm_gdfs = {utm: gpd.GeoDataFrame() for utm in get_valid_utm_zones()}

    logger.info(f"Found {len(ifn4_files)} IFN4 files and {len(sig_files)} SIG files")

    # Process each province
    for accdb_ifn4 in ifn4_files:
        region = re.findall("_(.*)", accdb_ifn4.stem)[0]
        logger.info(f"Processing province {region}")
        
        # Skip Canarias and Baleares as specified in original code
        if region in ["Canarias", "Baleares"]:
            logger.info(f"Skipping {region} (excluded region)")
            continue

        try:
            # Extract database tables
            plot_df = extract_province_data(
                accdb_ifn4, sig_files, region, temp_dir, 
                wood_density_db, ifn_species_codes, logger
            )
            
            if plot_df is None:
                continue

            # Process each UTM zone
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

        except Exception as e:
            logger.error(f"Error processing {region}: {str(e)}")
            continue

    # Reset indices
    for utm in utm_gdfs:
        utm_gdfs[utm] = utm_gdfs[utm].reset_index(drop=True)
        logger.info(f"UTM {utm}: {len(utm_gdfs[utm])} plots processed")

    return utm_gdfs


def extract_province_data(accdb_ifn4, sig_files, region, temp_dir, 
                         wood_density_db, ifn_species_codes, logger):
    """
    Extract and process biomass data for a single province.
    
    Returns:
        pd.DataFrame or None: Processed plot data or None if extraction fails
    """
    # Export database tables to CSV
    ifn4_out_file = f"{temp_dir}PCDatosMAP_{region}.csv"
    ifn4_pc_parcelas = f"{temp_dir}PCParcelas_{region}.csv"
    
    for table_name, output_file in [
        ("PCDatosMap", ifn4_out_file),
        ("PCParcelas", ifn4_pc_parcelas)
    ]:
        if not Path(output_file).exists():
            mdb_command = f"mdb-export {accdb_ifn4} {table_name} > {output_file}"
            os.system(mdb_command)

    # Get corresponding SIG file
    accdb_sig = [file for file in sig_files if region in str(file)]
    if not accdb_sig:
        logger.warning(f"No SIG file found for {region}")
        return None
    accdb_sig = accdb_sig[0]

    # Extract biomass/volume data
    sig_out_file = f"{temp_dir}Parcelas_exs_{region}.csv"
    if not Path(sig_out_file).exists():
        mdb_command = f"mdb-export {accdb_sig} Parcelas_exs > {sig_out_file}"
        os.system(mdb_command)
    
    plot_df = pd.read_csv(sig_out_file)
    
    if len(plot_df.index) == 0:
        logger.error(f'Empty stocks table for {region}')
        return None

    # Process biomass data based on available columns
    if "CA" not in list(plot_df.columns):
        # Calculate biomass from volume data
        logger.info('Calculating biomass from woody volume data...')
        plot_df = plot_df[["Estadillo", "VLE", "VCC", "Especie", "Provincia"]]
        plot_df["AGB"] = plot_df.apply(
            lambda x: compute_biomass(
                x["VCC"], x["VLE"], x["Especie"], 
                wood_density_db, ifn_species_codes
            ), axis=1
        )
        plot_df["BGB"] = np.nan
        plot_df = plot_df[["Estadillo", "AGB", "BGB", "Provincia"]]
    else:
        # Use reported biomass values
        logger.info('Using reported biomass data...')
        plot_df = plot_df[["Estadillo", "Provincia", "BA", "BR"]]
        plot_df = plot_df.rename(columns={"BA": "AGB", "BR": "BGB"})

    # Create compound index and remove NaN values
    plot_df['Index'] = list(zip(plot_df["Estadillo"], plot_df["Provincia"]))
    na_indices = plot_df[plot_df["AGB"].isnull()]["Index"].unique()
    plot_df = plot_df[~plot_df["Index"].isin(na_indices)]
    
    # Aggregate by plot
    plot_df = plot_df.groupby("Index").agg({"AGB": "sum", "BGB": "sum"}).reset_index()

    # Add spatial and temporal information
    plot_df = add_spatial_temporal_info(plot_df, ifn4_out_file, ifn4_pc_parcelas, region)
    
    return plot_df


def add_spatial_temporal_info(plot_df, ifn4_out_file, ifn4_pc_parcelas, region):
    """Add UTM coordinates, date, and region information to plot data."""
    # Add coordinates and UTM info
    ifn4_df = pd.read_csv(ifn4_out_file)
    ifn4_df = ifn4_df[["Estadillo", "CoorX", "CoorY", 'Huso', 'Provincia']]
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


def add_forest_type_data(utm_gdfs, data_dir, logger):
    """
    Add forest type information from MFE (Spanish Forest Map) data.
    
    Args:
        utm_gdfs (dict): Dictionary of UTM GeoDataFrames
        data_dir (str): Data directory path
        logger: Logger instance
        
    Returns:
        dict: Updated UTM GeoDataFrames with forest type information
    """
    logger.info("Adding forest type information from MFE data...")
    
    mfe_dir = f"{data_dir}/MFESpain/"
    enhanced_gdfs = {}
    
    for utm, plots in utm_gdfs.items():
        if len(plots) == 0:
            enhanced_gdfs[utm] = plots
            continue
            
        logger.info(f"Processing forest types for UTM {utm}")
        original_crs = plots.crs
        enhanced_plots = gpd.GeoDataFrame()
        
        # Iterate over MFE files to find spatial matches
        for mfe_file in Path(mfe_dir).glob('MFE*.shp'):
            try:
                # Check if dissolved version exists, otherwise use original
                dissolved_file = f'{mfe_dir}dissolved_{mfe_file.stem}.shp'
                if Path(dissolved_file).exists():
                    mfe_contour = gpd.read_file(dissolved_file)
                else:
                    mfe_contour = gpd.read_file(mfe_file)
                
                # Reproject plots to match MFE CRS
                plots_reprojected = plots.to_crs(mfe_contour.crs)
                
                # Find intersecting plots
                intersecting = plots_reprojected.geometry.intersects(
                    mfe_contour.geometry.unary_union
                )
                
                if np.any(intersecting):
                    logger.info(f"Found spatial match with {mfe_file.stem}")
                    matched_plots = plots_reprojected[intersecting]
                    
                    # Spatial join to add forest type
                    if 'FormArbol' in mfe_contour.columns:
                        mfe_simplified = mfe_contour[['FormArbol', 'geometry']]
                        mfe_simplified = mfe_simplified.rename(
                            columns={'FormArbol': 'ForestType'}
                        )
                        matched_plots = gpd.sjoin(
                            matched_plots, mfe_simplified, how='left'
                        ).drop(columns=['index_right'], errors='ignore')
                    
                    # Reproject back to original CRS
                    matched_plots = matched_plots.to_crs(original_crs)
                    enhanced_plots = pd.concat([enhanced_plots, matched_plots])
                    
            except Exception as e:
                logger.warning(f"Error processing {mfe_file}: {str(e)}")
                continue
        
        # If no forest type data was added, add empty ForestType column
        if 'ForestType' not in enhanced_plots.columns:
            plots['ForestType'] = np.nan
            enhanced_gdfs[utm] = plots
        else:
            enhanced_gdfs[utm] = enhanced_plots.reset_index(drop=True)
    
    return enhanced_gdfs


def export_data(utm_gdfs, output_dir, logger):
    """
    Export processed data in multiple formats.
    
    Args:
        utm_gdfs (dict): Dictionary of UTM GeoDataFrames
        output_dir (str): Output directory path
        logger: Logger instance
    """
    logger.info("Starting data export...")
    create_output_directory(output_dir)
    
    all_data = []
    
    # Export UTM-specific files and collect data for combined export
    for utm, gdf in utm_gdfs.items():
        if len(gdf) > 0:
            # Add BGB ratio calculation
            gdf = calculate_bgb_ratio(gdf)
            
            # Convert to target CRS
            gdf = gdf.to_crs(TARGET_CRS)
            
            # Export UTM-specific file
            utm_filename = f"{output_dir}/nfi4_utm{utm}_biomass.shp"
            gdf.to_file(utm_filename, driver='ESRI Shapefile')
            logger.info(f"Exported UTM {utm} data: {len(gdf)} plots -> {utm_filename}")
            
            # Collect for combined export
            all_data.append(gdf)
    
    # Export combined file with all UTMs
    if all_data:
        combined_gdf = pd.concat(all_data, ignore_index=True)
        combined_filename = f"{output_dir}/nfi4_all_biomass.shp"
        combined_gdf.to_file(combined_filename, driver='ESRI Shapefile')
        logger.info(f"Exported combined data: {len(combined_gdf)} plots -> {combined_filename}")
        
        # Export year-stratified files
        export_year_stratified(combined_gdf, output_dir, logger)


def export_year_stratified(gdf, output_dir, logger):
    """
    Export data stratified by year.
    
    Args:
        gdf (gpd.GeoDataFrame): Combined GeoDataFrame
        output_dir (str): Output directory path
        logger: Logger instance
    """
    logger.info("Exporting year-stratified data...")
    
    # Get all unique years in the data
    years = sorted(gdf['Year'].dropna().unique())
    logger.info(f"Found data for years: {years}")
    
    for year in years:
        year_data = gdf[gdf['Year'] == year]
        if len(year_data) > 0:
            year_filename = f"{output_dir}/nfi4_{year}_biomass.shp"
            year_data.to_file(year_filename, driver='ESRI Shapefile')
            logger.info(f"Exported {year} data: {len(year_data)} plots -> {year_filename}")


def main():
    """Main processing pipeline."""
    logger = setup_logging()
    logger.info("Starting NFI biomass processing pipeline...")
    
    try:
        # Load reference databases
        logger.info("Loading reference databases...")
        wood_density_db, ifn_species_codes = load_reference_databases(DATA_DIR)
        
        # Extract biomass data
        logger.info("Extracting biomass data from NFI files...")
        utm_gdfs = extract_biomass_data(
            DATA_DIR, TEMP_DIR, wood_density_db, ifn_species_codes, logger
        )
        
        # Add forest type information
        utm_gdfs = add_forest_type_data(utm_gdfs, DATA_DIR, logger)
        
        # Export data in all formats
        export_data(utm_gdfs, OUTPUT_DIR, logger)
        
        logger.info("NFI biomass processing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
