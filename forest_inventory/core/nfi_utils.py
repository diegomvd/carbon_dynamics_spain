"""
Utility functions for Spanish National Forest Inventory (NFI) data processing.

This module provides common functions for biomass calculation, coordinate system handling,
species data management, and geospatial operations for processing NFI4 data.

The utilities support:
- CRS handling for different Spanish regions and UTM zones
- Species name and wood density lookups
- Biomass calculations using allometric relationships
- Below-ground to above-ground biomass ratio calculations
- File I/O operations and directory management

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Shared utilities
from shared_utils import setup_logging, get_logger


def get_region_UTM(region, utmzone):
    """
    Get the appropriate CRS codes for a region and UTM zone.
    
    Handles the complex coordinate system situation in Spain where different
    regions use either ED50 or ETRS89 datums, and provides appropriate
    transformation targets.
    
    Args:
        region: Region name (province or autonomous community)
        utmzone: UTM zone as string ('28', '29', '30', '31')
        
    Returns:
        tuple: (current_crs, target_crs) EPSG codes
        
    Examples:
        >>> current, target = get_region_UTM('Madrid', '30')
        >>> print(f"Transform from {current} to {target}")
        Transform from EPSG:23030 to EPSG:25830
    """
    # Regions that historically used ED50 datum
    list_ED50 = [
        'Navarra', 'ACoruna', 'Lugo', 'Orense', 'Pontevedra',
        'Asturias', 'Cantabria', 'Murcia', 'Baleares', 
        'Euskadi', 'LaRioja', 'Madrid', 'Cataluna'
    ]

    # ED50 EPSG codes by UTM zone
    epsg_ed50 = {
        '28': 'EPSG:23028', '29': 'EPSG:23029', 
        '30': 'EPSG:23030', '31': 'EPSG:23031'
    }
    
    # ETRS89 EPSG codes by UTM zone
    epsg_etrs89 = {
        '28': 'EPSG:25828', '29': 'EPSG:25829', 
        '30': 'EPSG:25830', '31': 'EPSG:25831'
    }
    
    # Determine appropriate CRS based on region
    if region in list_ED50:
        return epsg_ed50[utmzone], epsg_etrs89[utmzone] 
    elif region == 'Canarias':
        return 'EPSG:4326', 'EPSG:25828'  # Special case for Canary Islands
    else:
        return epsg_etrs89[utmzone], epsg_etrs89[utmzone]


def get_species_name(species_code, spdb):
    """
    Get species name from species code using the species database.
    
    Args:
        species_code: Species code
        spdb: Species database
        
    Returns:
        str or None: Species name or None if not found
        
    Examples:
        >>> name = get_species_name(21, species_db)
    """
    try:
        species_name = spdb[spdb["CODIGO ESPECIE"] == species_code].reset_index()["NOMBRE IFN"][0]
    except:
        species_name = None
    return species_name


def get_wood_density(species_name, wddb):
    """
    Get wood density for a species from the Global Wood Density Database.
    
    Args:
        species_name: Scientific name of the species
        wddb: Wood density database
        
    Returns:
        float: Wood density value
        
    Examples:
        >>> density = get_wood_density("Pinus sylvestris", wood_db)
    """
    if species_name in wddb["Binomial"].values:
        wd = wddb[wddb["Binomial"] == species_name]["Wood density (g/cm^3), oven dry mass/fresh volume"].dropna().mean()
    else: 
        genus = species_name.split(' ')[0]
        wddb2 = wddb.copy()
        wddb2['Binomial'] = wddb2['Binomial'].apply(lambda name: name.split(' ')[0])
        if genus in wddb2["Binomial"].values:
            wd = wddb2[wddb2["Binomial"] == genus]["Wood density (g/cm^3), oven dry mass/fresh volume"].dropna().mean()
        else:
            wd = wddb["Wood density (g/cm^3), oven dry mass/fresh volume"].dropna().mean()
    return wd


def compute_biomass(volume_stem, volume_branches, species_code, wddb, spdb):
    """
    Compute above-ground biomass from volume data using wood density.
    
    Args:
        volume_stem: Stem volume
        volume_branches: Branch volume  
        species_code: Species code
        wddb: Wood density database
        spdb: Species database
        
    Returns:
        float: Above-ground biomass or NaN if calculation fails
        
    Examples:
        >>> agb = compute_biomass(0.5, 0.1, 21, wood_db, species_db)
        >>> print(f"AGB: {agb}")
    """
    species_name = get_species_name(species_code, spdb)
    try:
        wd = get_wood_density(species_name, wddb)
        agb = (volume_stem + volume_branches) * wd
    except:
        agb = np.nan    
    return agb


def load_reference_databases(data_dir):
    """
    Load reference databases needed for NFI processing.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        tuple: (wood_density_db, ifn_species_codes)
        
    Examples:
        >>> wood_db, species_db = load_reference_databases("/path/to/data")
    """
    wood_density_db = pd.read_excel(
        f"{data_dir}/GlobalWoodDensityDatabase.xls", 
        sheet_name='Data'
    )
    
    ifn_species_codes = pd.read_csv(
        f"{data_dir}/CODIGOS_IFN.csv", 
        delimiter=";"
    ).dropna()
    ifn_species_codes["CODIGO ESPECIE"] = ifn_species_codes["CODIGO ESPECIE"].apply(lambda c: int(c))
    
    return wood_density_db, ifn_species_codes


def calculate_bgb_ratio(gdf):
    """
    Calculate the below-ground to above-ground biomass ratio.
    
    Args:
        gdf: GeoDataFrame with AGB and BGB columns
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with added BGB_Ratio column
        
    Examples:
        >>> gdf_with_ratio = calculate_bgb_ratio(biomass_gdf)
    """
    gdf = gdf.copy()
    # Avoid division by zero and handle NaN values
    gdf['BGB_Ratio'] = np.where(
        (gdf['AGB'] > 0) & (~gdf['AGB'].isna()) & (~gdf['BGB'].isna()),
        gdf['BGB'] / gdf['AGB'],
        np.nan
    )
    return gdf


def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
        
    Examples:
        >>> create_output_directory("./results")
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        logger = get_logger('forest_inventory')
        logger.info(f"Created output directory: {output_dir}")


def get_valid_utm_zones():
    """
    Get list of valid UTM zones for processing (excluding 28).
    
    Returns:
        list: List of valid UTM zones
        
    Examples:
        >>> zones = get_valid_utm_zones()
        >>> print(zones)  # [29, 30, 31]
    """
    return [29, 30, 31]
