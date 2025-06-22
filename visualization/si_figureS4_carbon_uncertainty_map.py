#!/usr/bin/env python3
"""
Generate Figure S4: Biomass Uncertainty Map (2024)

Creates a simple map showing biomass uncertainty with north arrow, scale bar, and colorbar.

Author: Diego Bengochea
Component: Visualization Pipeline - Supporting Information
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, save_figure_multiple_formats,
    get_figure_output_path, validate_required_files
)

# Set up logging
logger = setup_logging('figure_si_04')

def find_uncertainty_raster(data_paths, year: int = 2024) -> Path:
    """
    Auto-detect uncertainty raster file for the specified year.
    
    Args:
        data_paths: CentralDataPaths instance
        year: Year for uncertainty data
        
    Returns:
        Path to uncertainty raster file
    """
    # Look in biomass_maps directory (with mask subdirectory)
    biomass_dir = data_paths.get_path('biomass_maps') / data_paths.subdirs['biomass_maps']['with_mask']
    
    # Pattern: TBD_S2_uncertainty_YEAR_100m_TBD_merged.tif
    uncertainty_pattern = f"TBD_S2_uncertainty_{year}_100m_TBD_merged.tif"
    uncertainty_file = biomass_dir / uncertainty_pattern
    
    if uncertainty_file.exists():
        logger.info(f"Found uncertainty raster: {uncertainty_file}")
        return uncertainty_file
    
    # Fallback: look in other biomass subdirectories
    for subdir_name in data_paths.subdirs['biomass_maps'].values():
        subdir_path = data_paths.get_path('biomass_maps') / subdir_name
        uncertainty_file = subdir_path / uncertainty_pattern
        if uncertainty_file.exists():
            logger.info(f"Found uncertainty raster: {uncertainty_file}")
            return uncertainty_file
    
    # Final fallback: look in processed directory
    processed_dir = data_paths.get_path('processed')
    for subdir in ['biomass_maps', 'uncertainty_maps', 'results']:
        search_dir = processed_dir / subdir
        if search_dir.exists():
            uncertainty_file = search_dir / uncertainty_pattern
            if uncertainty_file.exists():
                logger.info(f"Found uncertainty raster: {uncertainty_file}")
                return uncertainty_file
    
    raise FileNotFoundError(
        f"Uncertainty raster not found. Searched for pattern: {uncertainty_pattern}\n"
        f"Searched locations:\n"
        f"  - {biomass_dir}\n"
        f"  - Other biomass subdirectories\n"
        f"  - {processed_dir}/*/\n"
        f"Please ensure biomass model has generated uncertainty outputs."
    )

def add_north_arrow_and_scalebar(ax):
    """Add a north arrow and scale bar to a map - EXACT SAME LOGIC."""
    # Add north arrow (triangle shape)
    x, y = 2, 44.2
    triangle = mpatches.RegularPolygon((x, y), 3, radius=0.2, orientation=0, 
                                      color='black', zorder=10)
    ax.add_patch(triangle)
    ax.text(x, y-0.5, 'N', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add scale bar (100 km)
    scale_bar_km = 100
    scalebar_x = 0.8
    scalebar_y = 0.05
    scalebar_width = 0.15
    scalebar_height = 0.01
    
    ax.annotate(f"{scale_bar_km} km", 
                xy=(scalebar_x, scalebar_y), 
                xycoords='axes fraction',
                fontsize=8, 
                ha='center', 
                va='bottom')
    
    bar_rect = plt.Rectangle(
        (scalebar_x - scalebar_width*0.5, scalebar_y - scalebar_height),
        scalebar_width, scalebar_height,
        transform=ax.transAxes,
        color='black',
        zorder=10
    )
    ax.add_patch(bar_rect)

def load_uncertainty_raster(file_path: Path, bounds_path: Path):
    """Load uncertainty raster and mask to country boundaries - EXACT SAME LOGIC."""
    logger.info(f"Loading uncertainty raster: {file_path}")
    
    with rasterio.open(file_path) as src:
        # Load country boundaries
        bounds_shape = gpd.read_file(bounds_path)
        
        # Reproject shapefile if needed
        if bounds_shape.crs != src.crs:
            bounds_shape = bounds_shape.to_crs(src.crs)
        
        # Mask raster with shapefile
        data, masked_transform = mask(
            src,
            bounds_shape.geometry,
            crop=True,
            nodata=np.nan
        )
        data = data[0]  # Get first band
        transform = masked_transform
        
        # Handle NoData values
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        
        # Get CRS and bounds
        crs = src.crs
        bounds = src.bounds
    
    # Filter unrealistic values (keep only positive uncertainty values) - EXACT SAME LOGIC
    data = np.where(data < 0, np.nan, data)
    data = np.where(data > 500, np.nan, data)  # Remove extremely high values
    
    return data, transform, crs, bounds

def create_uncertainty_map(uncertainty_data, transform, crs, bounds, country_bounds_path: Path) -> plt.Figure:
    """Create the uncertainty map - EXACT SAME LOGIC."""
    logger.info("Creating biomass uncertainty map for 2024")
    
    # Constants from original - EXACT SAME LOGIC
    NON_FOREST_COLOR = '#F5F5F5'  # Light grey for non-forested areas
    MAP_EXTENT = [-9.5, 3.5, 35.5, 44.5]  # Spain extent
    
    # Load country boundaries for overlay
    country_bounds = gpd.read_file(country_bounds_path)
    if country_bounds.crs != crs:
        country_bounds = country_bounds.to_crs(crs)
    
    # Calculate extent from bounds - EXACT SAME LOGIC
    left, bottom, right, top = bounds
    extent = (left, right, bottom, top)
    
    # Create figure - 160mm width - EXACT SAME LOGIC
    fig_width_mm = 160
    fig_width_inch = fig_width_mm / 25.4  # Convert mm to inches
    aspect_ratio = (MAP_EXTENT[3] - MAP_EXTENT[2]) / (MAP_EXTENT[1] - MAP_EXTENT[0])
    fig_height_inch = fig_width_inch * aspect_ratio * 1.1  # Add some space for colorbar
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width_inch, fig_height_inch),
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set map extent - EXACT SAME LOGIC
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    
    # Add country outline - EXACT SAME LOGIC
    land = ShapelyFeature(country_bounds.geometry, ccrs.epsg(25830), 
                         facecolor=NON_FOREST_COLOR, edgecolor='0.3', linewidth=0.3)
    ax.add_feature(land, zorder=0)
    
    # Choose colormap - using rocket as suggested - EXACT SAME LOGIC
    cmap = 'magma_r'  # Alternative: plt.cm.magma or plt.cm.flare
    
    # Plot uncertainty data - EXACT SAME LOGIC
    img = ax.imshow(
        uncertainty_data,
        extent=extent,
        transform=ccrs.epsg(25830),
        cmap=cmap,
        vmin=0,
        vmax=np.nanpercentile(uncertainty_data, 95),  # Use 95th percentile for max
        zorder=1,
    )
    
    # Add grid lines - EXACT SAME LOGIC
    gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='-', linewidth=0.5)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.right_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    # Set grid locations - EXACT SAME LOGIC
    gl.xlocator = mticker.FixedLocator([-8, -6, -4, -2, 0, 2])
    gl.ylocator = mticker.FixedLocator([36, 37.5, 39, 40.5, 42, 43.5])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # Remove spine edges - EXACT SAME LOGIC
    for spine in ax.spines.values():
        spine.set_edgecolor('none')
    
    # Add north arrow and scale bar - EXACT SAME LOGIC
    add_north_arrow_and_scalebar(ax)
    
    # Add colorbar as inset - EXACT SAME LOGIC
    cbax = ax.inset_axes([0.02, 0.02, 0.03, 0.3])  # Position at bottom-left
    cbar = plt.colorbar(img, cax=cbax, orientation='vertical')
    cbar.set_label('Uncertainty (Mg/ha)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Remove any default labels - EXACT SAME LOGIC
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Print statistics - EXACT SAME LOGIC
    logger.info("Uncertainty statistics:")
    logger.info(f"  Min: {np.nanmin(uncertainty_data):.2f} Mg/ha")
    logger.info(f"  Max: {np.nanmax(uncertainty_data):.2f} Mg/ha")
    logger.info(f"  Mean: {np.nanmean(uncertainty_data):.2f} Mg/ha")
    logger.info(f"  95th percentile: {np.nanpercentile(uncertainty_data, 95):.2f} Mg/ha")
    
    return fig

def main():
    """Main function to create Figure S4."""
    logger.info("Starting Figure S4 creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    try:
        # Auto-detect uncertainty raster
        uncertainty_path = find_uncertainty_raster(data_paths, year=2024)
        
        # Country bounds path
        country_bounds_path = data_paths.get_path('reference_data') / "SpainPolygon" / "gadm41_ESP_0.shp"
        
        # Validate required files exist
        validate_required_files([uncertainty_path, country_bounds_path], 'figure_si_04')
        
        # Load uncertainty data
        uncertainty_data, transform, crs, bounds = load_uncertainty_raster(
            uncertainty_path, country_bounds_path
        )
        
        # Create figure
        fig = create_uncertainty_map(
            uncertainty_data, transform, crs, bounds, country_bounds_path
        )
        
        # Save figure
        output_path = get_figure_output_path(config, "Figure_S4_Biomass_Uncertainty_2024", 'supporting_info')
        save_figure_multiple_formats(fig, output_path, config, logger)
        
        plt.close()
        logger.info("Figure S4 created successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Required data not found: {e}")
        logger.error("Please ensure biomass model uncertainty outputs are available")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating Figure S4: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()