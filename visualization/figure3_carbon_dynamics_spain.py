#!/usr/bin/env python3
"""
Generate Figure 1: Carbon Status and Change (2017-2024)

This script creates a comprehensive figure with four panels:
- Panel A: Carbon storage map for 2024 (top-left)
- Panel B: Carbon change map showing raw difference (top-right)
- Panel C: Country-level carbon stocks trend (bottom-left) with smoothed uncertainty
- Panel D: Interannual carbon fluxes with cumulative trend (bottom-right) on same axis

Author: Diego Bengochea
Component: Visualization Pipeline
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from affine import Affine
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    save_figure_multiple_formats, load_trend_data,
    get_figure_output_path, validate_required_files
)

# Set up logging
logger = setup_logging('figure_01')

def load_raster(
    file_path: Path, 
    bounds_path: Path, 
    biomass_to_carbon: float = 0.5
) -> tuple:
    """
    Load a raster file, mask with country bounds, and convert to carbon values.
    
    Args:
        file_path: Path to the raster file
        bounds_path: Path to country boundary shapefile
        biomass_to_carbon: Conversion factor from biomass to carbon
        
    Returns:
        Tuple of (data, transform, crs, bounds)
    """
    logger.info(f"Loading raster: {file_path}")
    
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
    
    # Filter unrealistic values
    data = np.where(data > 1500, np.nan, data)  # Unrealistically high biomass values
    data = np.where(data < 0, 0.0, data)  # Set negative values to 0.0
    
    # Convert biomass to carbon
    data = data * biomass_to_carbon
    
    return data, transform, crs, bounds

def add_north_arrow_and_scalebar(ax):
    """Add a north arrow and scale bar to a map."""
    # Add north arrow (triangle shape without tail)
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

def plot_biomass_carbon_figure(
    carbon_data_2024,
    carbon_data_2017,
    transform,
    crs,
    bounds,
    trend_data,
    country_bounds_path: Path,
    config: dict
) -> plt.Figure:
    """
    Create Figure 1 with carbon map, temporal trend, and change maps.
    
    Args:
        carbon_data_2024: Carbon data for 2024
        carbon_data_2017: Carbon data for 2017
        transform: Raster transform
        crs: Coordinate reference system
        bounds: Raster bounds
        trend_data: Temporal trend DataFrame
        country_bounds_path: Path to country boundary shapefile
        config: Configuration dictionary
        
    Returns:
        Figure object
    """
    logger.info("Creating Figure 1: Carbon Status and Change")
    
    # Get figure parameters from config
    fig_params = config['figure_params']['figure1']
    constants = config['constants']
    
    # Load country boundaries
    country_bounds = gpd.read_file(country_bounds_path)
    
    # Reproject if needed
    if country_bounds.crs != crs:
        country_bounds = country_bounds.to_crs(crs)
    
    # Calculate extent from bounds
    left, bottom, right, top = bounds
    extent = (left, right, bottom, top)
    
    # Create a mask for areas inside Spain
    transform_affine = Affine.from_gdal(*transform.to_gdal())
    
    # Rasterize Spain boundary to create mask
    from rasterio import features
    shapes = [(geom, 1) for geom in country_bounds.geometry]
    spain_mask = features.rasterize(
        shapes=shapes,
        out_shape=carbon_data_2024.shape,
        transform=transform_affine,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    ).astype(bool)
    
    # Set NaN values inside Spain to 0 (non-forest) but keep outside Spain as NaN
    carbon_data_2024_masked = carbon_data_2024.copy()
    carbon_data_2024_masked[np.isnan(carbon_data_2024) & spain_mask] = 0
    
    carbon_data_2017_masked = carbon_data_2017.copy()
    carbon_data_2017_masked[np.isnan(carbon_data_2017) & spain_mask] = 0
    
    # Calculate change map
    carbon_change_absolute = carbon_data_2024_masked - carbon_data_2017_masked
    carbon_change_absolute[~spain_mask] = np.nan  # Keep outside as NaN
    
    # Calculate annual fluxes
    flux_data = trend_data.copy()
    flux_data['annual_flux'] = flux_data['carbon_mean'].diff()
    flux_data.loc[0, 'annual_flux'] = 0  # First year has no previous year
    
    # Calculate cumulative flux starting from 0 in 2017
    flux_data['cumulative_flux'] = 0  # Initialize first point to 0
    for i in range(1, len(flux_data)):
        flux_data.loc[i, 'cumulative_flux'] = flux_data.loc[i-1, 'cumulative_flux'] + flux_data.loc[i, 'annual_flux']
    
    # Create figure with journal-compliant size
    fig = plt.figure(figsize=fig_params['figsize'])
    
    # Create main gridspec for 2x2 layout
    gs_main = gridspec.GridSpec(2, 2, figure=fig, 
                               height_ratios=[1.3, 0.8],
                               wspace=0.35,
                               hspace=0.35)
    
    # Panel A: 2024 Carbon storage map (top-left)
    ax_carbon = fig.add_subplot(gs_main[0, 0], projection=ccrs.PlateCarree())
    ax_carbon.set_extent(fig_params['map_extent'], crs=ccrs.PlateCarree())
    
    # Add country outline
    land = ShapelyFeature(country_bounds.geometry, ccrs.epsg(25830), 
                         facecolor=fig_params['non_forest_color'], 
                         edgecolor='0.3', linewidth=0.15)
    ax_carbon.add_feature(land, zorder=0)
    
    # Create color map for carbon (grey to green to purple)
    carbon_colors = [fig_params['non_forest_color'], '#1a6837', '#7b3294']
    carbon_cmap = LinearSegmentedColormap.from_list('custom_green_purple', carbon_colors, N=256)
    
    # Plot carbon data
    img_carbon = ax_carbon.imshow(
        carbon_data_2024_masked,
        extent=extent,
        transform=ccrs.epsg(25830),
        cmap=carbon_cmap,
        vmin=0,
        vmax=fig_params['carbon_vmax'],
        zorder=1,
    )
    
    # Add grid lines
    gl = ax_carbon.gridlines(draw_labels=True, alpha=0.2, linestyle='-')
    gl.top_labels = True
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    gl.xlocator = mticker.FixedLocator([-8, -6, -4, -2, 0, 2])
    gl.ylocator = mticker.FixedLocator([36, 37.5, 39, 40.5, 42, 43.5])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    ax_carbon.set_xlabel('')
    ax_carbon.set_ylabel('')
    
    for spine in ax_carbon.spines.values():
        spine.set_edgecolor('none')
    
    add_north_arrow_and_scalebar(ax_carbon)
    ax_carbon.set_title('a) Carbon storage (2024)', fontsize=10, y=1.12)
    
    # Add colorbar
    cbax = ax_carbon.inset_axes([0.82, 0.15, 0.04, 0.35])
    cbar = plt.colorbar(img_carbon, cax=cbax, orientation='vertical')
    cbar.set_label('Mg C·ha⁻¹', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    # Panel B: Carbon change map (top-right)
    ax_change = fig.add_subplot(gs_main[0, 1], projection=ccrs.PlateCarree())
    ax_change.set_extent(fig_params['map_extent'], crs=ccrs.PlateCarree())
    
    # Add country outline
    ax_change.add_feature(land, zorder=0)
    
    # Red-White-Blue colormap
    change_colors = ['#D73027', '#FFFFFF', '#2166AC']
    change_cmap = LinearSegmentedColormap.from_list('red_white_blue', change_colors, N=256)
    
    # Use percentile for better visualization
    p10 = np.nanpercentile(carbon_change_absolute, 10)
    vmin = p10
    vmax = -vmin  # Symmetric limits
    
    # Make sure limits are at least ±20 for consistency
    vmin = min(vmin, -20)
    vmax = max(vmax, 20)
    
    img_change = ax_change.imshow(
        carbon_change_absolute,
        extent=extent,
        transform=ccrs.epsg(25830),
        cmap=change_cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    
    gl = ax_change.gridlines(draw_labels=True, alpha=0.2, linestyle='-')
    gl.top_labels = True
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    gl.xlocator = mticker.FixedLocator([-8, -6, -4, -2, 0, 2])
    gl.ylocator = mticker.FixedLocator([36, 37.5, 39, 40.5, 42, 43.5])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    ax_change.set_xlabel('')
    ax_change.set_ylabel('')
    
    for spine in ax_change.spines.values():
        spine.set_edgecolor('none')
    
    add_north_arrow_and_scalebar(ax_change)
    ax_change.set_title('b) Changes in carbon storage (2017-2024)', fontsize=10, y=1.12)
    
    # Add colorbar
    cbax = ax_change.inset_axes([0.82, 0.15, 0.04, 0.35])
    cbar = plt.colorbar(img_change, cax=cbax, orientation='vertical', ticks=[-20, 0, 20])
    cbar.set_label('Mg C·ha⁻¹', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    # Panel C: Country-level carbon stocks trend (bottom-left)
    ax_trend = fig.add_subplot(gs_main[1, 0])
    
    # Plot uncertainty bands
    ax_trend.fill_between(
        trend_data['year_mid'], 
        trend_data['carbon_low'], 
        trend_data['carbon_high'],
        alpha=0.3, color='#555555'
    )
    
    # Plot mean line
    ax_trend.plot(
        trend_data['year_mid'],
        trend_data['carbon_mean'],
        color='#333333',
        linewidth=1.5,
        marker='o',
        markersize=4
    )
    
    # Calculate and annotate percentage loss
    loss_2021_2022 = np.round(100*(trend_data.set_index('year').carbon_mean.loc[2022]/trend_data.set_index('year').carbon_mean.loc[2021] - 1),1)
    ax_trend.annotate(f'{loss_2021_2022}%', xy=(2022, 1200), 
                     xytext=(2020.2, 1000), arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    # Configure trend axis
    ax_trend.set_ylabel('Carbon stock (Mt C)', fontsize=9)
    ax_trend.set_xlabel('Year', fontsize=9)
    ax_trend.grid(True, alpha=0.3)
    ax_trend.set_ylim(800, 1600)
    
    # Set x-axis
    ax_trend.set_xlim(min(trend_data['year']), max(trend_data['year'])+1)
    ax_trend.set_xticks(trend_data['year'])
    year_labels = [str(trend_data['year'].iloc[0])] + [str(year)[2:] for year in trend_data['year'][1:]]
    ax_trend.set_xticklabels(year_labels)
    ax_trend.tick_params(axis='both', labelsize=8)
    
    ax_trend.set_title('c) Country-level carbon stocks\n(2017-2024)', fontsize=10, y=1.05)
    sns.despine(ax=ax_trend)
    
    # Panel D: Interannual flux bars and cumulative flux line (bottom-right)
    ax_flux = fig.add_subplot(gs_main[1, 1], sharex=ax_trend)
    
    # Create bars for annual fluxes
    bar_width = 0.4
    bar_colors = []
    for flux in flux_data['annual_flux']:
        if flux < 0:
            bar_colors.append('#D73027')  # Red for losses
        else:
            bar_colors.append('#2166AC')  # Blue for gains
    
    # Plot flux bars at midyear points
    bars = ax_flux.bar(
        flux_data['year_mid'],
        flux_data['annual_flux'],
        width=bar_width,
        color=bar_colors,
        alpha=0.7
    )
    
    # Plot cumulative flux line
    cumulative_color = '#333333'
    ax_flux.plot(
        flux_data['year_mid'],
        flux_data['cumulative_flux'],
        color=cumulative_color,
        linewidth=1.5,
        linestyle='-',
        marker='o',
        markersize=4
    )
    
    # Add horizontal line at zero
    ax_flux.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax_flux.tick_params(axis='both', labelsize=8)
    
    # Configure axes
    ax_flux.set_xlabel('Year', fontsize=9)
    ax_flux.set_ylabel('Carbon stock change (Mt C)', fontsize=9)
    ax_flux.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='none', edgecolor='black', label='Interannual change'),
        Line2D([0], [0], color=cumulative_color, marker='o', linestyle='-', 
              markersize=4, label='Cumulative change')
    ]
    
    ax_flux.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=8)
    ax_flux.set_title('d) Interannual carbon changes\n(2017-2024)', fontsize=10, y=1.05)
    sns.despine(ax=ax_flux)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to process data and create Figure 1."""
    logger.info("Starting Figure 1 creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Get constants from config
    constants = config['constants']
    start_year = constants['start_year']
    end_year = constants['end_year']
    biomass_to_carbon = config['figure_params']['figure1']['biomass_to_carbon']
    
    # Get data file paths TODO: adapt this logic to get all biomass files
    biomass_paths = {}
    for year in [start_year, end_year]:
        biomass_paths[year] = data_paths.get_biomass_path(
            biomass_type='TBD',
            year=year,
            measure='mean',
            with_mask=True
        )
    
    # Country bounds path
    country_bounds_path = SPAIN_BOUNDARIES_FILE
    
    # Validate required files exist
    required_files = list(biomass_paths.values()) + [country_bounds_path]
    validate_required_files(required_files, 'figure_01')
    
    # Load trend data # TODO: check inner working of this function
    trend_data = load_trend_data(data_paths, biomass_to_carbon=biomass_to_carbon)
    
    # Load raster data
    carbon_data_2024, transform, crs, bounds = load_raster(
        biomass_paths[end_year], country_bounds_path, biomass_to_carbon
    )
    carbon_data_2017, _, _, _ = load_raster(
        biomass_paths[start_year], country_bounds_path, biomass_to_carbon
    )
    
    # Create figure
    fig = plot_biomass_carbon_figure(
        carbon_data_2024,
        carbon_data_2017,
        transform,
        crs,
        bounds,
        trend_data,
        country_bounds_path,
        config
    )
    
    # Save figure
    output_path = get_figure_output_path(config, f"Figure1_Carbon_Status_Change_{start_year}_{end_year}")
    save_figure_multiple_formats(fig, output_path, config, logger)
    
    plt.close()
    logger.info("Figure 3 created successfully")

if __name__ == "__main__":
    main()