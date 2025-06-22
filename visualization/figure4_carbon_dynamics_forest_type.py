#!/usr/bin/env python3
"""
Generate Figure 2: Differential Vulnerability by Vegetation Characteristics

This script creates a comprehensive figure with four panels:
- Panel A: Forest type change bars with genus points (top-left)
- Panel B: Total carbon storage by forest type (top-right)
- Panel C: Area charts for height categories showing temporal trends (bottom-left)
- Panel D: Relative temporal trends of key genera (bottom-right)

Author: Diego Bengochea
Component: Visualization Pipeline
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.utils import (
    setup_logging, load_visualization_config, apply_style_config,
    setup_data_paths, save_figure_multiple_formats,
    get_figure_output_path, validate_required_files
)

# Set up logging
logger = setup_logging('figure_02')

def load_analysis_data(data_paths) -> tuple:
    """
    Load data from analysis CSV files and prepare for plotting.
    
    Args:
        data_paths: CentralDataPaths instance
        
    Returns:
        Tuple of DataFrames for forest_type_data, height_data, genus_data, clade_data
    """
    logger.info("Loading analysis data from CSV files...")
    
    # Define expected file paths
    analysis_outputs = data_paths.get_path('analysis_outputs')
    data_files = {
        'genus': analysis_outputs / "biomass_by_genus_year.csv",
        'clade': analysis_outputs / "biomass_by_clade_year.csv", 
        'height': analysis_outputs / "biomass_by_height_year.csv"
    }
    
    # Validate files exist
    validate_required_files(list(data_files.values()), 'figure_02')
    
    # Load dataframes
    dataframes = {}
    for key, file_path in data_files.items():
        logger.info(f"Loading {key} data from {file_path}")
        dataframes[key] = pd.read_csv(file_path)
    
    return dataframes['genus'], dataframes['clade'], dataframes['height']

def prepare_forest_type_data(genus_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare forest type data with genus breakdown for analysis.
    
    Args:
        genus_df: Genus-level biomass DataFrame
        config: Configuration dictionary
        
    Returns:
        Processed DataFrame with forest type and genus data
    """
    constants = config['constants']
    biomass_to_carbon = config['figure_params']['figure2']['biomass_to_carbon']
    start_year = constants['start_year']
    end_year = constants['end_year']
    
    # Convert biomass to carbon
    genus_df['Carbon_Tonnes'] = genus_df['Total_Biomass_Tonnes'] * biomass_to_carbon
    genus_df['Carbon_Mt'] = genus_df['Carbon_Tonnes'] / 1e6
    
    # Rename clade mappings
    clade_mappings = {
        'No arbolado': 'Woodlands\nand\nagroforestry',
        'Repoblaciones con especie desconocida': 'Mixed'
    }
    genus_df['Clade'] = genus_df['Clade'].replace(clade_mappings)
    
    # Get start and end year data
    start_year_genera = genus_df[genus_df['Year'] == start_year]
    end_year_genera = genus_df[genus_df['Year'] == end_year]
    
    # Merge to get both years' data for change calculation
    merged_genera = pd.merge(
        start_year_genera,
        end_year_genera,
        on=['Genus', 'Clade'],
        suffixes=('_start', '_end')
    )
    
    # Calculate changes
    merged_genera['carbon_change'] = merged_genera['Carbon_Mt_end'] - merged_genera['Carbon_Mt_start']
    merged_genera['carbon_pct_change'] = (merged_genera['carbon_change'] / merged_genera['Carbon_Mt_start']) * 100
    
    # Rename for clarity
    forest_type_data = merged_genera.rename(columns={
        'Clade': 'forest_type',
        'Carbon_Mt_start': 'carbon_2017',
        'Carbon_Mt_end': 'carbon_2024'
    })
    
    # Map forest types
    forest_type_data['forest_type'] = forest_type_data['forest_type'].replace({
        'Angiosperm': 'Broadleaved',
        'Gymnosperm': 'Conifer'
    })
    
    return forest_type_data

def prepare_height_data(height_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare height category data for area plotting.
    
    Args:
        height_df: Height-level biomass DataFrame
        
    Returns:
        Processed DataFrame with percentage calculations
    """
    # Convert to carbon
    biomass_to_carbon = 0.5
    height_df['carbon'] = height_df['biomass'] * biomass_to_carbon
    
    # Calculate percentages by year
    height_totals = height_df.groupby('year')['carbon'].sum().reset_index()
    height_data = pd.merge(height_df, height_totals, on='year', suffixes=('', '_total'))
    height_data['percentage'] = (height_data['carbon'] / height_data['carbon_total']) * 100
    
    return height_data

def prepare_genus_relative_data(genus_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare genus data for relative trend analysis.
    
    Args:
        genus_df: Genus-level biomass DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with relative values (2017=100%)
    """
    constants = config['constants']
    focus_genera = ['Eucaliptales', 'Quercus', 'Pinus', 'Fagus', 'Other']
    start_year = constants['start_year']
    
    # Filter to focus genera
    genus_data = genus_df[genus_df['Genus'].isin(focus_genera)]
    
    # Convert to carbon (already done if called after prepare_forest_type_data)
    if 'Carbon_Mt' not in genus_data.columns:
        biomass_to_carbon = config['figure_params']['figure2']['biomass_to_carbon']
        genus_data['Carbon_Mt'] = genus_data['Total_Biomass_Tonnes'] * biomass_to_carbon / 1e6
    
    # Pivot data
    pivot_genus = genus_data.pivot_table(index='Year', columns='Genus', values='Carbon_Mt').reset_index()
    
    # Get 2017 baseline values
    baseline_values = pivot_genus[pivot_genus['Year'] == start_year].iloc[0].drop('Year')
    
    # Create relative DataFrame
    rel_genus_data = []
    for _, row in pivot_genus.iterrows():
        year = row['Year']
        for genus in focus_genera:
            if genus in row and genus in baseline_values:
                absolute_value = row[genus]
                baseline = baseline_values[genus]
                if not pd.isna(absolute_value) and not pd.isna(baseline) and baseline > 0:
                    relative_value = (absolute_value / baseline) * 100
                    rel_genus_data.append({
                        'year': year,
                        'genus': genus,
                        'carbon': absolute_value,
                        'relative_value': relative_value
                    })
    
    return pd.DataFrame(rel_genus_data)

def create_forest_type_change_plot(ax, forest_data: pd.DataFrame, config: dict):
    """Create forest type change bar plot with genus points overlay."""
    forest_type_colors = config['figure_params']['figure2']['forest_type_colors']
    genus_colors = config['figure_params']['figure2']['genus_colors']
    
    # Forest types in reversed order for top-to-bottom reading
    forest_types_reversed = ['Woodlands\nand\nagroforestry', 'Mixed', 'Conifer', 'Broadleaved']
    focus_genera = ['Eucaliptales', 'Quercus', 'Pinus', 'Fagus', 'Other']
    
    # Aggregate data by forest type
    forest_agg = forest_data.groupby('forest_type').agg({
        'carbon_2017': 'sum',
        'carbon_2024': 'sum'
    }).reset_index()
    
    # Calculate changes
    forest_agg['carbon_change'] = forest_agg['carbon_2024'] - forest_agg['carbon_2017']
    forest_agg['carbon_pct_change'] = (forest_agg['carbon_change'] / forest_agg['carbon_2017']) * 100
    
    # Reorder to match forest_types_reversed
    forest_agg = forest_agg.set_index('forest_type').loc[forest_types_reversed].reset_index()
    
    # Create horizontal bars
    bars = ax.barh(
        forest_agg['forest_type'],
        forest_agg['carbon_pct_change'],
        color=[forest_type_colors[ft] for ft in forest_agg['forest_type']],
        alpha=0.7,
        height=0.6
    )
    
    # Add genus points overlay
    for forest_type in forest_types_reversed:
        type_genera = forest_data[forest_data['forest_type'] == forest_type]
        
        # Plot all genera as small black dots
        ax.scatter(
            type_genera['carbon_pct_change'],
            [forest_type] * len(type_genera),
            s=25,
            c='black',
            edgecolor='none',
            alpha=0.7,
            zorder=5
        )
        
        # Overlay focus genera with larger, colored points
        focus_genera_data = type_genera[type_genera['Genus'].isin(focus_genera)]
        if not focus_genera_data.empty:
            ax.scatter(
                focus_genera_data['carbon_pct_change'],
                [forest_type] * len(focus_genera_data),
                s=80,
                c=[genus_colors.get(genus, genus_colors['Other']) for genus in focus_genera_data['Genus']],
                edgecolor='black',
                linewidth=0.5,
                zorder=10
            )
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Add absolute change values
    for i, row in forest_agg.iterrows():
        forest_type = row['forest_type']
        change_value = row['carbon_change']
        
        # Special positioning for broadleaved to avoid overlap
        if forest_type == 'Broadleaved':
            x_pos = 5
            y_pos = 2.7
        elif forest_type == 'Conifer':
            x_pos = 5
            y_pos = i
        else:
            x_pos = 5
            y_pos = i
        
        ax.text(
            x_pos, y_pos,
            f"({change_value:.1f} Mt C)",
            va='center',
            ha='left',
            fontsize=8,
            color='black'
        )
    
    # Format plot
    ax.set_xlabel('Change 2017-2024 (%)', fontsize=9)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_xlim(-35, 25)
    ax.set_title('a) Carbon change by forest type (2017-2024)', fontsize=10, y=1.05)
    sns.despine(ax=ax)

def create_forest_type_storage_plot(ax, forest_data: pd.DataFrame, clade_df: pd.DataFrame, config: dict):
    """Create forest type carbon storage stacked bar plot."""
    constants = config['constants']
    forest_type_colors = config['figure_params']['figure2']['forest_type_colors']
    genus_colors = config['figure_params']['figure2']['genus_colors']
    biomass_to_carbon = config['figure_params']['figure2']['biomass_to_carbon']
    
    start_year = constants['start_year']
    end_year = constants['end_year']
    years = [start_year, end_year]
    
    # Process clade data
    clade_df['Carbon_Mt'] = clade_df['Total_Biomass_Tonnes'] * biomass_to_carbon / 1e6
    clade_df['Clade'] = clade_df['Clade'].replace({'No arbolado': 'Woodlands\nand\nagroforestry'})
    
    forest_types = ['Broadleaved', 'Conifer', 'Mixed', 'Forests', 'Woodlands\nand\nagroforestry']
    storage_data = []
    
    # Process all forest types
    for year in years:
        year_col = f'carbon_{year}'
        
        # Calculate forest totals
        broadleaved_total = forest_data[forest_data['forest_type'] == 'Broadleaved'][year_col].sum()
        conifer_total = forest_data[forest_data['forest_type'] == 'Conifer'][year_col].sum()
        year_clade_data = clade_df[clade_df['Year'] == year]
        mixed_total = year_clade_data[year_clade_data['Clade'] == 'Mixed']['Carbon_Mt'].sum()
        
        # Total forests
        total_forest = broadleaved_total + conifer_total + mixed_total
        storage_data.append({
            'forest_type': 'Forests',
            'genus': 'Total',
            'year': year,
            'carbon_value': total_forest
        })
        
        # Broadleaved breakdown
        broadleaved_data = forest_data[forest_data['forest_type'] == 'Broadleaved']
        eucalyptus_val = broadleaved_data[broadleaved_data['Genus'] == 'Eucaliptales'][year_col].sum()
        quercus_val = broadleaved_data[broadleaved_data['Genus'] == 'Quercus'][year_col].sum()
        fagus_val = broadleaved_data[broadleaved_data['Genus'] == 'Fagus'][year_col].sum()
        others_val = broadleaved_total - eucalyptus_val - quercus_val - fagus_val
        
        for genus, value in [('Eucalyptus', eucalyptus_val), ('Quercus', quercus_val), 
                           ('Fagus', fagus_val), ('Others', others_val)]:
            storage_data.append({
                'forest_type': 'Broadleaved',
                'genus': genus,
                'year': year,
                'carbon_value': value
            })
        
        # Conifer breakdown
        conifer_data = forest_data[forest_data['forest_type'] == 'Conifer']
        pinus_val = conifer_data[conifer_data['Genus'] == 'Pinus'][year_col].sum()
        others_val = conifer_total - pinus_val
        
        for genus, value in [('Pinus', pinus_val), ('Others', others_val)]:
            storage_data.append({
                'forest_type': 'Conifer',
                'genus': genus,
                'year': year,
                'carbon_value': value
            })
        
        # Mixed and Woodlands
        storage_data.append({
            'forest_type': 'Mixed',
            'genus': 'Total',
            'year': year,
            'carbon_value': mixed_total
        })
        
        woodlands_val = year_clade_data[year_clade_data['Clade'] == 'Woodlands\nand\nagroforestry']['Carbon_Mt'].sum()
        storage_data.append({
            'forest_type': 'Woodlands\nand\nagroforestry',
            'genus': 'Total',
            'year': year,
            'carbon_value': woodlands_val
        })
    
    # Convert to DataFrame and create plot
    storage_df = pd.DataFrame(storage_data)
    
    bar_width = 0.35
    bar_positions = {
        start_year: np.arange(len(forest_types)),
        end_year: np.arange(len(forest_types)) + bar_width
    }
    
    # Plot stacked bars for Broadleaved and Conifer
    stacked_types = ['Broadleaved', 'Conifer']
    for ft in stacked_types:
        ft_data = storage_df[storage_df['forest_type'] == ft]
        
        for year in years:
            year_data = ft_data[ft_data['year'] == year]
            bottom = 0
            position = bar_positions[year][forest_types.index(ft)]
            
            for _, row in year_data.iterrows():
                genus = row['genus']
                value = row['carbon_value']
                
                if genus == 'Eucalyptus':
                    color = genus_colors['Eucaliptales']
                elif genus in genus_colors:
                    color = genus_colors[genus]
                else:
                    color = '#999999'
                
                ax.bar(position, value, bar_width, bottom=bottom,
                      color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                bottom += value
            
            # Add year labels for Broadleaved
            if ft == 'Broadleaved':
                ax.text(position, bottom + 10, str(year), ha='center', va='bottom',
                       fontsize=8, rotation=90)
    
    # Plot non-stacked bars
    non_stacked_types = ['Forests', 'Mixed', 'Woodlands\nand\nagroforestry']
    for ft in non_stacked_types:
        ft_data = storage_df[storage_df['forest_type'] == ft]
        
        for year in years:
            year_data = ft_data[ft_data['year'] == year]
            position = bar_positions[year][forest_types.index(ft)]
            
            if not year_data.empty:
                value = year_data['carbon_value'].values[0]
                ax.bar(position, value, bar_width, color=forest_type_colors[ft],
                      alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Format plot
    ax.set_xticks(bar_positions[start_year] + bar_width/2)
    ax.set_xticklabels(forest_types)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylabel('Carbon stock (Mt C)', fontsize=9)
    ax.set_title('b) Carbon storage by forest type', fontsize=10, y=1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax)

def create_height_category_area_plot(ax, height_data: pd.DataFrame, config: dict):
    """Create area chart showing temporal carbon trends by height category."""
    constants = config['constants']
    years = constants['years']
    start_year = constants['start_year']
    end_year = constants['end_year']
    
    # Height category mapping
    height_display_names = {
        '0_2m': '0-2m',
        '2_6m': '2-6m',
        '6_10m': '6-10m',
        '10_20m': '10-20m',
        '20m_plus': '>20m'
    }
    
    # Get height column
    height_column = 'height_bin' if 'height_bin' in height_data.columns else 'height_category'
    height_categories = sorted(height_data[height_column].unique())
    
    # Create mapping
    height_display_map = {h: height_display_names.get(h, h) for h in height_categories}
    
    # Pivot data
    pivot_data = height_data.pivot_table(
        index='year',
        columns=height_column,
        values='percentage'
    ).reset_index()
    
    # Sort heights by numerical value
    def height_key(h):
        if h.startswith('20m'):
            return float('inf')
        return float(h.split('_')[0].replace('m', ''))
    
    plot_order = sorted(height_categories, key=height_key)
    
    # Create color palette
    height_colors = list(reversed(sns.color_palette("mako", n_colors=len(plot_order))))
    
    # Create stackplot
    ax.stackplot(
        pivot_data['year'],
        [pivot_data[height] for height in plot_order],
        labels=[height_display_map[h] for h in plot_order],
        colors=height_colors,
        alpha=0.8,
        edgecolor='white',
        linewidth=1.0
    )
    
    # Format plot
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Percentage of total carbon (%)', fontsize=9)
    ax.set_title('c) Carbon distribution by vegetation\nheight (2017-2024)', fontsize=10, y=1.05)
    
    # Format x-axis
    ax.set_xlim(start_year, end_year)
    ax.set_xticks(years)
    year_labels = [str(years[0])] + [str(year)[2:] for year in years[1:]]
    ax.set_xticklabels(year_labels)
    ax.tick_params(axis='both', labelsize=8)
    
    # Format y-axis
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Add direct labels
    if end_year in pivot_data['year'].values:
        last_year_idx = pivot_data[pivot_data['year'] == end_year].index[0]
        y_bottom = 0
        
        for height in plot_order:
            if height in pivot_data.columns:
                height_val = pivot_data.iloc[last_year_idx][height]
                y_pos = y_bottom + height_val/2
                label = height_display_map[height]
                
                if height == '20m_plus':
                    y_pos += 1.6
                
                ax.text(end_year - 0.3, y_pos, label, va='center', ha='right',
                       fontsize=8, color='black')
                y_bottom += height_val
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax)

def create_genus_relative_trends_plot(ax, genus_data: pd.DataFrame, config: dict):
    """Create line plot showing relative temporal carbon trends for key genera."""
    constants = config['constants']
    genus_colors = config['figure_params']['figure2']['genus_colors']
    focus_genera = ['Eucaliptales', 'Quercus', 'Pinus', 'Fagus', 'Other']
    years = constants['years']
    start_year = constants['start_year']
    end_year = constants['end_year']
    
    # Plot trends for each genus
    for genus in focus_genera:
        genus_subset = genus_data[genus_data['genus'] == genus].copy()
        
        if not genus_subset.empty:
            genus_subset = genus_subset.sort_values('year')
            
            # Plot line
            ax.plot(
                genus_subset['year'],
                genus_subset['relative_value'],
                color=genus_colors[genus],
                linewidth=1.5,
                alpha=0.7
            )
            
            # Plot points
            ax.scatter(
                genus_subset['year'],
                genus_subset['relative_value'],
                s=40,
                color=genus_colors[genus],
                edgecolor='black',
                linewidth=0.5,
                zorder=10,
                label=f"{genus_subset.iloc[0]['carbon']:.1f} Mt C in 2017"
            )
    
    # Add baseline at 100%
    ax.axhline(100, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Format plot
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Relative carbon stock (2017 = 100%)', fontsize=9)
    ax.set_title('d) Relative carbon trends by key genera\n(2017-2024)', fontsize=10, y=1.05)
    
    # Format axes
    ax.set_xlim(start_year, end_year+0.2)
    ax.set_xticks(years)
    year_labels = [str(years[0])] + [str(year)[2:] for year in years[1:]]
    ax.set_xticklabels(year_labels)
    ax.tick_params(axis='both', labelsize=8)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    
    # Add end value annotations
    for genus in focus_genera:
        end_data = genus_data[(genus_data['genus'] == genus) & (genus_data['year'] == end_year)]
        
        if not end_data.empty:
            end_value = end_data.iloc[0]['relative_value']
            ax.annotate(
                f"{end_value:.1f}%",
                xy=(end_year, end_value),
                xytext=(end_year + 0.1, end_value),
                fontsize=8,
                ha='left',
                va='center',
                color=genus_colors[genus]
            )
    
    sns.despine(ax=ax)

def add_shared_legend(fig, axes: list, config: dict):
    """Add shared legend for panels A and B."""
    genus_colors = config['figure_params']['figure2']['genus_colors']
    focus_genera = ['Eucaliptales', 'Quercus', 'Pinus', 'Fagus', 'Other']
    genus_display_names = {
        'Eucaliptales': 'Eucalyptus',
        'Quercus': 'Quercus',
        'Pinus': 'Pinus',
        'Fagus': 'Fagus',
        'Other': 'Other'
    }
    
    # Create legend elements
    legend_elements = []
    for genus in focus_genera:
        display_name = genus_display_names[genus]
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=genus_colors[genus],
                  markeredgecolor='black', markersize=8, label=display_name)
        )
    
    # Position legend
    ax1_pos = axes[0].get_position()
    ax2_pos = axes[1].get_position()
    legend_x = (ax1_pos.x0 + ax2_pos.x1) / 2
    legend_y = min(ax1_pos.y0, ax2_pos.y0) - 0.09
    
    fig.legend(
        handles=legend_elements,
        loc='center',
        bbox_to_anchor=(legend_x, legend_y),
        ncol=len(focus_genera),
        fontsize=8
    )

def create_figure2(forest_data: pd.DataFrame, height_data: pd.DataFrame, 
                  genus_data: pd.DataFrame, clade_df: pd.DataFrame, config: dict) -> plt.Figure:
    """Create Figure 2 with four panels showing differential vulnerability."""
    logger.info("Creating Figure 2: Differential Vulnerability by Vegetation Characteristics")
    
    # Create figure
    fig_params = config['figure_params']['figure2']
    fig = plt.figure(figsize=fig_params['figsize'])
    
    # 2x2 grid with adjusted spacing
    gs = gridspec.GridSpec(2, 2, figure=fig, 
                          width_ratios=[0.9, 1.1],
                          wspace=0.35, hspace=0.6)
    
    # Create panels
    ax1 = fig.add_subplot(gs[0, 0])
    create_forest_type_change_plot(ax1, forest_data, config)
    
    ax2 = fig.add_subplot(gs[0, 1])
    create_forest_type_storage_plot(ax2, forest_data, clade_df, config)
    
    # Add shared legend
    add_shared_legend(fig, [ax1, ax2], config)
    
    ax3 = fig.add_subplot(gs[1, 0])
    create_height_category_area_plot(ax3, height_data, config)
    
    ax4 = fig.add_subplot(gs[1, 1])
    create_genus_relative_trends_plot(ax4, genus_data, config)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    
    return fig

def main():
    """Main function to process data and create Figure 2."""
    logger.info("Starting Figure 2 creation")
    
    # Load configuration
    config = load_visualization_config()
    apply_style_config(config)
    
    # Set up data paths
    data_paths = setup_data_paths()
    
    # Load analysis data
    genus_df, clade_df, height_df = load_analysis_data(data_paths)
    
    # Prepare data for each panel
    forest_data = prepare_forest_type_data(genus_df, config)
    height_data = prepare_height_data(height_df)
    genus_data = prepare_genus_relative_data(genus_df, config)
    
    # Create figure
    fig = create_figure2(forest_data, height_data, genus_data, clade_df, config)
    
    # Save figure
    output_path = get_figure_output_path(config, "Figure2_Differential_Vulnerability")
    save_figure_multiple_formats(fig, output_path, config, logger)
    
    plt.close()
    logger.info("Figure 2 created successfully")

if __name__ == "__main__":
    main()