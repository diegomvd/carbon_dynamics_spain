#!/usr/bin/env python3
"""
Calculate biomass transition distributions between consecutive years.

This script analyzes pixel-level biomass transitions between consecutive years,
calculating summary statistics and optionally saving individual transition data.
Handles millions of transitions efficiently using optimized data structures.

Author: Diego Bengochea
"""

import numpy as np
import pandas as pd
import rasterio
import os
import re
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='analysis_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def find_biomass_files(data_dir):
    """
    Find and organize biomass files by year.
    
    Args:
        data_dir: Directory containing biomass files
        
    Returns:
        dict: Dictionary with years as keys and file paths as values
    """
    data_dir = Path(data_dir)
    biomass_data = {}
    
    # Get all TIF files matching the pattern
    pattern = "TBD_S2_mean_*_100m_TBD_merged.tif"
    tif_files = list(data_dir.glob(pattern))
    
    if not tif_files:
        logger.error(f"No TIF files found matching pattern {pattern} in {data_dir}")
        return {}
    
    for tif_file in tif_files:
        # Extract year from filename
        year_match = re.search(r'TBD_S2_mean_(\d{4})_100m_TBD_merged\.tif', tif_file.name)
        if year_match:
            year = int(year_match.group(1))
            biomass_data[year] = str(tif_file)
            logger.info(f"Found biomass file for year {year}: {tif_file.name}")
    
    return biomass_data


def load_raster_data(file_path, max_biomass=None):
    """
    Load raster data with quality control.
    
    Args:
        file_path: Path to raster file
        max_biomass: Maximum biomass threshold for filtering
        
    Returns:
        numpy array: Raster data with NaN for invalid values
    """
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        
        # Handle nodata values
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        
        # Apply maximum biomass threshold if specified
        if max_biomass is not None:
            data = np.where(data > max_biomass, np.nan, data)
        
        return data


def calculate_transition_statistics(year1_data, year2_data, year1, year2, config):
    """
    Calculate comprehensive transition statistics for a year pair.
    
    Args:
        year1_data: Biomass data for first year
        year2_data: Biomass data for second year
        year1: First year
        year2: Second year
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary with transition statistics and optionally raw data
    """
    pixel_area_ha = config['interannual']['transitions']['pixel_area_ha']
    biomass_to_carbon = config['interannual']['carbon_fluxes']['biomass_to_carbon']
    
    # Create valid mask (both years have valid data)
    valid_mask = ~np.isnan(year1_data) & ~np.isnan(year2_data)
    n_valid_pixels = np.sum(valid_mask)
    
    if n_valid_pixels == 0:
        logger.warning(f"No valid pixels for transition {year1}-{year2}")
        return None
    
    logger.info(f"Processing {n_valid_pixels:,} valid pixels for {year1}-{year2}")
    
    # Extract valid data
    x = year1_data[valid_mask].flatten()  # Initial biomass
    y = year2_data[valid_mask].flatten()  # Final biomass
    
    # Calculate changes
    biomass_changes = y - x  # Change in biomass (MgBM/ha)
    carbon_changes = biomass_changes * biomass_to_carbon * pixel_area_ha / 1e6  # Change in carbon (MtC)
    
    # Calculate various transition statistics
    stats = {
        'year_pair': f"{year1}-{year2}",
        'first_year': year1,
        'second_year': year2,
        'n_valid_pixels': n_valid_pixels,
        'total_area_ha': n_valid_pixels * pixel_area_ha,
        
        # Basic statistics
        'mean_initial_biomass': np.mean(x),
        'mean_final_biomass': np.mean(y),
        'mean_biomass_change': np.mean(biomass_changes),
        'total_carbon_change_Mt': np.sum(carbon_changes),
        
        # Change distribution
        'biomass_change_std': np.std(biomass_changes),
        'biomass_change_min': np.min(biomass_changes),
        'biomass_change_max': np.max(biomass_changes),
        'biomass_change_p25': np.percentile(biomass_changes, 25),
        'biomass_change_p50': np.percentile(biomass_changes, 50),
        'biomass_change_p75': np.percentile(biomass_changes, 75),
        'biomass_change_p95': np.percentile(biomass_changes, 95),
        'biomass_change_p05': np.percentile(biomass_changes, 5),
        
        # Gains and losses
        'n_pixels_gain': np.sum(biomass_changes > 0),
        'n_pixels_loss': np.sum(biomass_changes < 0),
        'n_pixels_stable': np.sum(np.abs(biomass_changes) < 1),  # <1 MgBM/ha change
        'total_gains_Mt': np.sum(carbon_changes[carbon_changes > 0]),
        'total_losses_Mt': np.sum(carbon_changes[carbon_changes < 0]),
        
        # Specific transition types (using biomass thresholds)
        'transitions_to_zero': np.sum((x > 10) & (y < 10)),  # Forest loss
        'transitions_from_zero': np.sum((x < 10) & (y > 10)),  # Forest gain
        'high_loss_transitions': np.sum(biomass_changes < -50),  # Major biomass loss
        'high_gain_transitions': np.sum(biomass_changes > 50),  # Major biomass gain
    }
    
    # Calculate transition percentages
    if stats['total_losses_Mt'] < 0:  # Only if there are losses
        # Transitions from high to low biomass as % of total losses
        mask_to_zero = (x > 10) & (y < 10) & (carbon_changes < 0)
        losses_to_zero = np.sum(carbon_changes[mask_to_zero])
        stats['pct_losses_to_zero'] = (losses_to_zero / stats['total_losses_Mt']) * 100
        
        # High biomass losses as % of total losses
        mask_high_loss = (biomass_changes < -50) & (carbon_changes < 0)
        high_losses = np.sum(carbon_changes[mask_high_loss])
        stats['pct_high_losses'] = (high_losses / stats['total_losses_Mt']) * 100
    else:
        stats['pct_losses_to_zero'] = 0
        stats['pct_high_losses'] = 0
    
    # Add raw transition data if requested
    if config['interannual']['transitions']['save_transition_data']:
        stats['raw_data'] = {
            'initial_biomass': x,
            'final_biomass': y,
            'biomass_changes': biomass_changes,
            'carbon_changes': carbon_changes
        }
        logger.info(f"  Including raw transition data: {len(x):,} transitions")
    
    return stats


def save_transition_results(all_results, config):
    """
    Save transition analysis results to files.
    
    Args:
        all_results: List of transition result dictionaries
        config: Configuration dictionary
        
    Returns:
        tuple: (summary_file, raw_data_file) - paths to saved files
    """
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary statistics
    summary_data = []
    raw_data_dict = {}
    
    for result in all_results:
        if result is None:
            continue
        
        # Extract summary statistics (exclude raw data)
        summary_stats = {k: v for k, v in result.items() if k != 'raw_data'}
        summary_data.append(summary_stats)
        
        # Collect raw data if present
        if 'raw_data' in result and config['interannual']['transitions']['save_transition_data']:
            year_pair = result['year_pair']
            raw_data_dict[year_pair] = result['raw_data']
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"transition_statistics_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary statistics saved to: {summary_file}")
    
    # Save raw data if requested
    raw_data_file = None
    if raw_data_dict and config['interannual']['transitions']['save_transition_data']:
        output_format = config['interannual']['transitions']['output_format']
        
        if output_format == 'npz':
            # Save as compressed NPZ (efficient for large arrays)
            raw_data_file = os.path.join(output_dir, f"transition_raw_data_{timestamp}.npz")
            
            # Flatten the nested structure for NPZ
            npz_data = {}
            for year_pair, data in raw_data_dict.items():
                for key, values in data.items():
                    npz_data[f"{year_pair}_{key}"] = values
            
            np.savez_compressed(raw_data_file, **npz_data)
            logger.info(f"Raw transition data saved to: {raw_data_file}")
            
        elif output_format == 'csv':
            # Save as CSV (warning: can be very large)
            logger.warning("Saving millions of transitions as CSV - this will be a large file!")
            raw_data_file = os.path.join(output_dir, f"transition_raw_data_{timestamp}.csv")
            
            combined_data = []
            for year_pair, data in raw_data_dict.items():
                for i in range(len(data['initial_biomass'])):
                    combined_data.append({
                        'year_pair': year_pair,
                        'initial_biomass': data['initial_biomass'][i],
                        'final_biomass': data['final_biomass'][i],
                        'biomass_change': data['biomass_changes'][i],
                        'carbon_change': data['carbon_changes'][i]
                    })
            
            combined_df = pd.DataFrame(combined_data)
            combined_df.to_csv(raw_data_file, index=False)
            logger.info(f"Raw transition data saved to: {raw_data_file}")
    
    return summary_file, raw_data_file


def create_simple_diagnostic_plot(all_results, config):
    """
    Create a simple diagnostic plot if requested.
    
    Args:
        all_results: List of transition result dictionaries
        config: Configuration dictionary
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract data for plotting
        year_pairs = []
        total_changes = []
        
        for result in all_results:
            if result is None:
                continue
            year_pairs.append(result['year_pair'])
            total_changes.append(result['total_carbon_change_Mt'])
        
        if len(year_pairs) < 2:
            logger.info("Not enough data for diagnostic plot")
            return
        
        # Simple bar plot of total carbon changes
        plt.figure(figsize=(10, 6))
        colors = ['red' if x > 0 else 'blue' for x in total_changes]
        bars = plt.bar(year_pairs, total_changes, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Year Pair')
        plt.ylabel('Total Carbon Change (Mt C)')
        plt.title('Net Carbon Change by Year Pair')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, total_changes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        # Save plot
        base_dir = config['data']['base_dir']
        output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
        plot_file = os.path.join(output_dir, 'transition_carbon_changes.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plot saved to: {plot_file}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping diagnostic plot")
    except Exception as e:
        logger.warning(f"Error creating diagnostic plot: {e}")


def main():
    """
    Main function to calculate transition distributions for all year pairs.
    """
    parser = argparse.ArgumentParser(description="Calculate biomass transition distributions")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--years', nargs='+', type=int, default=None, help='Specific years to process')
    parser.add_argument('--save-raw-data', action='store_true', help='Force saving of raw transition data')
    parser.add_argument('--no-plot', action='store_true', help='Skip diagnostic plot creation')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Override save raw data if specified
    if args.save_raw_data:
        config['interannual']['transitions']['save_transition_data'] = True
        logger.info("Forcing save of raw transition data")
    
    # Build input directory path
    base_dir = config['data']['base_dir']
    input_dir = os.path.join(base_dir, config['interannual']['differences']['input_biomass_dir'])
    
    logger.info(f"Input directory: {input_dir}")
    logger.info("Starting biomass transition distribution analysis...")
    
    # Find biomass files
    biomass_files = find_biomass_files(input_dir)
    
    if len(biomass_files) < 2:
        logger.error("Need at least 2 years of biomass data. Exiting.")
        return
    
    # Filter years if specified
    if args.years:
        filtered_files = {year: path for year, path in biomass_files.items() if year in args.years}
        if len(filtered_files) < 2:
            logger.error(f"Need at least 2 years from specified years {args.years}")
            return
        biomass_files = filtered_files
        logger.info(f"Processing specific years: {sorted(biomass_files.keys())}")
    
    detected_years = sorted(biomass_files.keys())
    logger.info(f"Processing years: {detected_years}")
    logger.info(f"Will calculate transitions for {len(detected_years)-1} consecutive year pairs")
    
    # Get parameters from config
    max_biomass = config['interannual']['transitions']['max_biomass_threshold']
    
    # Process each consecutive year pair
    all_results = []
    
    for i in range(len(detected_years) - 1):
        year1, year2 = detected_years[i], detected_years[i + 1]
        
        logger.info(f"Processing transition: {year1} -> {year2}")
        
        try:
            # Load raster data
            year1_data = load_raster_data(biomass_files[year1], max_biomass)
            year2_data = load_raster_data(biomass_files[year2], max_biomass)
            
            # Calculate transition statistics
            result = calculate_transition_statistics(year1_data, year2_data, year1, year2, config)
            all_results.append(result)
            
            # Clean up memory
            del year1_data, year2_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {year1}-{year2}: {e}")
            all_results.append(None)
    
    # Save results
    if any(r is not None for r in all_results):
        summary_file, raw_data_file = save_transition_results(all_results, config)
        
        # Create diagnostic plot if requested
        if not args.no_plot:
            create_simple_diagnostic_plot(all_results, config)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRANSITION ANALYSIS SUMMARY")
        logger.info("="*60)
        
        valid_results = [r for r in all_results if r is not None]
        logger.info(f"Successfully processed: {len(valid_results)} year pairs")
        
        for result in valid_results:
            logger.info(f"{result['year_pair']}: {result['total_carbon_change_Mt']:.2f} Mt C change "
                       f"({result['n_valid_pixels']:,} pixels)")
        
        logger.info(f"Summary statistics saved to: {summary_file}")
        if raw_data_file:
            logger.info(f"Raw transition data saved to: {raw_data_file}")
        
    else:
        logger.error("No valid transitions calculated")


if __name__ == "__main__":
    main()
