#!/usr/bin/env python3
"""
Calculate interannual carbon flux from Monte Carlo biomass samples.

This script loads Monte Carlo samples of annual biomass stocks and calculates
interannual carbon flux by randomly sampling from each year's distribution.
Computes statistics (mean, median, 95% CI) for the flux distributions and
optionally creates diagnostic visualizations.

Author: Diego Bengochea
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse
import yaml
import logging
from glob import glob

# Suppress warnings
warnings.filterwarnings('ignore')

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


def find_mc_samples_file(config, mc_file_path=None):
    """
    Find Monte Carlo samples file automatically or use provided path.
    
    Args:
        config: Configuration dictionary
        mc_file_path: Optional specific path to MC samples file
        
    Returns:
        str: Path to MC samples file or None if not found
    """
    if mc_file_path and os.path.exists(mc_file_path):
        return mc_file_path
    
    # Look for MC samples in output directory
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    
    # Find the most recent MC samples file
    pattern = os.path.join(output_dir, "country_biomass_mc_samples_*.npz")
    mc_files = glob(pattern)
    
    if mc_files:
        # Return the most recent file (based on timestamp in filename)
        mc_files.sort()
        latest_file = mc_files[-1]
        logger.info(f"Found MC samples file: {latest_file}")
        return latest_file
    else:
        logger.error(f"No MC samples files found in {output_dir}")
        return None


def load_mc_samples(npz_file):
    """
    Load Monte Carlo samples from NPZ file.
    
    Args:
        npz_file: Path to NPZ file with MC samples
        
    Returns:
        Dict with years as keys and numpy arrays of samples as values
    """
    try:
        # Load NPZ file
        data = np.load(npz_file)
        
        # Convert to dictionary with proper year keys
        samples = {}
        for key in data.keys():
            # Parse key format: "biomass_type_year" (e.g., "TBD_2020")
            if '_' in key:
                parts = key.split('_')
                if len(parts) >= 2:
                    year = parts[-1]  # Last part should be year
                    biomass_type = '_'.join(parts[:-1])  # Everything before year
                    
                    # Only keep TBD (total biomass) samples for flux calculation
                    if biomass_type == 'TBD':
                        samples[year] = data[key]
            else:
                # Handle case where key is just year or different format
                samples[key] = data[key]
        
        logger.info(f"Loaded MC samples for {len(samples)} years")
        
        # Brief summary
        for year, vals in samples.items():
            logger.info(f"  Year {year}: {len(vals)} samples, mean: {np.mean(vals):.2f} Mt")
        
        return samples
    
    except Exception as e:
        logger.error(f"Error loading MC samples: {e}")
        return None


def calculate_interannual_flux(mc_samples, config):
    """
    Calculate interannual carbon flux by random sampling from MC distributions.
    
    Args:
        mc_samples: Dict with years as keys and numpy arrays of MC samples as values
        config: Configuration dictionary
        
    Returns:
        Tuple of (DataFrame with flux results, dict with flux samples)
    """
    # Get parameters from config
    n_combinations = config['interannual']['carbon_fluxes']['n_combinations']
    biomass_to_carbon = config['interannual']['carbon_fluxes']['biomass_to_carbon']
    percentile_low = config['monte_carlo']['confidence_interval']['low_percentile']
    percentile_high = config['monte_carlo']['confidence_interval']['high_percentile']
    random_seed = config['monte_carlo']['random_seed']
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get sorted list of years
    years = sorted([int(year) for year in mc_samples.keys()])
    
    # Initialize results list
    flux_results = []
    
    # Store flux samples for visualization
    all_flux_samples = {}
    
    # Process each consecutive year pair
    for i in range(len(years) - 1):
        year1 = str(years[i])
        year2 = str(years[i + 1])
        year_pair = f"{year1}-{year2}"
        
        logger.info(f"Processing year pair: {year_pair}")
        
        # Get MC samples for each year
        samples1 = mc_samples[year1]
        samples2 = mc_samples[year2]
        
        # Cap number of combinations based on available samples
        actual_n_combinations = min(n_combinations, len(samples1) * 10)
        
        # Randomly sample from both distributions
        idx1 = np.random.choice(len(samples1), actual_n_combinations)
        idx2 = np.random.choice(len(samples2), actual_n_combinations)
        
        # Calculate flux samples (positive = source, negative = sink)
        # Formula: (biomass_year1 - biomass_year2) * conversion_factor
        # This gives carbon flux from the ecosystem (positive = carbon source)
        flux_samples = (samples1[idx1] - samples2[idx2]) * biomass_to_carbon
        
        # Store flux samples
        all_flux_samples[year_pair] = flux_samples
        
        # Calculate statistics
        mean_flux = np.mean(flux_samples)
        median_flux = np.median(flux_samples)
        lower_ci = np.percentile(flux_samples, percentile_low)
        upper_ci = np.percentile(flux_samples, percentile_high)
        std_dev = np.std(flux_samples)
        
        # Store results
        flux_results.append({
            'year_pair': year_pair,
            'first_year': year1,
            'second_year': year2,
            'mean_flux': mean_flux,
            'median_flux': median_flux,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'std_dev': std_dev,
            'mid_year': (float(year1) + float(year2)) / 2
        })
        
        # Print summary
        logger.info(f"  Mean flux: {mean_flux:.2f} Mt C/year")
        logger.info(f"  95% CI: [{lower_ci:.2f}, {upper_ci:.2f}] Mt C/year")
    
    # Create DataFrame
    flux_df = pd.DataFrame(flux_results)
    
    return flux_df, all_flux_samples


def create_diagnostic_plots(flux_samples, config):
    """
    Create diagnostic plots for flux distributions.
    
    Args:
        flux_samples: Dict with year pairs as keys and numpy arrays of flux samples as values
        config: Configuration dictionary
        
    Returns:
        str: Directory where plots were saved
    """
    logger.info("Creating diagnostic plots...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create output directory
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribution plots for each year pair
    plt.figure(figsize=(12, 8))
    
    # Calculate number of rows and columns for subplots
    n_pairs = len(flux_samples)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    
    for i, (year_pair, samples) in enumerate(flux_samples.items()):
        plt.subplot(rows, cols, i + 1)
        
        # Histogram with KDE
        sns.histplot(samples, kde=True, color='blue', alpha=0.6)
        
        # Add vertical lines for mean and 95% CI
        plt.axvline(np.mean(samples), color='red', linestyle='-', label='Mean')
        plt.axvline(np.percentile(samples, 2.5), color='black', linestyle='--', label='2.5%')
        plt.axvline(np.percentile(samples, 97.5), color='black', linestyle='--', label='97.5%')
        
        # Add zero line
        plt.axvline(0, color='green', linestyle=':', label='Zero')
        
        plt.title(f"Flux Distribution: {year_pair}")
        plt.xlabel("Carbon Flux (Mt C/year)")
        plt.ylabel("Frequency")
        
        # Add legend to first subplot only
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "carbon_flux_distributions.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    
    # 2. Combined violin plot
    plt.figure(figsize=(12, 6))
    
    # Extract data for violin plot
    violin_data = []
    labels = []
    
    for year_pair, samples in flux_samples.items():
        violin_data.append(samples)
        labels.append(year_pair)
    
    # Create violin plot
    sns.violinplot(data=violin_data, color="#FFC983")
    sns.despine()
    
    # Add zero line
    plt.axhline(0, color='black', linestyle=':', label='Zero')
    
    # Set x-axis labels
    plt.xticks(range(len(labels)), labels, rotation=45)
    
    plt.title("Interannual Carbon Flux Distributions")
    plt.xlabel("Period")
    plt.ylabel("Carbon Flux (Mt C/year)")
    
    plt.tight_layout()
    violin_plot_path = os.path.join(output_dir, "carbon_flux_violins.png")
    plt.savefig(violin_plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Diagnostic plots saved to {output_dir}")
    return output_dir


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Calculate interannual carbon fluxes")
    parser.add_argument('--config', default='analysis_config.yaml', help='Path to configuration file')
    parser.add_argument('--mc-samples', default=None, help='Path to Monte Carlo samples NPZ file')
    parser.add_argument('--no-plots', action='store_true', help='Skip diagnostic plot creation')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    logger.info("Starting interannual carbon flux calculation...")
    
    # Find or use provided MC samples file
    mc_samples_file = find_mc_samples_file(config, args.mc_samples)
    if not mc_samples_file:
        logger.error("Could not find Monte Carlo samples file. Please run country_biomass_timeseries.py first.")
        return
    
    # Load MC samples
    logger.info(f"Loading Monte Carlo samples from: {mc_samples_file}")
    mc_samples = load_mc_samples(mc_samples_file)
    
    if mc_samples is None:
        logger.error("Error: Could not load Monte Carlo samples. Exiting.")
        return
      
    # Calculate interannual flux
    logger.info("Calculating interannual carbon flux...")
    flux_df, flux_samples = calculate_interannual_flux(mc_samples, config)
    
    # Save results
    base_dir = config['data']['base_dir']
    output_dir = os.path.join(base_dir, config['output']['base_output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"spain_carbon_flux_{timestamp}.csv")
    flux_df.to_csv(output_file, index=False)
    logger.info(f"Flux results saved to: {output_file}")
    
    # Create diagnostic plots if requested
    if not args.no_plots and config['interannual']['carbon_fluxes']['create_diagnostics']:
        create_diagnostic_plots(flux_samples, config)
    
    # Print summary table
    logger.info("\nInterannual Carbon Flux Summary (positive = source, negative = sink):")
    summary_df = flux_df[['year_pair', 'mean_flux', 'lower_ci', 'upper_ci']].copy()
    summary_df.columns = ['Year Pair', 'Mean Flux (MtC/year)', 'Lower CI', 'Upper CI']
    logger.info("\n" + summary_df.to_string(index=False))
    
    logger.info("\nCarbon flux analysis complete!")


if __name__ == "__main__":
    main()
