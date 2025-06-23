"""
Carbon Flux Analysis Module

This module implements interannual carbon flux calculations from Monte Carlo biomass samples.
All algorithmic logic preserved exactly from original script:
- calculate_interannual_carbon_fluxes.py

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
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config
from shared_utils.central_data_paths_constants import *

warnings.filterwarnings('ignore')


class CarbonFluxAnalyzer:
    """
    Interannual carbon flux analysis from Monte Carlo biomass samples.
    
    Implements carbon flux calculations with exact preservation of original
    algorithms for Monte Carlo sampling and statistical analysis.
    """
    
    def __init__(self, config: Optional[Union[str, Path, Dict]] = None):
        """
        Initialize the carbon flux analyzer.
        
        Args:
            config: Configuration dictionary or path to config file
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config, component_name="biomass_analysis")
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = load_config(component_name="biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='carbon_flux_analysis'
        )
        
        self.logger.info("Initialized CarbonFluxAnalyzer")

    def find_mc_samples_file(self, mc_file_path: Optional[str] = None) -> Optional[str]:
        """
        Find Monte Carlo samples file automatically or use provided path.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            mc_file_path: Optional specific path to MC samples file
            
        Returns:
            str: Path to MC samples file or None if not found
        """
        if mc_file_path and os.path.exists(mc_file_path):
            return mc_file_path
        
        # Look for MC samples in output directory
        output_dir = ANALYSIS_OUTPUTS_DIR
        
        # Find the most recent MC samples file
        pattern = os.path.join(output_dir, "country_biomass_mc_samples_*.npz")
        mc_files = glob(pattern)
        
        if mc_files:
            # Return the most recent file (based on timestamp in filename)
            mc_files.sort()
            latest_file = mc_files[-1]
            self.logger.info(f"Found MC samples file: {latest_file}")
            return latest_file
        else:
            self.logger.error(f"No MC samples files found in {output_dir}")
            return None

    def load_mc_samples(self, npz_file: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load Monte Carlo samples from NPZ file.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            npz_file: Path to NPZ file containing MC samples
            
        Returns:
            dict: Dictionary with year keys and sample arrays, or None if error
        """
        try:
            self.logger.info(f"Loading Monte Carlo samples from: {npz_file}")
            
            # Load NPZ file
            data = np.load(npz_file)
            
            # Convert to dictionary
            mc_samples = {}
            for key in data.files:
                mc_samples[key] = data[key]
            
            self.logger.info(f"Loaded MC samples for {len(mc_samples)} entries")
            
            # Log sample info
            for key, samples in mc_samples.items():
                self.logger.info(f"  {key}: {len(samples)} samples, "
                               f"mean: {np.mean(samples):.2f} Mt, "
                               f"std: {np.std(samples):.2f} Mt")
            
            return mc_samples
            
        except Exception as e:
            self.logger.error(f"Error loading MC samples: {e}")
            return None

    def calculate_interannual_flux(self, mc_samples: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Calculate interannual carbon flux from Monte Carlo samples.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            mc_samples: Dictionary with biomass type_year keys and sample arrays
            
        Returns:
            Tuple of (flux_statistics_df, flux_samples_dict)
        """
        # Extract configuration parameters
        biomass_to_carbon = self.config['interannual']['carbon_fluxes']['biomass_to_carbon']
        n_combinations = self.config['interannual']['carbon_fluxes']['n_combinations']
        
        self.logger.info(f"Calculating interannual carbon flux with {n_combinations} random combinations")
        self.logger.info(f"Using biomass-to-carbon conversion factor: {biomass_to_carbon}")
        
        # Extract years and biomass types from keys
        entries = {}
        for key, samples in mc_samples.items():
            # Parse key format: "biomass_type_year"
            parts = key.split('_')
            if len(parts) >= 2:
                year = int(parts[-1])
                biomass_type = '_'.join(parts[:-1])
                
                if biomass_type not in entries:
                    entries[biomass_type] = {}
                entries[biomass_type][year] = samples
        
        # Focus on TBD (Total Biomass Density) for flux calculations
        if 'TBD' not in entries:
            self.logger.error("No TBD (Total Biomass Density) samples found for flux calculation")
            return pd.DataFrame(), {}
        
        tbd_samples = entries['TBD']
        years = sorted(tbd_samples.keys())
        
        if len(years) < 2:
            self.logger.error("Need at least 2 years of TBD samples for flux calculation")
            return pd.DataFrame(), {}
        
        self.logger.info(f"Processing flux for years: {years}")
        
        # Calculate flux for consecutive year pairs
        flux_results = []
        flux_samples_dict = {}
        
        for i in range(len(years) - 1):
            year1, year2 = years[i], years[i + 1]
            
            self.logger.info(f"Calculating flux: {year1} -> {year2}")
            
            samples1 = tbd_samples[year1]
            samples2 = tbd_samples[year2]
            
            # Ensure both years have the same number of samples
            min_samples = min(len(samples1), len(samples2))
            if min_samples < n_combinations:
                self.logger.warning(f"Only {min_samples} samples available, using all of them")
                n_combinations_actual = min_samples
            else:
                n_combinations_actual = n_combinations
            
            # Generate random combinations
            flux_samples = np.zeros(n_combinations_actual)
            
            np.random.seed(42)  # For reproducibility
            
            for j in range(n_combinations_actual):
                # Randomly sample from each year
                idx1 = np.random.randint(0, len(samples1))
                idx2 = np.random.randint(0, len(samples2))
                
                biomass1 = samples1[idx1]
                biomass2 = samples2[idx2]
                
                # Calculate flux: (biomass2 - biomass1) * biomass_to_carbon
                # Positive = source (biomass increase), Negative = sink (biomass decrease)
                flux = (biomass2 - biomass1) * biomass_to_carbon
                flux_samples[j] = flux
            
            # Calculate statistics
            year_pair = f"{year1}-{year2}"
            
            flux_stats = {
                'year_pair': year_pair,
                'year1': year1,
                'year2': year2,
                'mean_flux': np.mean(flux_samples),
                'median_flux': np.median(flux_samples),
                'std_flux': np.std(flux_samples),
                'min_flux': np.min(flux_samples),
                'max_flux': np.max(flux_samples),
                'lower_ci': np.percentile(flux_samples, 2.5),
                'upper_ci': np.percentile(flux_samples, 97.5),
                'q25': np.percentile(flux_samples, 25),
                'q75': np.percentile(flux_samples, 75),
                'n_samples': len(flux_samples)
            }
            
            flux_results.append(flux_stats)
            flux_samples_dict[year_pair] = flux_samples
            
            self.logger.info(f"  Flux {year_pair}: {flux_stats['mean_flux']:.2f} ± {flux_stats['std_flux']:.2f} MtC/year")
            self.logger.info(f"    95% CI: [{flux_stats['lower_ci']:.2f}, {flux_stats['upper_ci']:.2f}] MtC/year")
        
        # Convert to DataFrame
        flux_df = pd.DataFrame(flux_results)
        
        return flux_df, flux_samples_dict

    def create_diagnostic_plots(self, flux_samples: Dict[str, np.ndarray]) -> None:
        """
        Create diagnostic plots for carbon flux distributions.
        
        CRITICAL: This algorithm must be preserved exactly as in original.
        
        Args:
            flux_samples: Dictionary with year pair keys and flux sample arrays
        """
        if not self.config['interannual']['carbon_fluxes']['create_diagnostics']:
            self.logger.info("Diagnostic plot creation disabled in config")
            return
        
        if not flux_samples:
            self.logger.warning("No flux samples provided for diagnostic plots")
            return
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            n_pairs = len(flux_samples)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Carbon Flux Analysis Diagnostics', fontsize=16, fontweight='bold')
            
            # Flatten axes for easy iteration
            axes = axes.flatten()
            
            # Plot 1: Distribution of all flux values
            ax1 = axes[0]
            all_fluxes = np.concatenate(list(flux_samples.values()))
            ax1.hist(all_fluxes, bins=50, alpha=0.7, edgecolor='black')
            ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Carbon neutral')
            ax1.set_xlabel('Carbon Flux (MtC/year)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of All Carbon Flux Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plot by year pair
            ax2 = axes[1]
            data_for_box = []
            labels_for_box = []
            
            for year_pair, samples in flux_samples.items():
                data_for_box.append(samples)
                labels_for_box.append(year_pair)
            
            bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            
            # Color boxes
            colors = sns.color_palette("husl", len(bp['boxes']))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Year Pair')
            ax2.set_ylabel('Carbon Flux (MtC/year)')
            ax2.set_title('Carbon Flux Distribution by Year Pair')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Plot 3: Time series of mean flux with error bars
            ax3 = axes[2]
            year_pairs = list(flux_samples.keys())
            means = [np.mean(samples) for samples in flux_samples.values()]
            stds = [np.std(samples) for samples in flux_samples.values()]
            
            x_pos = range(len(year_pairs))
            ax3.errorbar(x_pos, means, yerr=stds, marker='o', linestyle='-', 
                        linewidth=2, markersize=8, capsize=5)
            ax3.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(year_pairs, rotation=45)
            ax3.set_xlabel('Year Pair')
            ax3.set_ylabel('Mean Carbon Flux (MtC/year)')
            ax3.set_title('Mean Carbon Flux Over Time (±1σ)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Cumulative flux
            ax4 = axes[3]
            cumulative_flux = np.cumsum(means)
            ax4.plot(x_pos, cumulative_flux, marker='o', linestyle='-', 
                    linewidth=2, markersize=8)
            ax4.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(year_pairs, rotation=45)
            ax4.set_xlabel('Year Pair')
            ax4.set_ylabel('Cumulative Carbon Flux (MtC)')
            ax4.set_title('Cumulative Carbon Flux')
            ax4.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            output_dir = ANALYSIS_OUTPUTS_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(output_dir, f"carbon_flux_diagnostics_{timestamp}.png")
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Diagnostic plots saved to: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating diagnostic plots: {e}")

    # ==================== MAIN ANALYSIS METHODS ====================
    
    def run_carbon_flux_analysis(self, mc_file_path: Optional[str] = None, create_diagnostics: Optional[bool] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, np.ndarray]]]:
        """
        Run complete carbon flux analysis from Monte Carlo samples.
        
        Args:
            mc_file_path: Optional specific path to MC samples file
            create_diagnostics: Optional override for diagnostic plot creation
            
        Returns:
            Tuple of (flux_statistics_df, flux_samples_dict)
        """
        self.logger.info("Starting interannual carbon flux calculation...")
        
        # Find MC samples file
        mc_samples_file = self.find_mc_samples_file(mc_file_path)
        if not mc_samples_file:
            self.logger.error("Could not find Monte Carlo samples file")
            return None, None
        
        # Load MC samples
        self.logger.info(f"Loading Monte Carlo samples from: {mc_samples_file}")
        mc_samples = self.load_mc_samples(mc_samples_file)
        
        if mc_samples is None:
            self.logger.error("Error: Could not load Monte Carlo samples")
            return None, None
        
        # Calculate interannual flux
        self.logger.info("Calculating interannual carbon flux...")
        flux_df, flux_samples = self.calculate_interannual_flux(mc_samples)
        
        if flux_df.empty:
            self.logger.error("No flux results calculated")
            return None, None
        
        # Create diagnostic plots if requested
        create_plots = create_diagnostics
        if create_plots is None:
            create_plots = self.config['interannual']['carbon_fluxes']['create_diagnostics']
        
        if create_plots:
            self.create_diagnostic_plots(flux_samples)
        
        return flux_df, flux_samples

    # ==================== RESULTS SAVING ====================
    
    def save_results(self, flux_df: pd.DataFrame, flux_samples: Optional[Dict[str, np.ndarray]] = None) -> str:
        """
        Save carbon flux analysis results to files.
        
        Args:
            flux_df: DataFrame with flux statistics
            flux_samples: Optional dictionary with flux samples for each year pair
            
        Returns:
            Path to main output file
        """
        output_dir = ANALYSIS_OUTPUTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save flux statistics
        output_file = os.path.join(output_dir, f"spain_carbon_flux_{timestamp}.csv")
        flux_df.to_csv(output_file, index=False)
        self.logger.info(f"Flux results saved to: {output_file}")
        
        # Save raw flux samples if provided and requested
        if flux_samples and self.config['output']['save_intermediate_results']:
            samples_file = os.path.join(output_dir, f"carbon_flux_samples_{timestamp}.npz")
            np.savez_compressed(samples_file, **flux_samples)
            self.logger.info(f"Flux samples saved to: {samples_file}")
        
        return output_file

    def print_summary(self, flux_df: pd.DataFrame) -> None:
        """
        Print summary table of carbon flux results.
        
        Args:
            flux_df: DataFrame with flux statistics
        """
        if flux_df.empty:
            self.logger.warning("No flux results to summarize")
            return
        
        self.logger.info("\nInterannual Carbon Flux Summary (positive = source, negative = sink):")
        
        # Create summary table
        summary_df = flux_df[['year_pair', 'mean_flux', 'lower_ci', 'upper_ci']].copy()
        summary_df.columns = ['Year Pair', 'Mean Flux (MtC/year)', 'Lower CI', 'Upper CI']
        
        # Format numbers to 2 decimal places
        for col in ['Mean Flux (MtC/year)', 'Lower CI', 'Upper CI']:
            summary_df[col] = summary_df[col].round(2)
        
        self.logger.info("\n" + summary_df.to_string(index=False))
        
        # Additional summary statistics
        mean_annual_flux = flux_df['mean_flux'].mean()
        total_period_flux = flux_df['mean_flux'].sum()
        
        self.logger.info(f"\nOverall Summary:")
        self.logger.info(f"  Mean annual flux: {mean_annual_flux:.2f} MtC/year")
        self.logger.info(f"  Total period flux: {total_period_flux:.2f} MtC")
        self.logger.info(f"  Number of periods: {len(flux_df)}")
