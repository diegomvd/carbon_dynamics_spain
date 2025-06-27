#!/usr/bin/env python3
"""
Recipe: Complete Analysis

Reproduces all analysis results from biomass maps:
1. Biomass pattern analysis (trends, aggregations, transitions)
2. Climate-biomass relationship analysis (ML modeling, climate data processing)
3. Model interpretability analysis (SHAP)

This recipe produces all analysis outputs and tabular results for the paper.

Usage:
    python 3_analysis_recipe.py [OPTIONS]

Examples:
    # Run complete analysis
    python 3_analysis_recipe.py
    
    # Specific analysis components
    python 3_analysis_recipe.py --analysis-only biomass
    python 3_analysis_recipe.py --analysis-only climate
    
    # Skip certain components
    python 3_analysis_recipe.py --skip-shap
    
    # Specific years
    python 3_analysis_recipe.py --years 2020 2021 2022

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

# Import utilities
from shared_utils.central_data_paths import CentralDataPaths
from shared_utils.config_utils import load_config
from shared_utils.logging_utils import setup_logging

# Import component scripts
from biomass_analysis.scripts.run_full_analysis import main as run_biomass_analysis_main
from climate_biomass_analysis.scripts.run_full_pipeline import main as run_climate_analysis_main


class AnalysisRecipe:
    """
    Recipe for complete analysis reproduction.
    
    Orchestrates biomass analysis and climate-biomass relationship analysis
    with centralized data management.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize analysis recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='analysis_recipe'
        )
        
        # Track stage results
        self.stage_results = {}
        
        self.logger.info("Initialized Analysis Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for analysis...")
        
        # Check for biomass maps - UPDATED PATH
        full_country_dir = BIOMASS_MAPS_FULL_COUNTRY_DIR
        if not full_country_dir.exists():
            self.logger.error(f"Biomass maps (full country) directory not found: {full_country_dir}")
            self.logger.error("Please run '2_biomass_estimation_recipe.py' first")
            return False
        
        # Check for merged biomass files
        merged_files = list(full_country_dir.rglob("*_merged.tif"))
        if not merged_files:
            self.logger.error("No merged biomass files found")
            return False
        
        self.logger.info(f"✅ Found {len(merged_files)} merged biomass files")
        
        # Check for raw climate data (for climate-biomass analysis)
        climate_raw_dir = CLIMATE_RAW_DIR
        if not climate_raw_dir.exists():
            self.logger.warning(f"Raw climate directory not found: {climate_raw_dir}")
            self.logger.warning("Climate-biomass analysis may not work properly")
        else:
            climate_files = list(climate_raw_dir.glob("*.grib")) + list(climate_raw_dir.glob("*.nc"))
            if climate_files:
                self.logger.info(f"✅ Found {len(climate_files)} raw climate files")
            else:
                self.logger.warning("Raw climate directory exists but no GRIB/NC files found")
        
        # Check for forest type data (for aggregation analysis)
        forest_types_file = FOREST_TYPES_TIERS_FILE
        if not forest_types_file.exists():
            self.logger.warning(f"Forest types file not found: {forest_types_file}")
            self.logger.warning("Forest type analysis may not work properly")
        else:
            self.logger.info("✅ Forest types data found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        # TODO: this should be updated with the new results directory structure.

        # Create analysis output directories
        self.data_paths.create_directories([
            'analysis_outputs',
            'tables',
            'ml_outputs'
        ])
        
        # Create analysis-specific subdirectories
        analysis_base = self.data_paths.get_path('analysis_outputs')
        analysis_subdirs = [
            'biomass_trends',
            'aggregation_analysis',
            'interannual_differences',
            'transition_analysis',
            'carbon_fluxes',
            'climate_analysis',
            'optimization_results',
            'shap_analysis'
        ]
        
        for subdir in analysis_subdirs:
            (analysis_base / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create processed bioclim directories - NEW STRUCTURE
        self.data_paths.create_directories(['bioclim'])
        bioclim_base = self.data_paths.get_path('bioclim')
        bioclim_subdirs = ['harmonized', 'variables', 'anomalies']
        
        for subdir in bioclim_subdirs:
            (bioclim_base / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Output directory structure created")
        self.logger.info(f"   📁 analysis_outputs/ - analysis results")
        self.logger.info(f"   📁 bioclim/ - processed climate data")
        self.logger.info(f"   📁 tables/ - tabular results")
        self.logger.info(f"   📁 ml_outputs/ - ML models and predictions")
    
    def run_biomass_analysis(self) -> bool:
        """
        Run biomass pattern analysis.
    
        Returns:
            bool: True if successful
        """
        stage_name = "Biomass Analysis"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed TODO: only checking for a single output on trends! THis should eig¡ther check for everythin ot for nothing
            analysis_dir = BIOMASS_COUNTRY_TIMESERIES_DIR
            existing_results = list(analysis_dir.glob("*.csv"))
            
            if existing_results:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_results)} result files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
 
    
            # Run the analysis
            result = run_biomass_analysis_main()
            
            stage_time = time.time() - stage_start
            success = result == 0
            
            self.stage_results[stage_name] = {
                'success': success,
                'duration_minutes': stage_time / 60,
                'result': result
            }

            if success:
                self.logger.info(f"{stage_name} completed successfully in {stage_time/60:.2f} minutes")
            else:
                self.logger.error(f"{stage_name} failed after {stage_time/60:.2f} minutes")
            
            return success
        
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"{stage_name} failed with error: {str(e)}")
            return False
    
    def run_climate_analysis(self) -> bool:
        """
        Run climate-biomass relationship analysis.
            
        Returns:
            bool: True if successful
        """
        stage_name = "Climate-Biomass Analysis"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            climate_analysis_dir = CLIMATE_BIOMASS_RESULTS_DIR
            ml_outputs_dir = CLIMATE_BIOMASS_MODELS_DIR
            
            existing_climate = list(climate_analysis_dir.glob("*.csv"))
            existing_ml = list(ml_outputs_dir.glob("*.pkl")) + list(ml_outputs_dir.glob("*.csv"))
            
            if existing_climate and existing_ml:
                self.logger.info(f"{stage_name} - Found existing outputs: {len(existing_climate + existing_ml)} files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
                
            # Run the analysis
            result = run_climate_analysis_main()
            
            stage_time = time.time() - stage_start
            success = result == 0
            
            self.stage_results[stage_name] = {
                'success': success,
                'duration_minutes': stage_time / 60,
                'result': result
            }
            
            if success:
                self.logger.info(f"{stage_name} completed successfully in {stage_time/60:.2f} minutes")
            else:
                self.logger.error(f"{stage_name} failed after {stage_time/60:.2f} minutes")
            
            return success
        
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"{stage_name} failed with error: {str(e)}")
            return False
    

def main():
    """Main entry point for analysis recipe."""
    start_time = time.time()
    
    # Initialize recipe
    recipe = AnalysisRecipe()
        
    # Validate prerequisites
    if not recipe.validate_prerequisites():
        recipe.logger.error("Prerequisites validation failed")
        recipe.logger.error("Please ensure biomass maps and climate data are available")
        sys.exit(1)
        
    # Create output structure
    recipe.create_output_structure()
        
    # Track overall success
    overall_success = True

    success = recipe.run_biomass_analysis()
    overall_success = overall_success and success

    success = recipe.run_climate_analysis()
    overall_success = overall_success and success
        
    # Validate outputs
    if overall_success:
        elapsed_time = time.time() - start_time
        recipe.logger.info(f"Analysis recipe completed successfully in {elapsed_time/60:.2f} minutes!")
    else:
        recipe.logger.error("Analysis recipe failed or output validation failed")
        sys.exit(1)
    
if __name__ == "__main__":
    main()