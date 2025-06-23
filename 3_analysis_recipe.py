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
    
    def __init__(self, data_root: str = "data", log_level: str = "INFO"):
        """
        Initialize analysis recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        # Setup centralized data paths
        self.data_paths = CentralDataPaths(data_root)
        
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
        full_country_dir = self.data_paths.get_biomass_maps_full_country_dir()
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
        climate_raw_dir = self.data_paths.get_path('climate_raw')
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
        forest_types_file = self.data_paths.get_path('forest_inventory') / "Forest_Types_Tiers.csv"
        if not forest_types_file.exists():
            self.logger.warning(f"Forest types file not found: {forest_types_file}")
            self.logger.warning("Forest type analysis may not work properly")
        else:
            self.logger.info("✅ Forest types data found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
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
    
    def run_biomass_analysis(self, years: Optional[List[int]] = None) -> bool:
        """
        Run biomass pattern analysis.
        
        Args:
            years: Specific years to process
            
        Returns:
            bool: True if successful
        """
        stage_name = "Biomass Analysis"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            analysis_dir = self.data_paths.get_path('analysis_outputs') / 'biomass_trends'
            existing_results = list(analysis_dir.glob("*.csv"))
            
            if existing_results:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_results)} result files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run biomass analysis with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for biomass analysis with new paths
                sys.argv = [
                    'run_full_analysis.py',
                    '--biomass-maps-dir', str(self.data_paths.get_biomass_maps_full_country_dir()),
                    '--forest-types-file', str(self.data_paths.get_path('forest_inventory') / "Forest_Types_Tiers.csv"),
                    '--output-dir', str(self.data_paths.get_path('analysis_outputs')),
                    '--tables-dir', str(self.data_paths.get_path('tables')),
                    '--log-level', 'INFO'
                ]
                
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                # Run the analysis
                result = run_biomass_analysis_main()
                
                stage_time = time.time() - stage_start
                success = result if result is not None else True
                
                self.stage_results[stage_name] = {
                    'success': success,
                    'duration_minutes': stage_time / 60,
                    'result': result
                }
                
                if success:
                    self.logger.info(f"✅ {stage_name} completed successfully in {stage_time/60:.2f} minutes")
                else:
                    self.logger.error(f"❌ {stage_name} failed after {stage_time/60:.2f} minutes")
                
                return success
                
            finally:
                sys.argv = old_argv
        
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"❌ {stage_name} failed with error: {str(e)}")
            return False
    
    def run_climate_analysis(self, years: Optional[List[int]] = None) -> bool:
        """
        Run climate-biomass relationship analysis.
        
        Args:
            years: Specific years to process
            
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
            climate_analysis_dir = self.data_paths.get_path('analysis_outputs') / 'climate_analysis'
            ml_outputs_dir = self.data_paths.get_path('ml_outputs')
            
            existing_climate = list(climate_analysis_dir.glob("*.csv"))
            existing_ml = list(ml_outputs_dir.glob("*.pkl")) + list(ml_outputs_dir.glob("*.csv"))
            
            if existing_climate and existing_ml:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_climate + existing_ml)} files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run climate analysis with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for climate analysis with new paths
                sys.argv = [
                    'run_full_pipeline.py',
                    '--raw-climate-dir', str(self.data_paths.get_path('climate_raw')),
                    '--bioclim-dir', str(self.data_paths.get_path('bioclim')),
                    '--biomass-maps-dir', str(self.data_paths.get_biomass_maps_full_country_dir()),
                    '--output-dir', str(self.data_paths.get_path('analysis_outputs') / 'climate_analysis'),
                    '--ml-output-dir', str(self.data_paths.get_path('ml_outputs')),
                    '--log-level', 'INFO'
                ]
                
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                # Run the analysis
                result = run_climate_analysis_main()
                
                stage_time = time.time() - stage_start
                success = result if result is not None else True
                
                self.stage_results[stage_name] = {
                    'success': success,
                    'duration_minutes': stage_time / 60,
                    'result': result
                }
                
                if success:
                    self.logger.info(f"✅ {stage_name} completed successfully in {stage_time/60:.2f} minutes")
                else:
                    self.logger.error(f"❌ {stage_name} failed after {stage_time/60:.2f} minutes")
                
                return success
                
            finally:
                sys.argv = old_argv
        
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"❌ {stage_name} failed with error: {str(e)}")
            return False
    
    def validate_outputs(self) -> bool:
        """
        Validate that outputs were created successfully.
        
        Returns:
            bool: True if outputs are valid
        """
        self.logger.info("Validating analysis outputs...")
        
        # Check analysis outputs
        analysis_base = self.data_paths.get_path('analysis_outputs')
        total_analysis_files = 0
        
        for subdir in ['biomass_trends', 'climate_analysis']:
            subdir_path = analysis_base / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.csv"))
                total_analysis_files += len(files)
                self.logger.info(f"✅ {subdir}: {len(files)} files")
            else:
                self.logger.warning(f"⚠️  {subdir} directory not found")
        
        if total_analysis_files == 0:
            self.logger.error("No analysis output files found")
            return False
        
        # Check tables
        tables_dir = self.data_paths.get_path('tables')
        if tables_dir.exists():
            table_files = list(tables_dir.glob("*.csv"))
            self.logger.info(f"✅ Tables: {len(table_files)} files")
        else:
            self.logger.warning("⚠️  Tables directory not found")
        
        # Check ML outputs
        ml_dir = self.data_paths.get_path('ml_outputs')
        if ml_dir.exists():
            ml_files = list(ml_dir.glob("*.pkl")) + list(ml_dir.glob("*.csv"))
            self.logger.info(f"✅ ML outputs: {len(ml_files)} files")
        else:
            self.logger.warning("⚠️  ML outputs directory not found")
        
        return True
    
    def print_summary(self) -> None:
        """Print summary of analysis results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ANALYSIS RECIPE SUMMARY")
        self.logger.info(f"{'='*60}")
        
        # Show results for each stage
        for stage_name, results in self.stage_results.items():
            status = "✅" if results['success'] else "❌"
            duration = results['duration_minutes']
            self.logger.info(f"  {status} {stage_name}: {duration:.2f} min")
        
        # Count outputs by directory
        output_dirs = [
            ('analysis_outputs', 'Analysis Results'),
            ('tables', 'Tables'),
            ('ml_outputs', 'ML Outputs')
        ]
        
        for dir_key, display_name in output_dirs:
            dir_path = self.data_paths.get_path(dir_key)
            if dir_path.exists():
                all_files = list(dir_path.rglob("*"))
                files_only = [f for f in all_files if f.is_file()]
                self.logger.info(f"📁 {display_name}: {len(files_only)} files in {dir_path}")
        
        # Show NEW data structure
        self.logger.info(f"📂 Analysis outputs in: {self.data_paths.data_root}")
        self.logger.info(f"   ├── processed/")
        self.logger.info(f"   │   └── bioclim/                     # NEW: processed climate data")
        self.logger.info(f"   │       ├── harmonized/              # Harmonized climate data")
        self.logger.info(f"   │       ├── variables/               # Bioclimatic variables")
        self.logger.info(f"   │       └── anomalies/               # Climate anomalies")
        self.logger.info(f"   └── results/")
        self.logger.info(f"       ├── analysis_outputs/")
        self.logger.info(f"       │   ├── biomass_trends/")
        self.logger.info(f"       │   ├── climate_analysis/")
        self.logger.info(f"       │   └── shap_analysis/")
        self.logger.info(f"       ├── tables/")
        self.logger.info(f"       └── ml_outputs/")
        
        self.logger.info(f"\n🎯 Analysis complete!")
        self.logger.info(f"   📊 Check results/ directory for all outputs")
        self.logger.info(f"   📋 Check tables/ for summary statistics")
        self.logger.info(f"   🤖 Check ml_outputs/ for models and predictions")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Complete Analysis (Biomass + Climate + SHAP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe reproduces all analysis results by executing:
1. Biomass pattern analysis (trends, aggregations, transitions)
2. Climate-biomass relationship analysis (ML modeling, climate data processing)
3. Model interpretability analysis (SHAP)

NEW DATA STRUCTURE:
  data/processed/bioclim/             # Processed climate data (NEW)
    ├── harmonized/                   # Harmonized climate data
    ├── variables/                    # Bioclimatic variables (bio1-bio19)
    └── anomalies/                    # Climate anomalies
  data/results/analysis_outputs/      # Analysis results
  data/results/tables/                # Tabular outputs
  data/results/ml_outputs/            # ML models and predictions

INPUTS FROM PREVIOUS RECIPES:
  data/processed/biomass_maps/full_country/  # Merged biomass maps (Recipe 2)
  data/raw/climate/                          # Raw climate GRIB files
  data/raw/forest_inventory/Forest_Types_Tiers.csv  # Forest type hierarchy

Examples:
  %(prog)s                              # Run complete analysis
  %(prog)s --analysis-only biomass      # Just biomass analysis
  %(prog)s --analysis-only climate      # Just climate analysis  
  %(prog)s --skip-shap                  # Skip SHAP analysis
  %(prog)s --years 2020 2021 2022       # Specific years
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements (FROM PREVIOUS RECIPES):
  - Merged biomass maps from biomass estimation recipe
  - Raw climate data (GRIB files) for climate-biomass analysis
  - Forest type hierarchy for aggregation analysis
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Root directory for data storage (default: data)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to process (default: all available)'
    )
    
    parser.add_argument(
        '--analysis-only',
        choices=['biomass', 'climate'],
        help='Run only specific analysis component'
    )
    
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Skip SHAP interpretability analysis'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution if a stage fails'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate prerequisites without running analysis'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for analysis recipe."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    log_level = 'ERROR' if args.quiet else args.log_level
    
    try:
        # Initialize recipe
        recipe = AnalysisRecipe(
            data_root=args.data_root,
            log_level=log_level
        )
        
        # Validate prerequisites
        if not recipe.validate_prerequisites():
            recipe.logger.error("❌ Prerequisites validation failed")
            recipe.logger.error("Please ensure biomass maps and climate data are available")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Track overall success
        overall_success = True
        
        # Run biomass analysis
        if not args.analysis_only or args.analysis_only == 'biomass':
            success = recipe.run_biomass_analysis(years=args.years)
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Biomass analysis failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Run climate analysis
        if not args.analysis_only or args.analysis_only == 'climate':
            success = recipe.run_climate_analysis(years=args.years)
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Climate analysis failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Validate outputs
        if overall_success and recipe.validate_outputs():
            recipe.print_summary()
            elapsed_time = time.time() - start_time
            recipe.logger.info(f"🎉 Analysis recipe completed successfully in {elapsed_time/60:.2f} minutes!")
        else:
            recipe.logger.error("❌ Analysis recipe failed or output validation failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Recipe interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Recipe failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()