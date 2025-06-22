#!/usr/bin/env python3
"""
Recipe: Analysis

Reproduces all analysis results from biomass maps including:
1. Biomass pattern analysis (trends, aggregations, transitions)
2. Climate-biomass relationship analysis (ML modeling, optimization)
3. SHAP analysis for model interpretability

This recipe generates all figures, tables, and analysis outputs from the paper.

Usage:
    python reproduce_analysis.py [OPTIONS]

Examples:
    # Run complete analysis
    python reproduce_analysis.py
    
    # Specific years only
    python reproduce_analysis.py --years 2020 2021 2022
    
    # Skip SHAP analysis (faster)
    python reproduce_analysis.py --skip-shap
    
    # Run only biomass analysis
    python reproduce_analysis.py --analysis-only biomass

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
from shared_utils.data_paths import CentralDataPaths
from shared_utils.config_utils import load_config
from shared_utils.logging_utils import setup_logging

# Import component orchestrators
from biomass_analysis.scripts.run_full_analysis import AnalysisPipelineOrchestrator as BiomassAnalysisOrchestrator
from climate_biomass_analysis.scripts.run_full_pipeline import main as run_climate_pipeline_main
from climate_biomass_analysis.scripts.run_shap_analysis import main as run_shap_analysis_main


class AnalysisRecipe:
    """
    Recipe for complete analysis reproduction.
    
    Orchestrates biomass analysis, climate-biomass analysis, and SHAP analysis
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
        
        # Check for biomass maps
        biomass_dir = (self.data_paths.get_path('biomass_maps') / 
                      self.data_paths.subdirs['biomass_maps']['with_mask'])
        if not biomass_dir.exists():
            self.logger.error(f"Biomass maps directory not found: {biomass_dir}")
            self.logger.error("Please run 'reproduce_biomass_estimation.py' first to generate biomass maps")
            return False
        
        # Check for merged biomass files
        merged_dir = (self.data_paths.get_path('biomass_maps') / 
                     self.data_paths.subdirs['biomass_maps']['merged'])
        if not merged_dir.exists():
            self.logger.error(f"Merged biomass directory not found: {merged_dir}")
            return False
        
        merged_files = list(merged_dir.glob("*_merged.tif"))
        if not merged_files:
            self.logger.error("No merged biomass files found")
            return False
        
        self.logger.info(f"✅ Found {len(merged_files)} merged biomass files")
        
        # Check for climate data (for climate-biomass analysis)
        climate_dir = self.data_paths.get_path('climate_variables')
        if not climate_dir.exists():
            self.logger.warning(f"Climate variables directory not found: {climate_dir}")
            self.logger.warning("Climate-biomass analysis may not work properly")
        else:
            self.logger.info("✅ Climate variables directory found")
        
        # Check for reference data
        spain_boundary = (self.data_paths.get_path('reference_data') / 
                         "SpainPolygon" / "gadm41_ESP_0.shp")
        if not spain_boundary.exists():
            self.logger.error(f"Spain boundary not found: {spain_boundary}")
            return False
        
        self.logger.info("✅ Spain boundary shapefile found")
        
        # Check for forest type data
        forest_inventory_dir = self.data_paths.get_path('forest_inventory')
        forest_types_file = forest_inventory_dir / "Forest_Types_Tiers.csv"
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
            'figures',
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
        
        # Create figure subdirectories
        figures_base = self.data_paths.get_path('figures')
        figure_subdirs = [
            'biomass_maps',
            'trend_analysis',
            'climate_relationships',
            'model_interpretability'
        ]
        
        for subdir in figure_subdirs:
            (figures_base / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Output directory structure created")
    
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
            
            # Prepare config overrides for centralized paths
            config_overrides = self.data_paths.get_component_config_overrides('biomass_analysis')
            
            # Initialize biomass analysis orchestrator
            orchestrator = BiomassAnalysisOrchestrator(
                config_path=None,  # Use default config
                log_level=self.logger.level
            )
            
            # Apply config overrides manually
            for key, value in config_overrides.items():
                keys = key.split('.')
                config_section = orchestrator.config
                for k in keys[:-1]:
                    if k not in config_section:
                        config_section[k] = {}
                    config_section = config_section[k]
                config_section[keys[-1]] = value
            
            # Run biomass analysis pipeline
            stage_kwargs = {
                'years': years,
                'biomass_types': ['TBD', 'AGBD', 'BGBD'],
                'skip_mask_creation': False,
                'save_raw_data': True,
                'create_diagnostics': True
            }
            
            results = orchestrator.run_full_pipeline(
                continue_on_error=False,
                **stage_kwargs
            )
            
            # Check for failures
            failed_stages = [k for k, v in results.items() if not v['success']]
            
            stage_time = time.time() - stage_start
            success = len(failed_stages) == 0
            
            self.stage_results[stage_name] = {
                'success': success,
                'duration_minutes': stage_time / 60,
                'result': results,
                'failed_stages': failed_stages
            }
            
            if success:
                self.logger.info(f"✅ {stage_name} completed successfully in {stage_time/60:.2f} minutes")
            else:
                self.logger.error(f"❌ {stage_name} failed in stages: {failed_stages}")
            
            return success
            
        except Exception as e:
            stage_time = time.time() - stage_start
            
            self.stage_results[stage_name] = {
                'success': False,
                'duration_minutes': stage_time / 60,
                'error': str(e)
            }
            
            self.logger.error(f"❌ {stage_name} failed with error: {str(e)}")
            return False
    
    def run_climate_biomass_analysis(self) -> bool:
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
            ml_dir = self.data_paths.get_path('ml_outputs')
            existing_results = list(ml_dir.glob("*.csv")) + list(ml_dir.glob("*.pkl"))
            
            if existing_results:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_results)} result files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Prepare config overrides for centralized paths
            config_overrides = self.data_paths.get_component_config_overrides('climate_biomass_analysis')
            
            # Run climate-biomass analysis
            import sys
            old_argv = sys.argv.copy()
            
            try:
                sys.argv = ['run_full_pipeline.py']
                
                # Run the analysis
                result = run_climate_pipeline_main()
                
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
    
    def run_shap_analysis(self) -> bool:
        """
        Run SHAP analysis for model interpretability.
        
        Returns:
            bool: True if successful
        """
        stage_name = "SHAP Analysis"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            shap_dir = self.data_paths.get_path('analysis_outputs') / 'shap_analysis'
            existing_results = list(shap_dir.glob("*.pkl")) + list(shap_dir.glob("*.png"))
            
            if existing_results:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_results)} result files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run SHAP analysis
            import sys
            old_argv = sys.argv.copy()
            
            try:
                sys.argv = ['run_shap_analysis.py']
                
                # Run the analysis
                result = run_shap_analysis_main()
                
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
        Validate that expected outputs were created.
        
        Returns:
            bool: True if outputs are valid
        """
        self.logger.info("Validating analysis outputs...")
        
        # Check analysis outputs
        analysis_dir = self.data_paths.get_path('analysis_outputs')
        if not analysis_dir.exists():
            self.logger.error(f"Analysis outputs directory not found: {analysis_dir}")
            return False
        
        # Count output files by type
        csv_files = list(analysis_dir.rglob("*.csv"))
        pkl_files = list(analysis_dir.rglob("*.pkl"))
        
        self.logger.info(f"✅ Found {len(csv_files)} CSV result files")
        self.logger.info(f"✅ Found {len(pkl_files)} pickle result files")
        
        # Check figures
        figures_dir = self.data_paths.get_path('figures')
        if figures_dir.exists():
            png_files = list(figures_dir.rglob("*.png"))
            pdf_files = list(figures_dir.rglob("*.pdf"))
            self.logger.info(f"✅ Found {len(png_files)} PNG figures")
            self.logger.info(f"✅ Found {len(pdf_files)} PDF figures")
        
        # Check tables
        tables_dir = self.data_paths.get_path('tables')
        if tables_dir.exists():
            table_files = list(tables_dir.rglob("*.csv")) + list(tables_dir.rglob("*.xlsx"))
            self.logger.info(f"✅ Found {len(table_files)} table files")
        
        # Basic validation - ensure we have some outputs
        total_outputs = len(csv_files) + len(pkl_files)
        if total_outputs < 5:  # Expect at least a few result files
            self.logger.warning(f"Found only {total_outputs} output files - this seems low")
        
        return total_outputs > 0
    
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
            
            # Show failed stages if any
            if not results['success'] and 'failed_stages' in results:
                failed = results['failed_stages']
                if failed:
                    self.logger.info(f"    Failed substages: {', '.join(failed)}")
        
        # Count outputs by directory
        output_dirs = [
            ('analysis_outputs', 'Analysis Results'),
            ('figures', 'Figures'),
            ('tables', 'Tables'),
            ('ml_outputs', 'ML Outputs')
        ]
        
        for dir_key, display_name in output_dirs:
            dir_path = self.data_paths.get_path(dir_key)
            if dir_path.exists():
                all_files = list(dir_path.rglob("*"))
                files_only = [f for f in all_files if f.is_file()]
                self.logger.info(f"📁 {display_name}: {len(files_only)} files in {dir_path}")
        
        # Show data structure
        self.logger.info(f"📂 Analysis outputs in: {self.data_paths.data_root}")
        self.logger.info(f"   └── results/")
        self.logger.info(f"       ├── analysis_outputs/")
        self.logger.info(f"       │   ├── biomass_trends/")
        self.logger.info(f"       │   ├── climate_analysis/")
        self.logger.info(f"       │   └── shap_analysis/")
        self.logger.info(f"       ├── figures/")
        self.logger.info(f"       ├── tables/")
        self.logger.info(f"       └── ml_outputs/")
        
        self.logger.info(f"\n🎯 Analysis complete!")
        self.logger.info(f"   📊 Check results/ directory for all outputs")
        self.logger.info(f"   📈 Review figures/ for visualization")
        self.logger.info(f"   📋 Check tables/ for summary statistics")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Complete Analysis (Biomass + Climate + SHAP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe reproduces all analysis results by executing:
1. Biomass pattern analysis (trends, aggregations, transitions)
2. Climate-biomass relationship analysis (ML modeling, optimization)
3. SHAP analysis for model interpretability

Examples:
  %(prog)s                              # Run complete analysis
  %(prog)s --years 2020 2021 2022       # Specific years only
  %(prog)s --skip-shap                  # Skip SHAP analysis (faster)
  %(prog)s --analysis-only biomass      # Run only biomass analysis
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements:
  - Biomass maps (run reproduce_biomass_estimation.py first)
  - Climate variables for climate-biomass analysis
  - Spain boundary shapefile
  - Forest type reference data
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
        choices=['biomass', 'climate', 'shap'],
        help='Run only specific analysis component'
    )
    
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Skip SHAP analysis (faster execution)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue if an analysis stage fails'
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
            recipe.logger.error("Please ensure required input data is available")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Track overall success
        overall_success = True
        
        # Run specific analysis or all analyses
        if args.analysis_only == 'biomass':
            success = recipe.run_biomass_analysis(years=args.years)
            overall_success = success
        elif args.analysis_only == 'climate':
            success = recipe.run_climate_biomass_analysis()
            overall_success = success
        elif args.analysis_only == 'shap':
            success = recipe.run_shap_analysis()
            overall_success = success
        else:
            # Run all analyses
            
            # 1. Biomass analysis
            success = recipe.run_biomass_analysis(years=args.years)
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Biomass analysis failed")
                sys.exit(1)
            overall_success = overall_success and success
            
            # 2. Climate-biomass analysis
            success = recipe.run_climate_biomass_analysis()
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Climate-biomass analysis failed")
                sys.exit(1)
            overall_success = overall_success and success
            
            # 3. SHAP analysis (optional)
            if not args.skip_shap:
                success = recipe.run_shap_analysis()
                if not success and not args.continue_on_error:
                    recipe.logger.error("❌ SHAP analysis failed")
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