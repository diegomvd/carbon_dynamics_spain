#!/usr/bin/env python3
"""
Recipe: Biomass Estimation

Reproduces biomass estimation results from canopy height predictions.
Executes the complete biomass model pipeline:

1. Allometry fitting (always runs - produces calibrated parameters)
2. Biomass estimation (forest type specific maps)  
3. Annual cropland masking
4. Forest type merging (country-wide maps)

This recipe produces the biomass maps used in the paper analysis.

Usage:
    python 2_biomass_estimation_recipe.py [OPTIONS]

Examples:
    # Run complete biomass estimation
    python 2_biomass_estimation_recipe.py
    
    # Specific years only
    python 2_biomass_estimation_recipe.py --years 2020 2021 2022
    
    # Continue on errors for testing
    python 2_biomass_estimation_recipe.py --continue-on-error

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
from biomass_model.scripts.run_full_pipeline import main as run_biomass_pipeline_main


class BiomassEstimationRecipe:
    """
    Recipe for biomass estimation reproduction.
    
    Provides a simple interface for reproducing biomass estimation results
    with centralized data management and error handling.
    """
    
    def __init__(self, data_root: str = "data", log_level: str = "INFO"):
        """
        Initialize biomass estimation recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        # Setup centralized data paths
        self.data_paths = CentralDataPaths(data_root)
        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='biomass_recipe'
        )
        
        # Track stage results
        self.stage_results = {}
        
        self.logger.info("Initialized Biomass Estimation Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for biomass estimation...")
        
        # Check for height maps (100m for biomass estimation) - UPDATED PATH
        height_100m_dir = self.data_paths.get_height_maps_100m_dir()
        if not height_100m_dir.exists():
            self.logger.error(f"Height maps (100m) directory not found: {height_100m_dir}")
            self.logger.error("Please run '1_canopy_height_prediction_recipe.py' first")
            return False
        
        height_files = list(height_100m_dir.glob("*.tif"))
        if not height_files:
            self.logger.error(f"No height map files found in {height_100m_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(height_files)} country-wide height maps (100m)")
        
        # Check for height maps (10m for allometry calibration) - NEW
        height_10m_dir = self.data_paths.get_height_maps_10m_dir()
        if not height_10m_dir.exists():
            self.logger.error(f"Height maps (10m) directory not found: {height_10m_dir}")
            self.logger.error("10m height maps needed for allometry calibration")
            return False
        
        height_10m_files = list(height_10m_dir.glob("*.tif"))
        if not height_10m_files:
            self.logger.error(f"No 10m height map files found in {height_10m_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(height_10m_files)} sanitized height maps (10m)")
        
        # Check for processed NFI data - UPDATED PATH
        nfi_processed_dir = self.data_paths.get_path('forest_inventory_processed')
        if not nfi_processed_dir.exists():
            self.logger.error(f"Processed NFI directory not found: {nfi_processed_dir}")
            self.logger.error("Please run '0_data_preparation_recipe.py' first")
            return False
        
        nfi_files = list(nfi_processed_dir.glob("*.shp"))
        if not nfi_files:
            self.logger.error(f"No processed NFI shapefiles found in {nfi_processed_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(nfi_files)} processed NFI shapefiles")
        
        # Check for forest type data - UPDATED PATHS
        forest_types_file = self.data_paths.get_path('forest_inventory') / "Forest_Types_Tiers.csv"
        if not forest_types_file.exists():
            self.logger.error(f"Forest types hierarchy file not found: {forest_types_file}")
            return False
        
        self.logger.info("✅ Forest types hierarchy file found")
        
        # Check for forest type maps - UPDATED PATH
        forest_type_maps_dir = self.data_paths.get_forest_type_maps_dir()
        if not forest_type_maps_dir.exists():
            self.logger.error(f"Forest type maps directory not found: {forest_type_maps_dir}")
            return False
        
        mfe_files = list(forest_type_maps_dir.glob("*.shp"))
        if not mfe_files:
            self.logger.error(f"No forest type map files found in {forest_type_maps_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(mfe_files)} forest type map files")
        
        # Check for Corine land cover data (for masking)
        corine_path = self.data_paths.get_corine_land_cover_file()
        if not corine_path.exists():
            self.logger.warning(f"Corine land cover data not found: {corine_path}")
            self.logger.warning("Annual cropland masking may not work properly")
        else:
            self.logger.info("✅ Corine land cover data found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        # Create main directories - NEW STRUCTURE
        self.data_paths.create_directories(['biomass_maps', 'allometries'])
        
        # Create biomass-specific subdirectories
        biomass_base = self.data_paths.get_path('biomass_maps')
        for subdir in self.data_paths.subdirs['biomass_maps'].values():
            (biomass_base / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create biomass type subdirectories in each
            for biomass_type in ['AGBD_MC_100m', 'BGBD_MC_100m', 'TBD_MC_100m']:
                (biomass_base / subdir / biomass_type).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Output directory structure created")
        self.logger.info(f"   📁 allometries/ - fitted allometric parameters")
        self.logger.info(f"   📁 biomass_maps/raw/ - no LC masking")
        self.logger.info(f"   📁 biomass_maps/per_forest_type/ - LC masked per forest type")
        self.logger.info(f"   📁 biomass_maps/full_country/ - merged country-wide")
    
    def run_biomass_estimation(self, 
                              years: Optional[List[int]] = None,
                              continue_on_error: bool = False) -> bool:
        """
        Run the biomass estimation pipeline.
        
        Args:
            years: Specific years to process
            continue_on_error: Continue if a stage fails
            
        Returns:
            bool: True if successful
        """
        stage_name = "Biomass Estimation Pipeline"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed - UPDATED PATH
            full_country_dir = self.data_paths.get_biomass_maps_full_country_dir()
            existing_outputs = []
            for biomass_type in ['AGBD_MC_100m', 'BGBD_MC_100m', 'TBD_MC_100m']:
                type_dir = full_country_dir / biomass_type
                if type_dir.exists():
                    existing_outputs.extend(list(type_dir.glob("*_merged.tif")))
            
            if existing_outputs:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_outputs)} merged files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run biomass estimation with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for biomass estimation with new paths
                sys.argv = [
                    'run_full_pipeline.py',
                    '--height-100m-dir', str(self.data_paths.get_height_maps_100m_dir()),
                    '--height-10m-dir', str(self.data_paths.get_height_maps_10m_dir()),
                    '--nfi-processed-dir', str(self.data_paths.get_path('forest_inventory_processed')),
                    '--forest-types-file', str(self.data_paths.get_path('forest_inventory') / "Forest_Types_Tiers.csv"),
                    '--forest-type-maps-dir', str(self.data_paths.get_forest_type_maps_dir()),
                    '--allometries-output-dir', str(self.data_paths.get_allometries_dir()),
                    '--output-dir', str(self.data_paths.get_path('biomass_maps')),
                    '--corine-file', str(self.data_paths.get_corine_land_cover_file()),
                    '--always-fit-allometry',  # NEW: always fit allometry (no existence check)
                    '--log-level', 'INFO'
                ]
                
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                if continue_on_error:
                    sys.argv.append('--continue-on-error')
                
                # Run the pipeline
                result = run_biomass_pipeline_main()
                
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
        self.logger.info("Validating biomass estimation outputs...")
        
        # Check all three output directories - NEW STRUCTURE
        stage_checks = [
            ('Raw outputs', self.data_paths.get_biomass_maps_raw_dir()),
            ('Per forest type', self.data_paths.get_biomass_maps_per_forest_type_dir()),
            ('Full country', self.data_paths.get_biomass_maps_full_country_dir())
        ]
        
        total_files = 0
        for stage_name, stage_dir in stage_checks:
            if not stage_dir.exists():
                self.logger.error(f"{stage_name} directory not found: {stage_dir}")
                return False
            
            # Count all biomass files (mean + uncertainty)
            stage_files = list(stage_dir.rglob("*.tif"))
            if not stage_files:
                self.logger.error(f"No biomass files found in {stage_name}")
                return False
            
            total_files += len(stage_files)
            self.logger.info(f"✅ {stage_name}: {len(stage_files)} files")
        
        self.logger.info(f"✅ Total biomass estimation outputs: {total_files} files")
        
        # Basic validation - check for both mean and uncertainty files
        full_country_dir = self.data_paths.get_biomass_maps_full_country_dir()
        mean_files = list(full_country_dir.rglob("*_mean_*.tif"))
        uncertainty_files = list(full_country_dir.rglob("*_uncertainty_*.tif"))
        
        if not mean_files or not uncertainty_files:
            self.logger.error("Missing mean or uncertainty files in final outputs")
            return False
        
        self.logger.info(f"✅ Found {len(mean_files)} mean files and {len(uncertainty_files)} uncertainty files")
        
        # Check allometry outputs - NEW
        allometries_dir = self.data_paths.get_allometries_dir()
        if allometries_dir.exists():
            allometry_files = list(allometries_dir.glob("*.csv"))
            self.logger.info(f"✅ Found {len(allometry_files)} allometry parameter files")
        else:
            self.logger.warning("⚠️  Allometries directory not found")
        
        return True
    
    def print_summary(self) -> None:
        """Print summary of biomass estimation results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BIOMASS ESTIMATION RECIPE SUMMARY")
        self.logger.info(f"{'='*60}")
        
        # Show results for each stage
        for stage_name, results in self.stage_results.items():
            status = "✅" if results['success'] else "❌"
            duration = results['duration_minutes']
            self.logger.info(f"  {status} {stage_name}: {duration:.2f} min")
        
        # Check output directories - NEW STRUCTURE
        stage_mappings = [
            ('Allometries', self.data_paths.get_allometries_dir()),
            ('Raw Maps', self.data_paths.get_biomass_maps_raw_dir()),
            ('Per Forest Type', self.data_paths.get_biomass_maps_per_forest_type_dir()),
            ('Full Country', self.data_paths.get_biomass_maps_full_country_dir())
        ]
        
        for stage_name, stage_dir in stage_mappings:
            if stage_dir.exists():
                if stage_name == 'Allometries':
                    stage_files = list(stage_dir.glob("*.csv"))
                else:
                    stage_files = list(stage_dir.rglob("*.tif"))
                self.logger.info(f"📁 {stage_name}: {len(stage_files)} files in {stage_dir}")
        
        # Show NEW data structure
        self.logger.info(f"📂 Data structure created in: {self.data_paths.data_root}")
        self.logger.info(f"   └── processed/")
        self.logger.info(f"       └── biomass_maps/                  # NEW structure")
        self.logger.info(f"           ├── raw/                       # No LC masking")
        self.logger.info(f"           ├── per_forest_type/           # LC masked per forest type")
        self.logger.info(f"           └── full_country/              # Merged country-wide")
        
        self.logger.info(f"\n🎯 Next steps:")
        self.logger.info(f"   1. Run '3_analysis_recipe.py' to analyze biomass patterns")
        self.logger.info(f"   2. Check allometry parameters in processed/allometries/")
        self.logger.info(f"   3. Check biomass maps quality in biomass_maps/ directories")
        self.logger.info(f"   4. Use full_country/ files for further analysis")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Biomass Estimation from Canopy Height Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe reproduces the biomass estimation results by executing:
1. Allometry fitting (always runs - produces calibrated parameters)
2. Biomass estimation (forest type specific maps)
3. Annual cropland masking  
4. Forest type merging (country-wide maps)

NEW DATA STRUCTURE:
  data/processed/allometries/             # Fitted allometric parameters (NEW)
    ├── fitted_parameters.csv             # Calibrated allometric equations
    ├── bgb_ratios.csv                    # Calculated BGB ratios
    └── fitting_summary.csv               # Model fit statistics
  data/processed/biomass_maps/            # Output structure
    ├── raw/                              # No LC masking
    ├── per_forest_type/                  # LC masked per forest type  
    └── full_country/                     # Merged country-wide

INPUTS FROM PREVIOUS RECIPES:
  data/processed/height_maps/100m/        # Primary height data (Recipe 1)
  data/processed/height_maps/10m/         # For allometry calibration (Recipe 1)
  data/processed/forest_inventory/        # Processed NFI plots (Recipe 0)
  data/raw/forest_inventory/Forest_Types_Tiers.csv  # Forest type hierarchy
  data/raw/forest_type_maps/              # MFE forest type maps (Recipe 0)

Examples:
  %(prog)s                              # Run complete pipeline
  %(prog)s --years 2020 2021 2022       # Specific years only
  %(prog)s --continue-on-error          # Continue despite failures
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements (FROM PREVIOUS RECIPES):
  - Height predictions from height modeling recipe (100m + 10m)
  - Processed forest inventory data from data preparation recipe
  - Forest type maps and hierarchy from data preparation recipe
  - Corine land cover data in data/raw/land_cover/ for annual cropland masking
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
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution if a stage fails'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate prerequisites without running pipeline'
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
    """Main entry point for biomass estimation recipe."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    log_level = 'ERROR' if args.quiet else args.log_level
    
    try:
        # Initialize recipe
        recipe = BiomassEstimationRecipe(
            data_root=args.data_root,
            log_level=log_level
        )
        
        # Validate prerequisites
        if not recipe.validate_prerequisites():
            recipe.logger.error("❌ Prerequisites validation failed")
            recipe.logger.error("Please ensure height maps and processed NFI data are available")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Run biomass estimation pipeline
        success = recipe.run_biomass_estimation(
            years=args.years,
            continue_on_error=args.continue_on_error
        )
        
        if not success:
            recipe.logger.error("❌ Biomass estimation pipeline failed")
            sys.exit(1)
        
        # Validate outputs
        if success and recipe.validate_outputs():
            recipe.print_summary()
            elapsed_time = time.time() - start_time
            recipe.logger.info(f"🎉 Biomass estimation recipe completed successfully in {elapsed_time/60:.2f} minutes!")
        else:
            recipe.logger.error("❌ Biomass estimation recipe failed or output validation failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Recipe interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Recipe failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()