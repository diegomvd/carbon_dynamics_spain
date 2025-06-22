#!/usr/bin/env python3
"""
Recipe: Biomass Estimation

Reproduces biomass estimation results from canopy height predictions.
Executes the complete biomass model pipeline:

1. Allometry fitting (optional - uses existing if available)
2. Biomass estimation (forest type specific maps)  
3. Annual cropland masking
4. Forest type merging (country-wide maps)

This recipe produces the biomass maps used in the paper analysis.

Usage:
    python reproduce_biomass_estimation.py [OPTIONS]

Examples:
    # Run complete biomass estimation
    python reproduce_biomass_estimation.py
    
    # Specific years only
    python reproduce_biomass_estimation.py --years 2020 2021 2022
    
    # Skip allometry fitting (use existing)
    python reproduce_biomass_estimation.py --skip-allometry
    
    # Continue on errors for testing
    python reproduce_biomass_estimation.py --continue-on-error

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
from shared_utils.config_utils import load_config, update_config_paths
from shared_utils.logging_utils import setup_logging

# Import component orchestrator
from biomass_model.scripts.run_full_pipeline import BiomassFullPipelineOrchestrator


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
        
        self.logger.info("Initialized Biomass Estimation Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for biomass estimation...")
        
        # Check for canopy height predictions
        height_dir = self.data_paths.get_path('height_predictions')
        if not height_dir.exists():
            self.logger.error(f"Height predictions directory not found: {height_dir}")
            self.logger.error("Please run 'reproduce_height_modeling.py' first to generate height predictions")
            return False
        
        # Check for at least one height file
        height_files = list(height_dir.rglob("*.tif"))
        if not height_files:
            self.logger.error(f"No height prediction files found in {height_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(height_files)} height prediction files")
        
        # Check for forest inventory data (allometries, forest types, etc.)
        forest_inventory_dir = self.data_paths.get_path('forest_inventory')
        if not forest_inventory_dir.exists():
            self.logger.error(f"Forest inventory directory not found: {forest_inventory_dir}")
            self.logger.error("Please ensure forest inventory data is available in data/raw/forest_inventory/")
            return False
        
        # Check for key files
        required_files = [
            "H_AGB_Allometries_Tiers_ModelCalibrated_Quantiles_15-85_OnlyPowerLaw.csv",
            "Forest_Types_Tiers.csv",
            "BGBRatios_Tiers.csv"
        ]
        
        missing_files = []
        for filename in required_files:
            file_path = forest_inventory_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            self.logger.error(f"Missing required forest inventory files: {missing_files}")
            return False
        
        self.logger.info("✅ Forest inventory data validation passed")
        
        # Check for forest type masks
        mask_dir = self.data_paths.get_path('forest_type_masks')
        if not mask_dir.exists():
            self.logger.warning(f"Forest type masks directory not found: {mask_dir}")
            self.logger.warning("Forest type masks will be created during processing if needed")
        else:
            mask_files = list(mask_dir.rglob("*.tif"))
            self.logger.info(f"✅ Found {len(mask_files)} forest type mask files")
        
        # Check for Corine land cover data
        corine_path = self.data_paths.get_path('reference_data') / "corine_land_cover" / "U2018_CLC2018_V2020_20u1.tif"
        if not corine_path.exists():
            self.logger.warning(f"Corine land cover data not found: {corine_path}")
            self.logger.warning("Annual cropland masking may not work properly")
        else:
            self.logger.info("✅ Corine land cover data found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        # Create main output directories
        self.data_paths.create_directories([
            'biomass_maps',
            'analysis_outputs',
            'figures',
            'tables'
        ])
        
        # Create biomass-specific subdirectories
        biomass_base = self.data_paths.get_path('biomass_maps')
        for subdir in self.data_paths.subdirs['biomass_maps'].values():
            (biomass_base / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create biomass type subdirectories
            for biomass_type in ['AGBD_MC_100m', 'BGBD_MC_100m', 'TBD_MC_100m']:
                (biomass_base / subdir / biomass_type).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Output directory structure created")
    
    def prepare_config_overrides(self) -> dict:
        """
        Prepare configuration overrides for centralized data paths.
        
        Returns:
            dict: Configuration overrides for biomass model component
        """
        overrides = self.data_paths.get_component_config_overrides('biomass_model')
        
        self.logger.debug(f"Config overrides prepared: {len(overrides)} settings")
        for key, value in overrides.items():
            self.logger.debug(f"  {key}: {value}")
        
        return overrides
    
    def run_biomass_estimation(self, 
                              years: Optional[List[int]] = None,
                              skip_allometry: bool = False,
                              continue_on_error: bool = False,
                              overwrite_allometry: bool = False) -> bool:
        """
        Run the biomass estimation pipeline.
        
        Args:
            years: Specific years to process
            skip_allometry: Skip allometry fitting stage
            continue_on_error: Continue if a stage fails
            overwrite_allometry: Overwrite existing allometry files
            
        Returns:
            bool: True if successful
        """
        self.logger.info("Starting biomass estimation pipeline...")
        
        try:
            # Initialize orchestrator with config overrides
            config_overrides = self.prepare_config_overrides()
            
            # For now, create a temporary config with overrides
            # In a full implementation, you might want to patch the config loading
            orchestrator = BiomassFullPipelineOrchestrator(
                config_path=None,  # Use default config
                log_level=self.logger.level
            )
            
            # Apply config overrides manually
            # Note: This is a simplified approach - in production you'd want 
            # a more sophisticated config merging mechanism
            for key, value in config_overrides.items():
                keys = key.split('.')
                config_section = orchestrator.config
                for k in keys[:-1]:
                    if k not in config_section:
                        config_section[k] = {}
                    config_section = config_section[k]
                config_section[keys[-1]] = value
            
            # Run the pipeline
            results = orchestrator.run_full_pipeline(
                skip_allometry=skip_allometry,
                continue_on_error=continue_on_error,
                years=years,
                overwrite_allometry=overwrite_allometry
            )
            
            # Print summary
            orchestrator.print_pipeline_summary()
            
            # Check for failures
            failed_stages = [k for k, v in results.items() if not v['success']]
            
            if failed_stages:
                if continue_on_error:
                    self.logger.warning(f"Pipeline completed with failures in: {failed_stages}")
                    return True  # Consider it successful if continue_on_error
                else:
                    self.logger.error(f"Pipeline failed in stages: {failed_stages}")
                    return False
            else:
                self.logger.info("✅ Biomass estimation pipeline completed successfully!")
                return True
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False
    
    def validate_outputs(self) -> bool:
        """
        Validate that expected outputs were created.
        
        Returns:
            bool: True if outputs are valid
        """
        self.logger.info("Validating biomass estimation outputs...")
        
        # Check for merged biomass maps (final outputs)
        merged_dir = self.data_paths.get_path('biomass_maps') / self.data_paths.subdirs['biomass_maps']['merged']
        
        if not merged_dir.exists():
            self.logger.error(f"Merged biomass directory not found: {merged_dir}")
            return False
        
        # Check for expected biomass files
        biomass_types = ['TBD', 'AGBD', 'BGBD']
        statistics = ['mean', 'uncertainty']
        expected_files = []
        
        for biomass_type in biomass_types:
            for stat in statistics:
                pattern = f"{biomass_type}_S2_{stat}_*_100m_merged.tif"
                files = list(merged_dir.glob(pattern))
                expected_files.extend(files)
        
        if not expected_files:
            self.logger.error("No merged biomass files found")
            return False
        
        self.logger.info(f"✅ Found {len(expected_files)} merged biomass files")
        
        # Check file sizes (basic validation)
        for file_path in expected_files[:3]:  # Check first few files
            if file_path.stat().st_size < 1000:  # Very small files likely indicate errors
                self.logger.warning(f"Suspiciously small output file: {file_path}")
        
        return True
    
    def print_summary(self) -> None:
        """Print summary of biomass estimation results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BIOMASS ESTIMATION RECIPE SUMMARY")
        self.logger.info(f"{'='*60}")
        
        # Check output directories
        merged_dir = self.data_paths.get_path('biomass_maps') / self.data_paths.subdirs['biomass_maps']['merged']
        if merged_dir.exists():
            biomass_files = list(merged_dir.glob("*.tif"))
            self.logger.info(f"📁 Final biomass maps: {len(biomass_files)} files in {merged_dir}")
        
        # Show data structure
        self.logger.info(f"📂 Data structure created in: {self.data_paths.data_root}")
        self.logger.info(f"   ├── processed/biomass_maps/")
        self.logger.info(f"   │   ├── biomass_no_LC_masking/")
        self.logger.info(f"   │   ├── with_annual_crop_mask/")
        self.logger.info(f"   │   └── biomass_maps_merged/")
        self.logger.info(f"   └── results/")
        
        self.logger.info(f"\n🎯 Next steps:")
        self.logger.info(f"   1. Run 'reproduce_analysis.py' to analyze biomass patterns")
        self.logger.info(f"   2. Check output quality in data/processed/biomass_maps/")
        self.logger.info(f"   3. Use merged files for further analysis")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Biomass Estimation from Canopy Height Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe reproduces the biomass estimation results by executing:
1. Allometry fitting (optional - uses existing if available)
2. Biomass estimation (forest type specific maps)
3. Annual cropland masking  
4. Forest type merging (country-wide maps)

Examples:
  %(prog)s                              # Run complete pipeline
  %(prog)s --years 2020 2021 2022       # Specific years only
  %(prog)s --skip-allometry             # Use existing allometries
  %(prog)s --continue-on-error          # Continue despite failures
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements:
  - Canopy height predictions (run reproduce_height_modeling.py first)
  - Forest inventory data in data/raw/forest_inventory/
  - Corine land cover data in data/raw/reference_data/
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
        '--skip-allometry',
        action='store_true',
        help='Skip allometry fitting (use existing allometry files)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline execution if a stage fails'
    )
    
    parser.add_argument(
        '--overwrite-allometry',
        action='store_true',
        help='Overwrite existing allometry files'
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
            recipe.logger.error("Please ensure required input data is available")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Run biomass estimation
        success = recipe.run_biomass_estimation(
            years=args.years,
            skip_allometry=args.skip_allometry,
            continue_on_error=args.continue_on_error,
            overwrite_allometry=args.overwrite_allometry
        )
        
        if success:
            # Validate outputs
            if recipe.validate_outputs():
                recipe.print_summary()
                elapsed_time = time.time() - start_time
                recipe.logger.info(f"🎉 Biomass estimation recipe completed successfully in {elapsed_time/60:.2f} minutes!")
            else:
                recipe.logger.error("❌ Output validation failed")
                sys.exit(1)
        else:
            recipe.logger.error("❌ Biomass estimation recipe failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Recipe interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Recipe failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()