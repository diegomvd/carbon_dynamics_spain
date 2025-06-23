#!/usr/bin/env python3
"""
Recipe: Data Preparation

Reproduces the data preparation pipeline including:
1. Forest inventory processing (NFI data to biomass shapefiles)
2. Sentinel-2 mosaic creation (summer mosaics from STAC catalog)
3. ALS PNOA processing (LiDAR tiles to training data, optional)

This recipe prepares the foundational datasets used throughout the analysis.

Usage:
    python 0_data_preparation_recipe.py [OPTIONS]

Examples:
    # Run complete data preparation
    python 0_data_preparation_recipe.py
    
    # Specific years for Sentinel-2 mosaics
    python 0_data_preparation_recipe.py --years 2020 2021 2022
    
    # Skip forest inventory processing
    python 0_data_preparation_recipe.py --skip-forest-inventory
    
    # Custom processing extent
    python 0_data_preparation_recipe.py --bbox 40.0 -10.0 44.0 -6.0

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

# Import utilities
from shared_utils.central_data_paths import CentralDataPaths
from shared_utils.config_utils import load_config
from shared_utils.logging_utils import setup_logging

# Import component scripts
from forest_inventory.scripts.run_nfi_processing import main as run_nfi_processing_main
from sentinel2_processing.scripts.run_postprocessing import main as run_sentinel2_postprocessing_main
from als_pnoa.scripts.run_pnoa_processing import main as run_pnoa_processing_main


class DataPreparationRecipe:
    """
    Recipe for data preparation reproduction.
    
    Orchestrates forest inventory processing and Sentinel-2 mosaic creation
    with centralized data management.
    """
    
    def __init__(self, data_root: str = "data", log_level: str = "INFO"):
        """
        Initialize data preparation recipe.
        
        Args:
            data_root: Root directory for data storage
            log_level: Logging level
        """
        # Setup centralized data paths
        self.data_paths = CentralDataPaths(data_root)
        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='data_prep_recipe'
        )
        
        # Track stage results
        self.stage_results = {}
        
        self.logger.info("Initialized Data Preparation Recipe")
        self.logger.info(f"Data root: {self.data_paths.data_root}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate that required input data exists.
        
        Returns:
            bool: True if prerequisites are met
        """
        self.logger.info("Validating prerequisites for data preparation...")
        
        # Check for NFI4 database files - UPDATED PATH
        nfi4_dir = self.data_paths.get_nfi4_database_dir()
        if not nfi4_dir.exists():
            self.logger.error(f"NFI4 database directory not found: {nfi4_dir}")
            self.logger.error("Expected structure: data/raw/forest_inventory/nfi4/*.accdb")
            return False
        
        nfi4_files = list(nfi4_dir.glob("*.accdb"))
        if not nfi4_files:
            self.logger.error(f"No NFI4 database files found in: {nfi4_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(nfi4_files)} NFI4 database files")
        
        # Check for forest type maps (MFE) - UPDATED PATH
        forest_type_maps_dir = self.data_paths.get_forest_type_maps_dir()
        if not forest_type_maps_dir.exists():
            self.logger.error(f"Forest type maps directory not found: {forest_type_maps_dir}")
            self.logger.error("Expected structure: data/raw/forest_type_maps/*.shp")
            return False
        
        mfe_files = list(forest_type_maps_dir.glob("*.shp"))
        if not mfe_files:
            self.logger.error(f"No forest type map files found in: {forest_type_maps_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(mfe_files)} forest type map files")
        
        # Check for Spain boundary - UPDATED PATH (now in forest_type_maps)
        spain_boundaries = [f for f in mfe_files if 'spain' in f.name.lower() or 'gadm' in f.name.lower()]
        if not spain_boundaries:
            self.logger.warning("No Spain boundary shapefile found in forest_type_maps")
            self.logger.warning("Processing may fail without country boundaries")
        else:
            self.logger.info(f"✅ Found Spain boundary: {spain_boundaries[0].name}")
        
        # Check for NFI4 codes file - UPDATED PATH
        nfi4_codes_file = self.data_paths.get_forest_inventory_codes_file()
        if not nfi4_codes_file.exists():
            self.logger.error(f"NFI4 codes file not found: {nfi4_codes_file}")
            return False
        
        self.logger.info(f"✅ NFI4 codes file found")
        
        # Check for forest types CSV
        forest_types_file = (self.data_paths.get_path('forest_inventory') / 
                           self.data_paths.subdirs['forest_inventory']['types'])
        if not forest_types_file.exists():
            self.logger.error(f"Forest types file not found: {forest_types_file}")
            return False
        
        self.logger.info(f"✅ Forest types file found")
        
        # Check for wood density database - UPDATED PATH
        wood_density_file = self.data_paths.get_wood_density_file()
        if not wood_density_file.exists():
            self.logger.error(f"Wood density database not found: {wood_density_file}")
            return False
        
        self.logger.info(f"✅ Wood density database found")
        
        # Check for internet connectivity (for Sentinel-2)
        self.logger.info("ℹ️  Sentinel-2 processing requires internet access for STAC catalog")
        self.logger.warning("Sentinel-2 processing may fail if no internet access")
        
        # Check for ALS canopy height data - UPDATED PATH
        als_data_dir = self.data_paths.get_als_data_dir()
        if not als_data_dir.exists():
            self.logger.warning(f"ALS canopy height data directory not found: {als_data_dir}")
            self.logger.warning("ALS PNOA processing will be skipped")
        else:
            als_files = list(als_data_dir.glob("NDSM-VEGETACION-*.tif"))
            if als_files:
                self.logger.info(f"✅ Found {len(als_files)} ALS canopy height files")
            else:
                self.logger.warning("ALS data directory exists but no NDSM files found")
        
        # Check for ALS tile metadata - UPDATED PATH
        als_metadata_dir = self.data_paths.get_als_metadata_dir()
        if not als_metadata_dir.exists():
            self.logger.warning(f"ALS tile metadata directory not found: {als_metadata_dir}")
        else:
            metadata_files = list(als_metadata_dir.rglob("*.shp"))
            if metadata_files:
                self.logger.info(f"✅ Found {len(metadata_files)} ALS metadata files")
            else:
                self.logger.warning("ALS metadata directory exists but no shapefiles found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        # Create main data directories
        self.data_paths.create_directories([
            'forest_inventory_processed',
            'sentinel2_processed',
            'als_canopy_height_processed',
            'sentinel2_mosaics'
        ])
        
        self.logger.info("✅ Output directory structure created")
    
    def run_forest_inventory_processing(self) -> bool:
        """
        Run forest inventory processing.
        
        Returns:
            bool: True if successful
        """
        stage_name = "Forest Inventory Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed - UPDATED PATH
            forest_out_dir = self.data_paths.get_path('forest_inventory_processed')
            existing_shapefiles = list(forest_out_dir.glob("*.shp"))
            
            if existing_shapefiles:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_shapefiles)} shapefiles")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run forest inventory processing with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for NFI processing with new paths
                sys.argv = [
                    'run_nfi_processing.py',
                    '--nfi4-dir', str(self.data_paths.get_nfi4_database_dir()),
                    '--forest-types-dir', str(self.data_paths.get_forest_type_maps_dir()),
                    '--codes-file', str(self.data_paths.get_forest_inventory_codes_file()),
                    '--wood-density-file', str(self.data_paths.get_wood_density_file()),
                    '--output-dir', str(forest_out_dir),
                    '--log-level', 'INFO'
                ]
                
                # Run the processing
                result = run_nfi_processing_main()
                
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
    
    def run_sentinel2_processing(self, 
                                years: Optional[List[int]] = None,
                                bbox: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Run Sentinel-2 mosaic processing.
        
        Args:
            years: Specific years to process
            bbox: Bounding box (south, west, north, east)
            
        Returns:
            bool: True if successful
        """
        stage_name = "Sentinel-2 Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
            existing_mosaics = list(mosaic_dir.glob("*.tif"))
            
            if existing_mosaics and len(existing_mosaics) >= 5:  # Expect multiple years
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_mosaics)} mosaics")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run Sentinel-2 processing
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for Sentinel-2 processing with new paths
                sys.argv = [
                    'run_postprocessing.py',
                    '--output-dir', str(mosaic_dir),
                    '--boundary-file', str(self.data_paths.get_forest_type_maps_dir() / "gadm41_ESP_0.shp"),
                    '--log-level', 'INFO'
                ]
                
                if years:
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                if bbox:
                    sys.argv.extend(['--bbox'] + [str(coord) for coord in bbox])
                
                # Run the processing
                result = run_sentinel2_postprocessing_main()
                
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
    
    def run_als_pnoa_processing(self) -> bool:
        """
        Run ALS PNOA processing to prepare training data.
        
        Returns:
            bool: True if successful
        """
        stage_name = "ALS PNOA Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if ALS data is available - UPDATED PATH
        als_data_dir = self.data_paths.get_als_data_dir()
        if not als_data_dir.exists():
            self.logger.info(f"ℹ️  Skipping {stage_name} - no ALS data found")
            self.stage_results[stage_name] = {
                'success': True,
                'duration_minutes': 0,
                'result': 'skipped_no_data'
            }
            return True
        
        stage_start = time.time()
        
        try:
            # Check if already completed - UPDATED PATH
            als_output_dir = self.data_paths.get_path('als_canopy_height_processed')
            existing_outputs = list(als_output_dir.glob("PNOA_*.tif"))
            
            if existing_outputs:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_outputs)} files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Run ALS processing with new paths
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for ALS processing with new paths
                sys.argv = [
                    'run_pnoa_processing.py',
                    '--als-data-dir', str(self.data_paths.get_als_data_dir()),
                    '--metadata-dir', str(self.data_paths.get_als_metadata_dir()),
                    '--sentinel2-dir', str(self.data_paths.get_path('sentinel2_processed')),
                    '--output-dir', str(als_output_dir),
                    '--log-level', 'INFO'
                ]
                
                # Run the processing
                result = run_pnoa_processing_main()
                
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
        self.logger.info("Validating data preparation outputs...")
        
        # Check forest inventory outputs - UPDATED PATH
        forest_out_dir = self.data_paths.get_path('forest_inventory_processed')
        forest_shapefiles = list(forest_out_dir.glob("*.shp"))
        
        if not forest_shapefiles:
            self.logger.error("No forest inventory shapefiles found")
            return False
        
        self.logger.info(f"✅ Found {len(forest_shapefiles)} forest inventory shapefiles")
        
        # Check Sentinel-2 outputs
        mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
        mosaic_files = list(mosaic_dir.glob("*.tif"))
        
        if not mosaic_files:
            self.logger.error("No Sentinel-2 mosaic files found")
            return False
        
        self.logger.info(f"✅ Found {len(mosaic_files)} Sentinel-2 mosaic files")
        
        # Check ALS PNOA outputs (optional) - UPDATED PATH
        als_output_dir = self.data_paths.get_path('als_canopy_height_processed')
        if als_output_dir.exists():
            als_files = list(als_output_dir.glob("PNOA_*.tif"))
            if als_files:
                self.logger.info(f"✅ Found {len(als_files)} ALS canopy height files")
            else:
                self.logger.info("ℹ️  No ALS canopy height files found (optional)")
        else:
            self.logger.info("ℹ️  ALS output directory not found (optional)")
        
        # Basic file size validation
        for file_path in (forest_shapefiles[:2] + mosaic_files[:2]):
            if file_path.stat().st_size < 1000:
                self.logger.warning(f"Suspiciously small output file: {file_path}")
        
        return True
    
    def print_summary(self) -> None:
        """Print summary of data preparation results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DATA PREPARATION RECIPE SUMMARY")
        self.logger.info(f"{'='*60}")
        
        # Show results for each stage
        for stage_name, results in self.stage_results.items():
            status = "✅" if results['success'] else "❌"
            duration = results['duration_minutes']
            self.logger.info(f"  {status} {stage_name}: {duration:.2f} min")
        
        # Check output directories - UPDATED PATHS
        forest_out_dir = self.data_paths.get_path('forest_inventory_processed')
        mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
        
        if forest_out_dir.exists():
            forest_files = list(forest_out_dir.glob("*.shp"))
            self.logger.info(f"📁 Forest inventory outputs: {len(forest_files)} files in {forest_out_dir}")
        
        if mosaic_dir.exists():
            mosaic_files = list(mosaic_dir.glob("*.tif"))
            self.logger.info(f"📁 Sentinel-2 mosaics: {len(mosaic_files)} files in {mosaic_dir}")
        
        # Check ALS PNOA outputs - UPDATED PATH
        als_output_dir = self.data_paths.get_path('als_canopy_height_processed')
        if als_output_dir.exists():
            als_files = list(als_output_dir.glob("PNOA_*.tif"))
            self.logger.info(f"📁 ALS canopy height data: {len(als_files)} files in {als_output_dir}")
        
        # Show NEW data structure
        self.logger.info(f"📂 Data structure created in: {self.data_paths.data_root}")
        self.logger.info(f"   ├── raw/")
        self.logger.info(f"   │   ├── forest_inventory/nfi4/           # NFI4 databases")
        self.logger.info(f"   │   ├── forest_type_maps/                # MFE shapefiles + boundaries")
        self.logger.info(f"   │   ├── wood_density/                    # Wood density database")
        self.logger.info(f"   │   └── als_canopy_height/               # ALS LiDAR data")
        self.logger.info(f"   │       ├── data/                        # NDSM files")
        self.logger.info(f"   │       └── tile_metadata/utm_XX/        # Coverage metadata")
        self.logger.info(f"   └── processed/")
        self.logger.info(f"       ├── forest_inventory/                # Processed NFI outputs")
        self.logger.info(f"       ├── sentinel2_mosaics/               # Summer mosaics")
        self.logger.info(f"       ├── sentinel2/                       # S2 intermediate data")
        self.logger.info(f"       └── als_canopy_height/               # ALS training data")
        
        self.logger.info(f"\n🎯 Next steps:")
        self.logger.info(f"   1. Run '1_canopy_height_prediction_recipe.py' to generate height predictions")
        self.logger.info(f"   2. Check data quality in processed directories")
        self.logger.info(f"   3. Verify Sentinel-2 mosaics cover your area of interest")
        if als_output_dir.exists() and list(als_output_dir.glob("PNOA_*.tif")):
            self.logger.info(f"   4. ALS canopy height training data ready for model training")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Data Preparation (Forest Inventory + Sentinel-2 Mosaics + ALS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe prepares foundational datasets by executing:
1. Forest inventory processing (NFI4 → biomass shapefiles)
2. Sentinel-2 mosaic creation (STAC catalog → summer mosaics)
3. ALS PNOA processing (LiDAR tiles → training data, optional)

NEW DATA STRUCTURE:
  data/raw/forest_inventory/nfi4/          # NFI4 databases (was IFN_4_SP)
  data/raw/forest_type_maps/               # MFE + boundaries (was separate)
  data/raw/wood_density/                   # Wood density database
  data/raw/als_canopy_height/data/         # ALS data (was pnoa_lidar)
  data/raw/als_canopy_height/tile_metadata/utm_XX/  # Metadata (was pnoa_coverage)

Examples:
  %(prog)s                              # Run complete data preparation
  %(prog)s --years 2020 2021 2022       # Specific years for S2 mosaics
  %(prog)s --skip-forest-inventory      # Skip NFI processing
  %(prog)s --skip-als-pnoa              # Skip ALS processing
  %(prog)s --bbox 40.0 -10.0 44.0 -6.0  # Custom processing extent
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements (NEW PATHS):
  - NFI4 database files in data/raw/forest_inventory/nfi4/*.accdb
  - nfi4_codes.csv in data/raw/forest_inventory/
  - MFE forest maps + Spain boundaries in data/raw/forest_type_maps/*.shp
  - Wood density DB in data/raw/wood_density/GlobalWoodDensityDatabase.xls
  - Internet connectivity for Sentinel-2 STAC catalog access
  - ALS LiDAR data in data/raw/als_canopy_height/data/ (optional)
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
        help='Specific years for Sentinel-2 processing (default: all available)'
    )
    
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('SOUTH', 'WEST', 'NORTH', 'EAST'),
        help='Bounding box for processing (south west north east)'
    )
    
    parser.add_argument(
        '--skip-forest-inventory',
        action='store_true',
        help='Skip forest inventory processing'
    )
    
    parser.add_argument(
        '--skip-sentinel2',
        action='store_true',
        help='Skip Sentinel-2 mosaic processing'
    )
    
    parser.add_argument(
        '--skip-als-pnoa',
        action='store_true',
        help='Skip ALS PNOA processing (training data preparation)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue if a processing stage fails'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate prerequisites without running processing'
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
    """Main entry point for data preparation recipe."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    log_level = 'ERROR' if args.quiet else args.log_level
    
    try:
        # Initialize recipe
        recipe = DataPreparationRecipe(
            data_root=args.data_root,
            log_level=log_level
        )
        
        # Validate prerequisites
        if not recipe.validate_prerequisites():
            recipe.logger.error("❌ Prerequisites validation failed")
            recipe.logger.error("Please ensure required input data is available in the NEW structure")
            sys.exit(1)
        
        if args.validate_only:
            recipe.logger.info("✅ Prerequisites validation passed - exiting")
            sys.exit(0)
        
        # Create output structure
        recipe.create_output_structure()
        
        # Track overall success
        overall_success = True
        
        # Run forest inventory processing
        if not args.skip_forest_inventory:
            success = recipe.run_forest_inventory_processing()
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Forest inventory processing failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Run Sentinel-2 processing
        if not args.skip_sentinel2:
            success = recipe.run_sentinel2_processing(
                years=args.years,
                bbox=tuple(args.bbox) if args.bbox else None
            )
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ Sentinel-2 processing failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Run ALS PNOA processing (after Sentinel-2)
        if not args.skip_als_pnoa:
            success = recipe.run_als_pnoa_processing()
            if not success and not args.continue_on_error:
                recipe.logger.error("❌ ALS PNOA processing failed")
                sys.exit(1)
            overall_success = overall_success and success
        
        # Validate outputs
        if overall_success and recipe.validate_outputs():
            recipe.print_summary()
            elapsed_time = time.time() - start_time
            recipe.logger.info(f"🎉 Data preparation recipe completed successfully in {elapsed_time/60:.2f} minutes!")
        else:
            recipe.logger.error("❌ Data preparation recipe failed or output validation failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Recipe interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Recipe failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()