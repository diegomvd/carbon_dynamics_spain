#!/usr/bin/env python3
"""
Recipe: Data Preparation

Reproduces the data preparation pipeline including:
1. Forest inventory processing (NFI data to biomass shapefiles)
2. Sentinel-2 mosaic creation (summer mosaics from STAC catalog)

This recipe prepares the foundational datasets used throughout the analysis.

Usage:
    python reproduce_data_preparation.py [OPTIONS]

Examples:
    # Run complete data preparation
    python reproduce_data_preparation.py
    
    # Specific years for Sentinel-2 mosaics
    python reproduce_data_preparation.py --years 2020 2021 2022
    
    # Skip forest inventory processing
    python reproduce_data_preparation.py --skip-forest-inventory
    
    # Custom processing extent
    python reproduce_data_preparation.py --bbox 40.0 -10.0 44.0 -6.0

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
from shared_utils.data_paths import CentralDataPaths
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
        
        # Check for forest inventory raw data
        forest_inventory_dir = self.data_paths.get_path('forest_inventory')
        if not forest_inventory_dir.exists():
            self.logger.error(f"Forest inventory directory not found: {forest_inventory_dir}")
            self.logger.error("Please place NFI4 data in data/raw/forest_inventory/")
            return False
        
        # Check for NFI database files
        nfi_files = list(forest_inventory_dir.glob("IFN_4_SP/*.accdb"))
        if not nfi_files:
            self.logger.error("No NFI4 database files (.accdb) found")
            self.logger.error("Please ensure IFN_4_SP/ directory contains NFI4 Access databases")
            return False
        
        self.logger.info(f"✅ Found {len(nfi_files)} NFI4 database files")
        
        # Check for MFE forest map data
        mfe_files = list(forest_inventory_dir.glob("MFESpain/*.shp"))
        if not mfe_files:
            self.logger.warning("No MFE forest map files found")
            self.logger.warning("Forest type integration may not work properly")
        else:
            self.logger.info(f"✅ Found {len(mfe_files)} MFE forest map files")
        
        # Check for reference data files
        required_ref_files = [
            "GlobalWoodDensityDatabase.xls",
            "CODIGOS_IFN.csv"
        ]
        
        missing_ref_files = []
        for filename in required_ref_files:
            file_path = forest_inventory_dir / filename
            if not file_path.exists():
                missing_ref_files.append(filename)
        
        if missing_ref_files:
            self.logger.error(f"Missing reference files: {missing_ref_files}")
            return False
        
        self.logger.info("✅ Reference data files validation passed")
        
        # Check for Spain boundary for Sentinel-2 processing
        spain_polygon = self.data_paths.get_path('reference_data') / "SpainPolygon" / "gadm41_ESP_0.shp"
        if not spain_polygon.exists():
            self.logger.error(f"Spain boundary shapefile not found: {spain_polygon}")
            self.logger.error("Please provide Spain boundary in data/raw/reference_data/SpainPolygon/")
            return False
        
        self.logger.info("✅ Spain boundary shapefile found")
        
        # Check internet connectivity for STAC catalog access
        try:
            import socket
            socket.create_connection(("earth-search.aws.element84.com", 443), timeout=10)
            self.logger.info("✅ Internet connectivity for STAC catalog confirmed")
        except Exception:
            self.logger.warning("⚠️  Cannot verify internet connectivity for STAC catalog")
            self.logger.warning("Sentinel-2 processing may fail if no internet access")
        
        # Check for PNOA LiDAR data (for ALS processing)
        pnoa_data_dir = self.data_paths.get_path('raw') / "pnoa_lidar" / "PNOA2_LIDAR_VEGETATION"
        if not pnoa_data_dir.exists():
            self.logger.warning(f"PNOA LiDAR data directory not found: {pnoa_data_dir}")
            self.logger.warning("ALS PNOA processing will be skipped")
        else:
            pnoa_files = list(pnoa_data_dir.glob("NDSM-VEGETACION-*.tif"))
            if pnoa_files:
                self.logger.info(f"✅ Found {len(pnoa_files)} PNOA LiDAR files")
            else:
                self.logger.warning("PNOA LiDAR directory exists but no NDSM files found")
        
        # Check for PNOA coverage polygons
        pnoa_coverage_dir = self.data_paths.get_path('raw') / "pnoa_coverage"
        if not pnoa_coverage_dir.exists():
            self.logger.warning(f"PNOA coverage directory not found: {pnoa_coverage_dir}")
        else:
            coverage_files = list(pnoa_coverage_dir.rglob("*.shp"))
            if coverage_files:
                self.logger.info(f"✅ Found {len(coverage_files)} PNOA coverage files")
            else:
                self.logger.warning("PNOA coverage directory exists but no shapefiles found")
        
        return True
    
    def create_output_structure(self) -> None:
        """Create necessary output directories."""
        self.logger.info("Creating output directory structure...")
        
        # Create main data directories
        self.data_paths.create_directories([
            'forest_inventory',
            'sentinel2_mosaics', 
            'reference_data'
        ])
        
        # Create forest inventory output subdirectories
        forest_out_dir = self.data_paths.get_path('forest_inventory') / "processed"
        forest_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ALS PNOA output directory
        als_output_dir = self.data_paths.get_path('processed') / "training_data_sentinel2_pnoa"
        als_output_dir.mkdir(parents=True, exist_ok=True)
        
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
            # Check if already completed
            forest_out_dir = self.data_paths.get_path('forest_inventory') / "processed"
            existing_shapefiles = list(forest_out_dir.glob("*.shp"))
            
            if existing_shapefiles:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_shapefiles)} shapefiles")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Prepare config overrides for centralized paths
            config_overrides = self.data_paths.get_component_config_overrides('forest_inventory')
            
            # Run forest inventory processing
            # Note: This is a simplified approach - in production you'd want proper config injection
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for NFI processing
                sys.argv = [
                    'run_nfi_processing.py',
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
        
        stage_start = time.time()
        
        try:
            # Check if PNOA data is available
            pnoa_data_dir = self.data_paths.get_path('raw') / "pnoa_lidar" / "PNOA2_LIDAR_VEGETATION"
            pnoa_coverage_dir = self.data_paths.get_path('raw') / "pnoa_coverage"
            
            if not pnoa_data_dir.exists() or not pnoa_coverage_dir.exists():
                self.logger.warning(f"⚠️  {stage_name} - PNOA data not available, skipping")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'skipped_no_pnoa_data'
                }
                return True
            
            # Check if already completed
            als_output_dir = self.data_paths.get_path('processed') / "training_data_sentinel2_pnoa"
            existing_pnoa = list(als_output_dir.glob("PNOA_*.tif")) if als_output_dir.exists() else []
            
            if existing_pnoa:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_pnoa)} PNOA files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Check that Sentinel-2 mosaics exist (prerequisite)
            mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
            mosaic_files = list(mosaic_dir.glob("*.tif")) if mosaic_dir.exists() else []
            
            if not mosaic_files:
                self.logger.error(f"❌ {stage_name} - No Sentinel-2 mosaics found, cannot proceed")
                self.stage_results[stage_name] = {
                    'success': False,
                    'duration_minutes': 0,
                    'error': 'missing_sentinel2_mosaics'
                }
                return False
            
            # Run ALS PNOA processing
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for ALS PNOA processing
                sys.argv = [
                    'run_pnoa_processing.py',
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
    
    def run_sentinel2_processing(self, years: Optional[List[int]] = None,
                                bbox: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Run Sentinel-2 mosaic processing.
        
        Args:
            years: Specific years to process
            bbox: Optional bounding box (south, west, north, east)
            
        Returns:
            bool: True if successful
        """
        stage_name = "Sentinel-2 Mosaic Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {stage_name}")
        self.logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        try:
            # Check if already completed
            mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
            existing_mosaics = list(mosaic_dir.glob("*.tif"))
            
            if existing_mosaics:
                self.logger.info(f"✅ {stage_name} - Found existing outputs: {len(existing_mosaics)} mosaic files")
                self.stage_results[stage_name] = {
                    'success': True,
                    'duration_minutes': 0,
                    'result': 'existing_outputs_used'
                }
                return True
            
            # Prepare config overrides for centralized paths
            config_overrides = self.data_paths.get_component_config_overrides('sentinel2_processing')
            
            # Run Sentinel-2 processing
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # Prepare arguments for Sentinel-2 processing
                sys.argv = ['run_postprocessing.py']
                
                if years:
                    # Note: This assumes the postprocessing script accepts years argument
                    # You may need to adjust based on actual script interface
                    sys.argv.extend(['--years'] + [str(y) for y in years])
                
                if bbox:
                    # Note: Adjust argument format based on actual script interface
                    sys.argv.extend(['--bbox'] + [str(x) for x in bbox])
                
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
    
    def validate_outputs(self) -> bool:
        """
        Validate that expected outputs were created.
        
        Returns:
            bool: True if outputs are valid
        """
        self.logger.info("Validating data preparation outputs...")
        
        # Check forest inventory outputs
        forest_out_dir = self.data_paths.get_path('forest_inventory') / "processed"
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
        
        # Check ALS PNOA outputs (optional)
        als_output_dir = self.data_paths.get_path('processed') / "training_data_sentinel2_pnoa"
        if als_output_dir.exists():
            pnoa_files = list(als_output_dir.glob("PNOA_*.tif"))
            if pnoa_files:
                self.logger.info(f"✅ Found {len(pnoa_files)} ALS PNOA training files")
            else:
                self.logger.info("ℹ️  No ALS PNOA files found (optional)")
        else:
            self.logger.info("ℹ️  ALS PNOA output directory not found (optional)")
        
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
        
        # Check output directories
        forest_out_dir = self.data_paths.get_path('forest_inventory') / "processed"
        mosaic_dir = self.data_paths.get_path('sentinel2_mosaics')
        
        if forest_out_dir.exists():
            forest_files = list(forest_out_dir.glob("*.shp"))
            self.logger.info(f"📁 Forest inventory outputs: {len(forest_files)} files in {forest_out_dir}")
        
        if mosaic_dir.exists():
            mosaic_files = list(mosaic_dir.glob("*.tif"))
            self.logger.info(f"📁 Sentinel-2 mosaics: {len(mosaic_files)} files in {mosaic_dir}")
        
        # Check ALS PNOA outputs
        als_output_dir = self.data_paths.get_path('processed') / "training_data_sentinel2_pnoa"
        if als_output_dir.exists():
            pnoa_files = list(als_output_dir.glob("PNOA_*.tif"))
            self.logger.info(f"📁 ALS PNOA training data: {len(pnoa_files)} files in {als_output_dir}")
        
        # Show data structure
        self.logger.info(f"📂 Data structure created in: {self.data_paths.data_root}")
        self.logger.info(f"   ├── raw/")
        self.logger.info(f"   │   ├── forest_inventory/")
        self.logger.info(f"   │   ├── pnoa_lidar/ (optional)")
        self.logger.info(f"   │   └── reference_data/")
        self.logger.info(f"   └── processed/")
        self.logger.info(f"       ├── sentinel2_mosaics/")
        self.logger.info(f"       └── training_data_sentinel2_pnoa/ (optional)")
        
        self.logger.info(f"\n🎯 Next steps:")
        self.logger.info(f"   1. Run 'reproduce_height_modeling.py' to generate height predictions")
        self.logger.info(f"   2. Check data quality in processed directories")
        self.logger.info(f"   3. Verify Sentinel-2 mosaics cover your area of interest")
        if als_output_dir.exists() and list(als_output_dir.glob("PNOA_*.tif")):
            self.logger.info(f"   4. ALS PNOA training data ready for height model training")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recipe: Data Preparation (Forest Inventory + Sentinel-2 Mosaics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This recipe prepares foundational datasets by executing:
1. Forest inventory processing (NFI4 → biomass shapefiles)
2. Sentinel-2 mosaic creation (STAC catalog → summer mosaics)
3. ALS PNOA processing (LiDAR tiles → training data, optional)

Examples:
  %(prog)s                              # Run complete data preparation
  %(prog)s --years 2020 2021 2022       # Specific years for S2 mosaics
  %(prog)s --skip-forest-inventory      # Skip NFI processing
  %(prog)s --skip-als-pnoa              # Skip ALS PNOA processing
  %(prog)s --bbox 40.0 -10.0 44.0 -6.0  # Custom processing extent
  %(prog)s --data-root /path/to/data    # Custom data directory

Requirements:
  - NFI4 database files in data/raw/forest_inventory/IFN_4_SP/
  - MFE forest maps in data/raw/forest_inventory/MFESpain/
  - Spain boundary in data/raw/reference_data/SpainPolygon/
  - Internet connectivity for Sentinel-2 STAC catalog access
  - PNOA LiDAR data in data/raw/pnoa_lidar/ (optional, for training data)
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
            recipe.logger.error("Please ensure required input data is available")
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