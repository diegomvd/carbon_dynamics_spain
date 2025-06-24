#!/usr/bin/env python3
"""
Complete climate-biomass analysis pipeline orchestrator.

Integrates climate data processing, bioclimatic variable calculation, biomass
integration, spatial analysis, and machine learning optimization into a single
coordinated pipeline with smart checkpointing and error recovery.

Usage:
    python run_full_pipeline.py

Author: Diego Bengochea
"""

import argparse
import sys
import time
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from climate_biomass_analysis.core.climate_raster_processing import ClimateProcessor
from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculator
from climate_biomass_analysis.core.biomass_integration import BiomassIntegrator
from climate_biomass_analysis.core.spatial_analysis import SpatialAnalyzer
from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline
from shared_utils import setup_logging, load_config
from shared_utils.central_data_paths_constants import *


class PipelineOrchestrator:
    """
    Complete climate-biomass analysis pipeline orchestrator.
    
    Manages the execution of all pipeline stages with checkpointing,
    error recovery, and progress monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='pipeline_orchestrator',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Initialize pipeline components
        self.climate_processor = None
        self.bioclim_calculator = None
        self.biomass_integrator = None
        self.spatial_analyzer = None
        self.optimizer = None
        
        # Pipeline state
        self.stage_results = {}
        self.start_time = None
        
        self.logger.info("Initialized PipelineOrchestrator")
    
    def check_stage_completion(self, stage_name: str, check_paths: List[str], 
                              check_patterns: Optional[List[str]] = None) -> bool:
        """
        Check if a pipeline stage has already been completed.
        
        Args:
            stage_name: Name of the pipeline stage
            check_paths: Paths to check for completion
            check_patterns: Optional file patterns to check
            
        Returns:
            True if stage is completed
        """
        for path in check_paths:
            if not os.path.exists(path):
                return False
            
            # For directories, check if they contain expected files
            if os.path.isdir(path) and check_patterns:
                for pattern in check_patterns:
                    files = glob.glob(os.path.join(path, pattern))
                    if not files:
                        return False
        
        self.logger.info(f"{stage_name}: ✅ Already completed")
        return True
    
    def stage_1_climate_processing(self) -> bool:
        """
        Stage 1: Climate data processing (GRIB to GeoTIFF conversion).
        
        Returns:
            True if successful
        """
        stage_name = "Stage 1: Climate Processing"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if stage already completed
        climate_outputs = CLIMATE_HARMONIZED_DIR
        check_paths = [climate_outputs]
        check_patterns = ["*.tif"]
        
        if self.check_stage_completion(stage_name, check_paths, check_patterns):
            return True
        
        # Initialize processor if needed
        if self.climate_processor is None:
            self.climate_processor = ClimateProcessor(config_path=None)
        
        try:
            self.logger.info(f"{stage_name}: Starting climate data processing...")
            start_time = time.time()
            
            # This would need to be implemented based on specific requirements
            # For now, assume climate data is already processed or skip this stage
            self.logger.info(f"{stage_name}: ⚠️  Skipping - implement based on data source requirements")
            
            stage_time = time.time() - start_time
            self.logger.info(f"{stage_name}: Completed in {stage_time/60:.2f} minutes")
            
            self.stage_results['climate_processing'] = {
                'completed': True,
                'duration_minutes': stage_time / 60
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
            return False
    
    def stage_2_bioclim_calculation(self) -> bool:
        """
        Stage 2: Bioclimatic variables calculation.
        
        Returns:
            True if successful
        """
        stage_name = "Stage 2: Bioclimatic Calculation"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if stage already completed
        bioclim_dir = BIOCLIM_VARIABLES_DIR
        anomaly_dir = BIOCLIM_ANOMALIES_DIR
        check_paths = [bioclim_dir, anomaly_dir]
        check_patterns = ["bio*.tif", "anomalies_*"]
        
        if self.check_stage_completion(stage_name, check_paths, check_patterns):
            return True
        
        # Initialize calculator if needed
        if self.bioclim_calculator is None:
            self.bioclim_calculator = BioclimCalculator(config_path=None)
        
        try:
            self.logger.info(f"{stage_name}: Starting bioclimatic variables calculation...")
            start_time = time.time()
            
            # Run bioclimatic calculation pipeline
            results = self.bioclim_calculator.run_bioclim_pipeline()
            
            if results is not None:
                stage_time = time.time() - start_time
                self.logger.info(f"{stage_name}: ✅ Completed successfully")
                self.logger.info(f"{stage_name}: Completed in {stage_time/60:.2f} minutes")
                
                self.stage_results['bioclim_calculation'] = {
                    'completed': True,
                    'duration_minutes': stage_time / 60,
                    'results': results
                }
                
                return True
            else:
                self.logger.error(f"{stage_name}: ❌ Failed - no results returned")
                return False
                
        except Exception as e:
            self.logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
            return False
    
    def stage_3_biomass_integration(self) -> bool:
        """
        Stage 3: Biomass-climate data integration.
        
        Returns:
            True if successful
        """
        stage_name = "Stage 3: Biomass Integration"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if stage already completed
        training_dataset = CLIMATE_BIOMASS_DATASET_FILE
        check_paths = [training_dataset]
        
        if self.check_stage_completion(stage_name, check_paths):
            return True
        
        # Initialize integrator if needed
        if self.biomass_integrator is None:
            self.biomass_integrator = BiomassIntegrator(config_path=None)
        
        try:
            self.logger.info(f"{stage_name}: Starting biomass-climate integration...")
            start_time = time.time()
            
            # Run biomass integration pipeline
            dataset = self.biomass_integrator.run_biomass_integration_pipeline()
            
            if dataset is not None and len(dataset) > 0:
                stage_time = time.time() - start_time
                self.logger.info(f"{stage_name}: ✅ Completed successfully")
                self.logger.info(f"  - Created dataset with {len(dataset)} data points")
                self.logger.info(f"  - Features: {len(dataset.columns)} columns")
                self.logger.info(f"{stage_name}: Completed in {stage_time/60:.2f} minutes")
                
                self.stage_results['biomass_integration'] = {
                    'completed': True,
                    'duration_minutes': stage_time / 60,
                    'n_data_points': len(dataset),
                    'n_features': len(dataset.columns)
                }
                
                return True
            else:
                self.logger.error(f"{stage_name}: ❌ Failed - no data points created")
                return False
                
        except Exception as e:
            self.logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
            return False
    
    def stage_4_spatial_analysis(self) -> bool:
        """
        Stage 4: Spatial analysis and clustering.
        
        Returns:
            True if successful
        """
        stage_name = "Stage 4: Spatial Analysis"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if stage already completed
        clustered_dataset = CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE
        check_paths = [clustered_dataset]
        
        if self.check_stage_completion(stage_name, check_paths):
            return True
        
        # Initialize analyzer if needed
        if self.spatial_analyzer is None:
            self.spatial_analyzer = SpatialAnalyzer(config_path=None)
        
        try:
            self.logger.info(f"{stage_name}: Starting spatial analysis...")
            start_time = time.time()
            
            # Run spatial analysis pipeline (returns DataFrame with clusters)
            df_clustered = self.spatial_analyzer.run_spatial_analysis_pipeline()
            
            if df_clustered is not None and 'cluster_id' in df_clustered.columns:
                stage_time = time.time() - start_time
                n_clusters = df_clustered['cluster_id'].nunique()
                
                self.logger.info(f"{stage_name}: ✅ Completed successfully")
                self.logger.info(f"  - Created {n_clusters} spatial clusters")
                self.logger.info(f"  - Dataset shape: {df_clustered.shape}")
                self.logger.info(f"{stage_name}: Completed in {stage_time/60:.2f} minutes")
                
                self.stage_results['spatial_analysis'] = {
                    'completed': True,
                    'duration_minutes': stage_time / 60,
                    'n_clusters': n_clusters,
                    'dataset_shape': df_clustered.shape
                }
                
                return True
            else:
                self.logger.error(f"{stage_name}: ❌ Failed - no clustering results or cluster_id column")
                return False
                
        except Exception as e:
            self.logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
            return False
    
    def stage_5_optimization(self) -> bool:
        """
        Stage 5: Machine learning optimization.
        
        Returns:
            True if successful
        """
        stage_name = "Stage 5: ML Optimization"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"{stage_name}")
        self.logger.info(f"{'='*60}")
        
        # Check if stage already completed
        optimization_dir = CLIMATE_BIOMASS_MODELS_DIR
        check_paths = [optimization_dir]
        check_patterns = ["optimization_summary.pkl", "individual_run_results.pkl"]
        
        if self.check_stage_completion(stage_name, check_paths, check_patterns):
            return True
        
        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = OptimizationPipeline(config_path=None)
        
        try:
            self.logger.info(f"{stage_name}: Starting machine learning optimization...")
            start_time = time.time()
            
            # Run optimization pipeline
            results = self.optimizer.run_optimization_pipeline()
            
            if results and 'summary' in results:
                stage_time = time.time() - start_time
                summary = results['summary']
                
                self.logger.info(f"{stage_name}: ✅ Completed successfully")
                self.logger.info(f"  - Validation R²: {summary['validation_r2']['mean']:.4f} ± {summary['validation_r2']['std']:.4f}")
                self.logger.info(f"  - Test R²: {summary['test_r2']['mean']:.4f} ± {summary['test_r2']['std']:.4f}")
                self.logger.info(f"  - Top features: {', '.join(summary['feature_analysis']['top_features'][:3])}")
                self.logger.info(f"{stage_name}: Completed in {stage_time/60:.2f} minutes")
                
                self.stage_results['optimization'] = {
                    'completed': True,
                    'duration_minutes': stage_time / 60,
                    'validation_r2_mean': summary['validation_r2']['mean'],
                    'test_r2_mean': summary['test_r2']['mean'],
                    'results': results
                }
                
                return True
            else:
                self.logger.error(f"{stage_name}: ❌ Failed - no optimization results")
                return False
                
        except Exception as e:
            self.logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
            return False
    
    def run_full_pipeline(self, stages: Optional[List[str]] = None, 
                         continue_on_error: bool = False) -> Dict[str, any]:
        """
        Run the complete climate-biomass analysis pipeline.
        
        Args:
            stages: Specific stages to run (None = all stages)
            continue_on_error: Continue pipeline even if a stage fails
            
        Returns:
            Dictionary with pipeline results
        """
        self.start_time = time.time()
        self.logger.info("🚀 Starting complete climate-biomass analysis pipeline...")
        
        # Define pipeline stages
        all_stages = [
            ('climate_processing', self.stage_1_climate_processing),
            ('bioclim_calculation', self.stage_2_bioclim_calculation),
            ('biomass_integration', self.stage_3_biomass_integration),
            ('spatial_analysis', self.stage_4_spatial_analysis),
            ('optimization', self.stage_5_optimization)
        ]
        
        # Filter stages if specified
        if stages:
            pipeline_stages = [(name, func) for name, func in all_stages if name in stages]
        else:
            pipeline_stages = all_stages
        
        # Execute stages
        completed_stages = []
        failed_stages = []
        
        for stage_name, stage_func in pipeline_stages:
            try:
                success = stage_func()
                
                if success:
                    completed_stages.append(stage_name)
                else:
                    failed_stages.append(stage_name)
                    if not continue_on_error:
                        break
                        
            except Exception as e:
                self.logger.error(f"Unexpected error in {stage_name}: {e}")
                failed_stages.append(stage_name)
                if not continue_on_error:
                    break
        
        # Pipeline summary
        total_time = time.time() - self.start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total execution time: {total_time/60:.2f} minutes")
        self.logger.info(f"Completed stages: {len(completed_stages)}/{len(pipeline_stages)}")
        
        if completed_stages:
            self.logger.info(f"✅ Successful stages: {', '.join(completed_stages)}")
        
        if failed_stages:
            self.logger.error(f"❌ Failed stages: {', '.join(failed_stages)}")
        
        # Detailed stage timings
        for stage_name, stage_data in self.stage_results.items():
            if stage_data.get('completed'):
                duration = stage_data.get('duration_minutes', 0)
                self.logger.info(f"  {stage_name}: {duration:.2f} minutes")
        
        pipeline_results = {
            'completed_stages': completed_stages,
            'failed_stages': failed_stages,
            'total_duration_minutes': total_time / 60,
            'stage_results': self.stage_results,
            'success': len(failed_stages) == 0
        }
        
        if pipeline_results['success']:
            self.logger.info("🎉 Pipeline completed successfully!")
        else:
            self.logger.error("💥 Pipeline completed with errors")
        
        return pipeline_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete climate-biomass analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    
    # Stage selection
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['climate_processing', 'bioclim_calculation', 'biomass_integration', 
                'spatial_analysis', 'optimization'],
        help='Specific stages to run (default: all stages)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for complete pipeline script."""
    args = parse_arguments()

    log_level = 'INFO'

    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config_path=args.config, log_level=log_level)
        
        # Determine stages to run
        if args.stages:
            stages_to_run = args.stages
        else:
            all_stages = ['climate_processing', 'bioclim_calculation', 'biomass_integration', 
                         'spatial_analysis', 'optimization']
            stages_to_run = all_stages
        
        orchestrator.logger.info(f"Pipeline stages to execute: {', '.join(stages_to_run)}")
        
        # Run pipeline
        results = orchestrator.run_full_pipeline(
            stages=stages_to_run
        )
        
        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except Exception as e:
        print(f"Pipeline orchestrator failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()