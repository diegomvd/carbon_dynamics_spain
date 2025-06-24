#!/usr/bin/env python3
"""
Complete biomass analysis pipeline orchestrator.

Integrates all biomass analysis components into a single coordinated pipeline
with smart stage detection, error recovery, and progress monitoring.

Usage:
    python run_full_analysis.py 

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_analysis.core.monte_carlo_analysis import MonteCarloAnalyzer
from biomass_analysis.core.aggregation_analysis import BiomassAggregator
from biomass_analysis.core.interannual_analysis import InterannualAnalyzer
from biomass_analysis.core.carbon_flux_analysis import CarbonFluxAnalyzer
from shared_utils import setup_logging


class AnalysisPipelineOrchestrator:
    """
    Complete biomass analysis pipeline orchestrator.
    
    Manages the execution of all pipeline stages with progress monitoring
    and error recovery.
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        # Setup logging
        self.logger = setup_logging(
            level=log_level,
            component_name='pipeline_orchestrator'
        )
        
        # Initialize analyzers
        try:
            self.monte_carlo = MonteCarloAnalyzer(config_path)
            self.aggregator = BiomassAggregator(config_path)
            self.interannual = InterannualAnalyzer(config_path)
            self.carbon_flux = CarbonFluxAnalyzer(config_path)
            
            # Use config from one of the analyzers
            self.config = self.monte_carlo.config
            
        except Exception as e:
            self.logger.error(f"Error initializing analyzers: {e}")
            raise
        
        # Pipeline state
        self.stage_results = {}
        self.start_time = None
        
        self.logger.info("Initialized AnalysisPipelineOrchestrator")
    
    def run_stage(self, stage_name: str, stage_func, *args, **kwargs) -> bool:
        """
        Run a pipeline stage with error handling and timing.
        
        Args:
            stage_name: Name of the stage
            stage_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STAGE: {stage_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        try:
            result = stage_func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            
            if result is not None:
                self.stage_results[stage_name] = {
                    'success': True,
                    'result': result,
                    'elapsed_time': elapsed
                }
                self.logger.info(f"✅ {stage_name} completed successfully in {elapsed:.1f}s")
                return True
            else:
                self.stage_results[stage_name] = {
                    'success': False,
                    'result': None,
                    'elapsed_time': elapsed
                }
                self.logger.error(f"❌ {stage_name} failed")
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            self.stage_results[stage_name] = {
                'success': False,
                'result': None,
                'elapsed_time': elapsed,
                'error': str(e)
            }
            self.logger.error(f"❌ {stage_name} failed with error: {e}")
            return False
    
    def run_country_analysis(self, biomass_types: Optional[List[str]] = None, years: Optional[List[int]] = None) -> bool:
        """Run country-level Monte Carlo analysis."""
        def _run():
            results, samples = self.monte_carlo.run_country_analysis(biomass_types, years)
            if results:
                # Save results
                summary_file, samples_file = self.monte_carlo.save_results(results, samples)
                return {'results': results, 'samples': samples, 'files': (summary_file, samples_file)}
            return None
        
        return self.run_stage("country_analysis", _run)
    
    def run_forest_type_analysis(self, years: Optional[List[int]] = None) -> bool:
        """Run forest type hierarchical analysis."""
        def _run():
            forest_df, genus_df, clade_df = self.aggregator.run_forest_type_analysis(years)
            if forest_df is not None:
                # Save results
                results_data = {'forest_type': forest_df, 'genus': genus_df, 'clade': clade_df}
                output_file = self.aggregator.save_results(results_data, 'forest_type')
                return {'forest_type': forest_df, 'genus': genus_df, 'clade': clade_df, 'file': output_file}
            return None
        
        return self.run_stage("forest_type_analysis", _run)
    
    def run_landcover_analysis(self, years: Optional[List[int]] = None) -> bool:
        """Run landcover analysis."""
        def _run():
            results = self.aggregator.run_landcover_analysis(years)
            if results:
                # Save results
                results_data = {'results': results}
                output_file = self.aggregator.save_results(results_data, 'landcover')
                return {'results': results, 'file': output_file}
            return None
        
        return self.run_stage("landcover_analysis", _run)
    
    def run_height_bin_analysis(self, years: Optional[List[int]] = None, skip_mask_creation: bool = False) -> bool:
        """Run height bin analysis."""
        def _run():
            results = self.aggregator.run_height_bin_analysis(years, skip_mask_creation)
            if results:
                # Save results
                results_data = {'results': results}
                output_file = self.aggregator.save_results(results_data, 'height_bin')
                return {'results': results, 'file': output_file}
            return None
        
        return self.run_stage("height_bin_analysis", _run)
    
    def run_difference_mapping(self, years: Optional[List[int]] = None) -> bool:
        """Run interannual difference mapping."""
        def _run():
            success = self.interannual.run_difference_mapping(years)
            return success if success else None
        
        return self.run_stage("difference_mapping", _run)
    
    def run_transition_analysis(self, years: Optional[List[int]] = None, save_raw_data: bool = False) -> bool:
        """Run transition distribution analysis."""
        def _run():
            results = self.interannual.run_transition_analysis(years, save_raw_data)
            if results:
                # Save results
                output_file = self.interannual.save_results(results, 'transitions')
                return {'results': results, 'file': output_file}
            return None
        
        return self.run_stage("transition_analysis", _run)
    
    def run_carbon_flux_analysis(self, create_diagnostics: bool = True) -> bool:
        """Run carbon flux analysis."""
        def _run():
            flux_df, flux_samples = self.carbon_flux.run_carbon_flux_analysis(create_diagnostics=create_diagnostics)
            if flux_df is not None:
                # Save results
                output_file = self.carbon_flux.save_results(flux_df, flux_samples)
                return {'flux_df': flux_df, 'flux_samples': flux_samples, 'file': output_file}
            return None
        
        return self.run_stage("carbon_flux_analysis", _run)
    
    def run_full_pipeline(self, stages: Optional[List[str]] = None, continue_on_error: bool = False, **stage_kwargs) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            stages: Optional list of specific stages to run
            continue_on_error: Whether to continue if a stage fails
            **stage_kwargs: Stage-specific keyword arguments
            
        Returns:
            dict: Pipeline execution results
        """
        self.start_time = time.time()
        
        # Define all available stages
        all_stages = [
            'country_analysis',
            'forest_type_analysis', 
            'landcover_analysis',
            'height_bin_analysis',
            'difference_mapping',
            'transition_analysis',
            'carbon_flux_analysis'
        ]
        
        # Use specified stages or all stages
        stages_to_run = stages if stages else all_stages
        
        self.logger.info(f"Starting biomass analysis pipeline")
        self.logger.info(f"Stages to run: {', '.join(stages_to_run)}")
        self.logger.info(f"Continue on error: {continue_on_error}")
        
        # Extract common parameters
        years = stage_kwargs.get('years')
        biomass_types = stage_kwargs.get('biomass_types')
        
        # Run each stage
        for stage in stages_to_run:
            if stage == 'country_analysis':
                success = self.run_country_analysis(biomass_types, years)
            elif stage == 'forest_type_analysis':
                success = self.run_forest_type_analysis(years)
            elif stage == 'landcover_analysis':
                success = self.run_landcover_analysis(years)
            elif stage == 'height_bin_analysis':
                skip_masks = stage_kwargs.get('skip_mask_creation', False)
                success = self.run_height_bin_analysis(years, skip_masks)
            elif stage == 'difference_mapping':
                success = self.run_difference_mapping(years)
            elif stage == 'transition_analysis':
                save_raw = stage_kwargs.get('save_raw_data', False)
                success = self.run_transition_analysis(years, save_raw)
            elif stage == 'carbon_flux_analysis':
                create_plots = stage_kwargs.get('create_diagnostics', True)
                success = self.run_carbon_flux_analysis(create_plots)
            else:
                self.logger.warning(f"Unknown stage: {stage}")
                continue
            
            if not success and not continue_on_error:
                self.logger.error(f"Pipeline stopped due to failure in {stage}")
                break
        
        return self.stage_results
    
    def print_pipeline_summary(self):
        """Print a summary of pipeline execution."""
        if not self.stage_results:
            self.logger.info("No pipeline results to summarize")
            return
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total pipeline time: {total_time/60:.1f} minutes")
        
        successful_stages = [k for k, v in self.stage_results.items() if v['success']]
        failed_stages = [k for k, v in self.stage_results.items() if not v['success']]
        
        self.logger.info(f"Successful stages: {len(successful_stages)}/{len(self.stage_results)}")
        
        # Stage details
        for stage_name, result in self.stage_results.items():
            status = "✅" if result['success'] else "❌"
            elapsed = result['elapsed_time']
            self.logger.info(f"  {status} {stage_name}: {elapsed:.1f}s")
            
            if not result['success'] and 'error' in result:
                self.logger.info(f"      Error: {result['error']}")
        
        if successful_stages:
            self.logger.info(f"\nSuccessful stages: {', '.join(successful_stages)}")
        
        if failed_stages:
            self.logger.info(f"Failed stages: {', '.join(failed_stages)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete biomass analysis pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: component config.yaml)'
    )
    


def main():
    """Main entry point for full analysis pipeline."""
    args = parse_arguments()
    
    # Set logging level
    log_level = 'INFO'
    
    # Initialize pipeline orchestrator
    try:
        orchestrator = AnalysisPipelineOrchestrator(args.config, log_level)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Prepare stage kwargs
    stage_kwargs = {
        'years': args.years,
        'biomass_types': args.biomass_types,
        'skip_mask_creation': args.skip_mask_creation,
        'save_raw_data': args.save_raw_data,
        'create_diagnostics': not args.no_diagnostics
    }
    
    # Run pipeline
    try:
        results = orchestrator.run_full_pipeline(
            stages=args.stages,
            continue_on_error=args.continue_on_error,
            **stage_kwargs
        )
        
        # Print summary
        orchestrator.print_pipeline_summary()
        
        # Determine exit code
        failed_stages = [k for k, v in results.items() if not v['success']]
        if failed_stages and not args.continue_on_error:
            orchestrator.logger.error("Pipeline failed due to stage failures")
            sys.exit(1)
        elif failed_stages:
            orchestrator.logger.warning("Pipeline completed with some stage failures")
        else:
            orchestrator.logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
