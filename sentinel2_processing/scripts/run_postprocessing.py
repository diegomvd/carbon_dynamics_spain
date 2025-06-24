#!/usr/bin/env python3
"""
Orchestrated Post-processing Script

Command-line interface for running multiple post-processing workflows
in sequence or individually. Provides flexible execution of downsampling,
merging, and analysis operations with comprehensive workflow management.

Usage Examples:
    # Run all workflows
    python scripts/run_postprocessing.py

Author: Diego Bengochea
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config

# Component imports
from sentinel2_processing.core.postprocessing import (
    DownsamplingMergingProcessor, MissingTilesAnalyzer,
    RobustnessAssessor, InterannualConsistencyAnalyzer
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Orchestrated Post-processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    if args.config and not Path(args.config).exists():
        print(f"Error: Configuration file does not exist: {args.config}")
        return False
    
    return True



def run_downsampling_workflow(config: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """Run downsampling and merging workflow."""
    logger = get_logger('sentinel2_processing')
    logger.info("RUNNING DOWNSAMPLING AND MERGING WORKFLOW")
    
    processor = DownsamplingMergingProcessor(config)
    return processor.run_complete_workflow()


def run_missing_analysis_workflow(config: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """Run missing tiles analysis workflow."""
    logger = get_logger('sentinel2_processing')
    logger.info("RUNNING MISSING TILES ANALYSIS")
    
    analyzer = MissingTilesAnalyzer(config)
    return analyzer.run_missing_analysis()


def run_robustness_workflow(config: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """Run robustness assessment workflow."""
    logger = get_logger('sentinel2_processing')
    logger.info("RUNNING ROBUSTNESS ASSESSMENT")
    
    assessor = RobustnessAssessor(config)
    
    # Set up robustness parameters
    bbox =  None
    start_date = None
    end_date = None
    
    return assessor.run_robustness_assessment(bbox, start_date, end_date)


def run_consistency_workflow(config: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """Run interannual consistency analysis workflow."""
    logger = get_logger('sentinel2_processing')
    logger.info("RUNNING INTERANNUAL CONSISTENCY ANALYSIS")
    
    analyzer = InterannualConsistencyAnalyzer(config)
    return analyzer.run_consistency_analysis()


def main() -> int:
    """
    Main entry point for orchestrated post-processing script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Setup logging
        setup_logging(level='INFO', log_file='sentinel2_full_postprocessing')
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)

        
        # Determine workflows to run
        workflows = ['missing','downsampling','consistency','robustness']
        
        # Track execution
        start_time = time.time()
        results = {}
        failed_workflows = []
        
        # Workflow execution functions
        workflow_functions = {
            'downsampling': run_downsampling_workflow,
            'missing': run_missing_analysis_workflow,
            'robustness': run_robustness_workflow,
            'consistency': run_consistency_workflow
        }
        
        # Execute workflows
        for workflow in workflows:
            logger.info(f"\n{'='*60}")
            logger.info(f"EXECUTING WORKFLOW: {workflow.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                workflow_func = workflow_functions[workflow]
                result = workflow_func(config, args)
                results[workflow] = result
                
                if result.get('success', False):
                    logger.info(f"{workflow.upper()} workflow completed successfully")
                else:
                    logger.error(f"{workflow.upper()} workflow failed: {result.get('error', 'Unknown error')}")
                    failed_workflows.append(workflow)
                
                    logger.error("Stopping execution due to workflow failure")
                    break
                    
            except Exception as e:
                logger.error(f"{workflow.upper()} workflow failed with exception: {str(e)}")
                failed_workflows.append(workflow)
                results[workflow] = {'success': False, 'error': str(e)}
                
                logger.error("Stopping execution due to workflow exception")
                break
    
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("POST-PROCESSING EXECUTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Duration: {duration/60:.1f} minutes")
        logger.info(f"Workflows Executed: {len(results)}/{len(workflows)}")
        
        for workflow, result in results.items():
            status = "SUCCESS" if result.get('success', False) else "FAILED"
            logger.info(f"{workflow.upper()}: {status}")
        
        if failed_workflows:
            logger.error(f"Failed workflows: {', '.join(failed_workflows)}")
            logger.info(f"{'='*60}")
            return 1
        else:
            logger.info("All workflows completed successfully!")
            logger.info(f"{'='*60}")
            return 0
        
    except Exception as e:
        logger.error(f"Post-processing orchestration failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())