#!/usr/bin/env python3
"""
Orchestrated Post-processing Script

Command-line interface for running multiple post-processing workflows
in sequence or individually. Provides flexible execution of downsampling,
merging, and analysis operations with comprehensive workflow management.

Usage Examples:
    # Run all workflows
    python scripts/run_postprocessing.py
    
    # Run specific workflows
    python scripts/run_postprocessing.py --workflows downsampling missing
    
    # Skip analysis workflows
    python scripts/run_postprocessing.py --skip-analysis
    
    # Custom scale factor and configuration
    python scripts/run_postprocessing.py --scale-factor 5 --config custom.yaml
    
    # Run only data processing (no analysis)
    python scripts/run_postprocessing.py --data-only
    
    # Run only analysis workflows
    python scripts/run_postprocessing.py --analysis-only

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
        epilog="""
Examples:
  %(prog)s                                    # Run all workflows
  %(prog)s --workflows downsampling missing  # Run specific workflows
  %(prog)s --skip-analysis                   # Skip analysis workflows
  %(prog)s --data-only                       # Only data processing
  %(prog)s --analysis-only                   # Only analysis workflows
  %(prog)s --config custom.yaml              # Use custom configuration
  %(prog)s --scale-factor 5 --continue-on-error  # Custom params with error handling

Available workflows: downsampling, missing, robustness, consistency

For more information, see the component documentation.
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Workflow selection
    parser.add_argument(
        '--workflows',
        nargs='+',
        choices=['downsampling', 'missing', 'robustness', 'consistency', 'all'],
        default=['all'],
        help='Workflows to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis workflows (robustness, consistency)'
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Run only data processing workflows (downsampling, missing)'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Run only analysis workflows (robustness, consistency)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--scale-factor',
        type=int,
        help='Downsampling scale factor (overrides config)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Sample size for consistency analysis (overrides config)'
    )
    
    # Directory overrides
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory for processing (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Base output directory (overrides config)'
    )
    
    # Execution options
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue with remaining workflows if one fails'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running workflows'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary information for each workflow'
    )
    
    # Robustness assessment options
    parser.add_argument(
        '--robustness-bbox',
        type=float,
        nargs=4,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help='Bounding box for robustness assessment'
    )
    
    parser.add_argument(
        '--robustness-year',
        type=int,
        help='Year for robustness assessment'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: console only)'
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
    
    if args.input_dir and not Path(args.input_dir).exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return False
    
    if args.output_dir and not Path(args.output_dir).parent.exists():
        print(f"Error: Output directory parent does not exist: {args.output_dir}")
        return False
    
    # Check for conflicting workflow options
    exclusive_options = [args.skip_analysis, args.data_only, args.analysis_only]
    if sum(exclusive_options) > 1:
        print("Error: Cannot specify multiple exclusive workflow options")
        return False
    
    if args.robustness_bbox and len(args.robustness_bbox) != 4:
        print("Error: Robustness bounding box must have exactly 4 coordinates")
        return False
    
    if args.sample_size and args.sample_size < 100:
        print("Error: Sample size must be at least 100")
        return False
    
    if args.log_file and not Path(args.log_file).parent.exists():
        print(f"Error: Log file directory does not exist: {Path(args.log_file).parent}")
        return False
    
    return True


def apply_config_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Apply command-line argument overrides to configuration.
    
    Args:
        config: Configuration dictionary
        args: Parsed arguments
        
    Returns:
        dict: Updated configuration
    """
    if args.scale_factor:
        config.setdefault('postprocessing', {}).setdefault('downsample', {})['scale_factor'] = args.scale_factor
    
    if args.sample_size:
        config.setdefault('consistency', {})['sample_size'] = args.sample_size
    
    if args.input_dir:
        config.setdefault('paths', {})['output_dir'] = args.input_dir
    
    if args.output_dir:
        # Update all relevant output directories
        config.setdefault('paths', {})['downsampled_dir'] = str(Path(args.output_dir) / 'downsampled')
        config.setdefault('paths', {})['merged_dir'] = str(Path(args.output_dir) / 'merged')
    
    return config


def determine_workflows(args: argparse.Namespace) -> List[str]:
    """
    Determine which workflows to run based on arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        List[str]: List of workflow names to execute
    """
    if args.analysis_only:
        return ['robustness', 'consistency']
    
    if args.data_only:
        return ['downsampling', 'missing']
    
    # Start with specified workflows
    workflows = args.workflows.copy()
    
    # Expand 'all' to specific workflows
    if 'all' in workflows:
        workflows = ['downsampling', 'missing', 'robustness', 'consistency']
    
    # Apply skip analysis filter
    if args.skip_analysis:
        workflows = [w for w in workflows if w not in ['robustness', 'consistency']]
    
    return workflows


def print_workflow_plan(workflows: List[str], config: dict, args: argparse.Namespace) -> None:
    """
    Print the execution plan for workflows.
    
    Args:
        workflows: List of workflow names
        config: Configuration dictionary
        args: Parsed arguments
    """
    print("\n" + "="*60)
    print("POST-PROCESSING EXECUTION PLAN")
    print("="*60)
    print(f"Workflows to execute: {', '.join(workflows)}")
    print(f"Continue on error: {'Yes' if args.continue_on_error else 'No'}")
    print(f"Summary only: {'Yes' if args.summary_only else 'No'}")
    
    if args.scale_factor:
        print(f"Scale factor: {args.scale_factor}")
    
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    
    if args.robustness_bbox:
        print(f"Robustness bbox: {args.robustness_bbox}")
    
    if args.robustness_year:
        print(f"Robustness year: {args.robustness_year}")
    
    print("="*60)


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
    bbox = args.robustness_bbox if args.robustness_bbox else None
    
    if args.robustness_year:
        from datetime import datetime
        start_date = datetime(args.robustness_year, 1, 1)
        end_date = datetime(args.robustness_year, 12, 31)
    else:
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
        setup_logging(level=args.log_level, log_file=args.log_file)
        logger = get_logger('sentinel2_processing')
        
        # Load and override configuration
        config = load_config(args.config)
        config = apply_config_overrides(config, args)
        
        # Determine workflows to run
        workflows = determine_workflows(args)
        
        if not workflows:
            logger.error("No workflows selected for execution")
            return 1
        
        # Print execution plan
        print_workflow_plan(workflows, config, args)
        
        # Dry run - just show plan
        if args.dry_run:
            print("\nDry run completed - no workflows executed")
            return 0
        
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
                    
                    if not args.continue_on_error:
                        logger.error("Stopping execution due to workflow failure")
                        break
                        
            except Exception as e:
                logger.error(f"{workflow.upper()} workflow failed with exception: {str(e)}")
                failed_workflows.append(workflow)
                results[workflow] = {'success': False, 'error': str(e)}
                
                if not args.continue_on_error:
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