#!/usr/bin/env python3
"""
Canopy Height Model Evaluation Script

Entry point for comprehensive evaluation of trained canopy height regression models.
Uses the ModelEvaluationPipeline from the core module to generate statistical analysis,
residual plots, prediction density visualizations, and height distribution comparisons.

Usage:
    python run_evaluation.py
    
Examples:
    # Evaluate with default configuration
    python run_evaluation.py
    
    # Custom configuration and checkpoint
    python run_evaluation.py --config custom_config.yaml --checkpoint model.ckpt

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Union

# Component imports
from canopy_height_model.core.model_evaluation import ModelEvaluationPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: use component config)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        help='Path(s) to model checkpoint(s) for evaluation'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the evaluation pipeline."""
    
    args = parse_arguments()
    model_evaluation = ModelEvaluationPipeline(args.config)
    success = model_evaluation.run_full_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)