#!/usr/bin/env python3
"""
Canopy Height Model Training Script

Main entry point for training canopy height estimation models using
PyTorch Lightning and TorchGeo.

Usage:
    python run_training.py
    
Examples:
    # Train with default configuration
    python run_training.py
    
    # Custom configuration
    python run_training.py --config custom_config.yaml
    
    # Resume from checkpoint
    python run_training.py --resume /path/to/checkpoint.ckpt

Author: Diego Bengochea
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Literal

# Add repo root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Component imports
from canopy_height_model.core.model_training import ModelTrainingPipeline
from shared_utils import setup_logging, log_pipeline_start, log_pipeline_end


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Canopy Height Deep Learning Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    return parser.parse_args()

def apply_argument_overrides(pipeline: ModelTrainingPipeline, args: argparse.Namespace) -> None:
    """Apply command line argument overrides to pipeline configuration."""
    config = pipeline.config
    
    if args.fast_dev_run:
        config['training']['max_epochs'] = 1
        pipeline.logger.info("Fast dev run enabled")


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    model_training = ModelTrainingPipeline(args.config,args.resume)
    success = model_training.run()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)