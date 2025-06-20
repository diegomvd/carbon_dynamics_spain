#!/usr/bin/env python3
"""
Canopy Height Model Training Script

Main entry point for training canopy height estimation models using
PyTorch Lightning and TorchGeo.

Usage:
    python run_training.py [OPTIONS]
    
Examples:
    # Train with default configuration
    python run_training.py
    
    # Custom configuration
    python run_training.py --config custom_config.yaml
    
    # Training and testing
    python run_training.py --mode train_test
    
    # Resume from checkpoint
    python run_training.py --resume /path/to/checkpoint.ckpt
    
    # Test only mode
    python run_training.py --mode test --checkpoint /path/to/checkpoint.ckpt

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
        epilog="""
Examples:
  %(prog)s                                    # Train with default config
  %(prog)s --config custom.yaml              # Custom configuration
  %(prog)s --mode train_test                 # Train and test
  %(prog)s --mode test --checkpoint model.ckpt  # Test only
  %(prog)s --resume checkpoint.ckpt          # Resume training
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'train_test'],
        default='train_test',
        help='Training mode (default: train_test)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for testing or resuming'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--max-epochs',
        type=int,
        help='Override maximum training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--accelerator',
        choices=['auto', 'gpu', 'cpu', 'mps'],
        help='Override accelerator type'
    )
    
    parser.add_argument(
        '--devices',
        type=int,
        help='Number of devices to use'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run fast development run (single batch)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without training'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if args.mode == 'test' and not args.checkpoint:
        print("Error: --checkpoint required for test mode")
        return False
    
    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return False
    
    if args.resume and not Path(args.resume).exists():
        print(f"Error: Resume checkpoint not found: {args.resume}")
        return False
    
    return True


def apply_argument_overrides(pipeline: ModelTrainingPipeline, args: argparse.Namespace) -> None:
    """Apply command line argument overrides to pipeline configuration."""
    config = pipeline.config
    
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
        pipeline.logger.info(f"Override max_epochs: {args.max_epochs}")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        pipeline.logger.info(f"Override batch_size: {args.batch_size}")
    
    if args.learning_rate:
        config['training']['lr'] = args.learning_rate
        pipeline.logger.info(f"Override learning_rate: {args.learning_rate}")
    
    if args.accelerator:
        config['training']['accelerator'] = args.accelerator
        pipeline.logger.info(f"Override accelerator: {args.accelerator}")
    
    if args.devices:
        config['training']['devices'] = args.devices
        pipeline.logger.info(f"Override devices: {args.devices}")
    
    if args.fast_dev_run:
        config['training']['max_epochs'] = 1
        pipeline.logger.info("Fast dev run enabled")


def main():
    """Main entry point."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        component_name='canopy_height_training',
        format_style='detailed' if args.log_level == 'DEBUG' else 'standard'
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing Canopy Height Training Pipeline...")
        pipeline = ModelTrainingPipeline(config_path=args.config)
        
        # Apply command line overrides
        apply_argument_overrides(pipeline, args)
        
        # Log pipeline start
        log_pipeline_start(logger, "Canopy Height DL Training", pipeline.config)
        
        # Validation only mode
        if args.validate_only:
            logger.info("Running configuration validation...")
            if pipeline.validate_configuration():
                logger.info("✅ Configuration validation successful")
                sys.exit(0)
            else:
                logger.error("❌ Configuration validation failed")
                sys.exit(1)
        
        # Handle resume training
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            pipeline.config['data']['checkpoint_path'] = args.resume
        
        # Run pipeline based on mode
        success = pipeline.run_full_pipeline(mode=args.mode)
        
        # Test specific checkpoint if provided
        if args.mode == 'test' and args.checkpoint:
            success = pipeline.run_testing(checkpoint_path=args.checkpoint)
        
        # Get and log training summary
        summary = pipeline.get_training_summary()
        logger.info("Training Summary:")
        for section, details in summary.items():
            if isinstance(details, dict):
                logger.info(f"  {section}:")
                for key, value in details.items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.info(f"  {section}: {details}")
        
        # Pipeline completion
        elapsed_time = time.time() - start_time
        log_pipeline_end(logger, "Canopy Height DL Training", success, elapsed_time)
        
        if success:
            logger.info("🎉 Training pipeline completed successfully!")
            if pipeline.best_checkpoint:
                logger.info(f"Best model checkpoint: {pipeline.best_checkpoint}")
            sys.exit(0)
        else:
            logger.error("💥 Training pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training pipeline failed with unexpected error: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
