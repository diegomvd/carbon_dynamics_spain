"""
Canopy height regression model prediction pipeline using PyTorch Lightning.

This script provides a streamlined prediction pipeline for generating canopy height
predictions using trained models. Handles large-scale inference with memory-efficient
processing, configurable patch sizes, and comprehensive error handling.

The pipeline supports various patch sizes and stride configurations for optimal
performance across different computational resources and accuracy requirements.

Author: Diego Bengochea
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Standard library imports
import torch

# PyTorch Lightning imports
from lightning.pytorch import Trainer

# Local imports
from config import load_config, setup_logging, create_output_directory
from s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from canopy_height_regression import CanopyHeightRegressionTask
from prediction_writer import CanopyHeightRasterWriter


class PredictionPipeline:
    """
    PyTorch Lightning prediction pipeline for canopy height regression.
    
    This class encapsulates the complete prediction workflow including data loading,
    model configuration, trainer setup, and execution of large-scale inference with
    memory-efficient processing strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the prediction pipeline with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all parameters
            
        Raises:
            FileNotFoundError: If checkpoint file is not found
        """
        self.config = config
        self.logger = setup_logging(config['logging']['level'])
        
        # Validate checkpoint path
        checkpoint_path = Path(config['prediction']['checkpoint_path'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create output directory
        self.output_dir = Path(config['prediction']['output_dir'])
        create_output_directory(self.output_dir)
        
        # Initialize components
        self._setup_datamodule()
        self._setup_model()
        self._setup_trainer()
        
        self.logger.info(f"Prediction pipeline initialized.")
        self.logger.info(f"Checkpoint: {checkpoint_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Patch size: {config['prediction']['patch_size']}")
        self.logger.info(f"Stride: {config['prediction']['stride']}")
        
    def _setup_datamodule(self) -> None:
        """Initialize prediction-specific datamodule configuration."""
        self.datamodule = S2PNOAVegetationDataModule(
            data_dir=str(self.config['data']['data_dir']),
            predict_patch_size=self.config['prediction']['patch_size']
        )
        
        self.logger.info(f"Datamodule configured for prediction with patch size: {self.config['prediction']['patch_size']}")
        
    def _setup_model(self) -> None:
        """Load the model from checkpoint."""
        self.model = CanopyHeightRegressionTask.load_from_checkpoint(
            self.config['prediction']['checkpoint_path'],
            nan_value_target=self.config['model']['nan_value_target'],
            nan_value_input=self.config['model']['nan_value_input']
        )
        self.model.eval()  # Set to evaluation mode
        
        self.logger.info("Model loaded from checkpoint and set to evaluation mode")
        
    def _setup_trainer(self) -> None:
        """Configure trainer specifically for prediction."""
        pred_writer = CanopyHeightRasterWriter(
            output_dir=str(self.output_dir),
            write_interval="batch"
        )
        
        # Determine accelerator
        accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.logger.info(f"Using accelerator: {accelerator}")
        
        self.trainer = Trainer(
            accelerator=accelerator,
            devices=self.config['compute']['devices'],
            callbacks=[pred_writer],
            enable_progress_bar=True,
            inference_mode=True,  # Important for memory efficiency
            logger=False  # No need for logging during prediction
        )
        
        self.logger.info("Trainer configured for prediction mode")
        
    def predict(self) -> Any:
        """
        Run predictions with comprehensive error handling and logging.
        
        Returns:
            Any: Predictions output from the trainer
            
        Raises:
            Exception: If prediction fails with detailed error information
        """
        try:
            self.logger.info("Starting prediction process...")
            
            predictions = self.trainer.predict(
                self.model,
                datamodule=self.datamodule
            )
            
            self.logger.info("Predictions completed successfully")
            self.logger.info(f"Results saved to: {self.output_dir}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise


def validate_configuration(config: Dict[str, Any]) -> None:
    """
    Validate prediction configuration parameters.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration parameters are invalid
        FileNotFoundError: If required files are not found
    """
    # Validate checkpoint path
    checkpoint_path = Path(config['prediction']['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Validate data directory
    data_dir = Path(config['data']['data_dir'])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Validate patch size
    patch_size = config['prediction']['patch_size']
    if patch_size <= 0 or patch_size % 256 != 0:
        raise ValueError(f"Patch size must be positive and divisible by 256, got: {patch_size}")
    
    # Validate stride
    stride = config['prediction']['stride']
    if stride <= 0 or stride > patch_size:
        raise ValueError(f"Stride must be positive and <= patch_size, got: {stride}")


def main():
    """Main entry point for the prediction pipeline."""
    try:
        # Load configuration
        config = load_config()
        logger = setup_logging(config['logging']['level'])
        logger.info("Starting Canopy Height Prediction Pipeline...")
        
        # Validate configuration
        validate_configuration(config)
        logger.info("Configuration validation passed")
        
        # Initialize and run prediction
        pipeline = PredictionPipeline(config)
        predictions = pipeline.predict()
        
        logger.info("Prediction pipeline completed successfully!")
        
    except FileNotFoundError as e:
        logger = setup_logging('ERROR')
        logger.error(f"File not found: {str(e)}")
        raise
    except ValueError as e:
        logger = setup_logging('ERROR')
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Prediction pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()