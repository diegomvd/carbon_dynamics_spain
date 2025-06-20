"""
Canopy height regression model training pipeline using PyTorch Lightning.

This script provides a comprehensive training pipeline for canopy height regression
using Sentinel-2 and PNOA vegetation data. Implements deep learning models with
balanced sampling, range-aware loss functions, and comprehensive evaluation metrics.

The pipeline supports training, testing, and prediction modes with proper
checkpointing, logging, and configuration management.

Author: Diego Bengochea
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, Dict, Any

# Standard library imports
import torch

# Third-party imports
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# Local imports
from config import load_config, setup_logging, create_output_directory
from s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from canopy_height_regression import CanopyHeightRegressionTask
from prediction_writer import CanopyHeightRasterWriter


class ModelTrainingPipeline:
    """
    PyTorch Lightning training pipeline for canopy height regression.
    
    This class encapsulates the complete training workflow including data loading,
    model configuration, trainer setup, and execution of training/testing/prediction.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mode: Literal['train', 'test', 'predict'] = 'train'
    ):
        """
        Initialize the training pipeline with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all parameters
            mode (str): Pipeline mode - 'train', 'test', or 'predict'
        """
        self.config = config
        self.mode = mode
        self.logger = setup_logging(config['logging']['level'])
        
        # Create checkpoint directory if it doesn't exist
        create_output_directory(self.config['data']['checkpoint_dir'])
        
        # Initialize components
        self._setup_datamodule()
        self._setup_model()
        self._setup_trainer()

    def _setup_datamodule(self) -> None:
        """Initialize the data module with configuration parameters."""
        self.datamodule = S2PNOAVegetationDataModule(
            data_dir=str(self.config['data']['data_dir']),
            predict_patch_size=self.config['training']['predict_patch_size']
        )
        
        self.logger.info(f"Datamodule initialized with data directory: {self.config['data']['data_dir']}")

    def _setup_model(self) -> None:
        """Initialize the model with configuration parameters."""
        self.model = CanopyHeightRegressionTask(
            model=self.config['model']['model_type'],
            backbone=self.config['model']['backbone'],
            in_channels=self.config['model']['in_channels'],
            num_outputs=self.config['model']['num_outputs'],
            target_range=self.config['model']['target_range'],
            nan_value_target=self.config['model']['nan_value_target'],
            nan_value_input=self.config['model']['nan_value_input'],
            lr=self.config['training']['learning_rate'],
            patience=self.config['training']['patience']
        )
        
        self.logger.info(f"Model initialized: {self.config['model']['model_type']} with {self.config['model']['backbone']} backbone")

    def _setup_trainer(self) -> None:
        """Initialize the PyTorch Lightning trainer with callbacks and loggers."""
        # Set up accelerator
        accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.logger.info(f"Using accelerator: {accelerator}")
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.config['data']['checkpoint_dir'],
            save_top_k=3,
            save_last=True
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=self.config['training']['early_stopping_patience']
        )
        
        pred_writer = CanopyHeightRasterWriter(
            output_dir="canopy_height_predictions_2",
            write_interval="batch"
        )
        
        # Logger
        csv_logger = CSVLogger(
            save_dir=self.config['data']['checkpoint_dir'],
            name='logs'
        )
        
        self.trainer = Trainer(
            check_val_every_n_epoch=self.config['training']['val_check_interval'],
            accelerator=accelerator,
            devices=self.config['compute']['devices'],
            callbacks=[checkpoint_callback, pred_writer],
            log_every_n_steps=self.config['training']['log_steps'],
            logger=csv_logger,
            max_epochs=self.config['training']['max_epochs'],
            accumulate_grad_batches=self.config['training']['accumulate_grad_batches'],
            num_sanity_val_steps=self.config['training']['num_sanity_val_steps']
        )
        
        self.logger.info("Trainer initialized with callbacks and logging")

    def train(self, checkpoint_path: Optional[str] = None, load_weights_only: bool = False) -> None:
        """
        Train the model with optional checkpoint loading.
        
        Args:
            checkpoint_path (str, optional): Optional path to checkpoint for resuming training
            load_weights_only (bool): If True, only load model weights (not optimizer state)
        """
        self.logger.info("Starting model training...")
        
        try:
            if checkpoint_path is None:
                self.trainer.fit(self.model, self.datamodule)
            else:
                if load_weights_only:
                    state_dict = torch.load(checkpoint_path, weights_only=True)['state_dict']
                    self.model.load_state_dict(state_dict)
                    self.trainer.fit(self.model, self.datamodule)
                    self.logger.info(f"Resumed training from weights: {checkpoint_path}")
                else:    
                    self.trainer.fit(self.model, self.datamodule, ckpt_path=checkpoint_path)
                    self.logger.info(f"Resumed training from checkpoint: {checkpoint_path}")
                    
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
    def test(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Test the model using a specific checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint for testing
            
        Returns:
            Dict[str, Any]: Test metrics dictionary
        """
        self.logger.info(f"Testing model with checkpoint: {checkpoint_path}")
        
        try:
            test_metrics = self.trainer.test(
                self.model,
                datamodule=self.datamodule,
                ckpt_path=checkpoint_path
            )
            
            self.logger.info("Testing completed successfully")
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            raise

    def predict(self, checkpoint_path: str) -> Any:
        """
        Generate predictions using a specific checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint for prediction
            
        Returns:
            Any: Predictions output
        """
        self.logger.info(f"Generating predictions with checkpoint: {checkpoint_path}")
        
        try:
            predictions = self.trainer.predict(
                self.model,
                datamodule=self.datamodule,
                ckpt_path=checkpoint_path
            )
            
            self.logger.info("Prediction completed successfully")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise


def run_training_mode(config: Dict[str, Any]) -> None:
    """
    Execute training mode pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    logger = setup_logging(config['logging']['level'])
    logger.info("=== TRAINING MODE ===")
    
    pipeline = ModelTrainingPipeline(config, mode='train')
    pipeline.train(
        checkpoint_path=config['data']['checkpoint_path'], 
        load_weights_only=config['training']['load_weights_only']
    )


def run_testing_mode(config: Dict[str, Any]) -> None:
    """
    Execute testing mode pipeline for multiple model configurations.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    logger = setup_logging(config['logging']['level'])
    logger.info("=== TESTING MODE ===")
    
    for model_config in config['test_models']:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing model: {model_config['name']}")
        logger.info(f"Checkpoint: {model_config['path']}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Create a new pipeline for each model to ensure separate logging
            test_pipeline = ModelTrainingPipeline(config, mode='test')
            
            # Create model-specific logger
            csv_logger = CSVLogger(
                save_dir=config['data']['checkpoint_dir'],
                name=f"test_metrics_{model_config['name']}",
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            # Update trainer with new logger
            test_pipeline.trainer.logger = csv_logger
            
            # Run test
            test_metrics = test_pipeline.test(checkpoint_path=model_config['path'])
            logger.info(f"Test metrics: {test_metrics}")
            
        except Exception as e:
            logger.error(f"Error testing checkpoint {model_config['path']}: {str(e)}")
            continue


def run_prediction_mode(config: Dict[str, Any]) -> None:
    """
    Execute prediction mode pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    logger = setup_logging(config['logging']['level'])
    logger.info("=== PREDICTION MODE ===")
    
    if not config['data']['checkpoint_path']:
        raise ValueError("Checkpoint path must be specified for prediction mode")
    
    pipeline = ModelTrainingPipeline(config, mode='predict')
    predictions = pipeline.predict(checkpoint_path=str(config['data']['checkpoint_path']))
    logger.info("Predictions generated successfully")


def main():
    """Main entry point for the training pipeline."""
    # Load configuration
    try:
        config = load_config()
        logger = setup_logging(config['logging']['level'])
        logger.info("Starting Canopy Height Deep Learning Pipeline...")
        
        # Choose operation mode
        mode: Literal['train', 'test', 'predict'] = 'train'
        
        # Execute based on mode
        if mode == 'train':
            run_training_mode(config)
        elif mode == 'test':
            run_testing_mode(config)
        elif mode == 'predict':
            run_prediction_mode(config)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'predict'")
            
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger = setup_logging('ERROR')
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
