"""
Model Training Pipeline

Main training pipeline for canopy height estimation using PyTorch Lightning.
Integrates data loading, model training, validation, and checkpointing.

Author: Diego Bengochea
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils import log_pipeline_start, log_pipeline_end

# Component imports (these would reference existing modules)
from .canopy_height_regression import CanopyHeightRegressionTask
from .datamodule import S2PNOAVegetationDataModule


class ModelTrainingPipeline:
    """
    Comprehensive training pipeline for canopy height estimation.
    
    Handles the complete training workflow including:
    - Data preparation and loading
    - Model initialization and configuration
    - Training with callbacks and logging
    - Validation and testing
    - Checkpoint management
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='canopy_height_dl',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Initialize components
        self.model = None
        self.datamodule = None
        self.trainer = None
        
        # Pipeline state
        self.start_time = None
        self.best_checkpoint = None
        
        self.logger.info("ModelTrainingPipeline initialized")
        self.logger.info(f"Configuration loaded from: {self.config['_meta']['config_file']}")
    
    def validate_configuration(self) -> bool:
        """
        Validate training configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        self.logger.info("Validating training configuration...")
        
        try:
            # Check required data directories
            data_dir = Path(self.config['data']['data_dir'])
            sentinel2_dir = data_dir / self.config['data']['sentinel2_dir']
            lidar_dir = data_dir / self.config['data']['pnoa_lidar_dir']
            
            if not sentinel2_dir.exists():
                self.logger.error(f"Sentinel-2 data directory not found: {sentinel2_dir}")
                return False
            
            if not lidar_dir.exists():
                self.logger.error(f"PNOA LiDAR directory not found: {lidar_dir}")
                return False
            
            # Check hardware requirements
            if self.config['training']['accelerator'] == 'gpu' and not torch.cuda.is_available():
                self.logger.warning("GPU requested but CUDA not available, switching to CPU")
                self.config['training']['accelerator'] = 'cpu'
            
            # Validate model parameters
            model_config = self.config['model']
            if model_config['in_channels'] <= 0:
                self.logger.error("Invalid number of input channels")
                return False
            
            if model_config['num_outputs'] <= 0:
                self.logger.error("Invalid number of outputs")
                return False
            
            # Validate training parameters
            train_config = self.config['training']
            if train_config['batch_size'] <= 0:
                self.logger.error("Invalid batch size")
                return False
            
            if train_config['lr'] <= 0:
                self.logger.error("Invalid learning rate")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return False
    
    def setup_data_module(self) -> bool:
        """
        Setup the data module for training.
        
        Returns:
            bool: True if setup succeeded
        """
        try:
            self.logger.info("Setting up data module...")
            
            # Create data module with configuration
            self.datamodule = S2PNOAVegetationDataModule(
                data_dir=self.config['data']['data_dir'],
                sentinel2_dir=self.config['data']['sentinel2_dir'],
                lidar_dir=self.config['data']['pnoa_lidar_dir'],
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['compute']['num_workers'],
                val_split=self.config['training']['val_split'],
                test_split=self.config['training']['test_split'],
                augmentation_enabled=self.config['training']['augmentation_enabled'],
                augmentation_probability=self.config['training']['augmentation_probability']
            )
            
            # Setup data module
            self.datamodule.setup()
            
            # Log dataset statistics
            self.logger.info(f"Training samples: {len(self.datamodule.train_dataset)}")
            self.logger.info(f"Validation samples: {len(self.datamodule.val_dataset)}")
            self.logger.info(f"Test samples: {len(self.datamodule.test_dataset)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up data module: {str(e)}")
            return False
    
    def setup_model(self) -> bool:
        """
        Setup the Lightning model.
        
        Returns:
            bool: True if setup succeeded
        """
        try:
            self.logger.info("Setting up model...")
            
            # Create model with configuration
            self.model = CanopyHeightRegressionTask(
                model=self.config['model']['model_type'],
                backbone=self.config['model']['backbone'],
                weights=self.config['model']['weights'],
                in_channels=self.config['model']['in_channels'],
                num_outputs=self.config['model']['num_outputs'],
                target_range=self.config['model']['target_range'],
                nan_value_target=self.config['model']['nan_value_target'],
                nan_value_input=self.config['model']['nan_value_input'],
                lr=self.config['training']['lr'],
                patience=self.config['training']['patience']
            )
            
            # Log model information
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"Model: {self.config['model']['model_type']} with {self.config['model']['backbone']} backbone")
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up model: {str(e)}")
            return False
    
    def setup_trainer(self) -> bool:
        """
        Setup the Lightning trainer with callbacks and logging.
        
        Returns:
            bool: True if setup succeeded
        """
        try:
            self.logger.info("Setting up trainer...")
            
            # Setup checkpoint directory
            checkpoint_dir = Path(self.config['data']['checkpoint_dir'])
            ensure_directory(checkpoint_dir)
            
            # Setup callbacks
            callbacks = []
            
            # Model checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='canopy_height_model_{epoch:02d}_{val_mae:.3f}',
                monitor='val_mae',
                mode='min',
                save_top_k=3,
                save_last=True,
                verbose=True
            )
            callbacks.append(checkpoint_callback)
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_mae',
                mode='min',
                patience=self.config['training']['patience'],
                verbose=True
            )
            callbacks.append(early_stopping)
            
            # Learning rate monitor
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor)
            
            # Setup logger
            logger = TensorBoardLogger(
                save_dir=checkpoint_dir.parent,
                name='tensorboard_logs',
                version=None
            )
            
            # Create trainer
            self.trainer = L.Trainer(
                max_epochs=self.config['training']['max_epochs'],
                accelerator=self.config['training']['accelerator'],
                devices=self.config['training']['devices'],
                precision=self.config['training']['precision'],
                callbacks=callbacks,
                logger=logger,
                enable_checkpointing=True,
                enable_progress_bar=True,
                enable_model_summary=True,
                deterministic=False,  # For better performance
                benchmark=True  # For consistent input sizes
            )
            
            self.logger.info(f"Trainer configured with {self.config['training']['accelerator']} accelerator")
            self.logger.info(f"Max epochs: {self.config['training']['max_epochs']}")
            self.logger.info(f"Checkpoint directory: {checkpoint_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up trainer: {str(e)}")
            return False
    
    def run_training(self) -> bool:
        """
        Execute the training process.
        
        Returns:
            bool: True if training completed successfully
        """
        try:
            self.logger.info("Starting model training...")
            
            # Fit the model
            self.trainer.fit(
                model=self.model,
                datamodule=self.datamodule
            )
            
            # Get best checkpoint path
            if hasattr(self.trainer.checkpoint_callback, 'best_model_path'):
                self.best_checkpoint = self.trainer.checkpoint_callback.best_model_path
                self.logger.info(f"Best checkpoint: {self.best_checkpoint}")
            
            # Check if training completed successfully
            if self.trainer.state.finished:
                self.logger.info("Training completed successfully")
                return True
            else:
                self.logger.warning("Training did not complete normally")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return False
    
    def run_testing(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Run testing on the best model.
        
        Args:
            checkpoint_path: Path to checkpoint for testing (uses best if None)
            
        Returns:
            bool: True if testing succeeded
        """
        try:
            self.logger.info("Starting model testing...")
            
            # Determine checkpoint to use
            if checkpoint_path:
                test_checkpoint = checkpoint_path
            elif self.best_checkpoint:
                test_checkpoint = self.best_checkpoint
            else:
                self.logger.error("No checkpoint available for testing")
                return False
            
            # Load model from checkpoint
            model = CanopyHeightRegressionTask.load_from_checkpoint(test_checkpoint)
            
            # Run test
            test_results = self.trainer.test(
                model=model,
                datamodule=self.datamodule,
                verbose=True
            )
            
            # Log test results
            if test_results:
                self.logger.info("Test Results:")
                for metric, value in test_results[0].items():
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during testing: {str(e)}")
            return False
    
    def run_full_pipeline(
        self, 
        mode: Literal['train', 'test', 'train_test'] = 'train_test'
    ) -> bool:
        """
        Run the complete training pipeline.
        
        Args:
            mode: Pipeline mode ('train', 'test', 'train_test')
            
        Returns:
            bool: True if pipeline completed successfully
        """
        self.start_time = time.time()
        
        # Log pipeline start
        log_pipeline_start(self.logger, "Canopy Height DL Training", self.config)
        
        try:
            # Validate configuration
            if not self.validate_configuration():
                return False
            
            # Setup data module
            if not self.setup_data_module():
                return False
            
            # Setup model
            if not self.setup_model():
                return False
            
            # Setup trainer
            if not self.setup_trainer():
                return False
            
            # Execute based on mode
            success = True
            
            if mode in ['train', 'train_test']:
                success = self.run_training()
                
                if not success:
                    self.logger.error("Training failed")
                    return False
            
            if mode in ['test', 'train_test']:
                success = self.run_testing()
                
                if not success:
                    self.logger.error("Testing failed")
                    return False
            
            # Pipeline completion
            elapsed_time = time.time() - self.start_time
            log_pipeline_end(self.logger, "Canopy Height DL Training", success, elapsed_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    def load_checkpoint_for_prediction(self, checkpoint_path: str) -> Optional[CanopyHeightRegressionTask]:
        """
        Load model from checkpoint for prediction.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model or None if failed
        """
        try:
            self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            
            model = CanopyHeightRegressionTask.load_from_checkpoint(checkpoint_path)
            model.eval()
            
            self.logger.info("Model loaded successfully for prediction")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training results.
        
        Returns:
            Dictionary with training summary
        """
        summary = {
            'config': {
                'model_type': self.config['model']['model_type'],
                'backbone': self.config['model']['backbone'],
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['lr'],
                'max_epochs': self.config['training']['max_epochs']
            }
        }
        
        if self.datamodule:
            summary['data'] = {
                'train_samples': len(self.datamodule.train_dataset),
                'val_samples': len(self.datamodule.val_dataset),
                'test_samples': len(self.datamodule.test_dataset)
            }
        
        if self.trainer and hasattr(self.trainer, 'callback_metrics'):
            summary['final_metrics'] = dict(self.trainer.callback_metrics)
        
        if self.best_checkpoint:
            summary['best_checkpoint'] = str(self.best_checkpoint)
        
        return summary
