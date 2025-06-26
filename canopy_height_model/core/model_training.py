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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils import log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import SENTINEL2_MOSAICS_DIR, ALS_CANOPY_HEIGHT_PROCESSED_DIR, PRETRAINED_HEIGHT_MODELS_DIR

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
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
            checkpoint_path: Path to checkpoint for resuming training
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='canopy_height_dl',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Pipeline state
        self.start_time = None
        self.best_checkpoint = None

        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = None
        
        self.logger.info("ModelTrainingPipeline initialized")
        self.logger.info(f"Configuration loaded from: {self.config['_meta']['config_file']}")
        
        # Initialize components
        self._setup_datamodule()
        self._setup_model()
        self._setup_trainer()
    
    def _setup_datamodule(self) -> None:
        """
        Setup the data module for training.
        """

        self.logger.info("Setting up data module...")
        
        # Create data module with configuration
        self.datamodule = S2PNOAVegetationDataModule(
            sentinel2_dir=str(SENTINEL2_MOSAICS_DIR),
            pnoa_dir=str(ALS_CANOPY_HEIGHT_PROCESSED_DIR),
            patch_size=self.config['training']['patch_size'],
            batch_size=self.config['training']['batch_size'],
            length=self.config['training'].get('length', 1000),
            num_workers=self.config['compute']['num_workers'],
            seed=self.config.get('seed', 42),
            predict_patch_size=self.config['prediction']['tile_size'],
            nan_target=self.config['model']['nan_value_target'],
            nan_input=self.config['model']['nan_value_input'],
            config_path=None  # Will use component default
        )
        
        self.logger.info("Data module setup completed successfully")
        self.logger.info(f"Sentinel-2 directory: {SENTINEL2_MOSAICS_DIR}")
        self.logger.info(f"PNOA directory: {ALS_CANOPY_HEIGHT_PROCESSED_DIR}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"Patch size: {self.config['training']['patch_size']}")
    
    def _setup_model(self) -> None:
        """
        Setup the Lightning model.
        """
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
        
        self.logger.info(f"Model: {self.config['model']['model_type']} with {self.config['model']['backbone']} backbone")
    
    def _setup_trainer(self) -> None:
        """
        Setup the Lightning trainer with callbacks and logging.
        """
        self.logger.info("Setting up trainer...")
        
        # Setup checkpoint directory
        checkpoint_dir = PRETRAINED_HEIGHT_MODELS_DIR
        ensure_directory(checkpoint_dir)

        if self.config['training']['accelerator'] == 'mps':
            accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            accelerator = 'cpu'
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='canopy_height_model_{epoch:02d}_{val_loss:.3f}',
            monitor='val_loss',
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['patience'],
            min_delta=0
        )
        callbacks.append(early_stopping)
        
        # # Setup logger
        # logger = TensorBoardLogger(
        #     save_dir=checkpoint_dir.parent,
        #     name='tensorboard_logs',
        #     version=None
        # )

        csv_logger = CSVLogger(
            save_dir=checkpoint_dir,
            name='logs'
        )
        
        # Create trainer
        self.trainer = L.Trainer(
            max_epochs=self.config['training']['max_epochs'],
            accelerator=accelerator,
            devices=self.config['training']['devices'],
            log_every_n_steps=self.config['training']['log_steps'],
            callbacks=callbacks,
            logger=csv_logger,
            accumulate_grad_batches=self.config['training']['accumulate_grad_batches'],
            num_sanity_val_steps=self.config['training']['num_sanity_val_steps'],
            check_val_every_n_epoch=self.config['training']['val_check_interval']
        )
        
        self.logger.info(f"Trainer configured with {self.config['training']['accelerator']} accelerator")
        self.logger.info(f"Max epochs: {self.config['training']['max_epochs']}")
        self.logger.info(f"Checkpoint directory: {checkpoint_dir}")

    
    def run(self) -> bool:
        """
        Execute the training process with proper checkpoint resumption support.
        
        Returns:
            bool: True if training completed successfully
        """
        try:
            self.logger.info("Starting model training...")
            
            # Handle checkpoint resumption properly
            if self.checkpoint_path is not None:
                checkpoint_path = Path(self.checkpoint_path)
                
                if not checkpoint_path.exists():
                    self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                    return False
                
                self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                
                # Check if we should load weights only or full checkpoint
                load_weights_only = self.config.get('training', {}).get('load_weights_only', False)
                
                if load_weights_only:
                    # Load only model weights, start fresh training
                    self.logger.info("Loading weights only, starting fresh training state")
                    try:
                        checkpoint = torch.load(checkpoint_path, weights_only=True)
                        self.model.load_state_dict(checkpoint['state_dict'])
                        self.trainer.fit(self.model, self.datamodule)
                    except Exception as e:
                        self.logger.error(f"Failed to load weights from checkpoint: {str(e)}")
                        return False
                else:
                    # Resume full training state (optimizer, scheduler, epoch, etc.)
                    self.logger.info("Resuming full training state from checkpoint")
                    try:
                        self.trainer.fit(
                            model=self.model,
                            datamodule=self.datamodule,
                            ckpt_path=str(checkpoint_path) 
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to resume from checkpoint: {str(e)}")
                        return False
            else:
                # Start training from scratch
                self.logger.info("Starting training from scratch (no checkpoint provided)")
                try:
                    self.trainer.fit(
                        model=self.model,
                        datamodule=self.datamodule
                    )
                except Exception as e:
                    self.logger.error(f"Training from scratch failed: {str(e)}")
                    return False
            
            # Get best checkpoint path if available
            if hasattr(self.trainer, 'checkpoint_callback') and hasattr(self.trainer.checkpoint_callback, 'best_model_path'):
                self.best_checkpoint = self.trainer.checkpoint_callback.best_model_path
                self.logger.info(f"Best checkpoint saved at: {self.best_checkpoint}")
            
            # Check if training completed successfully
            if hasattr(self.trainer, 'state') and self.trainer.state.finished:
                self.logger.info("Training completed successfully")
                return True
            else:
                self.logger.warning("Training did not complete normally")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return False

    
    