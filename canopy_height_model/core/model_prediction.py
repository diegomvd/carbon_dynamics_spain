"""
Model Prediction Pipeline

Simplified, framework-native prediction pipeline for canopy height estimation.
Uses TorchGeo and PyTorch Lightning as intended while maintaining configuration-driven flexibility.

Author: Diego Bengochea
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from lightning.pytorch import Trainer

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils import log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import HEIGHT_MODEL_CHECKPOINT_FILE

# Component imports
from .s2_pnoa_vegetation_datamodule import S2PNOAVegetationDataModule
from .canopy_height_regression import CanopyHeightRegressionTask
from .prediction_writer import CanopyHeightRasterWriter


class ModelPredictionPipeline:
    """
    Simplified prediction pipeline using TorchGeo and Lightning framework.
    
    This pipeline leverages the built-in capabilities of TorchGeo for data discovery
    and Lightning for prediction orchestration, resulting in cleaner, more maintainable
    code while preserving full configurability.
    
    Key Design Decisions:
    - Uses S2PNOAVegetationDataModule for automatic file discovery (no manual loops)
    - Uses Lightning Trainer.predict() as intended (no manual tiling logic)
    - Uses CanopyHeightRasterWriter callback for output handling
    - Configuration-driven for all parameters
    """
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the prediction pipeline.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint (overrides config)
            pattern: File pattern for input files (overrides config)
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='canopy_height_prediction',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Set checkpoint path
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = Path(HEIGHT_MODEL_CHECKPOINT_FILE)
                    
        # Validate checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Pipeline state
        self.start_time = None
        
        self.logger.info("ModelPredictionPipeline initialized")
        self.logger.info(f"Checkpoint: {self.checkpoint_path}")

    def run_full_pipeline(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Run the complete prediction pipeline.
        
        Args:
            input_path: Input directory containing Sentinel-2 rasters
            output_path: Output directory for predictions
            
        Returns:
            bool: True if processing succeeded
        """
        self.start_time = time.time()
        log_pipeline_start(self.logger, "Model Prediction Pipeline")
        
        try:
            input_dir = Path(input_path)
            output_dir = Path(output_path)
            
            # Validate input directory
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
            # Create output directory
            ensure_directory(output_dir)
            
            self.logger.info(f"Input directory: {input_dir}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Setup pipeline components
            self._setup_datamodule(input_dir)
            self._setup_model()
            self._setup_trainer(output_dir)
            
            # Run predictions
            self.logger.info("Starting predictions...")
            predictions = self.trainer.predict(
                self.model,
                datamodule=self.datamodule
            )
            
            # Log completion
            elapsed_time = time.time() - self.start_time
            self.logger.info(f"Predictions completed successfully in {elapsed_time:.1f} seconds")
            log_pipeline_end(self.logger, "Model Prediction Pipeline", True, elapsed_time)
            
            return True
            
        except Exception as e:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.logger.error(f"Prediction pipeline failed: {str(e)}")
            log_pipeline_end(self.logger, "Model Prediction Pipeline", False, elapsed_time)
            raise e

    def _setup_datamodule(self, input_dir: Path) -> None:
        """
        Initialize the data module for prediction.
        
        This uses TorchGeo's built-in file discovery and processing capabilities.
        The datamodule automatically finds all matching files in the input directory
        and handles the tiling/batching internally.
        """
        prediction_config = self.config.get('prediction', {})
        
        self.datamodule = S2PNOAVegetationDataModule(
            sentinel2_dir=str(input_dir),
            predict_patch_size=prediction_config.get('tile_size', 6144),
            batch_size=prediction_config.get('batch_size', 8),
            num_workers=prediction_config.get('num_workers', 4)
        )
        
        self.logger.info("Datamodule initialized for prediction")
        self.logger.info(f"  Tile size: {prediction_config.get('tile_size', 6144)}")
        self.logger.info(f"  Batch size: {prediction_config.get('batch_size', 8)}")
        self.logger.info(f"  Num workers: {prediction_config.get('num_workers', 4)}")
        
    def _setup_model(self) -> None:
        """
        Load the model from checkpoint.
        
        Uses Lightning's built-in checkpoint loading with proper device handling.
        """
        # Get model configuration
        model_config = self.config.get('model', {})
        
        self.model = CanopyHeightRegressionTask.load_from_checkpoint(
            self.checkpoint_path,
            nan_value_target=model_config.get('nan_value_target', -1.0),
            nan_value_input=model_config.get('nan_value_input', 0.0),
            config_path=None  # Model will use its own config
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        self.logger.info(f"Model loaded from checkpoint: {self.checkpoint_path}")
        
    def _setup_trainer(self, output_dir: Path) -> None:
        """
        Configure trainer for prediction with output handling.
        
        Uses Lightning's callback system for clean output handling.
        """
        # Get training configuration for device settings
        training_config = self.config.get('training', {})
        prediction_config = self.config.get('prediction', {})
        
        # Setup prediction writer callback
        pred_writer = CanopyHeightRasterWriter(
            output_dir=str(output_dir),
            write_interval="batch",
            output_dtype=prediction_config.get('output_dtype', 'float32'),
            compress=prediction_config.get('compress', 'lzw'),
            tiled=prediction_config.get('tiled', True)
        )
        
        # Auto-detect accelerator based on availability and config
        accelerator = self._get_accelerator(training_config.get('accelerator', 'auto'))
        
        self.trainer = Trainer(
            accelerator=accelerator,
            devices=training_config.get('devices', 'auto'),
            callbacks=[pred_writer],
            enable_checkpointing=False,
            enable_progress_bar=True,
            inference_mode=True,  # Memory optimization for prediction
            logger=False  # No need for logging during prediction
        )
        
        self.logger.info(f"Trainer configured for prediction")
        self.logger.info(f"  Accelerator: {accelerator}")
        self.logger.info(f"  Output format: {prediction_config.get('output_dtype', 'float32')}")
        
    def _get_accelerator(self, config_accelerator: str) -> str:
        """
        Determine the best accelerator to use based on availability and configuration.
        
        Args:
            config_accelerator: Accelerator preference from configuration
            
        Returns:
            str: Accelerator to use ('gpu', 'mps', 'cpu')
        """
        if config_accelerator == 'auto':
            # Auto-detect best available accelerator
            if torch.cuda.is_available():
                return 'gpu'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            # Use configured accelerator (with validation)
            if config_accelerator == 'gpu' and not torch.cuda.is_available():
                self.logger.warning("GPU requested but not available, falling back to CPU")
                return 'cpu'
            elif config_accelerator == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                self.logger.warning("MPS requested but not available, falling back to CPU")
                return 'cpu'
            else:
                return config_accelerator
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get summary of prediction pipeline configuration and state.
        
        Returns:
            Dictionary with prediction summary
        """
        prediction_config = self.config.get('prediction', {})
        model_config = self.config.get('model', {})
        
        summary = {
            'pipeline': {
                'checkpoint_path': str(self.checkpoint_path),
                'file_pattern': self.pattern,
                'framework': 'TorchGeo + Lightning'
            },
            'prediction_config': {
                'tile_size': prediction_config.get('tile_size', 6144),
                'batch_size': prediction_config.get('batch_size', 8),
                'num_workers': prediction_config.get('num_workers', 4),
                'output_dtype': prediction_config.get('output_dtype', 'float32'),
                'compress': prediction_config.get('compress', 'lzw')
            },
            'model_config': {
                'model_type': model_config.get('model_type', 'unet'),
                'backbone': model_config.get('backbone', 'efficientnet-b4'),
                'in_channels': model_config.get('in_channels', 10)
            }
        }
        
        if self.trainer:
            summary['runtime'] = {
                'accelerator': self.trainer.accelerator.__class__.__name__,
                'devices': str(self.trainer.num_devices)
            }
        
        return summary