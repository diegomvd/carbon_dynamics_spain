"""
Model Prediction Pipeline

Large-scale prediction pipeline for canopy height estimation using trained models.
Handles tiled prediction over large geographical areas with memory management.

Author: Diego Bengochea
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import DataLoader
import lightning as L

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils import find_files, validate_file_exists, log_pipeline_start, log_pipeline_end
from shared_utils.central_data_paths_constants import HEIGHT_MODEL_CHECKPOINT_FILE

# Component imports
from .canopy_height_regression import CanopyHeightRegressionTask


class ModelPredictionPipeline:
    """
    Pipeline for large-scale canopy height prediction.
    
    Handles:
    - Loading trained models from checkpoints
    - Tiled prediction over large rasters
    - Memory-efficient processing
    - Output formatting and saving
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='canopy_height_prediction',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Initialize components
        self.model = None
        self.device = None
        
        # Pipeline state
        self.start_time = None
        
        self.logger.info("ModelPredictionPipeline initialized")
    
    def load_model(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Load trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            validate_file_exists(checkpoint_path, "Model checkpoint")
            
            self.logger.info(f"Loading model from: {checkpoint_path}")
            
            # Load model
            self.model = CanopyHeightRegressionTask.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            
            # Setup device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.model = self.model.cuda()
                self.logger.info("Using GPU for prediction")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.model = self.model.to(self.device)
                self.logger.info("Using MPS (Apple Silicon) for prediction")
            else:
                self.device = torch.device('cpu')
                self.logger.info("Using CPU for prediction")
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def validate_input_raster(self, raster_path: Path) -> bool:
        """
        Validate input raster for prediction.
        
        Args:
            raster_path: Path to input raster
            
        Returns:
            bool: True if raster is valid
        """
        try:
            with rasterio.open(raster_path) as src:
                # Check number of bands
                expected_bands = self.config['model']['in_channels']
                if src.count != expected_bands:
                    self.logger.error(f"Expected {expected_bands} bands, found {src.count} in {raster_path}")
                    return False
                
                # Check data type
                if src.dtypes[0] not in ['float32', 'float64', 'uint16', 'int16']:
                    self.logger.warning(f"Unexpected data type {src.dtypes[0]} in {raster_path}")
                
                # Check spatial reference
                if src.crs is None:
                    self.logger.warning(f"No CRS found in {raster_path}")
                
                self.logger.debug(f"Raster validation passed: {raster_path.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error validating raster {raster_path}: {str(e)}")
            return False
    
    def create_prediction_tiles(
        self, 
        raster_shape: Tuple[int, int],
        tile_size: int,
        overlap: int
    ) -> List[Tuple[Window, Tuple[int, int]]]:
        """
        Create prediction tiles with overlap.
        
        Args:
            raster_shape: Shape of input raster (height, width)
            tile_size: Size of prediction tiles
            overlap: Overlap between tiles
            
        Returns:
            List of (window, tile_shape) tuples
        """
        height, width = raster_shape
        tiles = []
        
        step_size = tile_size - overlap
        
        for row in range(0, height, step_size):
            for col in range(0, width, step_size):
                # Calculate actual tile dimensions
                tile_height = min(tile_size, height - row)
                tile_width = min(tile_size, width - col)
                
                # Create window
                window = Window(col, row, tile_width, tile_height)
                tile_shape = (tile_height, tile_width)
                
                tiles.append((window, tile_shape))
        
        self.logger.debug(f"Created {len(tiles)} prediction tiles")
        return tiles
    
    def predict_tile(
        self, 
        input_data: np.ndarray,
        tile_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Predict canopy height for a single tile.
        
        Args:
            input_data: Input data array (bands, height, width)
            tile_shape: Expected tile shape
            
        Returns:
            Prediction array or None if failed
        """
        try:
            # Prepare input tensor
            input_tensor = torch.from_numpy(input_data).float()
            
            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Run prediction
            with torch.no_grad():
                prediction = self.model(input_tensor)
            
            # Convert to numpy
            prediction_np = prediction.cpu().numpy().squeeze()
            
            # Handle NaN values
            nan_value = self.config['model']['nan_value_target']
            prediction_np = np.where(np.isnan(prediction_np), nan_value, prediction_np)
            
            return prediction_np
            
        except Exception as e:
            self.logger.error(f"Error predicting tile: {str(e)}")
            return None
    
    def predict_raster(
        self,
        input_raster_path: Path,
        output_raster_path: Path
    ) -> bool:
        """
        Predict canopy height for entire raster using tiled approach.
        
        Args:
            input_raster_path: Path to input Sentinel-2 raster
            output_raster_path: Path to output prediction raster
            
        Returns:
            bool: True if prediction succeeded
        """
        try:
            self.logger.info(f"Predicting: {input_raster_path.name}")
            
            # Validate input
            if not self.validate_input_raster(input_raster_path):
                return False
            
            # Read input raster metadata
            with rasterio.open(input_raster_path) as src:
                input_profile = src.profile.copy()
                raster_shape = (src.height, src.width)
                input_transform = src.transform
                input_crs = src.crs
            
            # Create prediction tiles
            tile_size = self.config['prediction']['tile_size']
            overlap = self.config['prediction']['overlap']
            tiles = self.create_prediction_tiles(raster_shape, tile_size, overlap)
            
            # Setup output raster
            output_profile = input_profile.copy()
            output_profile.update({
                'count': 1,
                'dtype': self.config['prediction']['output_dtype'],
                'nodata': self.config['post_processing']['merge']['nodata_value'],
                'compress': self.config['prediction']['compress'],
                'tiled': self.config['prediction']['tiled']
            })
            
            # Ensure output directory
            ensure_directory(output_raster_path.parent)
            
            # Initialize output array
            output_array = np.full(
                raster_shape, 
                self.config['post_processing']['merge']['nodata_value'],
                dtype=output_profile['dtype']
            )
            
            # Process tiles
            successful_tiles = 0
            total_tiles = len(tiles)
            
            with rasterio.open(input_raster_path) as src:
                for i, (window, tile_shape) in enumerate(tiles):
                    if i % 10 == 0:  # Progress logging
                        self.logger.info(f"Processing tile {i+1}/{total_tiles}")
                    
                    # Read input tile
                    input_tile = src.read(window=window)
                    
                    # Skip if tile is empty/invalid
                    if input_tile.size == 0:
                        continue
                    
                    # Predict tile
                    prediction_tile = self.predict_tile(input_tile, tile_shape)
                    
                    if prediction_tile is not None:
                        # Handle overlap by taking average
                        tile_slice = (
                            slice(window.row_off, window.row_off + window.height),
                            slice(window.col_off, window.col_off + window.width)
                        )
                        
                        # Simple approach: overwrite (could be improved with overlap blending)
                        output_array[tile_slice] = prediction_tile
                        successful_tiles += 1
            
            # Write output raster
            with rasterio.open(output_raster_path, 'w', **output_profile) as dst:
                dst.write(output_array, 1)
                
                # Add metadata
                dst.update_tags(
                    MODEL_CHECKPOINT=str(HEIGHT_MODEL_CHECKPOINT_FILE),
                    TILE_SIZE=str(tile_size),
                    OVERLAP=str(overlap),
                    PROCESSED_TILES=str(successful_tiles),
                    TOTAL_TILES=str(total_tiles)
                )
            
            self.logger.info(f"Prediction completed: {successful_tiles}/{total_tiles} tiles successful")
            self.logger.info(f"Output saved to: {output_raster_path}")
            
            return successful_tiles > 0
            
        except Exception as e:
            self.logger.error(f"Error predicting raster: {str(e)}")
            return False
    
    def predict_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.tif"
    ) -> bool:
        """
        Predict canopy height for all rasters in a directory.
        
        Args:
            input_dir: Input directory containing Sentinel-2 rasters
            output_dir: Output directory for predictions
            file_pattern: File pattern to match
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            # Find input files
            input_files = find_files(input_path, file_pattern)
            
            if not input_files:
                self.logger.error(f"No input files found in {input_dir}")
                return False
            
            self.logger.info(f"Found {len(input_files)} files to process")
            
            # Process each file
            successful = 0
            failed = 0
            
            for input_file in input_files:
                # Generate output filename
                output_filename = f"canopy_height_pred_{input_file.stem}.tif"
                output_file = output_path / output_filename
                
                # Skip if output already exists
                if output_file.exists():
                    self.logger.info(f"Output exists, skipping: {output_filename}")
                    continue
                
                # Run prediction
                if self.predict_raster(input_file, output_file):
                    successful += 1
                else:
                    failed += 1
            
            self.logger.info(f"Directory prediction complete: {successful} successful, {failed} failed")
            return successful > 0
            
        except Exception as e:
            self.logger.error(f"Error predicting directory: {str(e)}")
            return False
    
    def run_prediction_pipeline(
        self,
        checkpoint_path: Union[str, Path],
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        is_directory: bool = False
    ) -> bool:
        """
        Run the complete prediction pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            input_path: Path to input raster(s)
            output_path: Path to output location
            is_directory: Whether input_path is a directory
            
        Returns:
            bool: True if pipeline succeeded
        """
        self.start_time = time.time()
        
        # Log pipeline start
        log_pipeline_start(self.logger, "Canopy Height Prediction", self.config)
        
        try:
            # Load model
            if not self.load_model(checkpoint_path):
                return False
            
            # Run prediction
            if is_directory:
                success = self.predict_directory(input_path, output_path)
            else:
                success = self.predict_raster(Path(input_path), Path(output_path))
            
            # Pipeline completion
            elapsed_time = time.time() - self.start_time
            log_pipeline_end(self.logger, "Canopy Height Prediction", success, elapsed_time)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline failed: {str(e)}")
            return False
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get summary of prediction pipeline.
        
        Returns:
            Dictionary with prediction summary
        """
        summary = {
            'config': {
                'tile_size': self.config['prediction']['tile_size'],
                'overlap': self.config['prediction']['overlap'],
                'batch_size': self.config['prediction']['batch_size'],
                'output_dtype': self.config['prediction']['output_dtype']
            }
        }
        
        if self.model:
            summary['model'] = {
                'model_type': self.config['model']['model_type'],
                'backbone': self.config['model']['backbone'],
                'in_channels': self.config['model']['in_channels'],
                'device': str(self.device) if self.device else 'unknown'
            }
        
        return summary
