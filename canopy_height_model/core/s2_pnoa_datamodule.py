"""
PyTorch Lightning DataModule for Sentinel-2 and PNOA vegetation data integration.

This module provides a comprehensive data loading and preprocessing pipeline for
canopy height regression using Sentinel-2 satellite imagery and PNOA vegetation
height models. Implements advanced augmentation strategies, balanced sampling,
and efficient data loading for large-scale deep learning training.

The DataModule handles data splitting, augmentation configuration, and provides
separate pipelines for training, validation, testing, and prediction modes.

Author: Diego Bengochea
"""

from typing import Dict, Optional, Union, Type, Tuple, Callable, List, Iterator

# Standard library imports
import torch
from torch import Tensor, Generator
from torch.utils.data import DataLoader, _utils

# Third-party imports
from lightning import LightningDataModule
import kornia.augmentation as K
from rasterio.crs import CRS

# TorchGeo imports
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.splits import random_grid_cell_assignment
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datamodules import GeoDataModule

# Shared utilities
from shared_utils import load_config, get_logger
from shared_utils.central_data_paths_constants import SENTINEL2_MOSAICS_DIR, ALS_CANOPY_HEIGHT_PROCESSED_DIR

# Component imports (these would be additional modules in the same core/ directory)
try:
    from .datasets import S2Mosaic, PNOAVegetation, KorniaIntersectionDataset
    from .balanced_geo_samplers import HeightDiversityBatchSampler
    from .pnoa_vegetation_transforms import (
        PNOAVegetationInput0InArtificialSurfaces, 
        PNOAVegetationRemoveAbnormalHeight, 
        PNOAVegetationLogTransform, 
        S2Scaling
    )
except ImportError:
    # Fallback imports if modules don't exist yet
    S2Mosaic = None
    PNOAVegetation = None
    KorniaIntersectionDataset = None
    HeightDiversityBatchSampler = None


class S2PNOAVegetationDataModule(GeoDataModule):
    """
    PyTorch Lightning DataModule for Sentinel-2 and PNOA vegetation data.
    
    This DataModule provides comprehensive data loading, preprocessing, and augmentation
    for canopy height regression tasks. Supports multiple modes (train/val/test/predict)
    with appropriate sampling strategies and augmentation pipelines.
    """

    def __init__(
        self,
        data_dir: str = None,
        patch_size: int = None,
        batch_size: int = None,
        length: int = None, 
        num_workers: int = None, 
        seed: int = None, 
        predict_patch_size: int = None,
        nan_target: float = None,
        nan_input: float = None,
        config_path: str = None
    ):
        """
        Initialize the S2PNOA vegetation DataModule.
        
        Args:
            data_dir (str, optional): Path to data directory
            patch_size (int, optional): Size of training patches
            batch_size (int, optional): Batch size for training
            length (int, optional): Length override for samplers
            num_workers (int, optional): Number of data loading workers
            seed (int, optional): Random seed for reproducibility
            predict_patch_size (int, optional): Patch size for prediction
            nan_target (float, optional): NaN value for target data
            nan_input (float, optional): NaN value for input data
            config_path (str, optional): Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        self.logger = get_logger('canopy_height_dl.datamodule')
        
        # Use provided values or fall back to configuration
        data_dir = data_dir or str(SENTINEL2_MOSAICS_DIR.parent)
        patch_size = patch_size or self.config['training'].get('patch_size', 512)
        batch_size = batch_size or self.config['training']['batch_size']
        length = length or self.config['training'].get('length', 1000)
        num_workers = num_workers or self.config['compute']['num_workers']
        seed = seed or 42
        predict_patch_size = predict_patch_size or self.config['prediction']['tile_size']
        nan_target = nan_target or self.config['model']['nan_value_target']
        nan_input = nan_input or self.config['model']['nan_value_input']

        super().__init__(
            GeoDataset, batch_size, patch_size, length, num_workers
        )
        
        # Store parameters in hparams for compatibility
        self.save_hyperparameters({
            'data_dir': data_dir,
            'patch_size': patch_size,
            'batch_size': batch_size,
            'length': length,
            'num_workers': num_workers,
            'seed': seed,
            'predict_patch_size': predict_patch_size,
            'nan_target': nan_target,
            'nan_input': nan_input
        })
        
        # Initialize samplers/datasets (will be set in setup())
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.train_batch_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        self.predict_sampler = None
        
        # Setup augmentations
        self._setup_augmentations()
        
        self.logger.info(f"DataModule initialized with data directory: {data_dir}")
        self.logger.info(f"Training patch size: {patch_size}, prediction patch size: {predict_patch_size}")
       
    def _setup_augmentations(self) -> None:
        """Initialize all augmentation pipelines with configuration parameters."""
        self.logger.info("Setting up augmentation pipelines...")
        
        # Training augmentations
        self.train_aug = {
            'image': K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90, p=0.5),
                data_keys=['image'],
                same_on_batch=False
            ),
            'mask': K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90, p=0.5),
                data_keys=['image'],
                same_on_batch=False
            )
        }
        
        # Add custom transformations if available
        try:
            self.train_aug['general'] = self._create_general_transforms()
        except Exception as e:
            self.logger.warning(f"Could not setup general transforms: {e}")
        
        # Validation/test augmentations (minimal)
        self.val_aug = {'image': None, 'mask': None}
        self.test_aug = {'image': None, 'mask': None}
        self.predict_aug = {'image': None}
        
        self.logger.info("Augmentation pipelines configured")
    
    def _create_general_transforms(self):
        """Create general transformations pipeline."""
        # This would use the custom transform classes
        # For now, return a simple pass-through
        return lambda x: x
    
    def _valid_attribute(self, name: str, fallback: str):
        """Get valid attribute with fallback."""
        if hasattr(self, name):
            return getattr(self, name)
        return getattr(self, fallback, {})
    
    def setup(self, stage: str) -> None:
        """
        Set up datasets for each stage.
        
        Args:
            stage (str): Current stage ('fit', 'test', 'predict')
        """
        self.logger.info(f"Setting up datasets for stage: {stage}")
        
        # Check if required dataset classes are available
        if S2Mosaic is None:
            self.logger.error("S2Mosaic dataset class not available - please copy datasets.py to core/")
            raise ImportError("Required dataset classes not available")
        
        s2 = S2Mosaic(self.hparams['data_dir'] + str(SENTINEL2_MOSAICS_DIR.name))
        
        if stage == 'predict':
            self._setup_predict(s2)
        else:
            self._setup_train_val_test(s2)

    def _setup_predict(self, s2) -> None:
        """
        Set up prediction dataset and sampler.
        
        Args:
            s2: Sentinel-2 mosaic dataset
        """
        self.predict_dataset = s2
        size = self.hparams['predict_patch_size']
        stride = size - 256  # Default stride calculation
        self.predict_sampler = GridGeoSampler(self.predict_dataset, size, stride)
        
        self.logger.info(f"Prediction setup: patch_size={size}, stride={stride}")

    def _setup_train_val_test(self, s2) -> None:
        """
        Set up training, validation and test datasets and samplers.
        
        Args:
            s2: Sentinel-2 mosaic dataset
        """
        if PNOAVegetation is None or KorniaIntersectionDataset is None:
            self.logger.error("Required dataset classes not available")
            raise ImportError("PNOAVegetation or KorniaIntersectionDataset not available")
        
        pnoa_vegetation = PNOAVegetation(self.hparams['data_dir'] + str(ALS_CANOPY_HEIGHT_PROCESSED_DIR.name))
        dataset = KorniaIntersectionDataset(s2, pnoa_vegetation)
        
        # Split dataset using configuration
        split_ratios = [0.7, 0.2, 0.1]  # Default split ratios
        if 'data' in self.config and 'split_ratios' in self.config['data']:
            split_ratios = self.config['data']['split_ratios']
        
        grid_size = 1000  # Default grid size
        if 'data' in self.config and 'grid_size' in self.config['data']:
            grid_size = self.config['data']['grid_size']
        
        splits = random_grid_cell_assignment(
            dataset,
            split_ratios,
            grid_size=grid_size,
            generator=torch.Generator().manual_seed(self.hparams['seed'])
        )
        self.train_dataset, self.val_dataset, self.test_dataset = splits
        
        # Set up samplers based on stage
        if self.trainer and self.trainer.training:
            if HeightDiversityBatchSampler is not None:
                self.train_batch_sampler = HeightDiversityBatchSampler(
                    self.train_dataset,
                    self.hparams['patch_size'],
                    self.hparams['batch_size'],
                    self.hparams['length']
                )
                self.logger.info("Using HeightDiversityBatchSampler for training")
            else:
                # Fallback to standard batch sampler
                self.train_batch_sampler = RandomBatchGeoSampler(
                    self.train_dataset,
                    self.hparams['patch_size'],
                    self.hparams['batch_size'],
                    self.hparams['length']
                )
                self.logger.warning("Using standard RandomBatchGeoSampler (HeightDiversityBatchSampler not available)")

        if self.trainer and (self.trainer.training or self.trainer.validating):
            self.val_sampler = GridGeoSampler(
                self.val_dataset,
                self.hparams['patch_size'],
                self.hparams['patch_size']
            )
        
        if self.trainer and self.trainer.testing:
            self.test_sampler = GridGeoSampler(
                self.test_dataset,
                self.hparams['patch_size'],
                self.hparams['patch_size']
            )
            
        self.logger.info("Train/val/test datasets and samplers configured")
        self.logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def _get_current_split(self) -> str:
        """
        Determine current split based on trainer state.
        
        Returns:
            str: Current split name ('train', 'val', 'test', 'predict')
        """
        if not self.trainer:
            return 'val'  # default
        
        if self.trainer.training:
            return 'train'
        elif self.trainer.validating or self.trainer.sanity_checking:
            return 'val'
        elif self.trainer.testing:
            return 'test'
        elif self.trainer.predicting:
            return 'predict'
        return 'val'  # default

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_batch_sampler is None:
            self.logger.error("Training batch sampler not initialized")
            raise RuntimeError("Training batch sampler not initialized - call setup() first")
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams['num_workers'],
            collate_fn=self.collate_geo,
            persistent_workers=True if self.hparams['num_workers'] > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_sampler is None:
            self.logger.error("Validation sampler not initialized")
            raise RuntimeError("Validation sampler not initialized - call setup() first")
        
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.hparams['batch_size'],
            num_workers=self.hparams['num_workers'],
            collate_fn=self.collate_geo,
            persistent_workers=True if self.hparams['num_workers'] > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_sampler is None:
            self.logger.error("Test sampler not initialized")
            raise RuntimeError("Test sampler not initialized - call setup() first")
        
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.hparams['batch_size'],
            num_workers=self.hparams['num_workers'],
            collate_fn=self.collate_geo,
            persistent_workers=True if self.hparams['num_workers'] > 0 else False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        if self.predict_sampler is None:
            self.logger.error("Prediction sampler not initialized")
            raise RuntimeError("Prediction sampler not initialized - call setup() first")
        
        return DataLoader(
            self.predict_dataset,
            sampler=self.predict_sampler,
            batch_size=self.hparams['batch_size'],
            num_workers=self.hparams['num_workers'],
            collate_fn=self.collate_geo,
            persistent_workers=True if self.hparams['num_workers'] > 0 else False
        )

    def transfer_batch_to_device(
        self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """
        Transfer batch to device with proper handling of different data types.
        
        Args:
            batch: Input batch
            device: Target device
            dataloader_idx: Dataloader index
            
        Returns:
            Batch transferred to device
        """
        if isinstance(batch, dict):
            batch['image'] = batch['image'].to(device)
            if not (self.trainer and self.trainer.predicting):
                if 'mask' in batch:
                    batch['mask'] = batch['mask'].float().to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """
        Apply augmentations after batch transfer to device.
        
        Args:
            batch (Dict[str, Tensor]): Input batch
            dataloader_idx (int): Dataloader index
            
        Returns:
            Dict[str, Tensor]: Augmented batch
        """
        if not self.trainer:
            return batch
            
        # Determine split and get corresponding augmentations
        split = self._get_current_split()
        aug = self._valid_attribute(f"{split}_aug", "val_aug")
        
        # Remove geo information that can't be processed by augmentations
        geo_keys = ['crs', 'bbox']
        geo_info = {k: batch.pop(k) for k in geo_keys if k in batch}
        
        # Apply augmentations
        if isinstance(aug, dict):
            if 'image' in aug and aug['image'] is not None and 'image' in batch:
                batch['image'] = aug['image']({'image': batch['image']})['image']
            if 'mask' in aug and aug['mask'] is not None and 'mask' in batch:
                batch['mask'] = aug['mask']({'image': batch['mask']})['image']
            
            if 'general' in aug and aug['general'] is not None:
                batch = aug['general'](batch)
        
        # Restore geo information
        batch.update(geo_info)
        
        return batch

    @staticmethod        
    def collate_crs_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        """Custom collate function for CRS objects."""
        return batch[0]

    @staticmethod
    def collate_bbox_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        """Custom collate function for BoundingBox objects."""
        return batch[0]

    @staticmethod
    def collate_geo(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        """
        Custom collate function for geographic data.
        
        Args:
            batch: Input batch
            collate_fn_map: Mapping of types to collate functions
            
        Returns:
            Collated batch
        """
        collate_map = {
            torch.Tensor: _utils.collate.collate_tensor_fn,
            CRS: S2PNOAVegetationDataModule.collate_crs_fn,
            BoundingBox: S2PNOAVegetationDataModule.collate_bbox_fn
        }
        return _utils.collate.collate(batch, collate_fn_map=collate_map)
