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


from .datasets import S2Mosaic, PNOAVegetation, KorniaIntersectionDataset
from .balanced_geo_samplers import HeightDiversityBatchSampler
from .pnoa_vegetation_transforms import (
    PNOAVegetationInput0InArtificialSurfaces, 
    PNOAVegetationRemoveAbnormalHeight, 
    PNOAVegetationLogTransform, 
    S2Scaling
)


class S2PNOAVegetationDataModule(GeoDataModule):
    """
    PyTorch Lightning DataModule for Sentinel-2 and PNOA vegetation data.
    
    This DataModule provides comprehensive data loading, preprocessing, and augmentation
    for canopy height regression tasks. Supports multiple modes (train/val/test/predict)
    with appropriate sampling strategies and augmentation pipelines.
    """

    def __init__(
        self,
        sentinel2_dir: str = None,
        pnoa_dir: str = None,
        patch_size: int = None,
        batch_size: int = None,
        length: int = None, 
        num_workers: int = 0, 
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
        sentinel2_dir = sentinel2_dir or str(SENTINEL2_MOSAICS_DIR)
        pnoa_dir = pnoa_dir or str(ALS_CANOPY_HEIGHT_PROCESSED_DIR)
        patch_size = patch_size or self.config['training'].get('patch_size', 256)
        batch_size = batch_size or self.config['training']['batch_size']
        length = length or self.config['training'].get('length', 100)
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
            'sentinel2_dir': sentinel2_dir,
            'pnoa_dir': pnoa_dir,
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
        
        self.logger.info(f"DataModule initialized:")
        self.logger.info(f"  Sentinel-2 directory: {sentinel2_dir}")
        self.logger.info(f"  PNOA directory: {pnoa_dir}")
        self.logger.info(f"Training patch size: {patch_size}, prediction patch size: {predict_patch_size}")
       
    def _setup_augmentations(self) -> None:
        """Initialize all augmentation pipelines with configuration parameters."""
        self.logger.info("Setting up augmentation pipelines...")
        
        # Load augmentation configuration
        aug_config = self.config['augmentation']
        
        # Base normalization and mask transforms - CRITICAL for PNOA data
        base_mask_transforms = K.AugmentationSequential(
            PNOAVegetationRemoveAbnormalHeight(
                hmax=aug_config['max_height_threshold'],
                nan_target=self.hparams['nan_target']
            ),
            PNOAVegetationInput0InArtificialSurfaces(),
            PNOAVegetationLogTransform(),
            data_keys=None,
            keepdim=True
        )
        
        # Training augmentations - RESTORED original sophisticated pipeline
        self.train_aug = {
            'general': K.AugmentationSequential(
                # Geometric augmentations
                K.RandomHorizontalFlip(p=aug_config['horizontal_flip_p']),
                K.RandomVerticalFlip(p=aug_config['vertical_flip_p']),
                K.RandomRotation(degrees=aug_config['rotation_degrees'], p=aug_config['rotation_p']),
                data_keys=None,
                keepdim=True,
                same_on_batch=False,
            ),
            'image': K.AugmentationSequential(
                S2Scaling(),  
                # Intensity augmentations for mosaiced data
                K.RandomGaussianNoise(
                    mean=0.0, 
                    std=aug_config['gaussian_noise_std'], 
                    p=aug_config['gaussian_noise_p']
                ),
                K.RandomBrightness(
                    brightness=aug_config['brightness_factor'], 
                    p=aug_config['brightness_p']
                ),
                K.RandomContrast(
                    contrast=aug_config['contrast_factor'], 
                    p=aug_config['contrast_p']
                ),
                data_keys=None,
                keepdim=True,
            ),
            'mask': K.AugmentationSequential(
                base_mask_transforms,
                K.RandomGaussianNoise(
                    mean=0.0,
                    std=aug_config['mask_noise_std'],
                    p=aug_config['mask_noise_p']
                ),
                data_keys=None,
                keepdim=True
            )
        }

        # Validation/Test augmentations 
        self.val_aug = self.test_aug = {
            'image': K.AugmentationSequential(S2Scaling(), data_keys=None, keepdim=True),
            'mask': base_mask_transforms 
        }
        
        # Prediction augmentations - Only S2 scaling needed
        self.predict_aug = {
            'image': K.AugmentationSequential(S2Scaling(), data_keys=None, keepdim=True)
        }
        
        self.logger.info("Augmentation pipelines configured")

    
    def _valid_attribute(self, name: str, fallback: str):
        """Get valid attribute with fallback."""
        if hasattr(self, name):
            attr = getattr(self, name)
            # Return the attribute if it exists and is not None
            if attr is not None:
                return attr
        return getattr(self, fallback, {})
    
    def setup(self, stage: str) -> None:
        """
        Set up datasets for each stage.
        
        Args:
            stage (str): Current stage ('fit', 'test', 'predict')
        """
        self.logger.info(f"Setting up datasets for stage: {stage}")
        
        # Check if required dataset classes are available
        if S2Mosaic is None or PNOAVegetation is None:
            self.logger.error("S2Mosaic dataset class not available - please copy datasets.py to core/")
            raise ImportError("Required dataset classes not available")
        
        s2 = S2Mosaic(self.hparams['sentinel2_dir'])
        
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
        
        pnoa_vegetation = PNOAVegetation(self.hparams['pnoa_dir'])
        dataset = KorniaIntersectionDataset(s2, pnoa_vegetation)
        
        # Split dataset using configuration
        split_ratios = [0.8, 0.1, 0.1]  # Default split ratios
        if 'data' in self.config and 'split_ratios' in self.config['data']:
            split_ratios = self.config['data']['split_ratios']
        
        grid_size = 64  # Default grid size
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
        aug = self._valid_attribute(f"{split}_aug", "aug")
        
        # Remove geo information that can't be processed by augmentations
        batch = {k: v for k, v in batch.items() if k not in ['crs', 'bbox']}
        
        # Apply augmentations based on split
        if 'image' in aug and 'image' in batch:
            batch['image'] = aug['image']({'image': batch['image']})['image']
        if 'mask' in aug and 'mask' in batch:
            batch['mask'] = aug['mask']({'image': batch['mask']})['image']
        
        # Apply augmentations
        if 'general' in aug:
            batch = aug['general'](batch)
            # Hardware-specific optimization - ensure tensors are on correct device
            if 'image' in batch and hasattr(batch['image'], 'device'):
                # Keep tensors on their current device (MPS, CUDA, or CPU)
                current_device = batch['image'].device
                for key in batch:
                    if hasattr(batch[key], 'to'):
                        batch[key] = batch[key].to(current_device)
        
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
