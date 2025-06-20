"""
Height-diversity aware batch sampling for canopy height regression training.

This module provides specialized batch samplers that prioritize patches with diverse
height distributions to improve model training on canopy height regression tasks.
Uses statistical moments to assess height diversity efficiently and applies
progressive relaxation of constraints for robust batch generation.

The samplers integrate with TorchGeo's sampling framework while adding domain-specific
logic for vegetation height analysis and balanced training data selection.

Author: Diego Bengochea
"""

from typing import List, Iterator, Tuple

# Third-party imports
import torch
import numpy as np

# TorchGeo imports
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.samplers.utils import get_random_bounding_box
from torchgeo.datasets.utils import BoundingBox

# Shared utilities
from shared_utils import load_config, get_logger


class HeightDiversityBatchSampler(RandomBatchGeoSampler):
    """
    Batch sampler that prioritizes patches with diverse height distributions.
    
    This sampler uses statistical moments to assess diversity efficiently and
    applies progressive constraint relaxation to ensure robust batch generation
    even when high-diversity patches are scarce.
    
    The sampler evaluates patches based on height range, standard deviation,
    and interquartile range to select training samples that provide diverse
    learning examples across different vegetation height scenarios.
    """
    
    def __init__(
        self,
        dataset,
        patch_size: int,
        batch_size: int,
        length: int,
        config_path: str = None,
        max_diversity_threshold: float = None,
        min_diversity_threshold: float = None,
        max_attempts: int = None,
        nan_value: float = None,
        initial_max_nodata_ratio: float = None,
        initial_max_low_veg_ratio: float = None
    ):
        """
        Initialize the height diversity batch sampler.
        
        Args:
            dataset: The dataset to sample from
            patch_size (int): Size of patches to sample
            batch_size (int): Number of patches per batch
            length (int): Length of the sampler
            config_path (str, optional): Path to configuration file
            max_diversity_threshold (float, optional): Maximum diversity threshold
            min_diversity_threshold (float, optional): Minimum diversity threshold  
            max_attempts (int, optional): Maximum attempts to find diverse patch
            nan_value (float, optional): Value indicating no-data points
            initial_max_nodata_ratio (float, optional): Initial maximum no-data ratio
            initial_max_low_veg_ratio (float, optional): Initial maximum low vegetation ratio
        """
        super().__init__(dataset, patch_size, batch_size, length)
        
        # Load configuration
        self.config = load_config(config_path, component_name="canopy_height_dl")
        self.logger = get_logger('canopy_height_dl.sampler')
        
        # Use provided values or fall back to configuration defaults
        sampling_config = self.config.get('sampling', {})
        self.max_diversity_threshold = max_diversity_threshold or sampling_config.get('max_diversity_threshold', 10.0)
        self.min_diversity_threshold = min_diversity_threshold or sampling_config.get('min_diversity_threshold', 3.0)
        self.max_attempts = max_attempts or sampling_config.get('max_attempts', 200)
        self.nan_value = nan_value or sampling_config.get('nan_value', -1.0)
        self.initial_max_nodata_ratio = initial_max_nodata_ratio or sampling_config.get('initial_max_nodata_ratio', 0.3)
        self.initial_max_low_veg_ratio = initial_max_low_veg_ratio or sampling_config.get('initial_max_low_veg_ratio', 0.7)
        
        self.dataset = dataset
        
        self.logger.info(f"Height diversity sampler initialized:")
        self.logger.info(f"  Diversity thresholds: {self.min_diversity_threshold} - {self.max_diversity_threshold}")
        self.logger.info(f"  Max attempts: {self.max_attempts}")
        self.logger.info(f"  Quality constraints: nodata<{self.initial_max_nodata_ratio}, low_veg<{self.initial_max_low_veg_ratio}")

    def _print_patch_stats(self, heights: torch.Tensor, score: float, progress: float) -> None:
        """
        Print detailed patch statistics ignoring nan values for debugging.
        
        Args:
            heights (torch.Tensor): Height values in the patch
            score (float): Calculated diversity score
            progress (float): Progress through sampling attempts (0-1)
        """            
        # Get valid heights (non-nan)
        valid_mask = heights != self.nan_value
        valid_heights = heights[valid_mask]
        
        if len(valid_heights) == 0:
            self.logger.debug("Empty patch (all nan values)")
            return
            
        # Calculate statistics on valid data
        stats = {
            "min_height": valid_heights.min().item(),
            "max_height": valid_heights.max().item(),
            "mean_height": valid_heights.mean().item(),
            "median_height": torch.median(valid_heights).item(),
            "std_height": valid_heights.std().item(),
            "diversity_score": score,
            "valid_data_ratio": valid_mask.float().mean().item(),
            "low_veg_ratio": ((valid_heights >= 0) & (valid_heights <= 1)).float().mean().item(),
        }
        
        # Height distribution in meaningful bins
        bins = torch.tensor([0, 1, 2, 4, 8, 12, 16, 20, 25, float('inf')])
        hist = torch.histogram(valid_heights, bins=bins)
        height_dist = {f"{bins[i].item():.1f}-{bins[i+1].item():.1f}m": 
                      f"{(hist.hist[i].item()/len(valid_heights)*100):.1f}%" 
                      for i in range(len(hist.hist))}
        
        self.logger.debug(f"\nPatch Statistics (progress: {progress:.2f}):")
        self.logger.debug(f"Basic Stats:")
        self.logger.debug(f"  Valid Data: {stats['valid_data_ratio']*100:.1f}%")
        self.logger.debug(f"  Low Veg (0-1m): {stats['low_veg_ratio']*100:.1f}%")
        self.logger.debug(f"  Height Range: {stats['min_height']:.1f}m - {stats['max_height']:.1f}m")
        self.logger.debug(f"  Mean Height: {stats['mean_height']:.1f}m")
        self.logger.debug(f"  Median Height: {stats['median_height']:.1f}m")
        self.logger.debug(f"  Height StdDev: {stats['std_height']:.1f}m")
        self.logger.debug(f"  Diversity Score: {stats['diversity_score']:.1f}")
        
        self.logger.debug("Height Distribution:")
        for range_str, percentage in height_dist.items():
            self.logger.debug(f"  {range_str}: {percentage}")
        self.logger.debug("-" * 50)

    def _calculate_diversity_score(self, heights: torch.Tensor) -> float:
        """
        Calculate diversity score using statistical moments.
        
        Higher scores indicate better height diversity combining standard deviation,
        height range, and interquartile range using geometric mean.
        
        Args:
            heights (torch.Tensor): Height values in the patch
            
        Returns:
            float: Diversity score (higher = more diverse)
        """
        # Get valid heights
        valid_mask = heights != self.nan_value
        valid_heights = heights[valid_mask]
        
        if len(valid_heights) == 0:
            return 0.0
            
        # Calculate basic statistics
        mean = valid_heights.mean()
        std = valid_heights.std()
        height_range = valid_heights.max() - valid_heights.min()
        
        # Get interquartile range
        q75, q25 = torch.quantile(valid_heights, torch.tensor([0.75, 0.25]))
        iqr = q75 - q25
        
        # Combine metrics into single score using geometric mean
        # Adding small epsilon to avoid zero scores
        eps = 1e-6
        diversity_score = (std * height_range * (iqr + eps)) ** (1/3)
        
        return diversity_score.item()

    def _check_patch_constraints(self, heights: torch.Tensor, progress: float) -> bool:
        """
        Check nodata and low vegetation constraints with progressive relaxation.
        
        Constraints become more lenient as sampling progresses to ensure
        batch completion even when high-quality patches are scarce.
        
        Args:
            heights (torch.Tensor): Height values in the patch
            progress (float): Progress through sampling attempts (0-1)
            
        Returns:
            bool: True if patch meets current constraints
        """
        valid_mask = heights != self.nan_value
        valid_ratio = valid_mask.float().mean()
        
        # Progressive relaxation of constraints
        max_nodata_ratio = self.initial_max_nodata_ratio
        max_low_veg_ratio = self.initial_max_low_veg_ratio
        
        if progress > 0.5:
            max_nodata_ratio *= 1.5
            max_low_veg_ratio *= 1.5
        if progress > 0.75:
            max_nodata_ratio *= 2
            max_low_veg_ratio *= 2
        if progress > 0.9:
            return True  # Accept any patch near the end
            
        if valid_ratio < (1 - max_nodata_ratio):
            return False
            
        valid_heights = heights[valid_mask]
        if len(valid_heights) == 0:
            return False
            
        low_veg_ratio = ((valid_heights >= 0) & (valid_heights <= 1.5)).float().mean()
        return low_veg_ratio <= max_low_veg_ratio

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """
        Iterator that yields batches of diverse patches.
        
        Searches for patches that meet diversity and quality thresholds,
        applying progressive relaxation to ensure batch completion.
        
        Yields:
            List[BoundingBox]: List of bounding boxes for diverse patches
        """        
        for batch_num in range(len(self)):
            # Choose initial random region
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)
            
            # Initialize lists to store selected bboxes
            selected_bboxes = []
            attempts = 0

            while len(selected_bboxes) < self.batch_size and attempts < self.max_attempts:
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                progress = float(attempts) / float(self.max_attempts)

                # Get patch data
                try:
                    # Access the target dataset (assuming vegetation data is at index 1)
                    if hasattr(self.dataset, 'datasets') and len(self.dataset.datasets) > 1:
                        patch_data = self.dataset.datasets[1][bounding_box]
                    else:
                        # Fallback if dataset structure is different
                        patch_data = self.dataset[bounding_box]
                    
                    if 'mask' not in patch_data:
                        attempts += 1
                        continue
                except Exception as e:
                    self.logger.debug(f"Error accessing patch data: {e}")
                    attempts += 1
                    continue
                
                # Check basic constraints
                if not self._check_patch_constraints(patch_data['mask'], progress):
                    attempts += 1
                    continue
                
                # Calculate diversity score
                score = self._calculate_diversity_score(patch_data['mask'])

                # Determine current diversity threshold with progressive relaxation
                diversity_threshold = self.max_diversity_threshold
                if progress > 0.25 and len(selected_bboxes) <= float(self.batch_size) * 0.25:
                    diversity_threshold = self.min_diversity_threshold + (self.max_diversity_threshold - self.min_diversity_threshold) * (1.0 - 0.25)
                if progress > 0.5 and len(selected_bboxes) <= float(self.batch_size) * 0.5:
                    diversity_threshold = self.min_diversity_threshold + (self.max_diversity_threshold - self.min_diversity_threshold) * (1.0 - 0.5)
                if progress > 0.75 and len(selected_bboxes) <= float(self.batch_size) * 0.75:
                    diversity_threshold = self.min_diversity_threshold
                if progress > 0.90 and len(selected_bboxes) <= float(self.batch_size):
                    diversity_threshold = 3.0
                    
                # Accept patch if it meets diversity threshold
                if score >= diversity_threshold: 
                    selected_bboxes.append(bounding_box)
                    
                    # Optional: print patch stats for debugging (only in debug mode)
                    if self.logger.level <= 10:  # DEBUG level
                        self._print_patch_stats(patch_data['mask'], score, progress)
                    
                attempts += 1
            
            # Fill remaining slots with random patches if needed
            if len(selected_bboxes) < self.batch_size:
                remaining = self.batch_size - len(selected_bboxes)
                self.logger.debug(f"Filling {remaining} remaining slots with random patches")
                for _ in range(remaining):
                    bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                    selected_bboxes.append(bounding_box)
                
            assert len(selected_bboxes) == self.batch_size, f"Invalid batch size: {len(selected_bboxes)}"
            
            if batch_num % 100 == 0:  # Log every 100 batches
                self.logger.debug(f"Generated batch {batch_num} with {len(selected_bboxes)} patches after {attempts} attempts")
                
            yield selected_bboxes
    
    def get_sampler_stats(self) -> dict:
        """
        Get statistics about the sampler configuration.
        
        Returns:
            dict: Dictionary with sampler statistics
        """
        return {
            'patch_size': self.size,
            'batch_size': self.batch_size,
            'length': len(self),
            'max_diversity_threshold': self.max_diversity_threshold,
            'min_diversity_threshold': self.min_diversity_threshold,
            'max_attempts': self.max_attempts,
            'nan_value': self.nan_value,
            'initial_max_nodata_ratio': self.initial_max_nodata_ratio,
            'initial_max_low_veg_ratio': self.initial_max_low_veg_ratio
        }
