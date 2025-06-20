"""
Core canopy height deep learning modules.

This package contains the core deep learning logic for canopy height estimation:
- ModelTrainingPipeline: PyTorch Lightning training workflow
- ModelPredictionPipeline: Large-scale prediction pipeline
- PostProcessingPipeline: Multi-step post-processing workflow
- CanopyHeightRegression: Core Lightning module
- DataModule: Data loading and augmentation

Author: Diego Bengochea
"""

from .model_training import ModelTrainingPipeline
from .model_prediction import ModelPredictionPipeline
from .postprocessing import PostProcessingPipeline

__all__ = [
    "ModelTrainingPipeline",
    "ModelPredictionPipeline",
    "PostProcessingPipeline"
]
