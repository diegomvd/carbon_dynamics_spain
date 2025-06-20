"""
Canopy Height Deep Learning Component

This component provides deep learning-based canopy height estimation using
Sentinel-2 and PNOA LiDAR data with PyTorch Lightning framework.

Components:
    core/: Core deep learning modules
    scripts/: Executable entry points  
    config.yaml: Component configuration

Features:
- PyTorch Lightning training pipeline
- Multi-scale prediction with TorchGeo
- Advanced post-processing workflows
- Distributed training support
- Comprehensive model evaluation

Author: Diego Bengochea
"""

from .core.model_training import ModelTrainingPipeline
from .core.model_prediction import ModelPredictionPipeline
from .core.postprocessing import PostProcessingPipeline

__version__ = "1.0.0"
__component__ = "canopy_height_dl"

__all__ = [
    "ModelTrainingPipeline",
    "ModelPredictionPipeline",
    "PostProcessingPipeline"
]
