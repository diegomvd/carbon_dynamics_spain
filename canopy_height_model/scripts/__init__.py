"""
Executable scripts for canopy height deep learning component.

This package contains command-line entry points for the deep learning
pipeline, providing easy access to training, prediction, and post-processing.

Scripts:
    run_training.py: Model training and evaluation
    run_prediction.py: Large-scale prediction
    run_postprocessing.py: Multi-step post-processing
    run_full_pipeline.py: Complete workflow orchestration

Author: Diego Bengochea
"""

from .run_training import main as run_training
from .run_prediction import main as run_prediction
from .run_postprocessing import main as run_postprocessing

__all__ = [
    "run_training",
    "run_prediction", 
    "run_postprocessing"
]
