"""
Executable scripts for biomass estimation component.

This package contains command-line entry points for the biomass estimation
pipeline, providing easy access to component functionality.

Scripts:
    run_biomass_estimation.py: Main biomass estimation pipeline
    run_allometry_fitting.py: Allometric relationship fitting
    run_masking.py: Annual cropland masking
    run_merging.py: Forest type merging

Author: Diego Bengochea
"""

# Package level imports for easy access
from .run_biomass_estimation import main as run_biomass_estimation
from .run_allometry_fitting import main as run_allometry_fitting
from .run_masking import main as run_masking
from .run_merging import main as run_merging

__all__ = [
    "run_biomass_estimation",
    "run_allometry_fitting", 
    "run_masking",
    "run_merging"
]
