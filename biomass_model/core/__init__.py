"""
Core biomass estimation modules.

This package contains the core processing logic for biomass estimation:
- BiomassEstimationPipeline: Main processing pipeline
- AllometryManager: Allometric relationship management  
- MonteCarloEstimator: Uncertainty quantification
- IO utilities: Raster I/O and data management
- Dask utilities: Distributed computing support

Author: Diego Bengochea
"""

from .biomass_estimation import BiomassEstimationPipeline
from .allometry import AllometryManager
from .monte_carlo import MonteCarloEstimator

__all__ = [
    "BiomassEstimationPipeline",
    "AllometryManager", 
    "MonteCarloEstimator"
]
