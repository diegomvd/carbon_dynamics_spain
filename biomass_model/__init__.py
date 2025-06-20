"""
Biomass Estimation Component

This component provides comprehensive biomass estimation capabilities for the
Iberian Carbon Assessment Pipeline, including:

- Allometric relationship fitting from NFI data
- Monte Carlo uncertainty quantification  
- Multi-scale biomass mapping
- Forest type specific processing
- Land cover masking and post-processing

Components:
    core/: Core processing modules
    scripts/: Executable entry points
    config.yaml: Component configuration

Author: Diego Bengochea
"""

from .core.biomass_estimation import BiomassEstimationPipeline
from .core.allometry import AllometryManager
from .core.monte_carlo import MonteCarloEstimator

__version__ = "1.0.0"
__component__ = "biomass_estimation"

__all__ = [
    "BiomassEstimationPipeline",
    "AllometryManager", 
    "MonteCarloEstimator"
]
