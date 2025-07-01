"""
Biomass Analysis Component

This component provides comprehensive biomass analysis capabilities for the
Iberian Carbon Assessment Pipeline, including:

- Country-level biomass time series with Monte Carlo uncertainty
- Hierarchical biomass aggregation by forest type, landcover, and height
- Interannual biomass difference mapping and transition analysis
- Carbon flux calculations from biomass changes

Components:
    core/: Core processing modules
    scripts/: Executable entry points
    config.yaml: Component configuration

Author: Diego Bengochea
"""

from .core.monte_carlo_analysis import MonteCarloAggregationPipeline
from .core.aggregation_analysis import BiomassAggregationPipeline
from .core.interannual_analysis import InterannualChangePipeline
from .core.carbon_flux_analysis import CarbonFluxPipeline

__version__ = "1.0.0"
__component__ = "biomass_analysis"

__all__ = [
    "MonteCarloAggregationPipeline",
    "BiomassAggregationPipeline",
    "InterannualChangePipeline", 
    "CarbonFluxPipeline"
]
