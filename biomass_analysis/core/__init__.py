"""
Core processing modules for biomass analysis.

This package contains the core processing classes for the biomass analysis
pipeline, including Monte Carlo uncertainty quantification, aggregation
analysis, interannual analysis, and carbon flux calculations.

Modules:
    monte_carlo_analysis: Country-level biomass analysis with Monte Carlo uncertainty
    aggregation_analysis: Hierarchical biomass aggregation by various classifications
    interannual_analysis: Interannual difference mapping and transition distributions
    carbon_flux_analysis: Carbon flux calculations from Monte Carlo samples

Author: Diego Bengochea
"""

from .monte_carlo_analysis import MonteCarloAnalyzer
from .aggregation_analysis import BiomassAggregator
from .interannual_analysis import InterannualAnalyzer
from .carbon_flux_analysis import CarbonFluxAnalyzer

__all__ = [
    "MonteCarloAnalyzer",
    "BiomassAggregator",
    "InterannualAnalyzer",
    "CarbonFluxAnalyzer"
]
