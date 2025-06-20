"""
Executable scripts for biomass analysis component.

This package contains command-line entry points for the biomass analysis
pipeline, providing easy access to component functionality.

Scripts:
    run_country_trend.py: Country-level biomass time series analysis
    run_forest_type_trend.py: Forest type biomass aggregation
    run_landcover_trend.py: Landcover biomass aggregation
    run_height_bin_trend.py: Height bin biomass aggregation
    run_interannual_differences.py: Interannual difference mapping
    run_transition_analysis.py: Biomass transition distributions
    run_carbon_fluxes.py: Carbon flux calculations
    run_full_analysis.py: Complete pipeline orchestrator

Author: Diego Bengochea
"""

# Package level imports for easy access
from .run_country_trend import main as run_country_trend
from .run_forest_type_trend import main as run_forest_type_trend
from .run_landcover_trend import main as run_landcover_trend
from .run_height_bin_trend import main as run_height_bin_trend
from .run_interannual_differences import main as run_interannual_differences
from .run_transition_analysis import main as run_transition_analysis
from .run_carbon_fluxes import main as run_carbon_fluxes
from .run_full_analysis import main as run_full_analysis

__all__ = [
    "run_country_trend",
    "run_forest_type_trend",
    "run_landcover_trend",
    "run_height_bin_trend",
    "run_interannual_differences",
    "run_transition_analysis",
    "run_carbon_fluxes",
    "run_full_analysis"
]
