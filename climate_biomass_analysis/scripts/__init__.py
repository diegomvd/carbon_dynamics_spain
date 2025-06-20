"""
Executable scripts for climate-biomass analysis component.

This package contains command-line entry points for the climate-biomass
analysis pipeline, providing easy access to component functionality.

Scripts:
    run_climate_processing.py: Climate data processing (GRIB to GeoTIFF)
    run_bioclim_calculation.py: Bioclimatic variables calculation
    run_biomass_integration.py: Biomass-climate data integration
    run_spatial_analysis.py: Spatial autocorrelation and clustering
    run_optimization.py: Machine learning optimization
    run_full_pipeline.py: Complete pipeline orchestrator

Author: Diego Bengochea
"""

# Package level imports for easy access
from .run_climate_processing import main as run_climate_processing
from .run_bioclim_calculation import main as run_bioclim_calculation
from .run_biomass_integration import main as run_biomass_integration
from .run_spatial_analysis import main as run_spatial_analysis
from .run_optimization import main as run_optimization
from .run_full_pipeline import main as run_full_pipeline

__all__ = [
    "run_climate_processing",
    "run_bioclim_calculation",
    "run_biomass_integration", 
    "run_spatial_analysis",
    "run_optimization",
    "run_full_pipeline"
]