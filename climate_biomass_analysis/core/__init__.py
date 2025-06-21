"""
Core processing modules for climate-biomass analysis.

This package contains the core processing classes for the climate-biomass
analysis pipeline, including climate data processing, bioclimatic variables
calculation, biomass integration, spatial analysis, ML optimization, and
SHAP analysis.

Modules:
    climate_raster_processing: GRIB to GeoTIFF conversion and harmonization
    bioclim_calculation: Bioclimatic variables calculation and anomaly analysis
    biomass_integration: Integration of biomass changes with climate data
    spatial_analysis: Spatial autocorrelation analysis and clustering
    optimization_pipeline: Machine learning optimization and feature selection
    shap_analysis: Comprehensive SHAP analysis for model interpretability

Author: Diego Bengochea
"""

from .climate_raster_processing import ClimateProcessor
from .bioclim_calculation import BioclimCalculator
from .biomass_integration import BiomassIntegrator
from .spatial_analysis import SpatialAnalyzer
from .optimization_pipeline import OptimizationPipeline
from .shap_analysis import ShapAnalyzer

__all__ = [
    "ClimateProcessor",
    "BioclimCalculator",
    "BiomassIntegrator", 
    "SpatialAnalyzer",
    "OptimizationPipeline",
    "ShapAnalyzer"
]