"""
Core processing modules for climate-biomass analysis.

This package contains the core processing classes for the climate-biomass
analysis pipeline, including climate data processing, bioclimatic variables
calculation, biomass integration, spatial analysis, ML optimization, and
SHAP analysis.

Modules:
    climate_raster_conversion: GRIB to GeoTIFF conversion and harmonization
    bioclim_calculation: Bioclimatic variables calculation and anomaly analysis
    biomass_integration: Integration of biomass changes with climate data
    spatial_analysis: Spatial autocorrelation analysis and clustering
    optimization_pipeline: Machine learning optimization and feature selection
    shap_analysis: Comprehensive SHAP analysis for model interpretability

Author: Diego Bengochea
"""

from .climate_raster_conversion import ClimateRasterConversionPipeline
from .bioclim_calculation import BioclimCalculationPipeline
from .biomass_integration import BiomassIntegrationPipeline
from .spatial_analysis import SpatialAnalysisPipeline
from .optimization_pipeline import OptimizationPipeline
from .shap_analysis import ShapAnalysisPipeline

__all__ = [
    "ClimateRasterConversionPipeline",
    "BioclimCalculationPipeline",
    "BiomassIntegrationPipeline", 
    "SpatialAnalysisPipeline",
    "OptimizationPipeline",
    "ShapAnalysisPipeline"
]