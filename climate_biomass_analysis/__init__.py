"""
Climate-Biomass Analysis Component

This component provides comprehensive climate-biomass relationship analysis
for the Iberian Carbon Assessment Pipeline, including:

- Climate data processing (GRIB to GeoTIFF conversion)
- Bioclimatic variables calculation (bio1-bio19)
- Biomass-climate data integration and harmonization
- Spatial autocorrelation analysis and clustering
- Machine learning optimization for predictor selection
- Bayesian optimization with spatial cross-validation
- SHAP analysis for model interpretability and feature importance

Pipeline Workflow:
    1. Climate Processing: Convert and harmonize climate data
    2. Bioclimatic Calculation: Calculate bio1-bio19 variables and anomalies
    3. Biomass Integration: Integrate biomass changes with climate anomalies
    4. Spatial Analysis: Compute spatial autocorrelation and create clusters
    5. ML Optimization: Optimize feature selection using Bayesian methods
    6. SHAP Analysis: Comprehensive model interpretability analysis

Components:
    core/: Core processing modules
    scripts/: Executable entry points
    config.yaml: Component configuration
    visualization_config.yaml: Visualization-specific configuration

Author: Diego Bengochea
"""

from .core.climate_raster_conversion import ClimateRasterConversionPipeline
from .core.bioclim_calculation import BioclimCalculationPipeline
from .core.biomass_integration import BiomassIntegrationPipeline
from .core.spatial_analysis import SpatialAnalysisPipeline
from .core.optimization_pipeline import OptimizationPipeline
from .core.shap_analysis import ShapAnalysisPipeline

__version__ = "1.1.0"
__component__ = "climate_biomass_analysis"

__all__ = [
    "ClimateRasterConversionPipeline",
    "BioclimCalculationPipeline",
    "BiomassIntegrationPipeline", 
    "SpatialAnalysisPipeline",
    "OptimizationPipeline",
    "ShapAnalysisPipeline"
]