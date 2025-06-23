"""
Core biomass estimation modules.

This package contains the core processing logic for biomass estimation:
- BiomassEstimationPipeline: Main processing pipeline
- AllometryManager: Allometric relationship management  
- MonteCarloEstimator: Uncertainty quantification
- Allometry fitting: Complete allometry fitting pipeline with hierarchical processing
- IO utilities: Raster I/O and data management
- Dask utilities: Distributed computing support

Author: Diego Bengochea
"""

from .biomass_estimation import BiomassEstimationPipeline
from .allometry import AllometryManager
from .monte_carlo import MonteCarloEstimator

# Import new allometry fitting modules
from .allometry_fitting import (
    run_allometry_fitting_pipeline, 
    save_allometry_results,
    fit_height_agb_allometry,
    calculate_bgb_ratios,
    process_hierarchical_allometries
)

from .allometry_utils import (
    AllometryResults,
    BGBRatioResults, 
    create_training_dataset,
    sample_height_at_points,
    validate_height_biomass_data,
    validate_ratio_data,
    remove_outliers,
    remove_ratio_outliers,
    process_hierarchy_levels
)

__all__ = [
    # Core pipeline components
    "BiomassEstimationPipeline",
    "AllometryManager", 
    "MonteCarloEstimator",
    
    # Allometry fitting pipeline
    "run_allometry_fitting_pipeline",
    "save_allometry_results", 
    "fit_height_agb_allometry",
    "calculate_bgb_ratios",
    "process_hierarchical_allometries",
    
    # Allometry utilities
    "AllometryResults",
    "BGBRatioResults",
    "create_training_dataset",
    "sample_height_at_points", 
    "validate_height_biomass_data",
    "validate_ratio_data",
    "remove_outliers",
    "remove_ratio_outliers",
    "process_hierarchy_levels"
]