"""
Core biomass estimation modules.

This package contains the core processing logic for biomass estimation:
- BiomassEstimationPipeline: Main processing pipeline
- AllometryManager: Allometric relationship management  
- MonteCarloEstimator: Uncertainty quantification
- Allometry fitting: Complete allometry fitting pipeline with hierarchical processing
- IO utilities: Raster I/O and data management
- Dask utilities: Distributed computing support

Updated to use centralized path constants instead of class-based approach.

Author: Diego Bengochea
"""

from .biomass_estimation import BiomassEstimationPipeline
from .allometry import AllometryManager
from .monte_carlo import MonteCarloEstimator

# Import allometry fitting modules
from .allometry_fitting import AllometryFittingPipeline

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

# Import I/O utilities
from .biomass_utils import BiomassUtils

# Import processing pipelines
from .land_use_masking import LandUseMaskingPipeline
from .forest_type_merging import ForestTypeMergingPipeline

__all__ = [
    # Core pipeline components
    "BiomassEstimationPipeline",
    "AllometryManager", 
    "MonteCarloEstimator",
    "AllometryFittingPipeline",
    
    # Allometry utilities
    "AllometryResults",
    "BGBRatioResults",
    "create_training_dataset",
    "sample_height_at_points", 
    "validate_height_biomass_data",
    "validate_ratio_data",
    "remove_outliers",
    "remove_ratio_outliers",
    "process_hierarchy_levels",
    
    # I/O utilities
    "RasterManager",
    
    # Processing pipelines
    "LandUseMaskingPipeline",
    "ForestTypeMergingPipeline", 
]
