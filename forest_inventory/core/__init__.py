"""
Forest Inventory Core Processing Modules

Core functionality for Spanish National Forest Inventory (NFI) data processing.
Contains the main processing algorithms and utility functions.

Modules:
    nfi_utils: Utility functions for biomass calculation, CRS handling, and data management
    nfi_processing: Main processing pipeline class

Author: Diego Bengochea
"""

# Import all utility functions
from .nfi_utils import (
    get_region_UTM,
    get_species_name,
    get_wood_density, 
    compute_biomass,
    load_reference_databases,
    calculate_bgb_ratio,
    create_output_directory,
    get_valid_utm_zones
)

# Import main processing pipeline
from .nfi_processing import NFIProcessingPipeline

__all__ = [
    # Utility functions
    "get_region_UTM",
    "get_species_name", 
    "get_wood_density",
    "compute_biomass",
    "load_reference_databases", 
    "calculate_bgb_ratio",
    "create_output_directory",
    "get_valid_utm_zones",
    
    # Main processing pipeline
    "NFIProcessingPipeline"
]