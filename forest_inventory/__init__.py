"""
Forest Inventory Processing Component

Spanish National Forest Inventory (NFI) biomass data processing pipeline.
Extracts biomass stocks from NFI4 data and integrates forest type information.

This component provides:
- NFI4 Access database processing (.accdb files)
- Volume to biomass conversion using Global Wood Density Database
- Forest type integration from Spanish Forest Map (MFE) data
- Multiple export formats (UTM-specific, combined, year-stratified)
- CRS handling for different Spanish regions

Key Features:
- Processes UTM zones 29, 30, 31 (excludes 28)
- Handles both volume-to-biomass conversion and direct biomass data
- Integrates forest type information from MFE shapefiles
- Calculates below-ground to above-ground biomass ratios
- Exports standardized shapefiles with metadata

Author: Diego Bengochea
"""

from .core.nfi_utils import (
    get_region_UTM,
    get_species_name, 
    get_wood_density,
    compute_biomass,
    load_reference_databases,
    calculate_bgb_ratio,
    create_output_directory,
    get_valid_utm_zones
)

# Import main processing pipeline (will be created in Phase 2)
from .core.nfi_processing import NFIProcessingPipeline

__version__ = "1.0.0"
__component__ = "forest_inventory"

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
    
    # Main pipeline (to be added in Phase 2)
    "NFIProcessingPipeline",
    
    # Component metadata
    "__version__",
    "__component__"
]

# Component configuration
DEFAULT_CONFIG_PATH = "config.yaml"
COMPONENT_NAME = "forest_inventory"

# Supported file formats
SUPPORTED_INPUT_FORMATS = ['.accdb']  # Access database files
SUPPORTED_OUTPUT_FORMATS = ['.shp']   # Shapefiles
SUPPORTED_CRS = ['EPSG:25829', 'EPSG:25830', 'EPSG:25831']  # UTM zones 29, 30, 31

# Data requirements
REQUIRED_SYSTEM_DEPENDENCIES = ['mdb-tools']
REQUIRED_DATA_FILES = [
    'GlobalWoodDensityDatabase.xls',
    'CODIGOS_IFN.csv'
]
REQUIRED_DIRECTORIES = [
    'IFN_4_SP',    # NFI4 database files
    'MFESpain'     # Spanish Forest Map shapefiles
]