"""
ALS PNOA Processing Component

Processes PNOA LiDAR tiles for training data preparation by selecting tiles 
that intersect with Sentinel-2 mosaics and standardizing their format.

This component provides:
- Spatial intersection analysis between Sentinel-2 and PNOA tiles
- Temporal filtering by target years
- Standardized file naming and format conversion
- Deduplication of redundant tiles

Key Features:
- Processes multiple UTM zones (29, 30, 31)
- Filters tiles by year using coverage polygon metadata
- Converts to standardized naming: PNOA_{year}_{tile_id}_epsg_25830.tif
- Quality control and validation

Author: Diego Bengochea
"""

from .core.pnoa_processor import PNOAProcessor

__version__ = "1.0.0"
__component__ = "als_pnoa"

__all__ = [
    "PNOAProcessor"
]

# Component configuration
DEFAULT_CONFIG_PATH = "config.yaml"
COMPONENT_NAME = "als_pnoa"

# Supported UTM zones for PNOA data
SUPPORTED_UTM_ZONES = [29, 30, 31]

# Target coordinate system (primary UTM zone for Spain)
DEFAULT_TARGET_CRS = "EPSG:25830"

# Expected file patterns
FILE_PATTERNS = {
    'sentinel2_input': 'sentinel2_mosaic_{year}_*.tif',
    'pnoa_coverage': '*.shp',
    'pnoa_data': 'NDSM-VEGETACION-*-COB2.tif',
    'output': 'PNOA_{year}_NDSM-VEGETACION-{utm}-{tile_id}-COB2_epsg_{crs_code}.tif'
}

# Data requirements
REQUIRED_FIELDS = {
    'coverage_shapefile': ['FECHA', 'PATH'],  # Year and file path fields
    'input_directories': [
        'sentinel2_mosaics',
        'pnoa_coverage', 
        'pnoa_lidar_data'
    ]
}

# Processing parameters
DEFAULT_YEARS = [2017, 2018, 2019, 2020, 2021]
DEFAULT_OUTPUT_FORMAT = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'compress': 'lzw',
    'nodata': -9999
}