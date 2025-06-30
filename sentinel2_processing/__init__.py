"""
Sentinel-2 Processing Component

Comprehensive pipeline for creating Sentinel-2 summer mosaics over Spain using
distributed computing with STAC catalog integration and optimized memory management.

This component provides:
- Large-scale distributed mosaic processing using Dask clusters
- STAC catalog integration for automated data discovery
- Scene Classification Layer (SCL) masking and optimal scene selection
- Spatial downsampling and merging for analysis-ready products
- Quality assurance through missing tile detection and gap analysis
- Robustness assessment for optimal scene selection parameters
- Interannual consistency analysis for temporal stability validation

Key Features:
- Processes Sentinel-2 L2A satellite imagery over Spain's territory
- Handles distributed computing with optimized cluster management
- Integrates with AWS STAC catalog for data access
- Creates summer mosaics (June-September) with configurable parameters
- Provides comprehensive post-processing and analysis workflows
- Exports standardized GeoTIFF mosaics with metadata

Author: Diego Bengochea
"""

# Import core processing pipelines
from .core.mosaic_processing import MosaicProcessingPipeline

# Import utility functions
from .core.s2_utils import (
    create_scl_mask,
    mask_scene,
    select_best_scenes,
    create_processing_tiles,
    create_processing_list,
    get_tile_bounding_box,
    search_catalog,
    load_dataset,
    create_dataset,
    setup_optimized_cluster,
    create_output_directory,
    generate_output_filename
)

# Import post-processing classes
from .core.postprocessing import (
    DownsamplingMergingProcessor,
    MissingTilesAnalyzer,
    RobustnessAssessor,
    InterannualConsistencyAnalyzer
)

# Import script entry points
from .scripts.run_mosaic_processing import main as run_mosaic_processing
from .scripts.run_downsampling import main as run_downsampling
from .scripts.run_missing_tiles import main as run_missing_tiles
from .scripts.run_robustness_analysis import main as run_robustness_analysis
from .scripts.run_consistency_analysis import main as run_consistency_analysis
from .scripts.run_postprocessing import main as run_postprocessing

__version__ = "1.0.0"
__component__ = "sentinel2_processing"

__all__ = [
    # Main processing pipeline
    "MosaicProcessingPipeline",
    
    # Utility functions
    "create_scl_mask",
    "mask_scene",
    "select_best_scenes",
    "create_processing_tiles",
    "create_processing_list",
    "get_tile_bounding_box",
    "search_catalog",
    "load_dataset",
    "create_dataset",
    "setup_optimized_cluster",
    "create_output_directory",
    "generate_output_filename",
    
    # Post-processing classes
    "DownsamplingMergingProcessor",
    "MissingTilesAnalyzer",
    "RobustnessAssessor",
    "InterannualConsistencyAnalyzer",
    
    # Script entry points
    "run_mosaic_processing",
    "run_downsampling",
    "run_missing_tiles",
    "run_robustness_assessment",
    "run_consistency_analysis",
    "run_postprocessing",
    
    # Component metadata
    "__version__",
    "__component__"
]

# Component configuration
DEFAULT_CONFIG_PATH = "config.yaml"
COMPONENT_NAME = "sentinel2_processing"

# Supported data formats
SUPPORTED_INPUT_COLLECTIONS = ['sentinel-2-l2a']  # STAC collections
SUPPORTED_OUTPUT_FORMATS = ['.tif']               # GeoTIFF outputs
SUPPORTED_CRS = ['EPSG:25830']                    # UTM zone 30N (Spain)

# Processing specifications
DEFAULT_TILE_SIZE = 12288                         # Processing tile size (12.288 km)
DEFAULT_SCALE_FACTOR = 10                         # Default downsampling factor
DEFAULT_N_SCENES = 12                             # Default scenes per mosaic
PROCESSING_MONTHS = [6, 7, 8, 9]                  # Summer months (June-September)

# Data requirements
REQUIRED_EXTERNAL_DEPENDENCIES = [
    'dask[distributed]',    # Distributed computing
    'rasterio',             # Geospatial raster I/O
    'xarray',               # N-dimensional arrays
    'rioxarray',            # Rasterio xarray extension
    'odc-stac',             # STAC data loading
    'pystac-client',        # STAC catalog client
    'geopandas',            # Geospatial vector data
    'odc.geo',              # ODC geospatial utilities
    'matplotlib',           # Plotting and visualization
    'seaborn',              # Statistical visualization
    'scipy',                # Scientific computing
    'tqdm',                 # Progress bars
    'psutil'                # System monitoring
]

REQUIRED_INPUT_DATA = [
    'spain_polygon.shp'     # Spain territory boundary shapefile
]

REQUIRED_STAC_ACCESS = [
    'https://earth-search.aws.element84.com/v1'  # AWS STAC catalog
]

# Processing capabilities
PROCESSING_FEATURES = [
    'Large-scale distributed processing across Spain',
    'STAC catalog integration for automated data discovery',
    'Scene Classification Layer (SCL) masking',
    'Optimal scene selection based on valid pixel coverage',
    'Memory-optimized cluster management',
    'Spatial downsampling and yearly merging',
    'Missing tile detection and gap analysis',
    'Robustness assessment for parameter optimization',
    'Interannual consistency validation',
    'Comprehensive error handling and recovery'
]

# Performance specifications
PERFORMANCE_SPECS = {
    'spain_coverage': 'Complete territorial coverage',
    'spatial_resolution': '10m native, configurable downsampling',
    'temporal_resolution': 'Annual summer composites',
    'processing_scale': 'Tile-based distributed processing',
    'memory_management': 'Optimized for large-scale satellite imagery',
    'cluster_support': 'Dask LocalCluster with resource monitoring'
}