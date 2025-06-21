"""
Sentinel-2 Processing Core Modules

Core functionality for Sentinel-2 satellite imagery processing including mosaic
creation, post-processing workflows, and analysis pipelines. Contains the main
processing algorithms and utility functions for large-scale distributed computing.

Modules:
    s2_utils: Utility functions for STAC integration, geospatial operations, and cluster management
    mosaic_processing: Main pipeline class for distributed mosaic creation
    postprocessing: Comprehensive post-processing workflows and analysis tools

Author: Diego Bengochea
"""

# Import all utility functions from s2_utils
from .s2_utils import (
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

# Import main processing pipeline
from .mosaic_processing import MosaicProcessingPipeline

# Import all post-processing classes
from .postprocessing import (
    DownsamplingMergingProcessor,
    MissingTilesAnalyzer,
    RobustnessAssessor,
    InterannualConsistencyAnalyzer
)

__all__ = [
    # Utility functions - Scene Classification and Masking
    "create_scl_mask",
    "mask_scene",
    "select_best_scenes",
    
    # Utility functions - Geospatial Operations
    "create_processing_tiles",
    "create_processing_list", 
    "get_tile_bounding_box",
    
    # Utility functions - STAC Integration
    "search_catalog",
    "load_dataset",
    "create_dataset",
    
    # Utility functions - Cluster Management
    "setup_optimized_cluster",
    
    # Utility functions - File Operations
    "create_output_directory",
    "generate_output_filename",
    
    # Main processing pipeline
    "MosaicProcessingPipeline",
    
    # Post-processing workflows
    "DownsamplingMergingProcessor",
    "MissingTilesAnalyzer",
    "RobustnessAssessor",
    "InterannualConsistencyAnalyzer"
]

# Core module capabilities
CORE_CAPABILITIES = {
    'mosaic_processing': {
        'description': 'Main distributed pipeline for Sentinel-2 mosaic creation',
        'features': [
            'STAC catalog integration',
            'Scene Classification Layer masking',
            'Optimal scene selection algorithms',
            'Distributed cluster management',
            'Memory-optimized processing'
        ]
    },
    'postprocessing': {
        'description': 'Comprehensive post-processing and analysis workflows',
        'features': [
            'Spatial downsampling and merging',
            'Missing tile detection and analysis',
            'Robustness assessment for parameter optimization',
            'Interannual consistency validation'
        ]
    },
    's2_utils': {
        'description': 'Core utility functions for Sentinel-2 processing',
        'features': [
            'Geospatial operations and coordinate transformations',
            'STAC catalog search with recursive cloud thresholds',
            'Distributed computing with optimized cluster setup',
            'File I/O and standardized naming conventions'
        ]
    }
}

# Algorithm specifications
ALGORITHM_SPECS = {
    'scene_selection': {
        'method': 'Valid pixel percentage ranking',
        'criteria': 'Scene Classification Layer (SCL) masking',
        'optimization': 'Median composite from best N scenes'
    },
    'distributed_processing': {
        'framework': 'Dask LocalCluster',
        'optimization': 'Memory monitoring and garbage collection',
        'scalability': 'Configurable workers and memory limits'
    },
    'geospatial_processing': {
        'crs': 'EPSG:25830 (UTM zone 30N)',
        'resolution': '10m native resolution',
        'tiling': 'Regular grid with configurable tile size'
    },
    'quality_assurance': {
        'gap_detection': 'Pattern-based missing file analysis',
        'robustness_testing': 'Statistical sampling with multiple iterations',
        'consistency_validation': 'Kolmogorov-Smirnov tests between years'
    }
}