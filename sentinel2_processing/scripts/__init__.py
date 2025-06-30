"""
Sentinel-2 Processing Executable Scripts

Command-line entry points for Sentinel-2 processing workflows. Provides comprehensive
CLI interfaces for all processing stages from main mosaic creation to post-processing
analysis with flexible parameter control and validation.

Scripts:
    run_mosaic_processing.py: Main distributed mosaic creation pipeline
    run_downsampling.py: Spatial downsampling and merging operations
    run_missing_analysis.py: Missing tile detection and gap analysis  
    run_robustness_analysis.py: Scene selection parameter optimization
    run_consistency_analysis.py: Interannual consistency validation
    run_postprocessing.py: Orchestrated post-processing workflows

Usage Examples:
    # Main mosaic processing
    python scripts/run_mosaic_processing.py --years 2021 2022 --tile-size 6144
    
    # Post-processing workflows
    python scripts/run_downsampling.py --scale-factor 5
    python scripts/run_missing_analysis.py --save-paths missing.txt
    python scripts/run_robustness_analysis.py --bbox -9.5 36.0 3.3 43.8
    python scripts/run_consistency_analysis.py --sample-size 5000
    
    # Orchestrated execution
    python scripts/run_postprocessing.py --workflows downsampling missing

Author: Diego Bengochea
"""

# Import main processing script
from .run_mosaic_processing import main as run_mosaic_processing

# Import post-processing scripts
from .run_downsampling import main as run_downsampling
from .run_missing_tiles import main as run_missing_tiles
from .run_robustness_analysis import main as run_robustness_analysis
from .run_consistency_analysis import main as run_consistency_analysis
from .run_postprocessing import main as run_postprocessing

__all__ = [
    "run_mosaic_processing",
    "run_downsampling", 
    "run_missing_tiles",
    "run_robustness_analysis",
    "run_consistency_analysis",
    "run_postprocessing"
]

