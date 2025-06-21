"""
Sentinel-2 Processing Executable Scripts

Command-line entry points for Sentinel-2 processing workflows. Provides comprehensive
CLI interfaces for all processing stages from main mosaic creation to post-processing
analysis with flexible parameter control and validation.

Scripts:
    run_mosaic_processing.py: Main distributed mosaic creation pipeline
    run_downsampling.py: Spatial downsampling and merging operations
    run_missing_analysis.py: Missing tile detection and gap analysis  
    run_robustness_assessment.py: Scene selection parameter optimization
    run_consistency_analysis.py: Interannual consistency validation
    run_postprocessing.py: Orchestrated post-processing workflows

Usage Examples:
    # Main mosaic processing
    python scripts/run_mosaic_processing.py --years 2021 2022 --tile-size 6144
    
    # Post-processing workflows
    python scripts/run_downsampling.py --scale-factor 5
    python scripts/run_missing_analysis.py --save-paths missing.txt
    python scripts/run_robustness_assessment.py --bbox -9.5 36.0 3.3 43.8
    python scripts/run_consistency_analysis.py --sample-size 5000
    
    # Orchestrated execution
    python scripts/run_postprocessing.py --workflows downsampling missing

Author: Diego Bengochea
"""

# Import main processing script
from .run_mosaic_processing import main as run_mosaic_processing

# Import post-processing scripts
from .run_downsampling import main as run_downsampling
from .run_missing_analysis import main as run_missing_analysis
from .run_robustness_assessment import main as run_robustness_assessment
from .run_consistency_analysis import main as run_consistency_analysis
from .run_postprocessing import main as run_postprocessing

__all__ = [
    "run_mosaic_processing",
    "run_downsampling", 
    "run_missing_analysis",
    "run_robustness_assessment",
    "run_consistency_analysis",
    "run_postprocessing"
]

# Script metadata and capabilities
AVAILABLE_SCRIPTS = [
    {
        "name": "run_mosaic_processing.py",
        "description": "Main distributed mosaic creation pipeline",
        "status": "available",
        "features": [
            "Distributed processing with Dask clusters",
            "STAC catalog integration",
            "Scene Classification Layer masking",
            "Optimal scene selection",
            "Memory-optimized cluster management"
        ],
        "parameters": [
            "years", "tile-size", "n-scenes", "output-dir", 
            "n-workers", "memory-per-worker", "validate-only", "summary"
        ]
    },
    {
        "name": "run_downsampling.py", 
        "description": "Spatial downsampling and merging operations",
        "status": "available",
        "features": [
            "Configurable spatial downsampling",
            "Yearly mosaic merging",
            "LZW compression optimization",
            "Progress tracking and statistics"
        ],
        "parameters": [
            "scale-factor", "downsample-only", "merge-only", 
            "input-dir", "downsampled-dir", "merged-dir"
        ]
    },
    {
        "name": "run_missing_analysis.py",
        "description": "Missing tile detection and gap analysis",
        "status": "available", 
        "features": [
            "Pattern-based gap detection",
            "Completeness statistics",
            "Missing file path generation",
            "Detailed reporting capabilities"
        ],
        "parameters": [
            "directory", "save-paths", "detailed-report", "summary-only"
        ]
    },
    {
        "name": "run_robustness_assessment.py",
        "description": "Scene selection parameter optimization",
        "status": "available",
        "features": [
            "Statistical sampling analysis",
            "STAC catalog integration",
            "Parameter optimization recommendations",
            "Custom spatial and temporal filtering"
        ],
        "parameters": [
            "bbox", "year", "min-scenes", "max-scenes", 
            "step", "save-results", "brief"
        ]
    },
    {
        "name": "run_consistency_analysis.py",
        "description": "Interannual consistency validation",
        "status": "available",
        "features": [
            "Kolmogorov-Smirnov statistical tests",
            "Multi-band spectral analysis",
            "Visualization generation",
            "Comprehensive reporting"
        ],
        "parameters": [
            "input-dir", "sample-size", "bands", "years",
            "exclude-years", "summary-only", "no-plots"
        ]
    },
    {
        "name": "run_postprocessing.py",
        "description": "Orchestrated post-processing workflows", 
        "status": "available",
        "features": [
            "Flexible workflow orchestration",
            "Continue-on-error execution",
            "Dry-run capabilities",
            "Cross-workflow parameter management"
        ],
        "parameters": [
            "workflows", "skip-analysis", "data-only", "analysis-only",
            "continue-on-error", "dry-run", "summary-only"
        ]
    }
]

# Execution modes and workflow patterns
EXECUTION_MODES = {
    'sequential': {
        'description': 'Run workflows in sequence',
        'example': 'run_mosaic_processing.py → run_downsampling.py → run_missing_analysis.py'
    },
    'parallel': {
        'description': 'Run independent analysis workflows in parallel',
        'example': 'run_robustness_assessment.py & run_consistency_analysis.py'
    },
    'orchestrated': {
        'description': 'Use run_postprocessing.py for managed execution',
        'example': 'run_postprocessing.py --workflows downsampling missing robustness'
    },
    'selective': {
        'description': 'Run specific workflows based on needs',
        'example': 'run_missing_analysis.py for quality assurance only'
    }
}

def list_available_scripts():
    """
    List all available scripts in this module.
    
    Returns:
        list: List of available script dictionaries
    """
    return AVAILABLE_SCRIPTS


def get_script_help(script_name: str) -> str:
    """
    Get help information for a specific script.
    
    Args:
        script_name: Name of the script
        
    Returns:
        str: Help text for the script
    """
    script_info = next((s for s in AVAILABLE_SCRIPTS if s["name"] == script_name), None)
    
    if not script_info:
        available_names = [s['name'] for s in AVAILABLE_SCRIPTS]
        return f"Script '{script_name}' not found. Available scripts: {available_names}"
    
    if script_info["status"] != "available":
        return f"Script '{script_name}' is {script_info['status']}."
    
    help_text = f"{script_info['description']}\n\n"
    help_text += f"Features:\n"
    for feature in script_info['features']:
        help_text += f"  - {feature}\n"
    help_text += f"\nKey Parameters: {', '.join(script_info['parameters'])}"
    
    return help_text


def get_execution_recommendations(use_case: str) -> str:
    """
    Get execution recommendations for specific use cases.
    
    Args:
        use_case: Description of the use case
        
    Returns:
        str: Recommended execution pattern
    """
    use_case_lower = use_case.lower()
    
    if 'quality' in use_case_lower or 'missing' in use_case_lower:
        return "run_missing_analysis.py for gap detection and completeness assessment"
    
    elif 'parameter' in use_case_lower or 'optimization' in use_case_lower:
        return "run_robustness_assessment.py for scene selection parameter optimization"
    
    elif 'consistency' in use_case_lower or 'temporal' in use_case_lower:
        return "run_consistency_analysis.py for interannual validation"
    
    elif 'analysis' in use_case_lower or 'post' in use_case_lower:
        return "run_postprocessing.py --analysis-only for comprehensive analysis workflows"
    
    elif 'processing' in use_case_lower or 'data' in use_case_lower:
        return "run_postprocessing.py --data-only for data processing workflows"
    
    else:
        return "run_postprocessing.py for complete workflow orchestration"