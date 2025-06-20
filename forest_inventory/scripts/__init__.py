"""
Forest Inventory Executable Scripts

Command-line entry points for Spanish National Forest Inventory (NFI) processing.
Provides easy access to the main processing pipeline with configuration options.

Scripts:
    run_nfi_processing.py: Main NFI biomass processing pipeline

Usage Examples:
    # Run with default configuration
    python scripts/run_nfi_processing.py
    
    # Run with custom configuration
    python scripts/run_nfi_processing.py --config custom_config.yaml
    
    # Process specific UTM zones only
    python scripts/run_nfi_processing.py --utm-zones 29 30
    
    # Enable verbose logging
    python scripts/run_nfi_processing.py --log-level DEBUG

Author: Diego Bengochea
"""

# Import main processing script
from .run_nfi_processing import main as run_nfi_processing

__all__ = [
    "run_nfi_processing"
]

# Script metadata
AVAILABLE_SCRIPTS = [
    {
        "name": "run_nfi_processing.py",
        "description": "Main NFI biomass processing pipeline",
        "status": "available"
    }
]

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
        return f"Script '{script_name}' not found. Available scripts: {[s['name'] for s in AVAILABLE_SCRIPTS]}"
    
    if script_info["status"] == "planned":
        return f"Script '{script_name}' is planned but not yet implemented."
    
    return script_info["description"]