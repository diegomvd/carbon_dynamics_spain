"""
Shared utilities for the Iberian Carbon Assessment Pipeline.

This package provides common functionality used across all components:
- Standardized logging configuration
- Configuration file loading utilities  
- Path handling utilities
- Common data validation functions

Author: Diego Bengochea
"""

from .logging_utils import setup_logging, get_logger, log_pipeline_start, log_pipeline_end, log_section
from .config_utils import load_config, validate_config
from .path_utils import ensure_directory, resolve_path, find_files

__version__ = "1.0.0"

__all__ = [
    "setup_logging",
    "get_logger",
    "log_pipeline_start", 
    "log_pipeline_end",
    "log_section",
    "load_config",
    "validate_config",
    "ensure_directory",
    "resolve_path",
    "find_files"
]
