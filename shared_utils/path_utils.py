"""
Path utilities for the Iberian Carbon Assessment Pipeline.

This module provides consistent path handling, file discovery, and directory
management across all pipeline components.

Author: Diego Bengochea
"""

import os
import glob
from pathlib import Path
from typing import List, Union, Optional, Generator
import logging


def ensure_directory(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        parents: Whether to create parent directories
        
    Returns:
        Path: Created directory path
        
    Examples:
        >>> output_dir = ensure_directory("results/biomass/2024")
        >>> temp_dir = ensure_directory(Path("temp"))
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def resolve_path(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve path to absolute path, optionally relative to base_path.
    
    Args:
        path: Path to resolve
        base_path: Base path for relative resolution (default: current directory)
        
    Returns:
        Path: Resolved absolute path
        
    Examples:
        >>> abs_path = resolve_path("data/input")
        >>> abs_path = resolve_path("../config", base_path="/project/component")
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base_path:
        base_path = Path(base_path)
        return (base_path / path).resolve()
    
    return path.resolve()


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    file_types: Optional[List[str]] = None
) -> List[Path]:
    """
    Find files matching pattern in directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        file_types: List of file extensions to filter by (e.g., ['.tif', '.tiff'])
        
    Returns:
        List[Path]: List of matching file paths
        
    Examples:
        >>> tif_files = find_files("data", "*.tif")
        >>> config_files = find_files(".", "config*.yaml", recursive=False)
        >>> raster_files = find_files("rasters", "*", file_types=['.tif', '.nc'])
    """
    directory = Path(directory)
    
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return []
    
    # Use glob pattern
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter by file types if specified
    if file_types:
        file_types = [ext.lower() for ext in file_types]
        files = [f for f in files if f.suffix.lower() in file_types]
    
    # Return only files (not directories)
    files = [f for f in files if f.is_file()]
    
    return sorted(files)


def find_directories(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True
) -> List[Path]:
    """
    Find directories matching pattern.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List[Path]: List of matching directory paths
        
    Examples:
        >>> data_dirs = find_directories(".", "*_data")
        >>> year_dirs = find_directories("results", "20??")
    """
    directory = Path(directory)
    
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return []
    
    # Use glob pattern
    if recursive:
        dirs = list(directory.rglob(pattern))
    else:
        dirs = list(directory.glob(pattern))
    
    # Return only directories
    dirs = [d for d in dirs if d.is_dir()]
    
    return sorted(dirs)


def get_file_groups_by_pattern(
    directory: Union[str, Path],
    pattern: str,
    group_key: callable = None
) -> dict:
    """
    Group files by pattern extraction.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files
        group_key: Function to extract grouping key from filename
        
    Returns:
        dict: Dictionary with groups as keys and file lists as values
        
    Examples:
        >>> # Group by year in filename
        >>> def extract_year(path):
        ...     return path.stem.split('_')[1]  # assumes format: prefix_2020_suffix
        >>> year_groups = get_file_groups_by_pattern("data", "*.tif", extract_year)
    """
    files = find_files(directory, pattern)
    
    if not group_key:
        # Default grouping by file extension
        group_key = lambda p: p.suffix
    
    groups = {}
    for file in files:
        try:
            key = group_key(file)
            if key not in groups:
                groups[key] = []
            groups[key].append(file)
        except Exception as e:
            logging.warning(f"Error grouping file {file}: {e}")
    
    return groups


def validate_file_exists(path: Union[str, Path], description: str = "") -> Path:
    """
    Validate that file exists and return Path object.
    
    Args:
        path: File path to validate
        description: Description for error messages
        
    Returns:
        Path: Validated file path
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Examples:
        >>> config_file = validate_file_exists("config.yaml", "Configuration file")
    """
    path = Path(path)
    
    if not path.exists():
        desc = f" ({description})" if description else ""
        raise FileNotFoundError(f"File not found{desc}: {path}")
    
    if not path.is_file():
        desc = f" ({description})" if description else ""
        raise ValueError(f"Path is not a file{desc}: {path}")
    
    return path


def validate_directory_exists(path: Union[str, Path], description: str = "") -> Path:
    """
    Validate that directory exists and return Path object.
    
    Args:
        path: Directory path to validate
        description: Description for error messages
        
    Returns:
        Path: Validated directory path
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        
    Examples:
        >>> data_dir = validate_directory_exists("data/input", "Input data directory")
    """
    path = Path(path)
    
    if not path.exists():
        desc = f" ({description})" if description else ""
        raise FileNotFoundError(f"Directory not found{desc}: {path}")
    
    if not path.is_dir():
        desc = f" ({description})" if description else ""
        raise ValueError(f"Path is not a directory{desc}: {path}")
    
    return path


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Get relative path from base directory.
    
    Args:
        path: Path to make relative
        base: Base directory
        
    Returns:
        Path: Relative path
        
    Examples:
        >>> rel_path = get_relative_path("/project/data/input.tif", "/project")
        >>> # Returns: data/input.tif
    """
    path = Path(path).resolve()
    base = Path(base).resolve()
    
    try:
        return path.relative_to(base)
    except ValueError:
        # Paths are not related, return absolute path
        return path


def safe_file_operation(operation: callable, *args, **kwargs):
    """
    Safely execute file operation with error handling.
    
    Args:
        operation: File operation function to execute
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of operation or None if failed
        
    Examples:
        >>> result = safe_file_operation(shutil.copy2, "source.txt", "dest.txt")
    """
    logger = logging.getLogger(__name__)
    
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.error(f"File operation failed: {operation.__name__}: {e}")
        return None


def get_available_space(path: Union[str, Path]) -> float:
    """
    Get available disk space in GB for given path.
    
    Args:
        path: Path to check available space for
        
    Returns:
        float: Available space in GB
        
    Examples:
        >>> space_gb = get_available_space("/data")
    """
    path = Path(path)
    
    # Find existing parent directory
    while not path.exists() and path.parent != path:
        path = path.parent
    
    if path.exists():
        stat = os.statvfs(path)
        available_bytes = stat.f_bavail * stat.f_frsize
        return available_bytes / (1024**3)  # Convert to GB
    
    return 0.0


def create_temp_directory(prefix: str = "iberian_carbon_", base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create temporary directory for processing.
    
    Args:
        prefix: Prefix for temporary directory name
        base_dir: Base directory for temporary files (default: system temp)
        
    Returns:
        Path: Created temporary directory
        
    Examples:
        >>> temp_dir = create_temp_directory("biomass_processing_")
    """
    import tempfile
    
    if base_dir:
        base_dir = Path(base_dir)
        ensure_directory(base_dir)
        temp_dir = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
    else:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
    
    return Path(temp_dir)
