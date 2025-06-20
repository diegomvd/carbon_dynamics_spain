"""
Standardized logging utilities for the Iberian Carbon Assessment Pipeline.

This module provides consistent logging configuration across all components
while allowing component-specific customization.

Author: Diego Bengochea
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = 'INFO',
    component_name: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    format_style: str = 'standard'
) -> logging.Logger:
    """
    Setup standardized logging configuration for pipeline components.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component_name: Name of the component (for logger identification)
        log_file: Optional file path for logging output
        format_style: Logging format style ('standard', 'detailed', 'simple')
        
    Returns:
        logging.Logger: Configured logger instance
        
    Examples:
        >>> logger = setup_logging('INFO', 'biomass_estimation')
        >>> logger = setup_logging('DEBUG', 'canopy_height_dl', 'training.log')
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Define format styles
    formats = {
        'standard': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        'simple': '%(levelname)s: %(message)s'
    }
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplication
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        formats.get(format_style, formats['standard']),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Get component-specific logger
    logger_name = f'iberian_carbon.{component_name}' if component_name else 'iberian_carbon'
    logger = logging.getLogger(logger_name)
    
    return logger


def get_logger(component_name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        logging.Logger: Component logger
        
    Examples:
        >>> logger = get_logger('biomass_estimation')
        >>> logger = get_logger('canopy_height_dl')
    """
    return logging.getLogger(f'iberian_carbon.{component_name}')


def log_pipeline_start(logger: logging.Logger, pipeline_name: str, config: dict = None) -> None:
    """
    Log standardized pipeline start message.
    
    Args:
        logger: Logger instance
        pipeline_name: Name of the pipeline being started
        config: Optional configuration dictionary to log key parameters
    """
    logger.info("=" * 80)
    logger.info(f"STARTING PIPELINE: {pipeline_name.upper()}")
    logger.info("=" * 80)
    
    if config:
        logger.info("Pipeline configuration:")
        # Log key configuration parameters (avoid logging sensitive data)
        for key, value in config.items():
            if key in ['data', 'processing', 'output']:
                logger.info(f"  {key}: {type(value).__name__} with {len(value)} parameters")
            elif not key.startswith('_'):  # Skip private config keys
                logger.info(f"  {key}: {value}")


def log_pipeline_end(logger: logging.Logger, pipeline_name: str, success: bool = True, elapsed_time: float = None) -> None:
    """
    Log standardized pipeline completion message.
    
    Args:
        logger: Logger instance
        pipeline_name: Name of the completed pipeline
        success: Whether pipeline completed successfully
        elapsed_time: Optional elapsed time in seconds
    """
    logger.info("=" * 80)
    
    if success:
        status_msg = f"✅ PIPELINE COMPLETED SUCCESSFULLY: {pipeline_name.upper()}"
    else:
        status_msg = f"❌ PIPELINE FAILED: {pipeline_name.upper()}"
    
    logger.info(status_msg)
    
    if elapsed_time:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        logger.info(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    logger.info("=" * 80)


def log_section(logger: logging.Logger, section_name: str) -> None:
    """
    Log a standardized section header.
    
    Args:
        logger: Logger instance
        section_name: Name of the section
    """
    logger.info(f"\n{'='*20} {section_name.upper()} {'='*20}")
