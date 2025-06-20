"""
Logging and system monitoring utilities for biomass estimation pipeline.

This module provides standardized logging configuration and system resource
monitoring functions. Includes memory usage tracking and timing utilities
for performance monitoring during long-running biomass processing tasks.

Author: Diego Bengochea
"""

import logging
import time
import psutil
import os
from contextlib import contextmanager
import threading


def setup_logging():
    """
    Set up standardized logging configuration for the biomass pipeline.
    
    Configures logging with INFO level and standardized timestamp format
    for consistent log output across all pipeline modules.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# Initialize module-level logger
logger = setup_logging()


def log_memory_usage(context="Current"):
    """
    Log current memory usage with contextual information.
    
    Monitors RSS (Resident Set Size) memory usage of the current process
    and logs it with descriptive context for tracking memory consumption
    during pipeline execution.
    
    Args:
        context (str): Description of where in the pipeline this is being called
                      (e.g., "Initial", "After loading data", "Final")
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        logger.info(f"{context} memory usage: {memory_mb:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not get memory usage for {context}: {e}")


@contextmanager
def timer(task_name):
    """
    Context manager for timing operations with automatic logging.
    
    Measures elapsed time for code blocks and automatically logs
    the duration with task identification for performance monitoring.
    
    Args:
        task_name (str): Descriptive name of the task being timed
                        (e.g., "Monte Carlo simulation", "Data loading")
    
    Yields:
        None: Context manager yields nothing
        
    Example:
        >>> with timer("Processing tile"):
        ...     # Long-running operation
        ...     process_data()
        Processing tile completed in 45.32 seconds
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{task_name} completed in {elapsed:.2f} seconds")