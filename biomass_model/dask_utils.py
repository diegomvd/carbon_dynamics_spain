"""
Distributed computing utilities for Dask cluster management.

This module provides context managers and utilities for creating and managing
Dask clusters for distributed biomass processing. Includes automatic cleanup
and garbage collection to prevent memory leaks in long-running pipelines.

Author: Diego Bengochea
"""

from dask.distributed import Client, LocalCluster
from contextlib import contextmanager
import gc


@contextmanager
def dask_cluster(num_workers, memory_limit, threads_per_worker=1):
    """
    Context manager for creating and managing a Dask LocalCluster.
    
    Sets up a local Dask cluster with specified resources and automatically
    handles cleanup when the context exits. Performs garbage collection
    after cluster shutdown to prevent memory accumulation.
    
    Args:
        num_workers (int): Number of worker processes to spawn
        memory_limit (str): Memory limit per worker (e.g., '4GB', '2048MB')
        threads_per_worker (int): Number of threads per worker process
        
    Yields:
        dask.distributed.Client: Connected Dask client for distributed computing
        
    Example:
        >>> with dask_cluster(4, '8GB', threads_per_worker=2) as client:
        ...     # Use client for distributed computations
        ...     result = some_dask_computation.compute()
    """
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        # protocol='tcp',     # Uncomment if specific protocol needed
        # timeout=60,         # Uncomment if custom timeout needed
    )
    client = Client(cluster)
    
    try:    
        yield client
    finally:
        # Ensure proper cleanup sequence
        client.close()
        cluster.close()
        gc.collect()  # Force garbage collection to free memory