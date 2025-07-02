"""
Dask Utilities for Distributed Computing

This module provides distributed computing support for biomass estimation
using Dask for large-scale raster processing.

Author: Diego Bengochea
"""

import os
import psutil
from typing import Dict, Any, Optional
from pathlib import Path
import dask
from dask.distributed import Client, LocalCluster
import dask.array as da
from contextlib import contextmanager
import gc

# Shared utilities
from shared_utils import get_logger


class DaskClusterManager:
    """
    Manager for Dask distributed computing clusters.
    
    Handles cluster setup, configuration, and resource management
    for large-scale biomass processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Dask cluster manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger('biomass_estimation.dask')
        
        # Cluster configuration
        self.num_workers = config['compute']['num_workers']
        self.memory_limit = config['compute']['memory_limit']
        self.threads_per_worker = config['compute']['threads_per_worker']
        
        # Cluster state
        self.cluster = None
        self.client = None
        
        self.logger.info("DaskClusterManager initialized")
    
    @contextmanager
    def create_cluster(self):
        """
        Create a fresh Dask cluster with aggressive cleanup.
        
        Yields:
            dask.distributed.Client: Fresh Dask client for processing
        """
        cluster = None
        client = None
        
        try:
            self.logger.info(f"Creating fresh Dask cluster with {self.num_workers} workers, "
                           f"{self.memory_limit} memory limit, {self.threads_per_worker} threads per worker")
            
            # Create fresh LocalCluster with original settings
            cluster = LocalCluster(
                n_workers=self.num_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
            )
            
            client = Client(cluster)
            
            self.logger.info(f"Dask dashboard available at: {client.dashboard_link}")
            
            yield client
            
        except Exception as e:
            self.logger.error(f"Error in Dask cluster operations: {str(e)}")
            raise
            
        finally:
            # Aggressive cleanup sequence 
            try:
                if client is not None:
                    self.logger.info("Closing Dask client...")
                    client.close()
                
                if cluster is not None:
                    self.logger.info("Closing Dask cluster...")
                    cluster.close()
                
                # Force garbage collection
                gc.collect()
                
                self.logger.info("Dask cluster cleanup completed")
                
            except Exception as cleanup_error:
                self.logger.error(f"Error during Dask cleanup: {str(cleanup_error)}")
