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
        
        # Cluster state
        self.cluster = None
        self.client = None
        
        self.logger.info("DaskClusterManager initialized")
    
    def setup_cluster(self, cluster_type: str = 'local') -> Optional[Client]:
        """
        Setup Dask cluster for distributed processing.
        
        Args:
            cluster_type: Type of cluster ('local', 'slurm', 'pbs')
            
        Returns:
            Dask client or None if setup failed
        """
        try:
            self.logger.info(f"Setting up {cluster_type} Dask cluster...")
            
            if cluster_type == 'local':
                return self._setup_local_cluster()
            elif cluster_type == 'slurm':
                return self._setup_slurm_cluster()
            elif cluster_type == 'pbs':
                return self._setup_pbs_cluster()
            else:
                self.logger.error(f"Unknown cluster type: {cluster_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error setting up Dask cluster: {str(e)}")
            return None
    
    def _setup_local_cluster(self) -> Optional[Client]:
        """Setup local Dask cluster."""
        try:
            # Configure Dask settings
            self._configure_dask_settings()
            
            # Determine optimal worker configuration
            worker_config = self._calculate_worker_config()
            
            # Create local cluster
            self.cluster = LocalCluster(
                n_workers=worker_config['n_workers'],
                threads_per_worker=worker_config['threads_per_worker'],
                memory_limit=worker_config['memory_per_worker'],
                processes=True,  # Use processes for better memory isolation
                dashboard_address=':8787',
                silence_logs=False
            )
            
            # Connect client
            self.client = Client(self.cluster)
            
            self.logger.info(f"Local cluster setup successful:")
            self.logger.info(f"  Workers: {worker_config['n_workers']}")
            self.logger.info(f"  Threads per worker: {worker_config['threads_per_worker']}")
            self.logger.info(f"  Memory per worker: {worker_config['memory_per_worker']}")
            self.logger.info(f"  Dashboard: {self.client.dashboard_link}")
            
            return self.client
            
        except Exception as e:
            self.logger.error(f"Error setting up local cluster: {str(e)}")
            return None
    
    def _setup_slurm_cluster(self) -> Optional[Client]:
        """Setup SLURM cluster (for HPC environments)."""
        try:
            from dask_jobqueue import SLURMCluster
            
            self.cluster = SLURMCluster(
                cores=self.config.get('slurm', {}).get('cores', 4),
                memory=self.config.get('slurm', {}).get('memory', '16GB'),
                queue=self.config.get('slurm', {}).get('queue', 'normal'),
                walltime=self.config.get('slurm', {}).get('walltime', '02:00:00'),
                job_extra=self.config.get('slurm', {}).get('job_extra', [])
            )
            
            # Scale cluster
            n_jobs = self.config.get('slurm', {}).get('n_jobs', 2)
            self.cluster.scale(jobs=n_jobs)
            
            self.client = Client(self.cluster)
            
            self.logger.info(f"SLURM cluster setup successful with {n_jobs} jobs")
            return self.client
            
        except ImportError:
            self.logger.error("dask-jobqueue not available for SLURM cluster")
            return None
        except Exception as e:
            self.logger.error(f"Error setting up SLURM cluster: {str(e)}")
            return None
    
    def _setup_pbs_cluster(self) -> Optional[Client]:
        """Setup PBS cluster (for HPC environments)."""
        try:
            from dask_jobqueue import PBSCluster
            
            self.cluster = PBSCluster(
                cores=self.config.get('pbs', {}).get('cores', 4),
                memory=self.config.get('pbs', {}).get('memory', '16GB'),
                queue=self.config.get('pbs', {}).get('queue', 'normal'),
                walltime=self.config.get('pbs', {}).get('walltime', '02:00:00'),
                job_extra=self.config.get('pbs', {}).get('job_extra', [])
            )
            
            # Scale cluster
            n_jobs = self.config.get('pbs', {}).get('n_jobs', 2)
            self.cluster.scale(jobs=n_jobs)
            
            self.client = Client(self.cluster)
            
            self.logger.info(f"PBS cluster setup successful with {n_jobs} jobs")
            return self.client
            
        except ImportError:
            self.logger.error("dask-jobqueue not available for PBS cluster")
            return None
        except Exception as e:
            self.logger.error(f"Error setting up PBS cluster: {str(e)}")
            return None
    
    def _configure_dask_settings(self) -> None:
        """Configure global Dask settings for optimal performance."""
        # Configure temporary directory
        temp_dir = Path.home() / 'tmp' / 'dask-worker-space'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        dask.config.set({
            'temporary-directory': str(temp_dir),
            'array.chunk-size': f"{self.config['compute']['chunk_size']}MiB",
            'array.slicing.split_large_chunks': True,
            'distributed.worker.memory.target': 0.8,
            'distributed.worker.memory.spill': 0.9,
            'distributed.worker.memory.pause': 0.95,
            'distributed.worker.memory.terminate': 0.98,
            'distributed.comm.timeouts.connect': '60s',
            'distributed.comm.timeouts.tcp': '60s',
            'distributed.worker.daemon': False
        })
        
        self.logger.debug("Dask configuration updated")
    
    def _calculate_worker_config(self) -> Dict[str, Any]:
        """
        Calculate optimal worker configuration based on system resources.
        
        Returns:
            Dictionary with worker configuration
        """
        # Get system information
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        total_cores = psutil.cpu_count()
        
        # Parse memory limit from config
        memory_limit_str = self.memory_limit
        if memory_limit_str.endswith('GB'):
            available_memory_gb = float(memory_limit_str[:-2])
        else:
            # Fallback to 80% of total memory
            available_memory_gb = total_memory_gb * 0.8
        
        # Calculate worker configuration
        if self.num_workers == 'auto':
            # Use 80% of cores, but at least 1 and at most 8
            n_workers = max(1, min(8, int(total_cores * 0.8)))
        else:
            n_workers = min(self.num_workers, total_cores)
        
        # Memory per worker
        memory_per_worker_gb = available_memory_gb / n_workers
        memory_per_worker = f"{memory_per_worker_gb:.1f}GB"
        
        # Threads per worker
        threads_per_worker = max(1, total_cores // n_workers)
        
        return {
            'n_workers': n_workers,
            'threads_per_worker': threads_per_worker,
            'memory_per_worker': memory_per_worker,
            'total_memory_gb': available_memory_gb,
            'total_cores_used': n_workers * threads_per_worker
        }
    
    def monitor_cluster_performance(self) -> Dict[str, Any]:
        """
        Monitor cluster performance and resource usage.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.client:
            return {}
        
        try:
            # Get cluster status
            scheduler_info = self.client.scheduler_info()
            
            # Calculate metrics
            total_workers = len(scheduler_info['workers'])
            total_cores = sum(w['nthreads'] for w in scheduler_info['workers'].values())
            
            # Memory usage
            total_memory = sum(w['memory_limit'] for w in scheduler_info['workers'].values())
            used_memory = sum(w['metrics']['memory'] for w in scheduler_info['workers'].values())
            memory_usage_pct = (used_memory / total_memory) * 100 if total_memory > 0 else 0
            
            # Task information
            task_info = self.client.scheduler_info()['tasks']
            
            metrics = {
                'total_workers': total_workers,
                'total_cores': total_cores,
                'total_memory_gb': total_memory / (1024**3),
                'used_memory_gb': used_memory / (1024**3),
                'memory_usage_percent': memory_usage_pct,
                'active_tasks': len([t for t in task_info.values() if t['state'] == 'processing']),
                'pending_tasks': len([t for t in task_info.values() if t['state'] == 'waiting']),
                'completed_tasks': len([t for t in task_info.values() if t['state'] == 'finished'])
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring cluster performance: {str(e)}")
            return {}
    
    def optimize_chunk_size(
        self, 
        array_shape: tuple, 
        dtype: str = 'float32',
        target_chunk_size_mb: int = 128
    ) -> tuple:
        """
        Calculate optimal chunk size for dask arrays.
        
        Args:
            array_shape: Shape of the array
            dtype: Data type
            target_chunk_size_mb: Target chunk size in MB
            
        Returns:
            Optimal chunk size tuple
        """
        try:
            # Calculate bytes per element
            if dtype in ['float32', 'int32']:
                bytes_per_element = 4
            elif dtype in ['float64', 'int64']:
                bytes_per_element = 8
            elif dtype in ['uint8', 'int8']:
                bytes_per_element = 1
            elif dtype in ['uint16', 'int16']:
                bytes_per_element = 2
            else:
                bytes_per_element = 4  # Default
            
            target_bytes = target_chunk_size_mb * 1024 * 1024
            target_elements = target_bytes // bytes_per_element
            
            # For 2D arrays (most common case)
            if len(array_shape) == 2:
                height, width = array_shape
                
                # Try square chunks first
                chunk_size = int(np.sqrt(target_elements))
                chunk_height = min(chunk_size, height)
                chunk_width = min(chunk_size, width)
                
                # Adjust if chunks are too small
                min_chunk_size = 256
                if chunk_height < min_chunk_size:
                    chunk_height = min(min_chunk_size, height)
                if chunk_width < min_chunk_size:
                    chunk_width = min(min_chunk_size, width)
                
                return (chunk_height, chunk_width)
            
            # For 3D arrays
            elif len(array_shape) == 3:
                depth, height, width = array_shape
                
                # Keep full depth, chunk height and width
                target_elements_2d = target_elements // depth
                chunk_size = int(np.sqrt(target_elements_2d))
                chunk_height = min(chunk_size, height)
                chunk_width = min(chunk_size, width)
                
                return (depth, chunk_height, chunk_width)
            
            else:
                # For other dimensions, use default
                return tuple(min(s, self.config['compute']['chunk_size']) for s in array_shape)
                
        except Exception as e:
            self.logger.error(f"Error calculating optimal chunk size: {str(e)}")
            return tuple(min(s, 1024) for s in array_shape)
    
    def close_cluster(self) -> None:
        """Close Dask cluster and clean up resources."""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.logger.info("Dask client closed")
            
            if self.cluster:
                self.cluster.close()
                self.cluster = None
                self.logger.info("Dask cluster closed")
                
        except Exception as e:
            self.logger.error(f"Error closing cluster: {str(e)}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster information.
        
        Returns:
            Dictionary with cluster information
        """
        if not self.client:
            return {'status': 'not_connected'}
        
        try:
            scheduler_info = self.client.scheduler_info()
            
            info = {
                'status': 'connected',
                'scheduler_address': self.client.scheduler.address,
                'dashboard_link': self.client.dashboard_link,
                'workers': {},
                'total_workers': len(scheduler_info['workers']),
                'total_cores': sum(w['nthreads'] for w in scheduler_info['workers'].values()),
                'total_memory_gb': sum(w['memory_limit'] for w in scheduler_info['workers'].values()) / (1024**3)
            }
            
            # Worker details
            for worker_id, worker_info in scheduler_info['workers'].items():
                info['workers'][worker_id] = {
                    'address': worker_info['address'],
                    'cores': worker_info['nthreads'],
                    'memory_limit_gb': worker_info['memory_limit'] / (1024**3),
                    'status': worker_info.get('status', 'unknown')
                }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting cluster info: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_cluster()
