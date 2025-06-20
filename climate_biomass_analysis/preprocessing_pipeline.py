"""
Climate-Biomass Analysis Preprocessing Pipeline.

Orchestrated pipeline that integrates climate data processing, bioclimatic variable 
calculation, biomass integration, and spatial analysis. Includes smart checkpointing
to allow resuming from interruptions.

Author: Diego Bengochea
"""

import os
import glob
import yaml
import logging
from pathlib import Path
import time

# Import our harmonized modules
from climate_raster_processing import process_climate_anomalies
from bioclim_calculation import run_bioclim_pipeline
from biomass_integration import run_biomass_integration_pipeline
from spatial_analysis import run_spatial_analysis_pipeline


def setup_logging():
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path="climate_biomass_config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise


def check_stage_completion(stage_name, check_paths, check_patterns=None):
    """
    Check if a pipeline stage has already been completed.
    
    Args:
        stage_name (str): Name of the pipeline stage
        check_paths (list): List of paths/directories to check
        check_patterns (list, optional): List of file patterns to check within directories
        
    Returns:
        bool: True if stage appears to be completed
    """
    logger = setup_logging()
    
    for i, path in enumerate(check_paths):
        if not os.path.exists(path):
            logger.info(f"  {stage_name}: Missing {path}")
            return False
        
        # If it's a directory, check for files
        if os.path.isdir(path):
            if check_patterns and i < len(check_patterns):
                pattern = check_patterns[i]
                files = glob.glob(os.path.join(path, pattern))
                if not files:
                    logger.info(f"  {stage_name}: No files matching '{pattern}' in {path}")
                    return False
                logger.info(f"  {stage_name}: Found {len(files)} files matching '{pattern}' in {path}")
            else:
                # Check if directory has any files
                files = os.listdir(path)
                if not files:
                    logger.info(f"  {stage_name}: Empty directory {path}")
                    return False
                logger.info(f"  {stage_name}: Directory {path} contains {len(files)} items")
        else:
            # It's a file
            logger.info(f"  {stage_name}: File exists {path}")
    
    logger.info(f"  {stage_name}: ✅ Stage appears complete")
    return True


def stage_1_climate_processing(config):
    """
    Stage 1: Climate raster processing (GRIB → TIF).
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if stage completed successfully
    """
    logger = setup_logging()
    stage_name = "Stage 1: Climate Processing"
    logger.info(f"\n{'='*60}")
    logger.info(f"{stage_name}")
    logger.info(f"{'='*60}")
    
    # Check if stage already completed
    check_paths = [config['paths']['climate_outputs']]
    check_patterns = ["*.tif"]
    
    if check_stage_completion(stage_name, check_paths, check_patterns):
        logger.info(f"{stage_name}: Skipping - already completed")
        return True
    
    # Run climate processing
    logger.info(f"{stage_name}: Starting climate raster processing...")
    start_time = time.time()
    
    try:
        process_climate_anomalies(config)
        
        # Verify outputs were created
        output_files = glob.glob(os.path.join(config['paths']['climate_outputs'], "*.tif"))
        if output_files:
            logger.info(f"{stage_name}: ✅ Completed successfully - created {len(output_files)} TIF files")
            logger.info(f"{stage_name}: Completed in {(time.time() - start_time)/60:.2f} minutes")
            return True
        else:
            logger.error(f"{stage_name}: ❌ Failed - no output files created")
            return False
            
    except Exception as e:
        logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
        return False


def stage_2_bioclim_calculation(config):
    """
    Stage 2: Bioclimatic variables calculation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if stage completed successfully
    """
    logger = setup_logging()
    stage_name = "Stage 2: Bioclim Calculation"
    logger.info(f"\n{'='*60}")
    logger.info(f"{stage_name}")
    logger.info(f"{'='*60}")
    
    # Check if stage already completed
    check_paths = [
        config['paths']['bioclim_dir'],
        config['paths']['anomaly_dir']
    ]
    check_patterns = ["bio*.tif", "anomalies_*"]
    
    if check_stage_completion(stage_name, check_paths, check_patterns):
        logger.info(f"{stage_name}: Skipping - already completed")
        return True
    
    # Run bioclim calculation
    logger.info(f"{stage_name}: Starting bioclimatic variables calculation...")
    start_time = time.time()
    
    try:
        results = run_bioclim_pipeline(config)
        
        if results is not None:
            # Verify outputs were created
            bioclim_files = glob.glob(os.path.join(config['paths']['bioclim_dir'], "bio*.tif"))
            anomaly_dirs = glob.glob(os.path.join(config['paths']['anomaly_dir'], "anomalies_*"))
            
            if bioclim_files and anomaly_dirs:
                logger.info(f"{stage_name}: ✅ Completed successfully")
                logger.info(f"  - Created {len(bioclim_files)} bioclim files")
                logger.info(f"  - Created {len(anomaly_dirs)} anomaly directories")
                logger.info(f"{stage_name}: Completed in {(time.time() - start_time)/60:.2f} minutes")
                return True
            else:
                logger.error(f"{stage_name}: ❌ Failed - insufficient output files created")
                return False
        else:
            logger.error(f"{stage_name}: ❌ Failed - function returned None")
            return False
            
    except Exception as e:
        logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
        return False


def stage_3_biomass_integration(config):
    """
    Stage 3: Biomass integration and ML dataset creation.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if stage completed successfully
    """
    logger = setup_logging()
    stage_name = "Stage 3: Biomass Integration"
    logger.info(f"\n{'='*60}")
    logger.info(f"{stage_name}")
    logger.info(f"{'='*60}")
    
    # Check if stage already completed
    check_paths = [config['paths']['training_dataset']]
    
    if check_stage_completion(stage_name, check_paths):
        logger.info(f"{stage_name}: Skipping - already completed")
        return True
    
    # Run biomass integration
    logger.info(f"{stage_name}: Starting biomass-climate integration...")
    start_time = time.time()
    
    try:
        results = run_biomass_integration_pipeline(config)
        
        if results is not None and len(results) > 0:
            logger.info(f"{stage_name}: ✅ Completed successfully")
            logger.info(f"  - Created dataset with {len(results)} data points")
            logger.info(f"  - Features: {len(results.columns)} columns")
            logger.info(f"{stage_name}: Completed in {(time.time() - start_time)/60:.2f} minutes")
            return True
        else:
            logger.error(f"{stage_name}: ❌ Failed - no data points created")
            return False
            
    except Exception as e:
        logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
        return False


def stage_4_spatial_analysis(config):
    """
    Stage 4: Spatial analysis and clustering.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if stage completed successfully
    """
    logger = setup_logging()
    stage_name = "Stage 4: Spatial Analysis"
    logger.info(f"\n{'='*60}")
    logger.info(f"{stage_name}")
    logger.info(f"{'='*60}")
    
    # Check if stage already completed
    check_paths = [config['paths']['clustered_dataset']]
    
    if check_stage_completion(stage_name, check_paths):
        logger.info(f"{stage_name}: Skipping - already completed")
        return True
    
    # Run spatial analysis
    logger.info(f"{stage_name}: Starting spatial analysis and clustering...")
    start_time = time.time()
    
    try:
        results = run_spatial_analysis_pipeline(config)
        
        if results is not None and 'cluster_id' in results.columns:
            n_clusters = results['cluster_id'].nunique()
            logger.info(f"{stage_name}: ✅ Completed successfully")
            logger.info(f"  - Created {n_clusters} spatial clusters")
            logger.info(f"  - Dataset with {len(results)} data points")
            logger.info(f"{stage_name}: Completed in {(time.time() - start_time)/60:.2f} minutes")
            return True
        else:
            logger.error(f"{stage_name}: ❌ Failed - no cluster_id column created")
            return False
            
    except Exception as e:
        logger.error(f"{stage_name}: ❌ Failed with error: {str(e)}")
        return False


def run_preprocessing_pipeline(config_path="climate_biomass_config.yaml"):
    """
    Run the complete preprocessing pipeline with smart checkpointing.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        bool: True if entire pipeline completed successfully
    """
    logger = setup_logging()
    logger.info("🚀 Starting Climate-Biomass Preprocessing Pipeline")
    logger.info("="*80)
    
    pipeline_start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Create output directories if they don't exist
        output_dirs = [
            config['paths']['climate_outputs'],
            config['paths']['harmonized_dir'],
            config['paths']['bioclim_dir'],
            config['paths']['anomaly_dir'],
            config['paths']['temp_resampled_dir']
        ]
        
        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)
        
        # Pipeline stages
        stages = [
            ("Climate Processing", stage_1_climate_processing),
            ("Bioclim Calculation", stage_2_bioclim_calculation),
            ("Biomass Integration", stage_3_biomass_integration),
            ("Spatial Analysis", stage_4_spatial_analysis)
        ]
        
        completed_stages = 0
        
        # Execute each stage
        for stage_name, stage_function in stages:
            success = stage_function(config)
            if success:
                completed_stages += 1
                logger.info(f"✅ {stage_name} completed successfully")
            else:
                logger.error(f"❌ {stage_name} failed - stopping pipeline")
                break
        
        # Pipeline summary
        total_time = time.time() - pipeline_start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPROCESSING PIPELINE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Stages completed: {completed_stages}/{len(stages)}")
        logger.info(f"Total pipeline time: {total_time/60:.2f} minutes")
        
        if completed_stages == len(stages):
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Final dataset ready: {config['paths']['clustered_dataset']}")
            
            # Provide next steps
            logger.info(f"\nNext steps:")
            logger.info(f"1. Review clustered dataset: {config['paths']['clustered_dataset']}")
            logger.info(f"2. Run optimization pipeline with the clustered dataset")
            logger.info(f"3. Analyze results and interpret model outputs")
            
            return True
        else:
            logger.error("❌ PIPELINE INCOMPLETE - some stages failed")
            logger.info(f"\nTo resume the pipeline:")
            logger.info(f"1. Check and fix any configuration issues")
            logger.info(f"2. Re-run this script - completed stages will be skipped")
            return False
            
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILED with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main function to run the preprocessing pipeline."""
    logger = setup_logging()
    
    try:
        success = run_preprocessing_pipeline()
        
        if success:
            logger.info("\nPreprocessing pipeline completed successfully! 🎉")
        else:
            logger.error("\nPreprocessing pipeline failed! ❌")
            logger.info("Check the logs above for details and re-run to resume from last completed stage.")
            
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user. You can resume by re-running this script.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
