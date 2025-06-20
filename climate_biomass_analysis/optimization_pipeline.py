"""
Bayesian optimization pipeline for biomass-climate modeling.

Performs multiple independent spatial optimization runs using Bayesian optimization
to select optimal climate predictors for biomass change prediction. Uses spatial
cross-validation to prevent overfitting and includes comprehensive model evaluation.

Author: Diego Bengochea
"""

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import r2_score
import time
import os
import pickle
from collections import Counter
import logging
import random
import yaml


def setup_logging():
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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


def load_spatial_data(config):
    """
    Load the dataset with spatial cluster assignments.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        pandas.DataFrame: Dataset with cluster_id column
    """
    logger = setup_logging()
    clustered_data_file = config['paths']['clustered_dataset']
    
    logger.info(f"Loading spatial data from {clustered_data_file}")
    
    try:
        # Load the data with cluster assignments
        df = pd.read_csv(clustered_data_file)
        
        # Check if cluster_id column exists
        if 'cluster_id' not in df.columns:
            raise ValueError("cluster_id column not found in the spatial data file")
        
        # Get the number of unique clusters
        num_clusters = df['cluster_id'].nunique()
        logger.info(f"Found {num_clusters} spatial clusters in the data")
        
        # Verify we have enough clusters for the strategy
        cv_config = config['cv_strategy']
        min_required = cv_config['test_blocks'] + cv_config['validation_blocks_range'][0]
        if num_clusters < min_required:
            raise ValueError(f"Not enough spatial clusters ({num_clusters}) for the partitioning strategy. "
                            f"Need at least {min_required} clusters.")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading spatial data: {e}")
        raise


def create_spatial_partition(df, run_seed, config):
    """
    Create a spatial partition for a single optimization run.
    
    Args:
        df (pandas.DataFrame): Dataset with cluster_id column
        run_seed (int): Random seed for this run's partition
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (train_mask, val_mask, test_mask) boolean arrays
    """
    logger = setup_logging()
    
    # Set seed for this run's partition
    np.random.seed(run_seed)
    random.seed(run_seed)
    
    # Extract CV configuration
    cv_config = config['cv_strategy']
    test_blocks = cv_config['test_blocks']
    validation_blocks_range = cv_config['validation_blocks_range']
    
    # Get all unique cluster IDs
    cluster_ids = df['cluster_id'].unique()
    
    # Shuffle the cluster IDs
    np.random.shuffle(cluster_ids)
    
    # Split clusters into test, validation, and training
    test_cluster_ids = cluster_ids[:test_blocks]
    
    # Randomly decide how many validation blocks for this run
    n_val_blocks = random.randint(validation_blocks_range[0], validation_blocks_range[1])
    val_cluster_ids = cluster_ids[test_blocks:test_blocks + n_val_blocks]
    train_cluster_ids = cluster_ids[test_blocks + n_val_blocks:]
    
    # Log the partition
    logger.info(f"Run seed {run_seed} partition:")
    logger.info(f"  Test blocks: {test_cluster_ids.tolist()} ({len(test_cluster_ids)} blocks)")
    logger.info(f"  Validation blocks: {val_cluster_ids.tolist()} ({len(val_cluster_ids)} blocks)")
    logger.info(f"  Training blocks: {train_cluster_ids.tolist()} ({len(train_cluster_ids)} blocks)")
    
    # Create masks for each partition
    test_mask = df['cluster_id'].isin(test_cluster_ids)
    val_mask = df['cluster_id'].isin(val_cluster_ids)
    train_mask = df['cluster_id'].isin(train_cluster_ids)
    
    # Log the dataset sizes
    logger.info(f"  Test set: {test_mask.sum()} instances")
    logger.info(f"  Validation set: {val_mask.sum()} instances")
    logger.info(f"  Training set: {train_mask.sum()} instances")
    
    return train_mask, val_mask, test_mask


def prepare_data(df, config):
    """
    Prepare data for modeling by standardizing and identifying columns.
    
    Args:
        df (pandas.DataFrame): Input dataset
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (df_standardized, predictor_columns, target_column)
    """
    logger = setup_logging()
    
    # Define columns to exclude from analysis
    exclude_cols = ['year_start', 'year_end', 'cluster_id', 'x', 'y']
    
    # Exclude bioclimatic variables as specified in config
    bio_prefixes_to_exclude = config['features']['exclude_bio_vars']
    
    # Find all columns that match the bio prefixes to exclude
    for col in df.columns:
        for prefix in bio_prefixes_to_exclude:
            if col.startswith(prefix):
                exclude_cols.append(col)
                break
    
    # Remove duplicates in exclude_cols if any
    exclude_cols = list(set(exclude_cols))
    
    # Log the excluded bioclimatic variables
    excluded_bio_vars = [col for col in exclude_cols if col.startswith('bio')]
    if excluded_bio_vars:
        logger.info(f"Excluding these bioclimatic variables: {', '.join(excluded_bio_vars)}")
    
    # Get analysis columns (all columns except excluded ones)
    analysis_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify target and predictor columns
    target_col = 'biomass_rel_change'
    predictor_cols = [col for col in analysis_cols if col != target_col]
    
    # Standardize all variables
    df_std = df.copy()
    for col in analysis_cols:
        df_std[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Print available features
    logger.info(f"Total number of features after exclusions: {len(predictor_cols)}")
    
    return df_std, predictor_cols, target_col


class CustomEarlyStopping:
    """Better implementation of early stopping without relying on unrealistic R² threshold."""
    
    def __init__(self, patience=40, min_trials=100):
        """
        Initialize early stopping callback.
        
        Args:
            patience (int): Number of trials without improvement before stopping
            min_trials (int): Minimum number of trials before early stopping can trigger
        """
        self.patience = patience
        self.min_trials = min_trials
        self.best_value = None
        self.best_trial = None
        self.stagnation_counter = 0
        self.should_stop = False
    
    def __call__(self, study, trial):
        """
        Callback function for Optuna study.
        
        Args:
            study: Optuna study object
            trial: Current trial object
        """
        logger = setup_logging()
        
        # Never stop before minimum trials
        if trial.number < self.min_trials:
            return
            
        # Check if we already decided to stop
        if self.should_stop:
            study.stop()
            return
            
        # Update best performance tracking
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.best_trial = study.best_trial.number
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Check stopping condition - based only on patience, no R² threshold
        if self.stagnation_counter >= self.patience:
            logger.info(f"Early stopping triggered after {trial.number+1} trials: "
                      f"No improvement for {self.patience} consecutive trials")
            logger.info(f"Best trial: #{self.best_trial} with R² = {self.best_value:.4f}")
            self.should_stop = True
            study.stop()


def run_optimization(X, y, train_mask, val_mask, test_mask, predictor_cols, config, run_seed=42):
    """
    Run Bayesian optimization for a single spatial partition.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        train_mask (numpy.ndarray): Training set mask
        val_mask (numpy.ndarray): Validation set mask
        test_mask (numpy.ndarray): Test set mask
        predictor_cols (list): List of predictor column names
        config (dict): Configuration dictionary
        run_seed (int): Random seed for this run
        
    Returns:
        tuple: (model_info, study) containing model results and optimization study
    """
    logger = setup_logging()
    
    # Extract configuration
    opt_config = config['optimization']
    max_features = opt_config['max_features']
    n_trials = opt_config['n_trials']
    early_stopping_config = opt_config['early_stopping']
    xgb_param_ranges = opt_config['xgb_params']
    
    # Extract the data for each partition
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    # Calculate the training set size for logging
    train_size = len(X_train)
    
    logger.info(f"Starting optimization with {train_size} training, {len(X_val)} validation, "
              f"and {len(X_test)} test instances")
    
    # Structure to store top models
    top_models = []
    
    def objective(trial):
        # Let the model decide how many features to use (1 to max_features)
        n_features = trial.suggest_int('n_features', 1, max_features)
        
        # Get total number of available predictors
        n_features_total = len(predictor_cols)
        
        # Use a combination-based method to ensure uniqueness
        available_indices = list(range(n_features_total))
        feature_indices = []
        
        # Randomly select n_features from available indices without replacement
        for i in range(n_features):
            if not available_indices:
                break  # Safety check
                
            # Select an index from the remaining available indices
            idx_position = trial.suggest_int(f'feature_pos_{i}', 0, len(available_indices) - 1)
            selected_idx = available_indices.pop(idx_position)
            feature_indices.append(selected_idx)
        
        # Sort indices for consistency
        feature_indices.sort()
        
        # Get the selected features
        X_train_selected = X_train[:, feature_indices]
        X_val_selected = X_val[:, feature_indices]
        
        # Configure XGBoost parameters with light regularization
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', *xgb_param_ranges['learning_rate']),
            'max_depth': trial.suggest_int('max_depth', *xgb_param_ranges['max_depth']),
            'min_child_weight': trial.suggest_int('min_child_weight', *xgb_param_ranges['min_child_weight']),
            'subsample': trial.suggest_float('subsample', *xgb_param_ranges['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *xgb_param_ranges['colsample_bytree']),
            'n_estimators': trial.suggest_int('n_estimators', *xgb_param_ranges['n_estimators']),
            'reg_alpha': trial.suggest_float('reg_alpha', *xgb_param_ranges['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *xgb_param_ranges['reg_lambda']),
            'seed': run_seed
        }
        
        # Train the model
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_selected, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_selected)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Store features and info
        trial.set_user_attr('features', [predictor_cols[i] for i in feature_indices])
        trial.set_user_attr('feature_indices', feature_indices)
        trial.set_user_attr('val_r2', val_r2)
        trial.set_user_attr('n_features', n_features)
        trial.set_user_attr('model', model)
        
        # Store promising models (we'll keep track of top 10% of models)
        top_models.append({
            'trial': trial.number,
            'model': model,
            'val_r2': val_r2,
            'features': [predictor_cols[i] for i in feature_indices],
            'feature_indices': feature_indices,
            'params': xgb_params,
            'n_features': n_features
        })
        
        # Sort and keep only top models to avoid memory issues
        # We'll dynamically determine top 10% as we go
        top_percentile = max(10, min(30, int(len(top_models) * 0.1)))  # At least 10, max 30 models
        top_models.sort(key=lambda x: x['val_r2'], reverse=True)
        if len(top_models) > top_percentile:
            top_models.pop()  # Remove the lowest performing model
            
        return val_r2  # We're maximizing R²

    # Create and run the study with early stopping
    # Configure TPE sampler with more exploration at the beginning
    sampler = optuna.samplers.TPESampler(
        seed=run_seed,
        n_startup_trials=30,  # More initial random trials
        n_ei_candidates=20,   # Consider more candidates
        prior_weight=1.0      # Balance prior and likelihood
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler
    )
    
    # Add early stopping callback
    early_stopping = CustomEarlyStopping(
        patience=early_stopping_config['patience'], 
        min_trials=early_stopping_config['min_trials']
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=n_trials, callbacks=[early_stopping])
    
    # Get best parameters and features
    best_params = {}
    for param_name, param_value in study.best_trial.params.items():
        if not param_name.startswith('feature_'):
            best_params[param_name] = param_value
    
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': run_seed
    })
    
    best_feature_indices = study.best_trial.user_attrs['feature_indices']
    best_features = study.best_trial.user_attrs['features']
    val_r2 = study.best_trial.user_attrs['val_r2']
    
    # Train final model on combined training and validation data
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    
    X_train_val_selected = X_train_val[:, best_feature_indices]
    X_test_selected = X_test[:, best_feature_indices]
    
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train_val_selected, y_train_val)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_selected)
    test_r2 = r2_score(y_test, y_pred)
    
    # Calculate feature importance
    feature_importance = best_model.feature_importances_
    importance_dict = {feature: importance for feature, importance in zip(best_features, feature_importance)}
    
    # Create model info dictionary
    model_info = {
        'model': best_model,
        'features': best_features,
        'feature_indices': best_feature_indices,
        'params': best_params,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'run_seed': run_seed,
        'trials_completed': len(study.trials),
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': importance_dict,
        'partition_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        },
        'num_features_used': len(best_features)
    }
    
    return model_info, study


def run_multiple_spatial_optimizations(df, config):
    """
    Run multiple independent optimizations with different spatial partitions.
    
    Args:
        df (pandas.DataFrame): Dataset with cluster_id column
        config (dict): Configuration dictionary
        
    Returns:
        list: Sorted list of model results
    """
    logger = setup_logging()
    
    # Extract configuration
    opt_config = config['optimization']
    n_runs = opt_config['n_runs']
    max_features = opt_config['max_features']
    n_trials = opt_config['n_trials']
    base_seed = opt_config['base_seed']
    
    # Create output directories
    output_config = config['output']
    models_dir = output_config['models_dir']
    analysis_data_dir = output_config['analysis_data_dir']
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(analysis_data_dir, exist_ok=True)
    
    # Prepare data
    df_std, predictor_cols, target_col = prepare_data(df, config)
    
    # Create X and y matrices
    X = df_std[predictor_cols].values
    y = df_std[target_col].values
    
    all_models = []
    partition_history = []
    
    start_time = time.time()
    
    for run in range(n_runs):
        run_start = time.time()
        run_seed = base_seed + run
        
        logger.info(f"\n===== Run {run+1}/{n_runs} (Seed: {run_seed}) =====")
        
        # Create a unique spatial partition for this run
        train_mask, val_mask, test_mask = create_spatial_partition(df, run_seed, config)
        
        # Save partition info for analysis
        partition_info = {
            'run': run + 1,
            'seed': run_seed,
            'train_instances': train_mask.sum(),
            'val_instances': val_mask.sum(),
            'test_instances': test_mask.sum()
        }
        partition_history.append(partition_info)
        
        # Run optimization
        logger.info(f"Running optimization with max {n_trials} trials and up to {max_features} features...")
        model_info, study = run_optimization(
            X, y, train_mask, val_mask, test_mask, predictor_cols, config, run_seed=run_seed
        )
        
        # Add run number to model info
        model_info['run'] = run + 1
        model_info['partition_info'] = partition_info
        
        # Print results
        logger.info(f"Run {run+1} completed after {model_info['trials_completed']} trials")
        logger.info(f"Test R²: {model_info['test_r2']:.4f}")
        logger.info(f"Selected features: {model_info['features']}")
        
        # Save this run's best model
        with open(f'{models_dir}/run_{run+1}_model.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        # Store model info
        all_models.append(model_info)
        
        # Calculate and print time
        run_time = (time.time() - run_start) / 60
        logger.info(f"Run {run+1} completed in {run_time:.2f} minutes")
        
        # Calculate overall progress
        elapsed_time = (time.time() - start_time) / 60
        avg_time_per_run = elapsed_time / (run + 1)
        remaining_runs = n_runs - (run + 1)
        estimated_time_remaining = avg_time_per_run * remaining_runs
        
        logger.info(f"Progress: {run+1}/{n_runs} runs completed")
        logger.info(f"Elapsed time: {elapsed_time:.2f} minutes")
        logger.info(f"Estimated time remaining: {estimated_time_remaining:.2f} minutes")
    
    # Calculate overall results
    total_time = (time.time() - start_time) / 60
    
    # Basic statistical analysis of results
    test_r2_values = [model['test_r2'] for model in all_models]
    avg_r2 = np.mean(test_r2_values)
    median_r2 = np.median(test_r2_values)
    best_r2 = max(test_r2_values)
    r2_std = np.std(test_r2_values)
    
    # Count trials saved by early stopping
    total_trials_completed = sum(model['trials_completed'] for model in all_models)
    max_possible_trials = n_runs * n_trials
    trials_saved = max_possible_trials - total_trials_completed
    time_saved = (trials_saved / total_trials_completed) * total_time if total_trials_completed > 0 else 0
    
    # Calculate feature count statistics
    feature_counts = [model['num_features_used'] for model in all_models]
    avg_features = np.mean(feature_counts)
    median_features = np.median(feature_counts)
    
    # Create histogram of feature counts
    feature_count_hist = Counter(feature_counts)
    
    logger.info("\n===== Multiple Optimization Results =====")
    logger.info(f"Total runs: {n_runs}")
    logger.info(f"Max features allowed: {max_features}")
    logger.info(f"Average features used: {avg_features:.2f}")
    logger.info(f"Median features used: {median_features}")
    logger.info(f"Feature count distribution: {dict(sorted(feature_count_hist.items()))}")
    logger.info(f"Total time: {total_time:.2f} minutes")
    logger.info(f"Average test R²: {avg_r2:.4f} ± {r2_std:.4f}")
    logger.info(f"Median test R²: {median_r2:.4f}")
    logger.info(f"Best test R²: {best_r2:.4f}")
    logger.info(f"Trials completed: {total_trials_completed}/{max_possible_trials}")
    logger.info(f"Estimated time saved by early stopping: {time_saved:.2f} minutes")
    
    # Quick feature frequency analysis
    all_selected_features = [model['features'] for model in all_models]
    feature_counter = Counter()
    for features in all_selected_features:
        feature_counter.update(features)
    
    # Print top 10 features by selection frequency
    logger.info("\nTop 10 features by selection frequency:")
    for feature, count in feature_counter.most_common(10):
        logger.info(f"- {feature}: selected in {count}/{n_runs} runs ({count/n_runs*100:.1f}%)")
    
    # Sort models by test R²
    sorted_models = sorted(all_models, key=lambda x: x['test_r2'], reverse=True)
    
    # Save minimal results data for later analysis
    results_data = {
        'models': all_models,
        'sorted_models': sorted_models,
        'feature_frequency': feature_counter,
        'partition_history': partition_history,
        'performance_stats': {
            'mean_r2': avg_r2,
            'median_r2': median_r2,
            'std_r2': r2_std,
            'best_r2': best_r2,
            'total_time': total_time,
            'trials_completed': total_trials_completed,
            'trials_saved': trials_saved
        }
    }
    
    with open(f'{analysis_data_dir}/spatial_optimization_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    
    return sorted_models


def main():
    """Main function to run the spatial optimization analysis."""
    logger = setup_logging()
    logger.info("Starting Bayesian optimization pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Load data with spatial clusters
        df = load_spatial_data(config)
        
        # Check if we need to create clusters
        if 'cluster_id' not in df.columns:
            logger.error("Spatial clustering hasn't been performed yet. Please run preprocessing pipeline first.")
            return
        
        logger.info(f"Starting {config['optimization']['n_runs']} independent spatial optimization runs...")
        sorted_models = run_multiple_spatial_optimizations(df, config)
        
        logger.info("\nMultiple optimization runs complete!")
        logger.info(f"All models have been saved to the '{config['output']['models_dir']}' directory.")
        logger.info(f"Basic results data saved to '{config['output']['analysis_data_dir']}/spatial_optimization_results.pkl'")
        
    except Exception as e:
        logger.error(f"Optimization pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
