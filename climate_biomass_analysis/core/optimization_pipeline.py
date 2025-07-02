"""
Optimization pipeline with restored feature selection logic.

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
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory
from shared_utils.central_data_paths_constants import *


class CustomEarlyStopping:
    """Early stopping implementation for Optuna optimization."""
    
    def __init__(self, patience: int = 40, min_trials: int = 100):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of trials without improvement before stopping
            min_trials: Minimum number of trials before early stopping can trigger
        """
        self.patience = patience
        self.min_trials = min_trials
        self.best_value = None
        self.best_trial = None
        self.stagnation_counter = 0
        self.should_stop = False
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback function for Optuna study.
        
        Args:
            study: Optuna study object
            trial: Current trial
        """
        if len(study.trials) < self.min_trials:
            return
        
        current_value = study.best_value
        
        if self.best_value is None or current_value > self.best_value:
            self.best_value = current_value
            self.best_trial = trial.number
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        if self.stagnation_counter >= self.patience:
            study.stop()
            self.should_stop = True


class OptimizationPipeline:
    """
    Bayesian optimization pipeline for biomass-climate modeling.
    
    This class performs multiple independent optimization runs to select
    optimal climate predictors and hyperparameters for biomass prediction.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the optimization pipeline.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='optimization_pipeline',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration sections
        self.opt_config = self.config['optimization']
        self.features_config = self.opt_config['features']
        self.hyperparams = self.opt_config['hyperparameters']
        self.cv_config = self.opt_config['cv_strategy']
        self.early_stopping_config = self.opt_config.get('early_stopping', {'patience': 40, 'min_trials': 100})
        
        self.max_features = self.opt_config.get('max_features', 10)  # Default to 15 like in old implementation
        
        self.logger.info("Initialized OptimizationPipeline")
    
    def run_full_pipeline(self) -> bool:
        """
        Execute the complete optimization pipeline.
        
        Returns:
            Dictionary with comprehensive optimization results
        """
        self.logger.info("Starting Bayesian optimization pipeline...")
        
        # Load spatial data
        df = self.load_spatial_data()
        
        # Prepare features
        df_processed, predictor_cols, target_col = self.prepare_data_for_modeling(df)
        
        self.logger.info(f"Dataset shape: {df_processed.shape}")
        self.logger.info(f"Features: {len(predictor_cols)}, Target: {target_col}")
        
        # Run multiple optimization runs
        n_runs = self.opt_config['n_runs']
        random_seeds = self.opt_config['random_seeds'][:n_runs]
        
        # Ensure we have enough seeds
        while len(random_seeds) < n_runs:
            random_seeds.append(random_seeds[-1] + 1000)

        # Save detailed results
        output_dir = CLIMATE_BIOMASS_MODELS_DIR
        ensure_directory(output_dir)

        for run_id in range(n_runs):
            try:
                run_results = self.run_single_optimization(
                    df_processed, predictor_cols, target_col, run_id, random_seeds[run_id]
                )
                output_file = output_dir / f'run_{run_id + 1}_model.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(run_results, f)  
               
            except Exception as e:
                self.logger.error(f"Error in optimization run {run_id + 1}: {e}")
                continue
        
        if not all_results:
            self.logger.error("All optimization runs failed")
            return False
               
        self.logger.info(f"Optimization completed successfully!")
        
        return True

    def prepare_data_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Prepare data for modeling by standardizing and identifying columns.
    
        Args:
            df: Input dataset
            
        Returns:
            Tuple of (standardized_df, predictor_columns, target_column)
        """
        # Define columns to exclude from analysis
        exclude_cols = ['year_start', 'year_end', 'cluster_id', 'x', 'y']
        
        # Exclude bioclimatic variables as specified in config
        bio_prefixes_to_exclude = self.features_config['exclude_bio_vars']
        
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
            self.logger.info(f"Excluding these bioclimatic variables: {', '.join(excluded_bio_vars)}")
        
        # Get analysis columns (all columns except excluded ones)
        analysis_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Identify target and predictor columns
        target_col = 'biomass_rel_change'
        predictor_cols = [col for col in analysis_cols if col != target_col]
        
        # But add zero-division protection
        df_std = df.copy()
        for col in analysis_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:  # Avoid division by zero
                df_std[col] = (df[col] - col_mean) / col_std
            else:
                self.logger.warning(f"Column {col} has zero standard deviation, keeping original values")
                df_std[col] = df[col] - col_mean  # Center but don't scale
        
        self.logger.info(f"Total number of features after exclusions: {len(predictor_cols)}")
        
        return df_std, predictor_cols, target_col
    
    def create_spatial_partition(
        self, 
        df: pd.DataFrame, 
        run_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a spatial partition for a single optimization run.
        
        Args:
            df: Dataset with cluster assignments
            run_seed: Random seed for this run
            
        Returns:
            Tuple of (train_indices, validation_indices, test_indices)
        """
        np.random.seed(run_seed)
        random.seed(run_seed)
        
        # Get unique clusters
        unique_clusters = df['cluster_id'].unique()
        np.random.shuffle(unique_clusters)
        
        # Partition clusters
        test_blocks = self.cv_config['test_blocks']
        val_blocks_range = self.cv_config['validation_blocks_range']
        val_blocks = np.random.randint(val_blocks_range[0], val_blocks_range[1] + 1)
        
        test_clusters = unique_clusters[:test_blocks]
        val_clusters = unique_clusters[test_blocks:test_blocks + val_blocks]
        train_clusters = unique_clusters[test_blocks + val_blocks:]
        
        # Get indices for each partition
        test_indices = df[df['cluster_id'].isin(test_clusters)].index.values
        val_indices = df[df['cluster_id'].isin(val_clusters)].index.values
        train_indices = df[df['cluster_id'].isin(train_clusters)].index.values
        
        self.logger.debug(f"Spatial partition - Train: {len(train_indices)}, "
                         f"Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def objective_function(
        self, 
        trial: optuna.Trial, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        predictor_cols: List[str],
        run_seed: int
    ) -> float:
        """
        Objective function for Optuna optimization.
                
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            predictor_cols: List of predictor column names
            run_seed: Random seed for this run
            
        Returns:
            Validation R² score
        """
        # Let the model decide how many features to use (1 to max_features)
        n_features = trial.suggest_int('n_features', 1, self.max_features)
        
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
        
        # Configure XGBoost parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', *self.hyperparams['learning_rate']),
            'max_depth': trial.suggest_int('max_depth', *self.hyperparams['max_depth']),
            'min_child_weight': trial.suggest_int('min_child_weight', *self.hyperparams['min_child_weight']),
            'subsample': trial.suggest_float('subsample', *self.hyperparams['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *self.hyperparams['colsample_bytree']),
            'n_estimators': trial.suggest_int('n_estimators', *self.hyperparams['n_estimators']),
            'reg_alpha': trial.suggest_float('reg_alpha', *self.hyperparams['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *self.hyperparams['reg_lambda']),
            'random_state': run_seed,
            'n_jobs': 1  # Use single thread to avoid conflicts
        }
        
        try:
            # Train the model
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train_selected, y_train)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_val_selected)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # Store features and info in trial attributes
            trial.set_user_attr('features', [predictor_cols[i] for i in feature_indices])
            trial.set_user_attr('feature_indices', feature_indices)
            trial.set_user_attr('val_r2', val_r2)
            trial.set_user_attr('n_features', n_features)
            
            # Handle invalid R² values
            if np.isnan(val_r2) or np.isinf(val_r2):
                return -1.0
            
            return val_r2  # We're maximizing R²
            
        except Exception as e:
            self.logger.warning(f"Error in trial {trial.number}: {e}")
            return -1.0
    
    def run_single_optimization(
        self,
        df: pd.DataFrame,
        predictor_cols: List[str],
        target_col: str,
        run_id: int,
        run_seed: int
    ) -> Dict[str, Any]:
        """
        Run a single optimization run.
        
        Args:
            df: Dataset with features and targets
            predictor_cols: List of predictor column names
            target_col: Target column name
            run_id: Run identifier
            run_seed: Random seed for this run
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting optimization run {run_id + 1} with seed {run_seed}")
        
        # Create spatial partition
        train_indices, val_indices, test_indices = self.create_spatial_partition(df, run_seed)
        
        # Prepare data
        X_train = df.loc[train_indices, predictor_cols].values
        y_train = df.loc[train_indices, target_col].values
        X_val = df.loc[val_indices, predictor_cols].values
        y_val = df.loc[val_indices, target_col].values
        X_test = df.loc[test_indices, predictor_cols].values
        y_test = df.loc[test_indices, target_col].values
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=run_seed,
                n_startup_trials=30,  # More initial random trials
                n_ei_candidates=20,   # Consider more candidates
                prior_weight=1.0      # Balance prior and likelihood
            )
        )
        
        # Set up early stopping
        early_stopping = CustomEarlyStopping(
            patience=self.early_stopping_config['patience'],
            min_trials=self.early_stopping_config['min_trials']
        )
        
        # Run optimization
        start_time = time.time()
        
        study.optimize(
            lambda trial: self.objective_function(trial, X_train, y_train, X_val, y_val, predictor_cols, run_seed),
            n_trials=self.opt_config['n_trials'],
            callbacks=[early_stopping]
        )
        
        optimization_time = time.time() - start_time
        
        # Get best parameters and features
        best_params = {}
        for param_name, param_value in study.best_trial.params.items():
            if not param_name.startswith('feature_'):
                best_params[param_name] = param_value
        
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': run_seed
        })
        
        best_feature_indices = study.best_trial.user_attrs['feature_indices']
        best_features = study.best_trial.user_attrs['features']
        val_r2 = study.best_trial.user_attrs['val_r2']
        
        # Train final model on combined training and validation data
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.concatenate((y_train, y_val))
        
        X_train_val_selected = X_train_val[:, best_feature_indices]
        X_test_selected = X_test[:, best_feature_indices]
        
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_train_val_selected, y_train_val)
        
        # Evaluate on test set
        y_test_pred = final_model.predict(X_test_selected)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate feature importance
        feature_importance = final_model.feature_importances_
        importance_dict = {feature: importance for feature, importance in zip(best_features, feature_importance)}
        
        # Create model info dictionary
        run_results = {
            'run_id': run_id,
            'run_seed': run_seed,
            'model': final_model,
            'features': best_features,
            'feature_indices': best_feature_indices,
            'params': best_params,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'trials_completed': len(study.trials),
            'optimization_time': optimization_time,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'feature_importance': importance_dict,
            'partition_info': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'num_features_used': len(best_features)
        }
        
        self.logger.info(f"Run {run_id + 1} completed: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}, "
                        f"Features used: {len(best_features)}")
        
        return run_results
    
    def analyze_optimization_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze results from multiple optimization runs.
        
        Args:
            all_results: List of results from individual runs
            
        Returns:
            Dictionary with aggregated analysis
        """
        # Extract metrics
        val_r2s = [r['val_r2'] for r in all_results]
        test_r2s = [r['test_r2'] for r in all_results]
        n_features = [r['num_features_used'] for r in all_results]
        
        # Feature selection frequency analysis
        all_features = []
        for result in all_results:
            all_features.extend(result['features'])
        
        feature_counts = Counter(all_features)
        total_runs = len(all_results)
        feature_frequencies = {feat: count/total_runs for feat, count in feature_counts.items()}
        
        # Sort features by frequency
        top_features = sorted(feature_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        summary = {
            'validation_r2': {
                'mean': np.mean(val_r2s),
                'std': np.std(val_r2s),
                'min': np.min(val_r2s),
                'max': np.max(val_r2s)
            },
            'test_r2': {
                'mean': np.mean(test_r2s),
                'std': np.std(test_r2s),
                'min': np.min(test_r2s),
                'max': np.max(test_r2s)
            },
            'features_per_run': {
                'mean': np.mean(n_features),
                'std': np.std(n_features),
                'min': np.min(n_features),
                'max': np.max(n_features)
            },
            'feature_analysis': {
                'feature_frequencies': feature_frequencies,
                'top_features': [feat for feat, freq in top_features[:20]],  # Top 20
                'feature_selection_counts': feature_counts
            },
            'run_summary': {
                'total_runs': total_runs,
                'successful_runs': len([r for r in all_results if r['val_r2'] > 0])
            }
        }
        
        return summary
    
    def load_spatial_data(self) -> pd.DataFrame:
        """Load the spatial dataset with cluster assignments."""
        # This method should load the clustered dataset
        # Implementation depends on where the data is stored
        clustered_dataset_path = CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE
        
        if not os.path.exists(clustered_dataset_path):
            raise FileNotFoundError(f"Clustered dataset not found: {clustered_dataset_path}")
        
        df = pd.read_csv(clustered_dataset_path)
        
        if 'cluster_id' not in df.columns:
            raise ValueError("Dataset must contain 'cluster_id' column")
        
        self.logger.info(f"Loaded dataset with {len(df)} data points and {df['cluster_id'].nunique()} clusters")
        
        return df
