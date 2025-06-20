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
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Shared utilities
from shared_utils import setup_logging, get_logger, load_config, ensure_directory


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
        
        # Extract configuration
        self.opt_config = self.config['optimization']
        self.cv_config = self.opt_config['cv_strategy']
        self.hyperparams = self.opt_config['hyperparameters']
        self.features_config = self.opt_config['features']
        self.early_stopping_config = self.opt_config['early_stopping']
        
        self.logger.info("Initialized OptimizationPipeline")
    
    def load_spatial_data(self) -> pd.DataFrame:
        """
        Load the dataset with spatial cluster assignments.
        
        Returns:
            Dataset with cluster_id column
        """
        clustered_data_file = self.config['data']['clustered_dataset']
        
        self.logger.info(f"Loading spatial data from {clustered_data_file}")
        
        try:
            # Load the data with cluster assignments
            df = pd.read_csv(clustered_data_file)
            
            # Check if cluster_id column exists
            if 'cluster_id' not in df.columns:
                raise ValueError("cluster_id column not found in the spatial data file")
            
            # Get the number of unique clusters
            num_clusters = df['cluster_id'].nunique()
            self.logger.info(f"Found {num_clusters} spatial clusters in the data")
            
            # Verify we have enough clusters for the strategy
            min_required = self.cv_config['test_blocks'] + self.cv_config['validation_blocks_range'][0]
            if num_clusters < min_required:
                raise ValueError(f"Not enough spatial clusters ({num_clusters}) for the partitioning strategy. "
                                f"Need at least {min_required} clusters.")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading spatial data: {e}")
            raise
    
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
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Prepare features for machine learning.
        
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
        
        # Standardize all variables if requested
        df_processed = df.copy()
        if self.features_config.get('standardize', True):
            for col in analysis_cols:
                col_mean = df[col].mean()
                col_std = df[col].std()
                if col_std > 0:  # Avoid division by zero
                    df_processed[col] = (df[col] - col_mean) / col_std
        
        # Remove highly correlated features if requested
        if self.features_config.get('remove_correlated', False):
            predictor_cols = self._remove_correlated_features(df_processed, predictor_cols)
        
        self.logger.info(f"Total number of features after processing: {len(predictor_cols)}")
        
        return df_processed, predictor_cols, target_col
    
    def _remove_correlated_features(
        self, 
        df: pd.DataFrame, 
        predictor_cols: List[str]
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            df: DataFrame with features
            predictor_cols: List of predictor column names
            
        Returns:
            Filtered list of predictor columns
        """
        threshold = self.features_config.get('correlation_threshold', 0.95)
        
        # Calculate correlation matrix
        corr_matrix = df[predictor_cols].corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        # Remove highly correlated features
        filtered_cols = [col for col in predictor_cols if col not in to_drop]
        
        if to_drop:
            self.logger.info(f"Removed {len(to_drop)} highly correlated features "
                           f"(correlation > {threshold})")
        
        return filtered_cols
    
    def objective_function(
        self, 
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation R² score
        """
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', *self.hyperparams['n_estimators'])
        max_depth = trial.suggest_int('max_depth', *self.hyperparams['max_depth'])
        learning_rate = trial.suggest_float('learning_rate', *self.hyperparams['learning_rate'])
        subsample = trial.suggest_float('subsample', *self.hyperparams['subsample'])
        colsample_bytree = trial.suggest_float('colsample_bytree', *self.hyperparams['colsample_bytree'])
        min_child_weight = trial.suggest_int('min_child_weight', *self.hyperparams['min_child_weight'])
        reg_alpha = trial.suggest_float('reg_alpha', *self.hyperparams['reg_alpha'])
        reg_lambda = trial.suggest_float('reg_lambda', *self.hyperparams['reg_lambda'])
        
        # Create and train model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=1  # Use single thread to avoid conflicts
        )
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            # Handle invalid R² values
            if np.isnan(r2) or np.isinf(r2):
                return -1.0
            
            return r2
            
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
            sampler=optuna.samplers.TPESampler(seed=run_seed)
        )
        
        # Set up early stopping
        early_stopping = CustomEarlyStopping(
            patience=self.early_stopping_config['patience'],
            min_trials=self.early_stopping_config['min_trials']
        )
        
        # Run optimization
        start_time = time.time()
        
        study.optimize(
            lambda trial: self.objective_function(trial, X_train, y_train, X_val, y_val),
            n_trials=self.opt_config['n_trials'],
            callbacks=[early_stopping]
        )
        
        optimization_time = time.time() - start_time
        
        # Get best parameters and retrain final model
        best_params = study.best_params
        
        final_model = xgb.XGBRegressor(**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_test_pred = final_model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Get feature importance
        feature_importance = dict(zip(predictor_cols, final_model.feature_importances_))
        
        results = {
            'run_id': run_id,
            'run_seed': run_seed,
            'best_params': best_params,
            'best_val_r2': study.best_value,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'n_trials': len(study.trials),
            'optimization_time_minutes': optimization_time / 60,
            'early_stopped': early_stopping.should_stop,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices)
        }
        
        self.logger.info(f"Run {run_id + 1} completed - Val R²: {study.best_value:.4f}, "
                        f"Test R²: {test_r2:.4f}, Trials: {len(study.trials)}")
        
        return results
    
    def analyze_optimization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze results from multiple optimization runs.
        
        Args:
            results: List of optimization results
            
        Returns:
            Summary statistics and analysis
        """
        # Extract metrics
        val_r2_scores = [r['best_val_r2'] for r in results]
        test_r2_scores = [r['test_r2'] for r in results]
        n_trials = [r['n_trials'] for r in results]
        
        # Feature importance analysis
        all_features = set()
        for r in results:
            all_features.update(r['feature_importance'].keys())
        
        feature_importance_matrix = pd.DataFrame(
            [r['feature_importance'] for r in results]
        ).fillna(0)
        
        # Calculate feature selection frequency and mean importance
        feature_stats = pd.DataFrame({
            'mean_importance': feature_importance_matrix.mean(),
            'std_importance': feature_importance_matrix.std(),
            'selection_frequency': (feature_importance_matrix > 0).mean(),
            'max_importance': feature_importance_matrix.max()
        }).sort_values('mean_importance', ascending=False)
        
        # Most frequently selected features
        top_features = feature_stats[feature_stats['selection_frequency'] > 0.5].index.tolist()
        
        # Parameter analysis
        all_params = set()
        for r in results:
            all_params.update(r['best_params'].keys())
        
        param_stats = {}
        for param in all_params:
            values = [r['best_params'].get(param, np.nan) for r in results]
            values = [v for v in values if not np.isnan(v)]
            if values:
                param_stats[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        summary = {
            'n_runs': len(results),
            'validation_r2': {
                'mean': np.mean(val_r2_scores),
                'std': np.std(val_r2_scores),
                'min': np.min(val_r2_scores),
                'max': np.max(val_r2_scores)
            },
            'test_r2': {
                'mean': np.mean(test_r2_scores),
                'std': np.std(test_r2_scores),
                'min': np.min(test_r2_scores),
                'max': np.max(test_r2_scores)
            },
            'optimization_efficiency': {
                'mean_trials': np.mean(n_trials),
                'std_trials': np.std(n_trials),
                'early_stopping_rate': np.mean([r['early_stopped'] for r in results])
            },
            'feature_analysis': {
                'total_features': len(all_features),
                'frequently_selected': len(top_features),
                'top_features': top_features[:10],  # Top 10
                'feature_stats': feature_stats.head(20).to_dict()  # Top 20
            },
            'parameter_analysis': param_stats
        }
        
        return summary
    
    def run_optimization_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline with multiple independent runs.
        
        Returns:
            Dictionary with comprehensive optimization results
        """
        self.logger.info("Starting Bayesian optimization pipeline...")
        
        # Load spatial data
        df = self.load_spatial_data()
        
        # Prepare features
        df_processed, predictor_cols, target_col = self.prepare_features(df)
        
        self.logger.info(f"Dataset shape: {df_processed.shape}")
        self.logger.info(f"Features: {len(predictor_cols)}, Target: {target_col}")
        
        # Run multiple optimization runs
        n_runs = self.opt_config['n_runs']
        random_seeds = self.opt_config['random_seeds'][:n_runs]
        
        # Ensure we have enough seeds
        while len(random_seeds) < n_runs:
            random_seeds.append(random_seeds[-1] + 1000)
        
        all_results = []
        
        for run_id in range(n_runs):
            try:
                run_results = self.run_single_optimization(
                    df_processed, predictor_cols, target_col, run_id, random_seeds[run_id]
                )
                all_results.append(run_results)
                
            except Exception as e:
                self.logger.error(f"Error in optimization run {run_id + 1}: {e}")
                continue
        
        if not all_results:
            raise RuntimeError("All optimization runs failed")
        
        # Analyze results
        self.logger.info("Analyzing optimization results...")
        summary = self.analyze_optimization_results(all_results)
        
        # Save detailed results
        output_dir = Path("optimization_results")
        ensure_directory(output_dir)
        
        # Save individual run results
        with open(output_dir / "individual_run_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save summary
        with open(output_dir / "optimization_summary.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        # Log summary statistics
        self.logger.info(f"Optimization completed successfully!")
        self.logger.info(f"Validation R² - Mean: {summary['validation_r2']['mean']:.4f} "
                        f"± {summary['validation_r2']['std']:.4f}")
        self.logger.info(f"Test R² - Mean: {summary['test_r2']['mean']:.4f} "
                        f"± {summary['test_r2']['std']:.4f}")
        self.logger.info(f"Top features: {', '.join(summary['feature_analysis']['top_features'][:5])}")
        
        return {
            'individual_results': all_results,
            'summary': summary,
            'output_directory': str(output_dir)
        }