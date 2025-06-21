"""
Comprehensive SHAP Analysis Pipeline

Combines all SHAP-related analyses into a single script:
1. Feature selection frequency analysis
2. SHAP importance calculation  
3. Permutation importance calculation
4. 1D SHAP PDP calculation with LOWESS smoothing
5. 2D SHAP interaction analysis

Author: Diego Bengochea (refactored)
"""

import numpy as np
import pandas as pd
import pickle
import os
import glob
import logging
import time
import shap
from datetime import timedelta
from collections import Counter
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.inspection import permutation_importance
from sklearn.cluster import HDBSCAN
from scipy.cluster import hierarchy
import json
import yaml
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_logging():
    """Set up standardized logging configuration."""
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

def load_models(models_dir='results/climate_biomass_analysis/models_remove_4'):
    """Load all models from subdirectories within the models directory."""
    logger = setup_logging()
    logger.info(f"Loading models from directory: {models_dir}")
    start_time = time.time()
    
    # Find all model subdirectories (models_XXXX)
    model_subdirs = [d for d in os.listdir(models_dir) 
                    if os.path.isdir(os.path.join(models_dir, d)) 
                    and d.startswith('models_')]
    
    logger.info(f"Found {len(model_subdirs)} model subdirectories")
    
    models = []
    model_count = 0
    
    # Process each subdirectory
    for subdir in model_subdirs:
        subdir_path = os.path.join(models_dir, subdir)
        model_files = glob.glob(os.path.join(subdir_path, '*_model.pkl'))
        
        for i, file_path in enumerate(model_files):
            if i % 500 == 0:  # Log progress every 500 models
                logger.info(f"Loading model {i+1}/{len(model_files)} from {subdir}")
            
            try:
                with open(file_path, 'rb') as f:
                    model_info = pickle.load(f)
                    # Add source directory information
                    model_info['source_dir'] = subdir
                    models.append(model_info)
                    model_count += 1
            except Exception as e:
                logger.error(f"Error loading model from {file_path}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Loaded {model_count} models in {timedelta(seconds=elapsed)}")
    
    # Check model performance
    r2_values = [model.get('test_r2', 0) for model in models]
    logger.info(f"Model R² range: {min(r2_values):.4f} - {max(r2_values):.4f}, mean: {np.mean(r2_values):.4f}")
    
    return models

def filter_models_by_r2(models, min_r2=0.2):
    """Filter models to keep only those with R² above the threshold."""
    logger = setup_logging()
    filtered_models = [model for model in models if model.get('test_r2', -1) >= min_r2]
    
    total = len(models)
    kept = len(filtered_models)
    removed = total - kept
    
    logger.info(f"Filtered models by R² >= {min_r2}:")
    logger.info(f"  - Original: {total} models")
    logger.info(f"  - Kept: {kept} models ({kept/total:.1%})")
    logger.info(f"  - Removed: {removed} models ({removed/total:.1%})")
    
    return filtered_models

def calculate_feature_frequencies(models):
    """Calculate feature selection frequency across all models."""
    logger = setup_logging()
    logger.info("Calculating feature selection frequencies")
    
    feature_counter = Counter()
    for model in models:
        feature_counter.update(model['features'])
    
    n_models = len(models)
    feature_selection_freq = {feature: count / n_models for feature, count in feature_counter.items()}
    
    # Create a DataFrame for easier handling
    freq_df = pd.DataFrame([
        {'feature': feature, 'frequency': freq} 
        for feature, freq in feature_selection_freq.items()
    ])
    
    # Sort by frequency for reporting
    freq_df = freq_df.sort_values('frequency', ascending=False)
    
    logger.info(f"Calculated frequencies for {len(freq_df)} unique features")
    
    # Log top features
    logger.info("\nTop 10 most frequently selected features:")
    for idx, row in freq_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: selected in {row['frequency']*100:.1f}% of models")
    
    return feature_selection_freq, freq_df

def calculate_shap_importance(models, df, target_col='biomass_rel_change', max_samples=1000, max_background=100):
    """
    Calculate SHAP-based feature importance for all models.
    
    Args:
        models: List of model information dictionaries
        df: Dataset with features
        target_col: Target column name
        max_samples: Maximum samples to use for SHAP calculations
        max_background: Maximum background samples for SHAP explainer
    
    Returns:
        Dictionary with feature importance values
    """
    logger = setup_logging()
    start_time = time.time()
    logger.info(f"Calculating SHAP-based feature importance using {max_samples} samples and {max_background} background samples")
    
    # Dictionary to store SHAP values by feature
    shap_values_by_feature = {}
    
    # Standardize the dataset
    df_std = df.copy()
    exclude_cols = ['year_start', 'year_end', 'cluster_id', 'x', 'y']
    for col in df.columns:
        if col not in exclude_cols:
            df_std[col] = (df[col] - df[col].mean()) / df[col].std()
    
    for m_idx, model_info in enumerate(models):
        if m_idx % 100 == 0:
            logger.info(f"Processing SHAP for model {m_idx+1}/{len(models)}")
            if m_idx > 0:
                elapsed = time.time() - start_time
                remaining = elapsed / m_idx * (len(models) - m_idx)
                logger.info(f"Elapsed: {timedelta(seconds=elapsed)}, Remaining: {timedelta(seconds=remaining)}")
        
        model = model_info['model']
        features = model_info['features']
        
        try:
            # Use a random subset of data for this model's analysis
            if len(df_std) > max_samples:
                model_df = df_std.sample(max_samples, random_state=m_idx+42)
            else:
                model_df = df_std
            
            # Extract model data
            X = model_df[features].values
            
            # Select background data for SHAP explainer
            if len(X) > max_background:
                background_indices = np.random.choice(len(X), max_background, replace=False)
                background_data = X[background_indices]
            else:
                background_data = X
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model, background_data)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP value for each feature
            for j, feature in enumerate(features):
                feature_shap = np.abs(shap_values[:, j])
                mean_abs_shap = np.mean(feature_shap)
                
                if feature not in shap_values_by_feature:
                    shap_values_by_feature[feature] = []
                
                shap_values_by_feature[feature].append(mean_abs_shap)
                
        except Exception as e:
            logger.error(f"Error calculating SHAP values for model {m_idx+1}: {e}")
            continue
    
    # Calculate average SHAP importance for each feature
    avg_normalized_shap = {}
    for feature, values in shap_values_by_feature.items():
        if values:
            avg_normalized_shap[feature] = np.mean(values)
    
    elapsed = time.time() - start_time
    logger.info(f"Calculated SHAP importance for {len(avg_normalized_shap)} features in {timedelta(seconds=elapsed)}")
    
    # Log top features by importance
    top_features = sorted(avg_normalized_shap.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nTop 10 features by SHAP importance:")
    for feature, importance in top_features:
        logger.info(f"  {feature}: {importance:.4f}")
    
    return shap_values_by_feature, avg_normalized_shap

def calculate_permutation_importance(models, df, target_col='biomass_rel_change', max_samples=5000, n_repeats=10):
    """
    Calculate permutation importance with subsampling for efficiency.
    """
    logger = setup_logging()
    logger.info(f"Calculating permutation importance with {max_samples} samples and {n_repeats} repeats")
    start_time = time.time()
    
    permutation_imp = {}
    
    try:
        # Create standardized version of dataset
        df_std = df.copy()
        for col in df.columns:
            if col not in ['year_start', 'year_end', 'cluster_id', 'x', 'y']:
                df_std[col] = (df[col] - df[col].mean()) / df[col].std()
        
        # Use a subset for permutation importance calculation
        n_samples = len(df_std)
        if n_samples > max_samples:
            df_subset = df_std.sample(max_samples, random_state=42)
            logger.info(f"Using {max_samples} samples ({max_samples/n_samples:.1%} of dataset) for permutation importance")
        else:
            df_subset = df_std
            logger.info(f"Using all {n_samples} samples for permutation importance")
        
        for i, model_info in enumerate(models):
            if i % 100 == 0:
                logger.info(f"Processing permutation importance for model {i+1}/{len(models)}")
                if i > 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed / i * (len(models) - i)
                    logger.info(f"Elapsed time: {timedelta(seconds=elapsed)}, Estimated remaining: {timedelta(seconds=remaining)}")
            
            model = model_info['model']
            features = model_info['features']
            
            # Extract data for these features
            X = df_subset[features].values
            y = df_subset[target_col].values
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=model_info.get('run_seed', 42)
            )
            
            # Normalize importances
            total_imp = np.sum(perm_importance.importances_mean)
            if total_imp > 0:
                normalized_imp = perm_importance.importances_mean / total_imp
            else:
                normalized_imp = perm_importance.importances_mean
            
            # Store results
            for feature, imp in zip(features, normalized_imp):
                if feature not in permutation_imp:
                    permutation_imp[feature] = []
                permutation_imp[feature].append(imp)
        
        # Calculate average permutation importance
        avg_permutation_imp = {f: np.mean(vals) for f, vals in permutation_imp.items()}
        
        elapsed = time.time() - start_time
        logger.info(f"Calculated permutation importance in {timedelta(seconds=elapsed)}")
        
        # Log top features by permutation importance
        top_features = sorted(avg_permutation_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 features by permutation importance:")
        for feature, importance in top_features:
            logger.info(f"  {feature}: {importance:.4f}")
        
        return permutation_imp, avg_permutation_imp
        
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}, {}

def calculate_shap_pdps(models, df_std, top_features, max_samples=3000, background_size=1000):
    """
    Calculate SHAP-based PDPs for top features in high-performance models.
    Preserves the original SHAP PDP calculation logic with LOWESS smoothing.
    
    Args:
        models: List of model information dictionaries
        df_std: Standardized dataset
        top_features: List of top features to analyze
        max_samples: Maximum samples to use for SHAP calculations
        background_size: Size of background dataset for SHAP explainer
    
    Returns:
        Dictionary with SHAP PDP data including LOWESS smoothed curves
    """
    logger = setup_logging()
    start_time = time.time()
    logger.info(f"Calculating SHAP PDPs for {len(top_features)} features across {len(models)} models")
    
    # Initialize dictionary to store PDP data
    pdp_data = {feature: [] for feature in top_features}
    pdp_lowess_data = {feature: [] for feature in top_features}
    
    # Track progress
    total_models = len(models)
    
    # Process each model
    for i, model_info in enumerate(models):
        if i % 50 == 0:  # Log progress
            logger.info(f"Processing model {i+1}/{total_models}")
            if i > 0:
                elapsed = time.time() - start_time
                remaining = elapsed / i * (total_models - i)
                logger.info(f"Elapsed: {timedelta(seconds=elapsed)}, Remaining: {timedelta(seconds=remaining)}")
        
        model = model_info['model']
        features = model_info['features']
        model_r2 = model_info.get('test_r2', 0)
        
        # Get subset of features that are in both the model and our top features
        common_features = [f for f in top_features if f in features]
        if not common_features:
            continue
            
        try:
            # Use a subset of data for this model
            if len(df_std) > max_samples:
                model_df = df_std.sample(max_samples, random_state=i+42)
            else:
                model_df = df_std
                
            # Extract model's feature data
            X = model_df[features].values
            
            # Create background data for SHAP explainer
            if len(X) > background_size:
                background_indices = np.random.choice(len(X), background_size, replace=False)
                background_data = X[background_indices]
            else:
                background_data = X
            
            # Create SHAP explainer with background data
            explainer = shap.TreeExplainer(model, background_data)
            shap_values = explainer.shap_values(X)
            
            # Process each feature in this model
            for feature in common_features:
                feature_idx = features.index(feature)
                
                # Extract feature values and corresponding SHAP values
                feature_values = X[:, feature_idx]
                feature_shap_values = shap_values[:, feature_idx]
                
                # Store raw data for this feature and model
                pdp_data[feature].append({
                    'feature_values': feature_values,
                    'shap_values': feature_shap_values,
                    'model_r2': model_r2,
                    'model_id': i
                })
                
        except Exception as e:
            logger.error(f"Error processing model {i+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Calculate LOWESS smoothed curves for each feature
    logger.info("Calculating LOWESS smoothed curves...")
    for feature in top_features:
        if feature not in pdp_data or not pdp_data[feature]:
            continue
            
        feature_data = pdp_data[feature]
        logger.info(f"Processing LOWESS for {feature}: collected data from {len(feature_data)} models")
        
        # Calculate LOWESS smoothed curve using the original logic
        smooth_x, smooth_y = calculate_lowess_for_feature(feature_data, x_range=(-2.5, 2.5), n_models=50)
        
        if smooth_x is not None:
            pdp_lowess_data[feature] = {
                'smooth_x': smooth_x,
                'smooth_y': smooth_y,
                'n_models': len(feature_data)
            }
            logger.info(f"  Successfully calculated LOWESS for {feature}")
        else:
            logger.warning(f"  Failed to calculate LOWESS for {feature}")
    
    elapsed = time.time() - start_time
    logger.info(f"SHAP PDP calculation completed in {timedelta(seconds=elapsed)}")
    
    return pdp_data, pdp_lowess_data

def calculate_lowess_for_feature(feature_data, x_range=(-2.5, 2.5), n_models=50):
    """
    Calculate LOWESS smoothed curve for a single feature.
    Preserves the original LOWESS calculation logic.
    
    Returns:
        smooth_x: Feature values
        smooth_y: Smoothed SHAP values
    """
    # Subsample models for efficiency
    n_models_to_show = min(n_models, len(feature_data))
    indices = np.random.choice(len(feature_data), n_models_to_show, replace=False)
    
    # Collect points from subsampled models only
    all_feature_values = []
    all_shap_values = []
    
    for model_idx in indices:
        model_data = feature_data[model_idx]
        feature_vals = model_data['feature_values']
        shap_vals = model_data['shap_values']
        
        # Collect for ensemble
        all_feature_values.extend(feature_vals)
        all_shap_values.extend(shap_vals)
    
    # Filter data to xlim range for LOWESS calculation
    all_feature_values = np.array(all_feature_values)
    all_shap_values = np.array(all_shap_values)
    
    mask = (all_feature_values >= x_range[0]) & (all_feature_values <= x_range[1])
    filtered_feature_values = all_feature_values[mask]
    filtered_shap_values = all_shap_values[mask]
    
    if len(filtered_feature_values) < 10:
        return None, None
    
    # Sort for LOWESS
    sorted_indices = np.argsort(filtered_feature_values)
    feature_vals_sorted = filtered_feature_values[sorted_indices]
    shap_vals_sorted = filtered_shap_values[sorted_indices]
    
    # Apply LOWESS smoothing to ensemble
    try:
        lowess_result = lowess(shap_vals_sorted, feature_vals_sorted, frac=0.3)
        smooth_x = lowess_result[:, 0]
        smooth_y = lowess_result[:, 1]
        return smooth_x, smooth_y
    except Exception as e:
        print(f"LOWESS failed: {e}")
        return None, None

def calculate_focused_shap_interactions(models, features, df_path, max_samples=1000, max_background=200):
    """
    Calculate SHAP interaction values for selected features using top-performing models.
    Preserves the original interaction calculation logic.
    
    Args:
        models: List of top-performing models
        features: List of features to analyze interactions for
        df_path: Path to the dataset
        max_samples: Maximum samples per model for interaction calculation
        max_background: Maximum background samples for SHAP explainer
    
    Returns:
        Dictionary containing interaction data
    """
    logger = setup_logging()
    logger.info(f"Calculating SHAP interactions for {len(features)} features using {len(models)} models")
    
    start_time = time.time()
    
    # Load and standardize dataset
    df = pd.read_csv(df_path)
    df_std = df.copy()
    exclude_cols = ['year_start', 'year_end', 'cluster_id', 'x', 'y']
    
    for col in df.columns:
        if col not in exclude_cols:
            df_std[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Storage for aggregated interaction data by feature pair
    pairwise_interactions = {}  # {(feat1, feat2): [interaction_values]}
    pairwise_feature_values = {}  # {(feat1, feat2): [(feat1_vals, feat2_vals)]}
    interaction_counts = {}  # {(feat1, feat2): n_models_contributing}
    
    # Initialize storage for all possible feature pairs
    n_features = len(features)
    for i in range(n_features):
        for j in range(n_features):  # Include diagonal for main effects
            pair = (features[i], features[j])
            pairwise_interactions[pair] = []
            pairwise_feature_values[pair] = []
            interaction_counts[pair] = 0
    
    # Process each model
    successful_models = 0
    skipped_models = 0
    
    for m_idx, model_info in enumerate(models):
        if m_idx % 50 == 0:
            elapsed = time.time() - start_time
            if m_idx > 0:
                remaining = elapsed / m_idx * (len(models) - m_idx)
                logger.info(f"Processing model {m_idx+1}/{len(models)} - "
                           f"Elapsed: {timedelta(seconds=elapsed)}, "
                           f"Remaining: {timedelta(seconds=remaining)}")
        
        model = model_info['model']
        model_features = model_info['features']
        
        # Find which of our target features are present in this model
        present_features = [f for f in features if f in model_features]
        
        if len(present_features) < 2:
            skipped_models += 1
            continue  # Need at least 2 features to calculate interactions
        
        try:
            # Prepare data for this model
            if len(df_std) > max_samples:
                model_df = df_std.sample(max_samples, random_state=m_idx+42)
            else:
                model_df = df_std
            
            # Extract all model features
            X_all_features = model_df[model_features].values
            
            # Prepare background data
            if len(X_all_features) > max_background:
                background_indices = np.random.choice(len(X_all_features), max_background, replace=False)
                background_data = X_all_features[background_indices]
            else:
                background_data = X_all_features
            
            # Create SHAP explainer for the full model
            try:
                explainer = shap.TreeExplainer(model, background_data, feature_dependence="tree_path_dependent")
            except:
                # Fallback: no background data, which forces tree_path_dependent mode
                logger.warning(f"Model {m_idx}: Using fallback TreeExplainer (no background)")
                explainer = shap.TreeExplainer(model)
            
            # Calculate interaction values for all features in this model
            shap_interaction_values = explainer.shap_interaction_values(X_all_features)
            
            # Extract interactions for pairs of our target features that are present
            for feat1 in present_features:
                for feat2 in present_features:
                    # Get indices in the model's feature list
                    feat1_model_idx = model_features.index(feat1)
                    feat2_model_idx = model_features.index(feat2)
                    
                    # Extract interaction values for this pair
                    interaction_vals = shap_interaction_values[:, feat1_model_idx, feat2_model_idx]
                    
                    # Extract feature values for this pair
                    feat1_vals = X_all_features[:, feat1_model_idx]
                    feat2_vals = X_all_features[:, feat2_model_idx]
                    
                    # Store in our aggregated storage
                    pair = (feat1, feat2)
                    pairwise_interactions[pair].extend(interaction_vals)
                    pairwise_feature_values[pair].extend(list(zip(feat1_vals, feat2_vals)))
                    
            # Count this model as contributing to present feature pairs
            for feat1 in present_features:
                for feat2 in present_features:
                    pair = (feat1, feat2)
                    interaction_counts[pair] += 1
            
            successful_models += 1
            
        except Exception as e:
            logger.error(f"Error calculating interactions for model {m_idx}: {e}")
            skipped_models += 1
            continue
    
    if successful_models == 0:
        logger.error("No successful interaction calculations!")
        return None
    
    logger.info(f"Successfully processed {successful_models} models")
    logger.info(f"Skipped {skipped_models} models (insufficient features)")
    
    # Create aggregated interaction matrix and data structures
    mean_interaction_matrix = np.zeros((n_features, n_features))
    all_interactions_dict = {}  # Store all interaction values by feature pair
    all_feature_values_dict = {}  # Store all feature value pairs
    
    # Calculate mean interactions and store data
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            pair = (feat1, feat2)
            
            if len(pairwise_interactions[pair]) > 0:
                interaction_values = np.array(pairwise_interactions[pair])
                mean_interaction_matrix[i, j] = np.mean(interaction_values)
                
                # Store for detailed analysis
                all_interactions_dict[pair] = interaction_values
                all_feature_values_dict[pair] = np.array(pairwise_feature_values[pair])
            else:
                mean_interaction_matrix[i, j] = 0.0  # No data for this pair
                all_interactions_dict[pair] = np.array([])
                all_feature_values_dict[pair] = np.array([]).reshape(0, 2)
    
    # Calculate interaction strengths (off-diagonal only)
    interaction_strengths = {}
    for i in range(n_features):
        for j in range(i+1, n_features):  # Only upper triangle
            pair = (features[i], features[j])
            if len(all_interactions_dict[pair]) > 0:
                interaction_strength = np.mean(np.abs(all_interactions_dict[pair]))
                interaction_strengths[pair] = interaction_strength
            else:
                interaction_strengths[pair] = 0.0
    
    # Sort interactions by strength
    sorted_interactions = sorted(interaction_strengths.items(), key=lambda x: x[1], reverse=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Calculated SHAP interactions in {timedelta(seconds=elapsed)}")
    
    results = {
        'features': features,
        'mean_interaction_matrix': mean_interaction_matrix,
        'all_interactions_dict': all_interactions_dict,
        'all_feature_values_dict': all_feature_values_dict,
        'interaction_strengths': interaction_strengths,
        'sorted_interactions': sorted_interactions,
        'interaction_counts': interaction_counts,
        'n_models_used': successful_models,
        'n_models_skipped': skipped_models
    }
    
    return results

def get_top_features_and_models(models, r2_threshold=0.2, top_percentile=10, min_selection_freq=0.3):
    """
    Get top features and top-performing models for interaction analysis.
    """
    logger = setup_logging()
    logger.info("Selecting top features and models for interaction analysis")
    
    # Filter models by R² threshold
    filtered_models = [model for model in models if model.get('test_r2', -1) >= r2_threshold]
    logger.info(f"Filtered to {len(filtered_models)} models with R² >= {r2_threshold}")
    
    # Calculate feature frequencies in filtered models
    feature_counter = Counter()
    for model in filtered_models:
        feature_counter.update(model['features'])
    
    feature_freq = {feature: count / len(filtered_models) 
                   for feature, count in feature_counter.items()}
    
    # Get top features (selected frequently)
    top_features = [(f, freq) for f, freq in feature_freq.items() if freq >= min_selection_freq]
    top_features.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Top features (selected >{min_selection_freq*100}% of time):")
    for i, (feature, freq) in enumerate(top_features[:10], 1):
        logger.info(f"  {i:2d}. {feature:<30} : {freq*100:5.1f}%")
    
    # Get top percentile of models by R²
    sorted_models = sorted(filtered_models, key=lambda x: x.get('test_r2', 0), reverse=True)
    n_top_models = max(1, int(len(filtered_models) * top_percentile / 100))
    top_models = sorted_models[:n_top_models]
    
    logger.info(f"Selected top {top_percentile}% models: {n_top_models} models")
    logger.info(f"R² range: {min(m['test_r2'] for m in top_models):.4f} - {max(m['test_r2'] for m in top_models):.4f}")
    
    # Extract just the feature names for return
    top_feature_names = [f[0] for f in top_features]
    
    return top_models, top_feature_names

def save_results(results, output_dir='results/shap_analysis'):
    """Save all analysis results to the specified directory."""
    logger = setup_logging()
    logger.info(f"Saving comprehensive SHAP analysis results to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each analysis component
    for key, data in results.items():
        if key.endswith('_df'):
            # Save DataFrames as CSV
            filename = f"{key}.csv"
            data.to_csv(os.path.join(output_dir, filename), index=False)
        elif key.endswith('_summary'):
            # Save summaries as JSON
            filename = f"{key}.json"
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(data, f, indent=4)
        else:
            # Save everything else as pickle
            filename = f"{key}.pkl"
            with open(os.path.join(output_dir, filename), 'wb') as f:
                pickle.dump(data, f)
        
        logger.info(f"Saved {key} to {filename}")
    
    logger.info(f"All results saved to {output_dir}")

def run_comprehensive_shap_analysis(config_path="climate_biomass_config.yaml"):
    """
    Run the complete comprehensive SHAP analysis pipeline.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: All analysis results
    """
    logger = setup_logging()
    logger.info("Starting comprehensive SHAP analysis pipeline...")
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Use default paths if config doesn't have SHAP section
        models_dir = config.get('paths', {}).get('models_dir', 'results/climate_biomass_analysis/models_remove_4')
        dataset_path = config.get('paths', {}).get('clustered_dataset', 'results/climate_biomass_analysis/biomass_climate_data_with_clusters_20250509_100039.csv')
        output_dir = config.get('paths', {}).get('shap_analysis_dir', 'results/shap_analysis')
        
        # Analysis parameters
        r2_threshold = 0.2
        max_samples = 1000
        max_background = 100
        
        # Step 1: Load models
        logger.info("="*60)
        logger.info("STEP 1: Loading models")
        logger.info("="*60)
        models = load_models(models_dir)
        
        # Filter models by R²
        filtered_models = filter_models_by_r2(models, min_r2=r2_threshold)
        
        # Step 2: Load dataset
        logger.info("="*60)
        logger.info("STEP 2: Loading dataset")
        logger.info("="*60)
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Step 3: Feature selection frequency analysis
        logger.info("="*60)
        logger.info("STEP 3: Feature selection frequency analysis")
        logger.info("="*60)
        feature_selection_freq, freq_df = calculate_feature_frequencies(filtered_models)
        
        # Step 4: SHAP importance analysis
        logger.info("="*60)
        logger.info("STEP 4: SHAP importance analysis")
        logger.info("="*60)
        shap_values_by_feature, avg_shap_importance = calculate_shap_importance(
            filtered_models, df, max_samples=max_samples, max_background=max_background
        )
        
        # Step 5: Permutation importance analysis
        logger.info("="*60)
        logger.info("STEP 5: Permutation importance analysis")
        logger.info("="*60)
        permutation_importance, avg_permutation_importance = calculate_permutation_importance(
            filtered_models, df, max_samples=5000, n_repeats=10
        )
        
        # Step 6: SHAP PDP analysis
        logger.info("="*60)
        logger.info("STEP 6: SHAP PDP analysis")
        logger.info("="*60)
        # Get top 6 features for PDP analysis
        top_features = freq_df.head(6)['feature'].tolist()
        logger.info(f"Analyzing PDPs for top 6 features: {top_features}")
        
        # Standardize dataset for PDP analysis
        df_std = df.copy()
        exclude_cols = ['year_start', 'year_end', 'cluster_id', 'x', 'y']
        for col in df.columns:
            if col not in exclude_cols:
                df_std[col] = (df[col] - df[col].mean()) / df[col].std()
        
        pdp_data, pdp_lowess_data = calculate_shap_pdps(
            filtered_models, df_std, top_features, max_samples=3000, background_size=1000
        )
        
        # Step 7: SHAP interaction analysis
        logger.info("="*60)
        logger.info("STEP 7: SHAP interaction analysis")
        logger.info("="*60)
        # Get top models and features for interaction analysis
        top_models, interaction_features = get_top_features_and_models(
            models, r2_threshold=0.2, top_percentile=10, min_selection_freq=0.3
        )
        
        # Use top 6 features for computational efficiency
        selected_features = interaction_features[:6]
        logger.info(f"Analyzing interactions for features: {selected_features}")
        
        interaction_results = calculate_focused_shap_interactions(
            models=top_models,
            features=selected_features,
            df_path=dataset_path,
            max_samples=1000,
            max_background=200
        )
        
        # Compile all results
        results = {
            'feature_selection_freq': feature_selection_freq,
            'feature_frequencies_df': freq_df,
            'shap_values_by_feature': shap_values_by_feature,
            'avg_shap_importance': avg_shap_importance,
            'permutation_importance': permutation_importance,
            'avg_permutation_importance': avg_permutation_importance,
            'pdp_data': pdp_data,
            'pdp_lowess_data': pdp_lowess_data,
            'interaction_results': interaction_results,
            'analysis_summary': {
                'n_total_models': len(models),
                'n_filtered_models': len(filtered_models),
                'r2_threshold': r2_threshold,
                'top_features': top_features,
                'interaction_features': selected_features,
                'dataset_shape': df.shape,
                'analysis_time': time.time() - start_time
            }
        }
        
        # Save all results
        save_results(results, output_dir)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("COMPREHENSIVE SHAP ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Total execution time: {timedelta(seconds=total_time)}")
        logger.info(f"Processed {len(filtered_models)} models (R² >= {r2_threshold})")
        logger.info(f"Analyzed {len(top_features)} features for PDPs")
        logger.info(f"Analyzed {len(selected_features)} features for interactions")
        logger.info(f"All results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive SHAP analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """Main function to run the comprehensive SHAP analysis."""
    try:
        results = run_comprehensive_shap_analysis()
        print("\n🎉 Comprehensive SHAP analysis completed successfully!")
        print("📊 Generated outputs:")
        print("   1. ✅ Feature selection frequencies")
        print("   2. ✅ SHAP importance values")
        print("   3. ✅ Permutation importance values")
        print("   4. ✅ SHAP PDP data with LOWESS curves")
        print("   5. ✅ 2D SHAP interaction matrices")
        print(f"📁 Results saved to: results/shap_analysis/")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
