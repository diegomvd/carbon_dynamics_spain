"""
SHAP Analysis Core Module

This module provides comprehensive SHAP analysis functionality for the climate-biomass
analysis pipeline, including feature selection frequency analysis, SHAP importance
calculation, permutation importance, 1D PDP calculation with LOWESS smoothing,
and 2D SHAP interaction analysis.

Author: Diego Bengochea
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
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_utils import setup_logging, load_config, ensure_directory
from shared_utils.central_data_paths_constants import *


class ShapAnalysisPipeline:
    """
    Comprehensive SHAP Analysis Pipeline
    
    Combines all SHAP-related analyses into a single class:
    1. Feature selection frequency analysis
    2. SHAP importance calculation  
    3. Permutation importance calculation
    4. 1D SHAP PDP calculation with LOWESS smoothing
    5. 2D SHAP interaction analysis
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the SHAP analyzer.
        
        Args:
            config_path: Path to configuration file. If None, uses component default.
        """
        # Load configuration
        self.config = load_config(config_path, component_name="climate_biomass_analysis")
        
        # Setup logging
        self.logger = setup_logging(
            level=self.config['logging']['level'],
            component_name='shap_analyzer',
            log_file=self.config['logging'].get('log_file')
        )
        
        # Extract configuration sections
        self.shap_config = self.config['shap_analysis']        
        self.logger.info("Initialized ShapAnalyzer")
    
    def run_full_pipeline(self) -> bool:
        """
        Run the complete SHAP analysis pipeline.
        
        Returns:
            Dict containing all analysis results
        """
        self.logger.info("Starting comprehensive SHAP analysis pipeline...")
        start_time = time.time()
        
        try:
            # Extract configuration
            models_dir = CLIMATE_BIOMASS_MODELS_DIR
            dataset_path = CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE
            output_dir = CLIMATE_BIOMASS_SHAP_OUTPUT_DIR
            
            # Analysis parameters
            r2_threshold = self.shap_config['model_filtering']['r2_threshold']
            max_samples = self.shap_config['analysis']['shap_max_samples']
            max_background = self.shap_config['analysis']['shap_max_background']
            
            # Step 1: Load models
            self.logger.info("="*60)
            self.logger.info("STEP 1: Loading models")
            self.logger.info("="*60)
            models = self.load_models(models_dir)
            
            # Filter models by R²
            filtered_models = self.filter_models_by_r2(models, min_r2=r2_threshold)
            
            # Step 2: Load dataset
            self.logger.info("="*60)
            self.logger.info("STEP 2: Loading dataset")
            self.logger.info("="*60)
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Step 3: Feature frequency analysis
            self.logger.info("="*60)
            self.logger.info("STEP 3: Feature frequency analysis")
            self.logger.info("="*60)
            freq_df = self.calculate_feature_frequencies(filtered_models)
            
            # Step 4: SHAP importance calculation
            self.logger.info("="*60)
            self.logger.info("STEP 4: SHAP importance calculation")
            self.logger.info("="*60)
            avg_shap_importance = self.calculate_shap_importance(filtered_models, df, max_samples, max_background)
            
            # Step 5: Permutation importance calculation
            self.logger.info("="*60)
            self.logger.info("STEP 5: Permutation importance calculation")
            self.logger.info("="*60)
            perm_max_samples = self.shap_config['analysis']['perm_max_samples']
            perm_n_repeats = self.shap_config['analysis']['perm_n_repeats']
            avg_permutation_importance = self.calculate_permutation_importance(
                filtered_models, df, perm_max_samples, perm_n_repeats
            )
            
            # Step 6: PDP analysis
            self.logger.info("="*60)
            self.logger.info("STEP 6: PDP analysis with LOWESS")
            self.logger.info("="*60)
            pdp_config = self.shap_config['analysis']
            pdp_data, pdp_lowess_data = self.calculate_pdp_with_lowess(
                filtered_models, df,
                max_samples=pdp_config['pdp_max_samples'],
                background_size=pdp_config['pdp_background_size'],
                n_top_features=pdp_config['pdp_n_top_features'],
                x_range=pdp_config['pdp_x_range'],
                lowess_frac=pdp_config['pdp_lowess_frac'],
                n_models_subsample=pdp_config['pdp_n_models_subsample']
            )
            
            # Step 7: Interaction analysis
            self.logger.info("="*60)
            self.logger.info("STEP 7: Interaction analysis")
            self.logger.info("="*60)
            interaction_config = self.shap_config['analysis']
            interaction_results = self.calculate_interaction_analysis(
                filtered_models, df,
                max_samples=interaction_config['interaction_max_samples'],
                max_background=interaction_config['interaction_max_background'],
                n_top_features=interaction_config['interaction_n_top_features']
            )
            
            # Step 8: Save results
            self.logger.info("="*60)
            self.logger.info("STEP 8: Saving results")
            self.logger.info("="*60)
            
            # Create output directory
            ensure_directory(output_dir)
            
            # Save feature frequencies
            freq_df.to_csv(os.path.join(output_dir, 'feature_frequencies.csv'), index=False)
            
            # Save SHAP importance
            with open(os.path.join(output_dir, 'shap_importance.pkl'), 'wb') as f:
                pickle.dump(avg_shap_importance, f)
            
            # Save permutation importance
            with open(os.path.join(output_dir, 'permutation_importance.pkl'), 'wb') as f:
                pickle.dump(avg_permutation_importance, f)
            
            # Save PDP data
            with open(os.path.join(output_dir, 'shap_pdp_data.pkl'), 'wb') as f:
                pickle.dump(pdp_data, f)
            
            # Save PDP LOWESS data
            with open(os.path.join(output_dir, 'pdp_lowess_data.pkl'), 'wb') as f:
                pickle.dump(pdp_lowess_data, f)
            
            # Save interaction results
            with open(os.path.join(output_dir, 'focused_shap_interactions.pkl'), 'wb') as f:
                pickle.dump(interaction_results, f)
            
            # Create analysis summary
            end_time = time.time()
            analysis_time = end_time - start_time
            
            summary = {
                'analysis_completed': True,
                'analysis_time_seconds': analysis_time,
                'analysis_time_formatted': str(timedelta(seconds=int(analysis_time))),
                'models_loaded': len(models),
                'models_filtered': len(filtered_models),
                'dataset_shape': df.shape,
                'features_analyzed': len(freq_df),
                'features_with_shap': len(avg_shap_importance),
                'features_with_permutation': len(avg_permutation_importance),
                'features_with_pdp': len(pdp_data),
                'interaction_samples': interaction_results.get('n_samples', 0),
                'output_directory': output_dir
            }
            
            # Save summary
            with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Log completion
            self.logger.info("="*60)
            self.logger.info("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info(f"Analysis time: {summary['analysis_time_formatted']}")
            self.logger.info(f"Models processed: {summary['models_filtered']}/{summary['models_loaded']}")
            self.logger.info(f"Features analyzed: {summary['features_analyzed']}")
            self.logger.info(f"Results saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {str(e)}")
            raise

    def load_models(self, models_dir: str) -> Dict[str, Any]:
        """Load all models from subdirectories within the models directory."""
        self.logger.info(f"Loading models from {models_dir}")
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        models = {}
        model_files = glob.glob(os.path.join(models_dir, '**', 'run_*.pkl'), recursive=True)
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        self.logger.info(f"Found {len(model_files)} model files")
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Extract model identifier from path
                relative_path = os.path.relpath(model_file, models_dir)
                model_id = os.path.dirname(relative_path)
                
                models[model_id] = model_data
                
            except Exception as e:
                self.logger.warning(f"Failed to load model {model_file}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(models)} models")
        return models

    def filter_models_by_r2(self, models: Dict[str, Any], min_r2: float = 0.2) -> Dict[str, Any]:
        """Filter models by minimum R² threshold."""
        self.logger.info(f"Filtering models with R² >= {min_r2}")
        
        filtered_models = {}
        
        for model_id, model_data in models.items():
            try:
                # Extract R² from model data
                if 'test_r2' in model_data:
                    r2 = model_data['test_r2']
                elif 'val_r2' in model_data:
                    r2 = model_data['val_r2']
                else:
                    self.logger.warning(f"No R² found for model {model_id}")
                    continue
                
                if r2 >= min_r2:
                    filtered_models[model_id] = model_data
                    
            except Exception as e:
                self.logger.warning(f"Error processing model {model_id}: {e}")
                continue
        
        self.logger.info(f"Filtered models: {len(filtered_models)}/{len(models)} passed R² threshold")
        return filtered_models

    def calculate_feature_frequencies(self, models: Dict[str, Any]) -> pd.DataFrame:
        """Calculate feature selection frequencies across models."""
        self.logger.info("Calculating feature selection frequencies")
        
        all_features = []
        total_models = len(models)
        
        for model_id, model_data in models.items():
            if 'features' in model_data:
                all_features.extend(model_data['features'])
            else:
                self.logger.warning(f"No features found for model {model_id}")
        
        if not all_features:
            raise ValueError("No features found in any model")
        
        # Count feature occurrences
        feature_counts = Counter(all_features)
        
        # Calculate frequencies
        freq_data = []
        for feature, count in feature_counts.items():
            frequency = count / total_models
            freq_data.append({
                'feature': feature,
                'count': count,
                'frequency': frequency
            })
        
        freq_df = pd.DataFrame(freq_data).sort_values('frequency', ascending=False)
        
        self.logger.info(f"Calculated frequencies for {len(freq_df)} unique features")
        return freq_df

    def calculate_shap_importance(self, models: Dict[str, Any], df: pd.DataFrame, 
                                 max_samples: int = 1000, max_background: int = 100) -> Dict[str, float]:
        """Calculate average SHAP importance across models."""
        self.logger.info("Calculating SHAP importance across models")
        
        # Prepare data
        feature_columns = [col for col in df.columns if col not in ['biomass_rel_change', 'cluster_id']]
        X = df[feature_columns]
        
        # Limit samples for computational efficiency
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        # Background data for SHAP
        if len(X) > max_background:
            bg_indices = np.random.choice(len(X), max_background, replace=False)
            X_background = X.iloc[bg_indices]
        else:
            X_background = X
        
        shap_importances = {}
        valid_models = 0
        
        for model_id, model_data in models.items():
            try:
                model = model_data['model']
                selected_features = model_data['features']
                
                # Get feature indices
                feature_indices = [feature_columns.index(feat) for feat in selected_features 
                                 if feat in feature_columns]
                
                if not feature_indices:
                    continue
                
                # Prepare data for this model
                X_model = X_sample.iloc[:, feature_indices]
                X_bg_model = X_background.iloc[:, feature_indices]
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model, X_bg_model)
                shap_values = explainer.shap_values(X_model)
                
                # Calculate mean absolute SHAP values
                mean_shap = np.mean(np.abs(shap_values), axis=0)
                
                # Store importance for each feature
                for i, feat in enumerate(selected_features):
                    if feat in feature_columns:
                        if feat not in shap_importances:
                            shap_importances[feat] = []
                        shap_importances[feat].append(mean_shap[i])
                
                valid_models += 1
                
            except Exception as e:
                self.logger.warning(f"SHAP calculation failed for model {model_id}: {e}")
                continue
        
        # Average across models
        avg_shap_importance = {}
        for feat, importance_list in shap_importances.items():
            avg_shap_importance[feat] = np.mean(importance_list)
        
        self.logger.info(f"Calculated SHAP importance using {valid_models} models")
        return avg_shap_importance

    def calculate_permutation_importance(self, models: Dict[str, Any], df: pd.DataFrame,
                                       max_samples: int = 5000, n_repeats: int = 10) -> Dict[str, float]:
        """Calculate average permutation importance across models."""
        self.logger.info("Calculating permutation importance across models")
        
        # Prepare data
        feature_columns = [col for col in df.columns if col not in ['biomass_rel_change', 'cluster_id']]
        X = df[feature_columns]
        y = df['biomass_rel_change']
        
        # Limit samples for computational efficiency
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        perm_importances = {}
        valid_models = 0
        
        for model_id, model_data in models.items():
            try:
                model = model_data['model']
                selected_features = model_data['features']
                
                # Get feature indices
                feature_indices = [feature_columns.index(feat) for feat in selected_features 
                                 if feat in feature_columns]
                
                if not feature_indices:
                    continue
                
                # Prepare data for this model
                X_model = X_sample.iloc[:, feature_indices]
                
                # Calculate permutation importance
                perm_result = permutation_importance(
                    model, X_model, y_sample, 
                    n_repeats=n_repeats, random_state=42, scoring='r2'
                )
                
                # Store importance for each feature
                for i, feat in enumerate(selected_features):
                    if feat in feature_columns:
                        if feat not in perm_importances:
                            perm_importances[feat] = []
                        perm_importances[feat].append(perm_result.importances_mean[i])
                
                valid_models += 1
                
            except Exception as e:
                self.logger.warning(f"Permutation importance failed for model {model_id}: {e}")
                continue
        
        # Average across models
        avg_perm_importance = {}
        for feat, importance_list in perm_importances.items():
            avg_perm_importance[feat] = np.mean(importance_list)
        
        self.logger.info(f"Calculated permutation importance using {valid_models} models")
        return avg_perm_importance

    def calculate_pdp_with_lowess(self, models: Dict[str, Any], df: pd.DataFrame,
                                 max_samples: int = 3000, background_size: int = 1000,
                                 n_top_features: int = 6, x_range: List[float] = [-2.5, 2.5],
                                 lowess_frac: float = 0.3, n_models_subsample: int = 50) -> Tuple[Dict, Dict]:
        """Calculate 1D PDP SHAP values with LOWESS smoothing."""
        self.logger.info("Calculating PDP with LOWESS smoothing")
        
        # Get top features by frequency
        freq_df = self.calculate_feature_frequencies(models)
        top_features = freq_df.head(n_top_features)['feature'].tolist()
        
        # Prepare data
        feature_columns = [col for col in df.columns if col not in ['biomass_rel_change', 'cluster_id']]
        X = df[feature_columns]
        
        # Limit samples
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        # Background data
        if len(X) > background_size:
            bg_indices = np.random.choice(len(X), background_size, replace=False)
            X_background = X.iloc[bg_indices]
        else:
            X_background = X
        
        # Create x values for PDP
        x_values = np.linspace(x_range[0], x_range[1], 100)
        
        pdp_data = {}
        pdp_lowess_data = {}
        
        for feature in top_features:
            if feature not in feature_columns:
                continue
                
            self.logger.info(f"Calculating PDP for {feature}")
            
            # Store all SHAP values for this feature
            all_shap_values = []
            all_x_values = []
            
            # Subsample models for efficiency
            model_items = list(models.items())
            if len(model_items) > n_models_subsample:
                model_items = np.random.choice(model_items, n_models_subsample, replace=False)
            
            for model_id, model_data in model_items:
                try:
                    model = model_data['model']
                    selected_features = model_data['features']
                    
                    if feature not in selected_features:
                        continue
                    
                    # Get feature indices
                    feature_indices = [feature_columns.index(feat) for feat in selected_features 
                                     if feat in feature_columns]
                    feature_idx_in_model = selected_features.index(feature)
                    
                    # Prepare data for this model
                    X_model = X_sample.iloc[:, feature_indices]
                    X_bg_model = X_background.iloc[:, feature_indices]
                    
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(model, X_bg_model)
                    
                    # Calculate PDP
                    for x_val in x_values:
                        X_modified = X_model.copy()
                        X_modified.iloc[:, feature_idx_in_model] = x_val
                        
                        shap_values = explainer.shap_values(X_modified)
                        mean_shap = np.mean(shap_values[:, feature_idx_in_model])
                        
                        all_shap_values.append(mean_shap)
                        all_x_values.append(x_val)
                
                except Exception as e:
                    self.logger.warning(f"PDP calculation failed for {feature} in model {model_id}: {e}")
                    continue
            
            if not all_shap_values:
                continue
            
            # Store raw data
            pdp_data[feature] = {
                'x_values': all_x_values,
                'shap_values': all_shap_values
            }
            
            # Calculate LOWESS smoothing
            if len(all_x_values) > 10:  # Need sufficient data points
                try:
                    # Sort by x values
                    sorted_data = sorted(zip(all_x_values, all_shap_values))
                    x_sorted, y_sorted = zip(*sorted_data)
                    
                    # Apply LOWESS
                    lowess_result = lowess(y_sorted, x_sorted, frac=lowess_frac, return_sorted=True)
                    
                    pdp_lowess_data[feature] = {
                        'x_lowess': lowess_result[:, 0],
                        'y_lowess': lowess_result[:, 1]
                    }
                    
                except Exception as e:
                    self.logger.warning(f"LOWESS calculation failed for {feature}: {e}")
        
        self.logger.info(f"Calculated PDP for {len(pdp_data)} features")
        return pdp_data, pdp_lowess_data

    def calculate_interaction_analysis(self, models: Dict[str, Any], df: pd.DataFrame,
                                     feature1: str = "bio12", feature2: str = "bio12_3yr",
                                     max_samples: int = 1000, max_background: int = 200,
                                     n_top_features: int = 6) -> Dict[str, Any]:
        """Calculate 2D SHAP interaction analysis."""
        self.logger.info(f"Calculating interaction analysis for {feature1} vs {feature2}")
        
        # Filter models that include both features
        r2_threshold_interactions = self.shap_config['model_filtering'].get('r2_threshold_interactions', 0.2)
        top_percentile_interactions = self.shap_config['model_filtering'].get('top_percentile_interactions', 10)
        min_selection_freq_interactions = self.shap_config['model_filtering'].get('min_selection_freq_interactions', 0.3)
        
        # Get models with high R² and both features
        valid_models = {}
        r2_values = []
        
        for model_id, model_data in models.items():
            features = model_data.get('features', [])
            if feature1 in features and feature2 in features:
                r2 = model_data.get('test_r2', model_data.get('val_r2', 0))
                if r2 >= r2_threshold_interactions:
                    valid_models[model_id] = model_data
                    r2_values.append(r2)
        
        if not valid_models:
            self.logger.warning("No models found with both interaction features")
            return {}
        
        # Select top models by R²
        if len(valid_models) > 10:  # Limit to top models for efficiency
            r2_threshold_top = np.percentile(r2_values, 100 - top_percentile_interactions)
            valid_models = {k: v for k, v in valid_models.items() 
                           if v.get('test_r2', v.get('val_r2', 0)) >= r2_threshold_top}
        
        # Prepare data
        feature_columns = [col for col in df.columns if col not in ['biomass_rel_change', 'cluster_id']]
        X = df[feature_columns]
        
        # Limit samples
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        # Background data
        if len(X) > max_background:
            bg_indices = np.random.choice(len(X), max_background, replace=False)
            X_background = X.iloc[bg_indices]
        else:
            X_background = X
        
        # Calculate interactions across models
        all_interactions = []
        all_feature1_values = []
        all_feature2_values = []
        
        for model_id, model_data in valid_models.items():
            try:
                model = model_data['model']
                selected_features = model_data['features']
                
                # Get feature indices
                feature_indices = [feature_columns.index(feat) for feat in selected_features 
                                 if feat in feature_columns]
                feat1_idx = selected_features.index(feature1)
                feat2_idx = selected_features.index(feature2)
                
                # Prepare data for this model
                X_model = X_sample.iloc[:, feature_indices]
                X_bg_model = X_background.iloc[:, feature_indices]
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model, X_bg_model)
                shap_values = explainer.shap_values(X_model)
                interaction_values = explainer.shap_interaction_values(X_model)
                
                # Extract interaction values
                interactions = interaction_values[:, feat1_idx, feat2_idx]
                feat1_vals = X_model.iloc[:, feat1_idx].values
                feat2_vals = X_model.iloc[:, feat2_idx].values
                
                all_interactions.extend(interactions)
                all_feature1_values.extend(feat1_vals)
                all_feature2_values.extend(feat2_vals)
                
            except Exception as e:
                self.logger.warning(f"Interaction calculation failed for model {model_id}: {e}")
                continue
        
        if not all_interactions:
            self.logger.warning("No successful interaction calculations")
            return {}
        
        interaction_results = {
            'feature1': feature1,
            'feature2': feature2,
            'feature1_values': all_feature1_values,
            'feature2_values': all_feature2_values,
            'interaction_values': all_interactions,
            'n_models_used': len(valid_models),
            'n_samples': len(all_interactions)
        }
        
        self.logger.info(f"Calculated interactions using {len(valid_models)} models, {len(all_interactions)} samples")
        return interaction_results

    