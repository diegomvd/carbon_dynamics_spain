# Climate-Biomass Analysis Component

Comprehensive climate-biomass relationship analysis for the Iberian Carbon Assessment Pipeline, including climate data processing, bioclimatic variables calculation, biomass-climate integration, spatial analysis, machine learning optimization, and SHAP analysis for model interpretability.

## Overview

This component analyzes the relationships between climate variables and biomass changes using advanced spatial modeling techniques. The pipeline integrates:

- **Climate data processing** - GRIB to GeoTIFF conversion with proper georeferencing
- **Bioclimatic variables calculation** - Bio1-Bio19 variables and anomaly analysis  
- **Biomass-climate integration** - Spatial alignment and data harmonization
- **Spatial autocorrelation analysis** - Semivariogram calculation and clustering
- **Machine learning optimization** - Bayesian optimization with spatial cross-validation
- **SHAP analysis** - Comprehensive model interpretability and feature importance analysis

## Structure

```
climate_biomass_analysis/
├── config.yaml                        # Component configuration
├── visualization_config.yaml          # Visualization-specific configuration
├── __init__.py                        # Package initialization
├── README.md                          # This file
├── core/                              # Core processing modules
│   ├── __init__.py
│   ├── climate_raster_processing.py   # GRIB to GeoTIFF conversion
│   ├── bioclim_calculation.py         # Bioclimatic variables calculation
│   ├── biomass_integration.py         # Biomass-climate data integration
│   ├── spatial_analysis.py            # Spatial autocorrelation & clustering
│   ├── optimization_pipeline.py       # ML optimization & feature selection
│   └── shap_analysis.py               # SHAP analysis for interpretability
└── scripts/                           # Executable entry points
    ├── __init__.py
    ├── run_climate_processing.py       # Climate data processing
    ├── run_bioclim_calculation.py      # Bioclimatic variables
    ├── run_biomass_integration.py      # Data integration
    ├── run_spatial_analysis.py         # Spatial analysis
    ├── run_optimization.py             # ML optimization
    ├── run_shap_analysis.py            # SHAP analysis
    └── run_full_pipeline.py            # Complete pipeline orchestrator
```

## Quick Start

### Generate Paper Figures (Easy Mode)
```bash
# Generate all publication figures from repository root
python generate_paper_figures.py

# High-quality figures for publication
python generate_paper_figures.py --high-quality

# Specific figures only
python generate_paper_figures.py --figures importance main
```

### Environment Setup

```bash
# Create conda environment
conda env create -f ../environments/climate_analysis.yml
conda activate climate_analysis

# Install repository for absolute imports
pip install -e ../..
```

### Complete Pipeline

```bash
# Run full pipeline with default configuration
python scripts/run_full_pipeline.py

# Run specific stages only
python scripts/run_full_pipeline.py --stages bioclim_calculation biomass_integration

# Continue on errors and use custom config
python scripts/run_full_pipeline.py --continue-on-error --config my_config.yaml

# Dry run to see what would be executed
python scripts/run_full_pipeline.py --dry-run
```

## Individual Processing Steps

### 1. Climate Data Processing
```bash
# Process GRIB files to GeoTIFF format
python scripts/run_climate_processing.py \
  --input-dir /path/to/grib/files \
  --output-dir /path/to/geotiff/output \
  --target-crs EPSG:25830

# With reference grid alignment
python scripts/run_climate_processing.py \
  --input-dir ./climate_data \
  --output-dir ./climate_outputs \
  --reference-grid ./reference.tif \
  --validate-outputs
```

### 2. Bioclimatic Variables Calculation
```bash
# Calculate bioclimatic variables and anomalies
python scripts/run_bioclim_calculation.py

# Calculate only reference period bioclimatic variables
python scripts/run_bioclim_calculation.py --mode reference

# Custom time periods with rolling years
python scripts/run_bioclim_calculation.py \
  --reference-start 1981 --reference-end 2010 \
  --analysis-start 2017 --analysis-end 2024 \
  --rolling-years

# Calculate specific variables only
python scripts/run_bioclim_calculation.py \
  --variables bio1 bio12 bio15 \
  --continue-on-error
```

### 3. Biomass-Climate Integration
```bash
# Integrate biomass changes with climate anomalies
python scripts/run_biomass_integration.py

# Custom input directories
python scripts/run_biomass_integration.py \
  --biomass-dir ./biomass_differences \
  --anomaly-dir ./climate_anomalies \
  --output-dataset ./training_data.csv
```

### 4. Spatial Analysis
```bash
# Perform spatial autocorrelation analysis and clustering
python scripts/run_spatial_analysis.py

# Custom dataset and clustering parameters
python scripts/run_spatial_analysis.py \
  --input-dataset ./training_data.csv \
  --n-clusters 25 \
  --autocorr-threshold 50
```

### 5. Machine Learning Optimization
```bash
# Run Bayesian optimization for feature selection
python scripts/run_optimization.py

# Custom optimization parameters
python scripts/run_optimization.py \
  --n-runs 5 \
  --n-trials 500 \
  --patience 30 \
  --output-dir ./ml_results

# Quick test run
python scripts/run_optimization.py \
  --single-run \
  --validate-only

# Exclude specific bioclimatic variables
python scripts/run_optimization.py \
  --exclude-bio-vars bio8 bio9 bio18 bio19 \
  --correlation-threshold 0.9
```

### 6. SHAP Analysis (New!)
```bash
# Run comprehensive SHAP analysis
python scripts/run_shap_analysis.py

# Custom parameters
python scripts/run_shap_analysis.py \
  --models-dir ./results/models \
  --dataset ./ml_dataset_with_clusters.csv \
  --output-dir ./shap_results

# SHAP analysis with custom sampling
python scripts/run_shap_analysis.py \
  --shap-max-samples 2000 \
  --pdp-n-top-features 8 \
  --r2-threshold 0.3

# Validate inputs before running
python scripts/run_shap_analysis.py \
  --validate-inputs \
  --dry-run
```

## Configuration

### Main Configuration (`config.yaml`)

Key configuration sections:

#### Data Paths
```yaml
data:
  climate_outputs: "climate_outputs"           # Climate GeoTIFF files
  harmonized_dir: "harmonized_climate"         # Harmonized rasters
  bioclim_dir: "bioclimatic_variables"         # Bio1-Bio19 variables
  anomaly_dir: "climate_anomalies"             # Climate anomalies
  biomass_diff_dir: "biomass_differences"      # Biomass change maps
  training_dataset: "ml_training_dataset.csv"  # Final ML dataset
  clustered_dataset: "ml_dataset_with_clusters.csv"
```

#### Time Periods
```yaml
time_periods:
  reference:
    start_year: 1981
    end_year: 2010
    rolling: true                             # Sep-Aug rolling years
  analysis:
    start_year: 2017
    end_year: 2024
    rolling: true
```

#### Machine Learning
```yaml
optimization:
  n_trials: 1000                             # Trials per optimization run
  n_runs: 10                                 # Independent optimization runs
  cv_strategy:
    method: "spatial"                         # Spatial cross-validation
    test_blocks: 2                            # Spatial blocks for testing
  features:
    exclude_bio_vars: ['bio8', 'bio9', 'bio18', 'bio19']
    standardize: true
    correlation_threshold: 0.95
```

#### SHAP Analysis (New!)
```yaml
shap_analysis:
  paths:
    models_dir: "results/climate_biomass_analysis/models"
    clustered_dataset: "results/climate_biomass_analysis/ml_dataset_with_clusters.csv"
    shap_analysis_dir: "results/climate_biomass_analysis/shap_analysis"
  
  model_filtering:
    r2_threshold: 0.2                         # Minimum R² for model inclusion
    r2_threshold_interactions: 0.2            # R² threshold for interaction analysis
  
  analysis:
    shap_max_samples: 1000                    # Max samples for SHAP calculation
    perm_max_samples: 5000                    # Max samples for permutation importance
    pdp_n_top_features: 6                     # Top features for PDP analysis
    interaction_feature_1: "bio12"            # First interaction feature
    interaction_feature_2: "bio12_3yr"        # Second interaction feature
```

### Visualization Configuration (`visualization_config.yaml`)

Separate configuration for visualization settings including plot styling, colors, DPI, and output formats.

## Pipeline Workflow

### Stage 1: Climate Processing
Convert raw climate data (GRIB/NetCDF) to harmonized GeoTIFF rasters with proper georeferencing and coordinate system alignment.

### Stage 2: Bioclimatic Variables
Calculate standard bioclimatic variables (bio1-bio19) for reference period and compute anomalies for analysis period using rolling ecological years.

### Stage 3: Biomass Integration
Integrate biomass change maps with climate anomalies, performing spatial resampling and creating the final machine learning dataset.

### Stage 4: Spatial Analysis
Analyze spatial autocorrelation patterns and create spatial clusters for proper cross-validation accounting for spatial dependencies.

### Stage 5: ML Optimization
Run Bayesian optimization with spatial cross-validation to select optimal climate predictors and hyperparameters for biomass prediction.

### Stage 6: SHAP Analysis
Perform comprehensive model interpretability analysis including:
- Feature selection frequency analysis
- SHAP importance calculation
- Permutation importance assessment
- Partial dependence plots with LOWESS smoothing
- 2D interaction analysis

## Outputs

### Optimization Results
```
optimization_results/
├── individual_run_results.pkl         # Detailed results from all runs
├── optimization_summary.pkl           # Aggregated analysis
├── effective_config.json              # Configuration used
└── feature_importance_analysis.png    # Visualization plots
```

### SHAP Analysis Results (New!)
```
shap_analysis/
├── feature_frequencies_df.csv         # Feature selection frequencies
├── avg_shap_importance.pkl            # Average SHAP importance values
├── avg_permutation_importance.pkl     # Average permutation importance
├── pdp_data.pkl                       # Raw PDP data
├── pdp_lowess_data.pkl                # LOWESS smoothed PDP curves
├── interaction_results.pkl            # 2D interaction analysis
├── analysis_summary.json              # Analysis metadata
└── effective_shap_config.json         # Configuration used
```

### Publication Figures
```
figures/
├── importance_barplots.png            # Feature importance comparison
├── r2_histogram.png                   # Model performance distribution
├── importance_scatter.png             # SHAP vs permutation scatter
├── pdp_panel.png                      # Partial dependence plots
└── main_figure.png                    # Main manuscript figure
```

## Features

### Climate Data Processing
- **Multi-format support**: GRIB, NetCDF, GeoTIFF input formats
- **Coordinate transformations**: Automatic CRS detection and reprojection
- **Grid harmonization**: Align multiple climate datasets to common grid
- **Quality validation**: Comprehensive output validation

### Bioclimatic Variables
- **Complete Bio1-Bio19 suite**: All standard bioclimatic variables
- **Rolling year support**: Sep-Aug ecological years for Mediterranean climate
- **Temporal flexibility**: Configurable reference and analysis periods
- **Anomaly calculation**: Climate departures from long-term means

### Spatial Analysis
- **Memory-efficient semivariograms**: Handle large raster datasets
- **Autocorrelation range estimation**: Knee detection and 95% sill methods
- **Optimal clustering**: Elbow method for cluster number selection
- **Validation metrics**: Cluster separation analysis

### Machine Learning
- **Bayesian optimization**: Efficient hyperparameter search with TPE sampler
- **Spatial cross-validation**: Account for spatial autocorrelation
- **Multiple independent runs**: Robust feature selection analysis
- **Early stopping**: Adaptive termination to prevent overfitting
- **Feature analysis**: Selection frequency and importance ranking

### SHAP Analysis (New!)
- **Comprehensive interpretability**: Multiple importance metrics
- **Partial dependence plots**: 1D effects with LOWESS smoothing
- **Interaction analysis**: 2D feature interactions with heatmaps
- **Model ensemble analysis**: Results across multiple optimization runs
- **Publication-ready figures**: Automated figure generation

### Quality Control
- **Input validation**: Check data integrity and spatial alignment
- **Progress monitoring**: Detailed logging and progress tracking
- **Error recovery**: Continue processing despite individual failures
- **Checkpointing**: Resume interrupted pipelines

## Advanced Usage

### Custom SHAP Analysis
```python
from climate_biomass_analysis.core.shap_analysis import ShapAnalyzer

analyzer = ShapAnalyzer()

# Override analysis parameters
analyzer.shap_config['analysis']['shap_max_samples'] = 2000
analyzer.shap_config['analysis']['pdp_n_top_features'] = 8

# Run specific analysis components
freq_df = analyzer.calculate_feature_frequencies(models)
shap_importance = analyzer.calculate_shap_importance(models, dataset)
pdp_data, pdp_lowess = analyzer.calculate_pdp_with_lowess(models, dataset)

# Run full analysis
results = analyzer.run_comprehensive_shap_analysis()
```

### Custom Visualization
```python
from climate_biomass_analysis.visualization_pipeline import (
    load_shap_results, plot_importance_barplots, plot_main_figure
)

# Load results
results = load_shap_results("results/shap_analysis")

# Generate custom figures
config = load_visualization_config("custom_viz_config.yaml")
plot_importance_barplots(results, config, variable_mapping)
```

### Custom ML Optimization
```python
from climate_biomass_analysis.core.optimization_pipeline import OptimizationPipeline

optimizer = OptimizationPipeline()

# Override hyperparameter ranges
optimizer.hyperparams['n_estimators'] = [100, 1000]
optimizer.hyperparams['max_depth'] = [5, 15]

# Custom feature exclusion
optimizer.features_config['exclude_bio_vars'] = ['bio8', 'bio9']

results = optimizer.run_optimization_pipeline()
```

## Dependencies

Core packages (managed via conda environment):
- `xarray` + `rioxarray` for multidimensional arrays and raster I/O
- `rasterio` for geospatial raster processing
- `geopandas` for vector data handling
- `pandas` for tabular data processing
- `numpy` + `scipy` for numerical computing
- `scikit-learn` for machine learning utilities
- `xgboost` for gradient boosting models
- `optuna` for Bayesian optimization
- `shap` for model interpretability (NEW!)
- `statsmodels` for LOWESS smoothing (NEW!)
- `matplotlib` + `seaborn` for visualization
- `earthkit-data` for GRIB file handling
- `kneed` for knee point detection

## Integration

This component integrates with:
- **Canopy Height Model**: Uses height predictions as input for biomass estimation
- **Biomass Estimation**: Analyzes climate drivers of biomass changes
- **Data Preprocessing**: Uses processed climate and biomass data
- **Visualization Pipeline**: Generates publication-quality figures

## Troubleshooting

### Common Issues

#### SHAP Analysis Fails
```bash
# Check if optimization results exist
ls results/climate_biomass_analysis/models/

# Validate model files
python scripts/run_shap_analysis.py --validate-inputs --dry-run

# Reduce memory requirements
python scripts/run_shap_analysis.py --shap-max-samples 500 --pdp-max-samples 1000
```

#### Missing Dependencies
```bash
# Install additional packages for SHAP analysis
pip install shap statsmodels

# Update conda environment
conda env update -f ../environments/climate_analysis.yml
```

#### Figure Generation Issues
```bash
# Check SHAP results exist
python generate_paper_figures.py --dry-run

# Generate specific figures only
python generate_paper_figures.py --figures importance scatter
```

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline  
**Version**: 1.1.0 (Added SHAP Analysis)