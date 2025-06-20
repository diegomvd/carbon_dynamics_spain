# Climate-Biomass Analysis Component

Comprehensive climate-biomass relationship analysis for the Iberian Carbon Assessment Pipeline, including climate data processing, bioclimatic variables calculation, biomass-climate integration, spatial analysis, and machine learning optimization.

## Overview

This component analyzes the relationships between climate variables and biomass changes using advanced spatial modeling techniques. The pipeline integrates:

- **Climate data processing** - GRIB to GeoTIFF conversion with proper georeferencing
- **Bioclimatic variables calculation** - Bio1-Bio19 variables and anomaly analysis  
- **Biomass-climate integration** - Spatial alignment and data harmonization
- **Spatial autocorrelation analysis** - Semivariogram calculation and clustering
- **Machine learning optimization** - Bayesian optimization with spatial cross-validation

## Structure

```
climate_biomass_analysis/
├── config.yaml                        # Component configuration
├── __init__.py                        # Package initialization
├── README.md                          # This file
├── core/                              # Core processing modules
│   ├── __init__.py
│   ├── climate_raster_processing.py   # GRIB to GeoTIFF conversion
│   ├── bioclim_calculation.py         # Bioclimatic variables calculation
│   ├── biomass_integration.py         # Biomass-climate data integration
│   ├── spatial_analysis.py            # Spatial autocorrelation & clustering
│   └── optimization_pipeline.py       # ML optimization & feature selection
└── scripts/                           # Executable entry points
    ├── __init__.py
    ├── run_climate_processing.py       # Climate data processing
    ├── run_bioclim_calculation.py      # Bioclimatic variables
    ├── run_biomass_integration.py      # Data integration
    ├── run_spatial_analysis.py         # Spatial analysis
    ├── run_optimization.py             # ML optimization
    └── run_full_pipeline.py            # Complete pipeline orchestrator
```

## Usage

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

### Individual Processing Steps

#### 1. Climate Data Processing
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

#### 2. Bioclimatic Variables Calculation
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

#### 3. Biomass-Climate Integration
```bash
# Integrate biomass changes with climate anomalies
python scripts/run_biomass_integration.py

# Custom input directories
python scripts/run_biomass_integration.py \
  --biomass-dir ./biomass_differences \
  --anomaly-dir ./climate_anomalies \
  --output-dataset ./training_data.csv
```

#### 4. Spatial Analysis
```bash
# Perform spatial autocorrelation analysis and clustering
python scripts/run_spatial_analysis.py

# Custom dataset and clustering parameters
python scripts/run_spatial_analysis.py \
  --input-dataset ./training_data.csv \
  --n-clusters 25 \
  --autocorr-threshold 50
```

#### 5. Machine Learning Optimization
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

## Configuration

Key configuration sections in `config.yaml`:

### Data Paths
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

### Time Periods
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

### Machine Learning
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

## Pipeline Workflow

### Stage 1: Climate Processing
- Convert GRIB files to properly georeferenced GeoTIFFs
- Reproject to target CRS (EPSG:25830 for Spain)
- Clip to Spain boundary
- Harmonize raster grids

### Stage 2: Bioclimatic Calculation
- Calculate Bio1-Bio19 variables from monthly temperature/precipitation
- Compute reference period climatology (1981-2010)
- Calculate climate anomalies for analysis period (2017-2024)
- Support both calendar and rolling (Sep-Aug) years

### Stage 3: Biomass Integration
- Resample biomass difference maps to climate resolution
- Harmonize spatial grids between biomass and climate data
- Extract data points where both datasets have valid values
- Create ML training dataset with coordinates and features

### Stage 4: Spatial Analysis
- Calculate spatial autocorrelation using semivariograms
- Estimate autocorrelation range for spatial independence
- Create spatial clusters for cross-validation
- Validate cluster separation exceeds autocorrelation threshold

### Stage 5: ML Optimization
- Multiple independent Bayesian optimization runs
- Spatial cross-validation to prevent overfitting
- XGBoost hyperparameter optimization
- Feature importance analysis and selection frequency

## Outputs

### Bioclimatic Variables
```
bioclimatic_variables/
├── bio1_1981_2010.tif                 # Annual mean temperature
├── bio12_1981_2010.tif                # Annual precipitation
└── ...                                # All bio1-bio19 variables
```

### Climate Anomalies
```
climate_anomalies/
├── anomalies_2017/
│   ├── bio1_anomaly_2017.tif
│   └── bio12_anomaly_2017.tif
├── anomalies_2018/
└── ...                                # One directory per year
```

### ML Training Data
```
ml_training_dataset.csv                # Integrated biomass-climate data
ml_dataset_with_clusters.csv           # With spatial cluster assignments
```

### Optimization Results
```
optimization_results/
├── individual_run_results.pkl         # Detailed results from all runs
├── optimization_summary.pkl           # Aggregated analysis
├── effective_config.json              # Configuration used
└── feature_importance_analysis.png    # Visualization plots
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

### Quality Control
- **Input validation**: Check data integrity and spatial alignment
- **Progress monitoring**: Detailed logging and progress tracking
- **Error recovery**: Continue processing despite individual failures
- **Checkpointing**: Resume interrupted pipelines

## Advanced Usage

### Custom Bioclimatic Variables
```python
from climate_biomass_analysis.core.bioclim_calculation import BioclimCalculator

calculator = BioclimCalculator()
calculator.bio_variables = ['bio1', 'bio12', 'bio15']  # Subset only

results = calculator.calculate_bioclim_variables(
    temp_files, precip_files, output_dir,
    start_year=2000, end_year=2020, rolling=True
)
```

### Advanced Spatial Clustering
```python
from climate_biomass_analysis.core.spatial_analysis import SpatialAnalyzer

analyzer = SpatialAnalyzer()

# Custom autocorrelation analysis
autocorr_results = analyzer.analyze_spatial_autocorrelation(
    raster_path="./biomass_change.tif",
    output_dir="./spatial_analysis"
)

# Create clusters with custom parameters
clustered_df = analyzer.create_spatial_clusters(
    dataset_path="./training_data.csv",
    output_path="./clustered_data.csv", 
    autocorr_range_km=75.0
)
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
- `earthkit-data` for GRIB file handling
- `kneed` for knee point detection

## Integration

This component integrates with:
- **Biomass Model**: Uses biomass difference maps as target variables
- **Data Preprocessing**: May use preprocessed climate data
- **Canopy Height Model**: Can incorporate height-derived variables
- **Analysis Output**: Provides ML models for biomass prediction

## Performance Optimization

### Memory Management
- Strategic raster sampling for large datasets
- Chunk-based processing for memory efficiency
- Automatic cleanup of intermediate files
- Memory-efficient semivariogram computation

### Parallel Processing
- Multi-threaded XGBoost training
- Parallel optimization runs
- Vectorized raster operations
- Dask integration for large-scale processing

### Storage Optimization
- Compressed GeoTIFF outputs (LZW compression)
- Efficient file formats (Parquet for tabular data)
- Overview pyramids for visualization
- Smart caching of intermediate results

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce sample fractions in spatial analysis
- **Slow raster processing**: Check for proper chunking and compression
- **CRS misalignment**: Verify coordinate systems across datasets
- **Missing climate data**: Check file patterns and date ranges

### Performance Tips
- **Use SSD storage** for faster I/O during raster processing
- **Increase memory** allocation for large-scale optimization
- **Enable compression** to reduce storage requirements
- **Monitor progress** with detailed logging

### Data Quality
- **Check spatial alignment** between biomass and climate data
- **Validate time periods** match between datasets
- **Verify cluster separation** exceeds autocorrelation range
- **Review optimization convergence** across multiple runs

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline  
**Version**: 1.0.0