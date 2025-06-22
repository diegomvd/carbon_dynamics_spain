# Iberian Carbon Assessment Pipeline - Orchestration System

A simple, recipe-based orchestration system for reproducing research results from the Iberian Carbon Assessment project.

## Overview

This orchestration system provides **four simple recipes** that reproduce different stages of the research pipeline:

1. **`reproduce_data_preparation.py`** - Forest inventory + Sentinel-2 mosaics
2. **`reproduce_height_modeling.py`** - Canopy height predictions + post-processing  
3. **`reproduce_biomass_estimation.py`** - Biomass estimation from height maps
4. **`reproduce_analysis.py`** - Complete analysis (biomass patterns + climate relationships)

Each recipe is a **standalone script** that handles prerequisites, calls component orchestrators, and validates outputs - perfect for reviewers and collaborators to reproduce specific research steps.

## Quick Start

```bash
# 1. Set up data structure
mkdir data
# Place your data in data/ following the structure below

# 2. Run recipes in order
python reproduce_data_preparation.py
python reproduce_height_modeling.py  
python reproduce_biomass_estimation.py
python reproduce_analysis.py

# 3. Check results in data/results/
```

## Data Structure

The orchestration system uses a **centralized data structure** under the `data/` directory:

```
data/
├── raw/                           # Input data
│   ├── forest_inventory/          # NFI4 databases, MFE shapefiles
│   │   ├── IFN_4_SP/             # NFI4 Access databases (.accdb)
│   │   ├── MFESpain/             # Forest map shapefiles
│   │   ├── GlobalWoodDensityDatabase.xls
│   │   ├── Forest_Types_Tiers.csv
│   │   └── ...
│   ├── climate/                   # Climate GRIB files
│   ├── pnoa_lidar/               # PNOA LiDAR data (optional)
│   │   └── PNOA2_LIDAR_VEGETATION/
│   ├── pnoa_coverage/            # PNOA coverage polygons (optional)
│   │   ├── Huso_29/
│   │   ├── Huso_30/
│   │   └── Huso_31/
│   └── reference_data/            # Boundary shapefiles, etc.
│       ├── SpainPolygon/
│       └── corine_land_cover/
├── processed/                     # Intermediate outputs
│   ├── sentinel2_mosaics/         # Summer mosaics
│   ├── height_predictions/        # Canopy height maps
│   ├── biomass_maps/             # Biomass estimation outputs
│   ├── climate_variables/        # Bioclimatic variables
│   ├── forest_type_masks/        # Processing masks
│   └── training_data_sentinel2_pnoa/  # ALS training data (optional)
├── results/                       # Final analysis outputs
│   ├── figures/                  # All plots and visualizations
│   ├── tables/                   # Summary statistics
│   ├── analysis_outputs/         # Analysis results (CSV, shapefiles)
│   └── ml_outputs/              # ML models and predictions
└── models/                       # Model checkpoints
    └── height_model_checkpoint.pkl
```

## Recipe Details

### Recipe 1: Data Preparation

**Purpose**: Prepare foundational datasets (forest inventory + Sentinel-2 mosaics + ALS training data)

```bash
# Basic usage
python reproduce_data_preparation.py

# Options
python reproduce_data_preparation.py --years 2020 2021 2022  # Specific years
python reproduce_data_preparation.py --skip-forest-inventory # Skip NFI processing
python reproduce_data_preparation.py --skip-als-pnoa        # Skip ALS processing
python reproduce_data_preparation.py --bbox 40.0 -10.0 44.0 -6.0  # Custom extent
```

**Prerequisites**:
- NFI4 database files in `data/raw/forest_inventory/IFN_4_SP/`
- MFE forest maps in `data/raw/forest_inventory/MFESpain/`
- Spain boundary in `data/raw/reference_data/SpainPolygon/`
- Internet connectivity for Sentinel-2 STAC catalog
- PNOA LiDAR data in `data/raw/pnoa_lidar/` (optional, for training data)

**Outputs**:
- Processed forest inventory shapefiles
- Sentinel-2 summer mosaics
- ALS PNOA training data (if available)
- Ready for height modeling

### Recipe 2: Height Modeling

**Purpose**: Generate canopy height predictions from Sentinel-2 mosaics

```bash
# Basic usage
python reproduce_height_modeling.py

# Options  
python reproduce_height_modeling.py --checkpoint /path/to/model.ckpt  # Custom model
python reproduce_height_modeling.py --years 2020 2021 2022           # Specific years
python reproduce_height_modeling.py --prediction-only                # Skip training
```

**Prerequisites**:
- Sentinel-2 mosaics (from Recipe 1)
- Pre-trained model checkpoint in `data/models/`
- PyTorch environment (GPU recommended)

**Outputs**:
- Height prediction patches
- Merged and sanitized height maps
- Final country-wide height mosaics
- Ready for biomass estimation

### Recipe 3: Biomass Estimation

**Purpose**: Estimate biomass from canopy height predictions

```bash
# Basic usage
python reproduce_biomass_estimation.py

# Options
python reproduce_biomass_estimation.py --skip-allometry      # Use existing allometries
python reproduce_biomass_estimation.py --years 2020 2021 2022  # Specific years
python reproduce_biomass_estimation.py --continue-on-error   # Continue despite failures
```

**Prerequisites**:
- Canopy height predictions (from Recipe 2)
- Forest inventory data with allometric relationships
- Corine land cover data for masking

**Outputs**:
- Forest type specific biomass maps (AGBD, BGBD, TBD)
- Annual cropland masked versions
- Country-wide merged biomass maps
- Monte Carlo uncertainty estimates
- Ready for analysis

### Recipe 4: Analysis

**Purpose**: Complete analysis of biomass patterns and climate relationships

```bash
# Basic usage - runs all analyses
python reproduce_analysis.py

# Options
python reproduce_analysis.py --analysis-only biomass    # Just biomass analysis
python reproduce_analysis.py --analysis-only climate    # Just climate analysis  
python reproduce_analysis.py --skip-shap                # Skip SHAP analysis
python reproduce_analysis.py --years 2020 2021 2022     # Specific years
```

**Prerequisites**:
- Biomass maps (from Recipe 3)
- Climate variables for climate-biomass analysis
- Reference data (boundaries, forest types)

**Outputs**:
- Biomass trend analysis
- Forest type and landcover aggregations
- Interannual difference maps
- Carbon flux calculations
- Climate-biomass ML models
- SHAP interpretability analysis
- All figures and tables from the paper

## Advanced Usage

### Custom Data Location

```bash
# Use custom data directory
python reproduce_biomass_estimation.py --data-root /path/to/my/data
```

### Validation Only

```bash
# Check prerequisites without running
python reproduce_height_modeling.py --validate-only
```

### Error Handling

```bash
# Continue processing despite stage failures
python reproduce_analysis.py --continue-on-error
```

### Logging Control

```bash
# Debug logging
python reproduce_biomass_estimation.py --log-level DEBUG

# Quiet mode (errors only)
python reproduce_analysis.py --quiet
```

## Integration with Existing Components

The orchestration system **does not change** existing component architecture. It simply:

1. **Uses centralized data paths** via `shared_utils/data_paths.py`
2. **Calls existing component orchestrators** with path overrides
3. **Provides user-friendly interfaces** for reproduction

Original component scripts remain fully functional and can still be run independently.

## Centralized Path Management

The `shared_utils/data_paths.py` module provides:

- **Standardized data structure** across all components
- **Path builder functions** for common file types
- **Config override generation** for component integration
- **Directory creation utilities**

Example usage:
```python
from shared_utils.data_paths import CentralDataPaths

# Initialize centralized paths
data_paths = CentralDataPaths("data")

# Get standardized paths
biomass_path = data_paths.get_biomass_path("TBD", 2020, "mean")
height_path = data_paths.get_height_prediction_path(2020)

# Create directory structure
data_paths.create_directories()

# Get config overrides for components
overrides = data_paths.get_component_config_overrides('biomass_model')
```

## Component Orchestrators

Each component has internal orchestrators that the recipes call:

- **`biomass_model/scripts/run_full_pipeline.py`** - Complete biomass pipeline
- **`biomass_analysis/scripts/run_full_analysis.py`** - Biomass pattern analysis  
- **`climate_biomass_analysis/scripts/run_full_pipeline.py`** - Climate-biomass analysis
- **`sentinel2_processing/scripts/run_postprocessing.py`** - Sentinel-2 processing
- **`forest_inventory/scripts/run_nfi_processing.py`** - Forest inventory processing

## Troubleshooting

### Common Issues

**Prerequisites validation fails**:
- Check data directory structure matches expected layout
- Verify all required input files are present
- Ensure file permissions allow reading

**Recipe execution fails**:
- Run with `--validate-only` first to check prerequisites
- Use `--log-level DEBUG` for detailed error information
- Check individual component documentation for specific requirements

**Output validation fails**:
- Verify sufficient disk space for outputs
- Check write permissions in output directories
- Review log files for processing errors

**Memory/performance issues**:
- Close other applications to free memory
- Use SSD storage for better I/O performance
- Consider processing subset of years for testing

### Getting Help

1. **Run validation first**: Use `--validate-only` to check prerequisites
2. **Check logs**: All recipes provide detailed logging with `--log-level DEBUG`
3. **Component docs**: Refer to individual component README files for specific issues
4. **Test with subset**: Use `--years` to process smaller datasets for testing

## System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **Memory**: 16GB RAM (32GB+ recommended)
- **Storage**: 100GB+ available space
- **Network**: High-bandwidth connection for Sentinel-2 data

### Recommended Setup
- **Environment**: Conda with component-specific environments
- **Hardware**: Multi-core CPU, GPU for deep learning
- **Storage**: SSD for temporary and output directories

### Software Dependencies
- Component-specific conda environments (see `environments/` directory)
- System dependencies: `mdb-tools` for Access database processing
- Optional: CUDA for GPU acceleration in height modeling

## Design Philosophy

This orchestration system follows key principles:

- **Simplicity**: Four clear entry points, no complex configuration
- **Autonomy**: Components remain independent and unchanged
- **Robustness**: Recipes handle errors gracefully and validate prerequisites  
- **Flexibility**: Users can run individual steps or customize execution
- **Reproducibility**: Consistent data structure and clear dependencies

The goal is to make it **easy for reviewers and collaborators** to reproduce research results without needing to understand the full pipeline complexity.

---

**Need help?** Check component-specific documentation in each `[component]/README.md` file for detailed usage instructions.