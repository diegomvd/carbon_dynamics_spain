# Biomass Analysis Component

Comprehensive biomass analysis pipeline for the Iberian Carbon Assessment Pipeline, including time series analysis, aggregation studies, and carbon flux calculations.

## Overview

This component provides analysis capabilities for biomass model outputs:

- **Country-level analysis** - Time series with Monte Carlo uncertainty quantification
- **Aggregation analysis** - By forest type, landcover, and height ranges
- **Interannual analysis** - Difference mapping and transition distributions  
- **Carbon flux analysis** - Flux calculations from biomass changes

## Structure

```
biomass_analysis/
├── config.yaml                    # Component configuration
├── __init__.py                    # Package initialization
├── README.md                      # This file
├── core/                          # Core processing modules
│   ├── __init__.py
│   ├── monte_carlo_analysis.py    # Monte Carlo uncertainty algorithms
│   ├── aggregation_analysis.py    # Forest type/landcover/height aggregations
│   ├── interannual_analysis.py    # Difference maps & transitions
│   └── carbon_flux_analysis.py    # Carbon flux calculations
└── scripts/                       # Executable entry points
    ├── __init__.py
    ├── run_country_trend.py        # Country-level analysis
    ├── run_forest_type_trend.py    # Forest type analysis
    ├── run_landcover_trend.py      # Landcover analysis
    ├── run_height_bin_trend.py     # Height bin analysis
    ├── run_interannual_differences.py # Difference mapping
    ├── run_transition_analysis.py  # Transition distributions
    ├── run_carbon_fluxes.py        # Carbon flux analysis
    └── run_full_analysis.py        # Pipeline orchestrator
```

## Usage

### Environment Setup

```bash
# Create conda environment
conda env create -f ../environments/biomass_analysis.yml
conda activate biomass_analysis

# Install repository for absolute imports
pip install -e ../..
```

### Complete Pipeline

```bash
# Run full analysis pipeline
python scripts/run_full_analysis.py

# Run specific analyses only
python scripts/run_full_analysis.py --stages country_analysis forest_type_analysis

# Continue on errors and use custom config
python scripts/run_full_analysis.py --continue-on-error --config custom_config.yaml
```

### Individual Analyses

#### Country-level Time Series
```bash
# Country-level biomass analysis with Monte Carlo uncertainty
python scripts/run_country_trend.py

# Specific biomass types and years
python scripts/run_country_trend.py --biomass-types TBD AGBD --years 2020 2021 2022

# Enable parallel processing
python scripts/run_country_trend.py --parallel --workers 4
```

#### Aggregation Analyses
```bash
# Forest type hierarchical analysis
python scripts/run_forest_type_trend.py

# Landcover group analysis  
python scripts/run_landcover_trend.py

# Height bin analysis (creates masks if needed)
python scripts/run_height_bin_trend.py

# Skip mask creation and use existing masks
python scripts/run_height_bin_trend.py --skip-mask-creation
```

#### Interannual Analyses
```bash
# Create interannual difference maps
python scripts/run_interannual_differences.py

# Transition distribution analysis
python scripts/run_transition_analysis.py

# Save raw transition data
python scripts/run_transition_analysis.py --save-raw-data
```

#### Carbon Flux Analysis
```bash
# Calculate carbon fluxes from Monte Carlo samples
python scripts/run_carbon_fluxes.py

# Skip diagnostic plots
python scripts/run_carbon_fluxes.py --no-plots

# Use specific Monte Carlo samples file
python scripts/run_carbon_fluxes.py --mc-samples /path/to/samples.npz
```

## Configuration

Key configuration sections in `config.yaml`:

### Data Paths
```yaml
data:
  base_dir: "/path/to/data"
  biomass_maps_dir: "biomass_maps_merged_with_annual_crop_mask"
  forest_type_masks_dir: "forest_type_masks/100m"
  corine_raster_path: "corine_land_cover/U2018_CLC2018_V2020_20u1.tif"
```

### Analysis Parameters
```yaml
analysis:
  target_years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
  pixel_area_ha: 1
```

### Monte Carlo Settings
```yaml
monte_carlo:
  num_samples: 1000
  spatial_block_size: 800  # 80km spatial blocks
  parallel_processing:
    enabled: false
    num_workers: 4
```

## Outputs

### Country Analysis
- `country_biomass_timeseries_YYYYMMDD_HHMMSS.csv` - Time series statistics
- `country_biomass_mc_samples_YYYYMMDD_HHMMSS.npz` - Monte Carlo samples

### Aggregation Analysis
- `biomass_by_forest_type_year.csv` - Forest type level results
- `biomass_by_genus_year.csv` - Genus level aggregation
- `biomass_by_clade_year.csv` - Clade level aggregation
- `biomass_by_landcover_year.csv` - Landcover group results
- `biomass_by_height_year.csv` - Height bin results

### Interannual Analysis
- `TBD_S2_raw_change_*_100m.tif` - Raw difference maps (Mg/ha)
- `TBD_S2_relative_change_symmetric_*_100m.tif` - Relative difference maps (%)
- `biomass_transition_statistics_YYYYMMDD_HHMMSS.csv` - Transition statistics

### Carbon Flux Analysis
- `spain_carbon_flux_YYYYMMDD_HHMMSS.csv` - Flux statistics
- `carbon_flux_diagnostics_YYYYMMDD_HHMMSS.png` - Diagnostic plots

## Features

### Monte Carlo Uncertainty
- **Spatial correlation** - 80km spatial blocks for realistic uncertainty
- **Parallel processing** - Optional multi-core processing
- **Quality control** - Boundary masking and threshold filtering

### Hierarchical Aggregation
- **Forest types** - Individual types → genus → clade levels
- **Landcover groups** - Urban, agricultural, natural classifications
- **Height bins** - Automatic mask creation from canopy height data

### Interannual Analysis
- **Difference mapping** - Raw and relative symmetric differences
- **Transition analysis** - Comprehensive pixel-level statistics
- **Quality control** - Biomass threshold filtering

### Carbon Flux Analysis
- **Monte Carlo sampling** - Random combinations from uncertainty distributions
- **Statistical analysis** - Mean, confidence intervals, percentiles
- **Diagnostic plots** - Distribution analysis and time series visualization

## Dependencies

Core packages (managed via conda environment):
- `numpy` + `pandas` for data processing
- `rasterio` + `rioxarray` for raster I/O
- `geopandas` for vector data handling
- `matplotlib` + `seaborn` for plotting
- `tqdm` for progress monitoring

## Integration

This component integrates with:
- **Biomass Model**: Uses biomass predictions as input
- **Data Preprocessing**: Uses forest type masks and boundary data
- **Shared Utils**: Uses common logging and configuration utilities

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline
