# Biomass Estimation Pipeline

Comprehensive forest biomass estimation pipeline with integrated allometry fitting and Monte Carlo uncertainty quantification.

## Overview

- **Allometry Fitting**: Train height-AGB and BGB-AGB relationships from NFI data + height maps
- **Biomass Estimation**: Apply allometries to height maps with Monte Carlo uncertainty  
- **Post-Processing**: Mask agricultural areas and merge forest types into country-wide maps

## Quick Start

```bash
# 1. Install dependencies
conda env create -f biomass_dask.yml
conda activate biomass-estimation-dask

# 2. Configure paths
cp config.yaml.example config.yaml
cp config_fitting.yaml.example config_fitting.yaml
# Edit both configs with your paths

# 3. Full pipeline
python fit_allometries.py           # Fit allometries (optional if using pre-fitted)
python biomass_estimation.py        # Generate biomass maps
python mask_annual_cropland.py      # Remove agricultural areas  
python merge_foresttypes_biomass.py # Create country-wide maps
```

## Pipeline Architecture

```
NFI Data + Height Maps → Fit Allometries → Apply to Height Maps → Post-Process
        ↓                      ↓                    ↓               ↓
  fit_allometries.py    biomass_estimation.py   mask_annual_    merge_foresttypes_
  (config_fitting.yaml)     (config.yaml)       cropland.py      biomass.py
```

## Configuration

### Allometry Fitting (`config_fitting.yaml`)
```yaml
data:
  nfi_data_dir: '/path/to/nfi/processed/'         # NFI shapefiles by year
  height_maps_dir: '/path/to/height_maps/'        # Country-wide height maps
  target_years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

height_agb:
  quantiles: [0.15, 0.85]    # Confidence bounds for allometry
  min_samples: 10            # Min samples per forest type

bgb_agb:
  min_samples: 10            # Min samples for ratio calculation
  percentiles: [5, 50, 95]   # BGB ratio percentiles
```

### Biomass Estimation (`config.yaml`)
```yaml
data:
  input_data_dir: '/path/to/height_tiles/'       # Tiled height maps
  masks_dir: '/path/to/forest_type_masks/'       # Forest type masks
  allometries_dir: '/path/to/fitted_allometries.csv'
  bgb_coeffs_dir: '/path/to/fitted_ratios.csv'

monte_carlo:
  num_samples: 250           # MC samples per pixel

processing:
  annual_crop_values: [12, 13, 14]  # Corine LC codes to mask
```

## Input Data Structure

### For Allometry Fitting
```
nfi_data_dir/
├── nfi4_2020_biomass.shp    # NFI plots: AGB, BGB, BGB_Ratio, ForestType, coords
├── nfi4_2021_biomass.shp
└── nfi4_YYYY_biomass.shp

height_maps_dir/
├── canopy_height_2020_100m.tif    # Country-wide height maps
├── canopy_height_2021_100m.tif
└── canopy_height_YYYY_100m.tif
```

### For Biomass Estimation
```
input_data_dir/
├── canopy_height_2020_100m_tile1.tif    # Tiled height maps
├── canopy_height_2020_100m_tile2.tif
└── ...

masks_dir/
├── canopy_height_2020_100m_tile1_code12.tif  # Forest type masks
├── canopy_height_2020_100m_tile1_code23.tif
└── ...
```

## Output Structure

### Fitted Allometries
```
H_AGB_Allometries_Tiers_ModelCalibrated_Quantiles_15-85_OnlyPowerLaw.csv
BGBRatios_Tiers.csv
```

### Biomass Maps
```
biomass_no_LC_masking/
├── AGBD_MC_100m/
│   ├── AGBD_S2_mean_2020_100m_code12.tif     # Per forest type
│   └── AGBD_S2_uncertainty_2020_100m_code12.tif
├── BGBD_MC_100m/
└── TBD_MC_100m/

with_annual_crop_mask/              # After masking agricultural areas
└── (same structure, masked)

biomass_maps_merged/                # Final country-wide maps
├── mean/
│   ├── AGBD_S2_mean_2020_100m_merged.tif
│   └── BGBD_S2_mean_2020_100m_merged.tif
└── uncertainty/
```

## Scripts & Components

| Script | Purpose | Config |
|--------|---------|---------|
| `fit_allometries.py` | Train height-AGB & BGB-AGB relationships | `config_fitting.yaml` |
| `biomass_estimation.py` | Main biomass estimation pipeline | `config.yaml` |
| `mask_annual_cropland.py` | Remove agricultural areas | `config.yaml` |
| `merge_foresttypes_biomass.py` | Create country-wide maps | `config.yaml` |

### Core Components
- **Allometry Fitting** (`fit_allometries.py`): Samples height maps at NFI locations, fits hierarchical relationships
- **Fitting Utilities** (`fitting_utils.py`): Data validation, outlier removal, hierarchy processing
- **Allometry Management** (`allometry.py`): Forest type hierarchy and parameter retrieval
- **Monte Carlo Engine** (`monte_carlo.py`): Uncertainty quantification through statistical sampling  
- **I/O Operations** (`io_utils.py`): Raster reading/writing with dask chunking
- **Distributed Computing** (`dask_utils.py`): Cluster management for large datasets

## Allometry Fitting Features

### Dynamic Training Data Creation
- Automatically samples country-wide height maps at NFI plot coordinates
- Processes all available years (2017-2024) where both NFI and height data exist
- No preprocessed training files needed

### Hierarchical Processing
- Fits relationships at multiple levels: General → Clade → Family → Genus → ForestType
- Independent processing: Different forest types can have different data availability
- Fallback system ensures all forest types get appropriate allometric relationships

### Dual Allometry Types
- **Height→AGB**: Quantile regression (15th, 50th, 85th percentiles) for confidence intervals
- **BGB→AGB Ratios**: Statistical percentiles (5th, mean, 95th) using existing NFI ratio data

### Quality Control
- Outlier removal using Elliptic Envelope method
- Statistical validation and quality filters
- Comprehensive logging and verification

### Compatible Output
- CSV formats designed for seamless integration with existing biomass estimation pipeline
- Maintains exact column structure expected by downstream processing

## Advanced Features

### Independent Processing
Forest types can have different data coverage:
- Type A: Height-AGB allometry fitted ✓, BGB ratios calculated ✗  
- Type B: Height-AGB allometry fitted ✗, BGB ratios calculated ✓
- Type C: Both fitted ✓✓
- Type D: Neither fitted ✗✗

### Monte Carlo Uncertainty
- 250 samples per pixel for robust uncertainty quantification
- Normal distribution assumptions with percentile-based confidence intervals
- Efficient vectorized processing with dask

### Hierarchical Fallback
- Specific forest type → Genus → Family → Clade → General
- Ensures every pixel gets appropriate biomass estimates
- Transparent logging of fallback decisions

## Requirements

- Python 3.8+
- Key packages: `dask`, `xarray`, `rasterio`, `geopandas`, `scikit-learn`
- See `biomass_dask.yml` for complete unified environment

