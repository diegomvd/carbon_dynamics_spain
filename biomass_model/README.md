# Biomass Estimation Component

Comprehensive biomass estimation pipeline for the Iberian Carbon Assessment using allometric relationships and Monte Carlo uncertainty quantification.

## Overview

This component estimates above-ground biomass (AGB), below-ground biomass (BGB), and total biomass for forest areas using:

- **Allometric relationships** fitted from NFI data
- **Canopy height predictions** from Sentinel-2 data
- **Monte Carlo uncertainty quantification** (250 samples per pixel)
- **Forest type specific processing** with hierarchical fallback
- **Land cover masking** to exclude agricultural areas

## Structure

```
biomass_model/
├── config.yaml                    # Component configuration
├── core/                          # Core processing modules
│   ├── biomass_estimation.py      # Main pipeline class
│   ├── allometry.py               # Allometric relationship management
│   ├── monte_carlo.py             # Monte Carlo uncertainty estimation
│   ├── io_utils.py                # Raster I/O utilities
│   └── dask_utils.py              # Distributed computing support
├── scripts/                       # Executable entry points
│   ├── run_biomass_estimation.py  # Main pipeline script
│   ├── run_allometry_fitting.py   # Allometry fitting script
│   ├── run_masking.py             # Annual cropland masking
│   └── run_merging.py             # Forest type merging
└── README.md                      # This file
```

## Usage

### Environment Setup

```bash
# Create conda environment
conda env create -f ../environments/biomass_estimation.yml
conda activate biomass_estimation

# Install repository for absolute imports
pip install -e ../..
```

### Main Pipeline

```bash
# Run full biomass estimation pipeline
python scripts/run_biomass_estimation.py

# Custom configuration
python scripts/run_biomass_estimation.py --config my_config.yaml

# Specific years only
python scripts/run_biomass_estimation.py --years 2020 2021 2022

# Test mode (single forest type)
python scripts/run_biomass_estimation.py --test-mode --year 2020 --forest-type 12
```

### Individual Processing Steps

```bash
# 1. Allometry fitting (if needed)
python scripts/run_allometry_fitting.py

# 2. Biomass estimation (creates forest type specific maps)
python scripts/run_biomass_estimation.py

# 3. Annual cropland masking
python scripts/run_masking.py --input-dir ./biomass_raw --output-dir ./biomass_masked

# 4. Forest type merging (country-wide maps)
python scripts/run_merging.py --input-dir ./biomass_masked --output-dir ./biomass_merged
```

## Configuration

Key configuration sections in `config.yaml`:

- **`data`**: Input/output paths for height data, masks, allometries
- **`processing`**: Height thresholds, target years, file patterns
- **`monte_carlo`**: Number of samples, random seed, distribution type
- **`compute`**: Dask workers, memory limits, chunk sizes
- **`output`**: File formats, compression, naming conventions

## Outputs

### Biomass Maps
- **AGBD**: Above-ground biomass density (Mg/ha)
- **BGBD**: Below-ground biomass density (Mg/ha)
- **TBD**: Total biomass density (Mg/ha)

### Statistical Measures
- **mean**: Mean biomass estimate
- **uncertainty**: Standard deviation across Monte Carlo samples

### File Organization
```
biomass_no_LC_masking/              # Before land cover masking
├── AGBD_MC_100m/
│   ├── AGBD_S2_mean_2020_100m_code12.tif
│   └── AGBD_S2_uncertainty_2020_100m_code12.tif
├── BGBD_MC_100m/
└── TBD_MC_100m/

with_annual_crop_mask/              # After masking agricultural areas
└── (same structure, masked)

biomass_maps_merged/                # Country-wide merged maps
├── AGBD_S2_mean_2020_100m_merged.tif
└── AGBD_S2_uncertainty_2020_100m_merged.tif
```

## Features

### Allometric Relationships
- **Hierarchical fallback**: ForestType → Genus → Family → Clade → General
- **Quantile regression**: 15th, 50th, 85th percentiles for confidence intervals
- **BGB ratios**: Forest type specific below-ground to above-ground ratios

### Monte Carlo Uncertainty
- **250 samples** per pixel for robust uncertainty quantification
- **Normal distribution** sampling of allometric parameters
- **Efficient vectorized** processing with Dask

### Distributed Computing
- **Dask integration** for large-scale processing
- **Automatic resource management** based on system capabilities
- **Progress monitoring** and performance optimization

### Quality Control
- **Input validation** for all data sources
- **Raster alignment** checking
- **Memory management** for large datasets
- **Comprehensive logging** with component-specific loggers

## Dependencies

Core packages (managed via conda environment):
- `dask` + `distributed` for parallel processing
- `rasterio` + `rioxarray` for raster I/O
- `geopandas` for vector data
- `scikit-learn` for statistical processing
- `numpy` + `scipy` for numerical computing

## Integration

This component integrates with:
- **Canopy Height Model**: Uses height predictions as input
- **Data Preprocessing**: Uses forest type masks from preprocessing
- **Biomass Analysis**: Provides input for trend analysis and comparisons

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline
