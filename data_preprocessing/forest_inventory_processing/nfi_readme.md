# Spanish National Forest Inventory (NFI) Biomass Processing

Harmonized pipeline for processing Spanish NFI4 data to extract forest biomass stocks and integrate forest type information.

## Features

- Extracts biomass data from NFI4 Access databases (.accdb files)
- Converts wood volume to biomass using Global Wood Density Database
- Adds forest type information from Spanish Forest Map (MFE) data
- Calculates below-ground to above-ground biomass ratios
- Exports data in multiple formats (UTM-specific, combined, year-stratified)

## Requirements

### Software Dependencies
```bash
conda env create -f env_nfi.yml
conda activate nfi
```

### System Dependencies
- **mdb-tools** (for Access database export):
  - Ubuntu/Debian: `sudo apt-get install mdb-tools`
  - macOS: `brew install mdb-tools`
  - Windows: Use WSL or alternative

## Data Structure

Expected data directory structure:
```
DATA_DIR/
├── IFN_4_SP/
│   ├── Ifn4_*.accdb          # NFI plot data by province
│   └── Sig_*.accdb           # Biomass/volume data by province
├── MFESpain/
│   └── MFE*.shp              # Spanish Forest Map data
├── GlobalWoodDensityDatabase.xls
└── CODIGOS_IFN.csv           # Species codes
```

## Usage

1. **Configure paths** in `process_nfi_biomass.py`:
```python
DATA_DIR = "/path/to/nfi/data/"
OUTPUT_DIR = "/path/to/output/"
```

2. **Run pipeline**:
```bash
python process_nfi_biomass.py
```

## Outputs

The pipeline generates shapefiles with biomass data:

### UTM-specific files
- `nfi4_utm29_biomass.shp`
- `nfi4_utm30_biomass.shp` 
- `nfi4_utm31_biomass.shp`

### Combined data
- `nfi4_all_biomass.shp` (all UTMs merged, EPSG:25830)

### Year-stratified data
- `nfi4_2020_biomass.shp`
- `nfi4_2021_biomass.shp`
- `nfi4_YYYY_biomass.shp` (one file per survey year)

## Output Fields

Each shapefile contains:
- `AGB`: Above-ground biomass (Mg/ha)
- `BGB`: Below-ground biomass (Mg/ha) 
- `BGB_Ratio`: BGB/AGB ratio
- `Year`: Survey year
- `ForestType`: Forest type from MFE data
- `Region`: Province/region name
- `Index`: Compound plot identifier

## Files

- `process_nfi_biomass.py` - Main processing pipeline
- `nfi_utils.py` - Utility functions for biomass calculations
- `env_nfi.yml` - Conda environment specification
- `all_forest_types.csv` - Reference list of forest types (optional)

## Notes

- Only processes UTM zones 29, 30, 31 (excludes 28)
- Automatically handles both volume-to-biomass conversion and direct biomass data
- All outputs use EPSG:25830 coordinate system
- Plots without forest type data are included with `ForestType = NaN`