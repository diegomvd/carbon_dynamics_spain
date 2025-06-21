# Forest Inventory Component

Spanish National Forest Inventory (NFI) biomass data processing pipeline for the Iberian Carbon Assessment. Extracts biomass stocks from NFI4 Access databases and integrates forest type information.

## Overview

This component processes Spanish National Forest Inventory (NFI4) data to:

- **Access database processing** - Extract data from `.accdb` files (IFN4 and SIG databases)
- **Volume to biomass conversion** - Using Global Wood Density Database and allometric relationships
- **Forest type integration** - Add MFE (Spanish Forest Map) forest type information
- **CRS handling** - Process multiple UTM zones (29, 30, 31) across Spain
- **Biomass ratio calculations** - Calculate below-ground to above-ground biomass ratios
- **Multiple export formats** - UTM-specific, combined, and year-stratified outputs

## Structure

```
forest_inventory/
├── config.yaml                    # Component configuration
├── __init__.py                    # Package initialization  
├── core/                          # Core processing modules
│   ├── __init__.py
│   ├── nfi_processing.py          # Main processing pipeline class
│   └── nfi_utils.py               # Utility functions for biomass calculations
├── scripts/                       # Executable entry points
│   ├── __init__.py
│   └── run_nfi_processing.py      # Main processing script
└── README.md                      # This file
```

## Installation

### Environment Setup

```bash
# Create conda environment
conda env create -f ../environments/data_preprocessing.yml
conda activate data_preprocessing

# Install repository for absolute imports
pip install -e ../..
```

### System Requirements

**Critical requirement**: Install `mdb-tools` for Access database processing:

```bash
# Ubuntu/Debian
sudo apt-get install mdb-tools

# macOS  
brew install mdb-tools

# Windows
# Use WSL or alternative Access export tools
```

### Data Requirements

Required reference files in data directory:
- `GlobalWoodDensityDatabase.xls` - Global wood density database
- `CODIGOS_IFN.csv` - NFI species codes mapping
- `IFN_4_SP/` - Directory containing NFI4 Access database files (`.accdb`)
- `MFESpain/` - Spanish Forest Map shapefiles for forest type data

## Usage

### Main Pipeline

```bash
# Run with default configuration
python scripts/run_nfi_processing.py

# Run with custom configuration  
python scripts/run_nfi_processing.py --config custom_config.yaml

# Process specific data directory
python scripts/run_nfi_processing.py --data-dir /path/to/nfi/data --output-dir /path/to/results

# Process specific UTM zones only
python scripts/run_nfi_processing.py --utm-zones 29 30

# Enable debug logging
python scripts/run_nfi_processing.py --log-level DEBUG
```

### Validation and Testing

```bash
# Validate inputs without processing
python scripts/run_nfi_processing.py --validate-only

# Show processing summary
python scripts/run_nfi_processing.py --summary

# Force processing despite validation warnings
python scripts/run_nfi_processing.py --force
```

## Configuration

Key configuration sections in `config.yaml`:

- **`data`**: Input paths for NFI databases, reference files, and MFE data
- **`output`**: Output directories and file naming templates
- **`processing`**: UTM zones, CRS settings, biomass calculation parameters
- **`logging`**: Log levels and output configuration

Example configuration structure:
```yaml
data:
  base_dir: '/path/to/data'
  ifn4_dir: 'IFN_4_SP'
  mfe_dir: 'MFESpain'
  wood_density_file: 'GlobalWoodDensityDatabase.xls'
  species_codes_file: 'CODIGOS_IFN.csv'

processing:
  valid_utm_zones: [29, 30, 31]  # Excludes zone 28
  target_crs: 'EPSG:4326'        # Output coordinate system
```

## Outputs

### Biomass Shapefiles

**UTM-specific exports:**
- `biomass_plots_utm29.shp` - UTM zone 29 biomass data
- `biomass_plots_utm30.shp` - UTM zone 30 biomass data  
- `biomass_plots_utm31.shp` - UTM zone 31 biomass data

**Combined export:**
- `biomass_plots_combined.shp` - All UTM zones merged to target CRS

**Year-stratified exports:**
- `biomass_plots_2009.shp` - 2009 inventory data
- `biomass_plots_2010.shp` - 2010 inventory data
- (etc. for all available years)

### Data Fields

Key attributes in output shapefiles:
- `AGB` - Above-ground biomass (Mg/ha)
- `BGB` - Below-ground biomass (Mg/ha)  
- `TotalBiomass` - Total biomass (Mg/ha)
- `BGB_Ratio` - Below-ground to above-ground ratio
- `ForestType` - MFE forest type classification
- `Species` - Dominant tree species
- `Year` - Inventory year
- `UTM_Zone` - Original UTM zone

## Features

### Biomass Calculations
- **Volume to biomass conversion** using wood density values
- **Species-specific calculations** with fallback to genus/family level
- **Below-ground biomass ratios** calculated from NFI measurements
- **Quality control** with outlier detection and validation

### Forest Type Integration
- **Spatial join** with MFE forest type polygons
- **Forest type hierarchies** (species → genus → family → clade)
- **Missing data handling** with appropriate fallback mechanisms

### Multi-UTM Processing
- **Automatic UTM zone detection** from plot coordinates
- **CRS transformations** between UTM zones and target projection
- **Zone-specific exports** plus combined multi-zone output
- **Coordinate validation** and quality checks

### Data Quality Assurance
- **Input validation** for all required files and directories
- **Database integrity checks** for Access files
- **Coordinate system validation** across datasets
- **Missing data reports** and handling strategies

## Dependencies

Core packages (managed via conda environment):
- `pandas` + `geopandas` for data processing
- `numpy` for numerical computations
- `rasterio` for geospatial operations
- `fiona` + `shapely` for vector data handling
- `xlrd` for Excel file reading

**System dependency**: `mdb-tools` for Access database export

## Integration

This component integrates with:
- **Biomass Model**: Provides NFI reference data for allometry fitting
- **Data Preprocessing**: Provides forest type masks and validation data
- **Shared Utils**: Uses common logging and configuration utilities

## Troubleshooting

### Common Issues
- **Access database errors**: Ensure `mdb-tools` is properly installed
- **CRS transformation errors**: Verify UTM zone assignments for plots
- **Missing forest types**: Check MFE shapefile coverage and projection
- **Species matching errors**: Review species code mappings in `CODIGOS_IFN.csv`

### Performance Tips
- **Process UTM zones separately** for large datasets to reduce memory usage
- **Use SSD storage** for faster Access database processing
- **Check system memory** - large datasets may require 8GB+ RAM
- **Monitor temp directory** space during processing

### Data Quality Checks
- **Verify database completeness** - check for missing years or UTM zones
- **Validate coordinate systems** between NFI and MFE datasets
- **Review biomass outliers** - very high/low values may indicate errors
- **Check forest type coverage** - ensure MFE data covers all plot locations

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline