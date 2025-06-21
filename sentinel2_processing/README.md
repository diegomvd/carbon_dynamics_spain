# Sentinel-2 Processing Component

Comprehensive pipeline for creating Sentinel-2 summer mosaics over Spain using distributed computing with STAC catalog integration and optimized memory management.

## Overview

This component implements a complete end-to-end pipeline for processing Sentinel-2 L2A satellite imagery to create analysis-ready summer mosaics across Spain's territory. The system includes distributed processing capabilities, comprehensive post-processing workflows, and extensive quality assurance tools.

Key features:
- **Distributed processing** with Dask clusters for large-scale geospatial computing
- **STAC catalog integration** for automated Sentinel-2 data discovery and access
- **Scene Classification Layer (SCL) masking** with optimal scene selection algorithms
- **Memory-optimized processing** with automatic cluster management and garbage collection
- **Comprehensive post-processing** including spatial downsampling and yearly merging
- **Quality assurance workflows** with missing tile detection and gap analysis
- **Parameter optimization** through robustness assessment and consistency validation

## Workflow

```
STAC Catalog → Scene Selection → Distributed Processing → Post-processing → Analysis
   (AWS)         (SCL Masking)    (Dask Clusters)      (Downsampling)    (QA/QC)
```

### Main Pipeline Steps

1. **Mosaic Processing** - Create summer mosaics using distributed computing
2. **Spatial Processing** - Downsample and merge tiles into analysis-ready products
3. **Quality Assurance** - Detect missing tiles and assess processing completeness
4. **Parameter Optimization** - Assess robustness with varying scene numbers
5. **Consistency Validation** - Analyze interannual spectral consistency

## Structure

```
sentinel2_processing/
├── config.yaml                    # Component configuration
├── __init__.py                    # Main component exports
├── core/                          # Core processing modules
│   ├── __init__.py
│   ├── s2_utils.py                # Utility functions for STAC, masking, clustering
│   ├── mosaic_processing.py       # Main distributed processing pipeline
│   └── postprocessing.py          # Post-processing and analysis workflows
├── scripts/                       # Executable entry points
│   ├── __init__.py
│   ├── run_mosaic_processing.py   # Main mosaic creation pipeline
│   ├── run_downsampling.py        # Spatial downsampling and merging
│   ├── run_missing_analysis.py    # Missing tile detection and analysis
│   ├── run_robustness_assessment.py # Scene selection optimization
│   ├── run_consistency_analysis.py  # Interannual consistency validation
│   └── run_postprocessing.py      # Orchestrated workflow execution
└── README.md                      # This file
```

## Installation

### Environment Setup

```bash
# Create conda environment for Sentinel-2 processing
conda env create -f ../environments/data_preprocessing.yml
conda activate data_preprocessing

# Install repository for absolute imports
pip install -e ../..
```

### System Requirements

- **Memory**: Minimum 16GB RAM (32GB+ recommended for large-scale processing)
- **Storage**: 100GB+ available disk space
- **Network**: High-bandwidth internet connection for STAC catalog access
- **Python**: 3.8+ with scientific computing stack

### Dependencies

Core dependencies automatically handled by conda environment:
- `dask[distributed]` - Distributed computing framework
- `rasterio` - Geospatial raster I/O
- `xarray`, `rioxarray` - N-dimensional array processing
- `odc-stac` - STAC data loading and processing
- `pystac-client` - STAC catalog client
- `geopandas` - Geospatial vector data handling
- `matplotlib`, `seaborn` - Visualization and analysis
- `scipy` - Statistical computing

## Usage

### Quick Start

```bash
# 1. Update configuration paths
vim config.yaml  # Set spain_polygon and output directories

# 2. Run complete processing pipeline
python scripts/run_mosaic_processing.py --years 2021 2022

# 3. Run post-processing workflows
python scripts/run_postprocessing.py
```

### Main Mosaic Processing

```bash
# Basic mosaic creation
python scripts/run_mosaic_processing.py

# Process specific years with custom parameters
python scripts/run_mosaic_processing.py --years 2021 2022 2023 --n-scenes 15

# Use custom tile size and compute resources
python scripts/run_mosaic_processing.py --tile-size 6144 --n-workers 16 --memory-per-worker 30GB

# Validate configuration without processing
python scripts/run_mosaic_processing.py --validate-only

# Show processing summary
python scripts/run_mosaic_processing.py --summary
```

### Post-processing Workflows

```bash
# Complete post-processing pipeline
python scripts/run_postprocessing.py

# Spatial downsampling and merging only
python scripts/run_downsampling.py --scale-factor 10

# Missing tile analysis
python scripts/run_missing_analysis.py --save-paths missing_files.txt

# Scene selection optimization
python scripts/run_robustness_assessment.py --bbox -9.5 36.0 3.3 43.8 --year 2022

# Interannual consistency analysis
python scripts/run_consistency_analysis.py --sample-size 5000

# Orchestrated execution with specific workflows
python scripts/run_postprocessing.py --workflows downsampling missing --continue-on-error
```

### Advanced Usage

```bash
# Custom spatial processing
python scripts/run_downsampling.py --downsample-only --scale-factor 5
python scripts/run_downsampling.py --merge-only

# Robustness assessment for different regions
python scripts/run_robustness_assessment.py --bbox -2.0 35.0 4.0 44.0 --min-scenes 5 --max-scenes 25

# Consistency analysis with year filtering
python scripts/run_consistency_analysis.py --years 2019 2021 2022 --exclude-years 2020

# Dry run for workflow planning
python scripts/run_postprocessing.py --dry-run --workflows all
```

## Configuration

### Main Configuration (`config.yaml`)

Key configuration sections:

#### Processing Parameters
```yaml
processing:
  n_scenes: 12                    # Number of best scenes per mosaic
  years: [2019, 2021, 2022, 2023] # Years to process
  tile_size: 12288                # Processing tile size (pixels)
  cloud_threshold: 1              # Initial cloud cover threshold (%)
```

#### Paths (Update Required)
```yaml
paths:
  spain_polygon: "/path/to/spain/polygon.shp"     # REQUIRED: Spain boundary
  output_dir: "/path/to/sentinel2/output/"        # REQUIRED: Main output directory
  downsampled_dir: "/path/to/downsampled/"        # Downsampled outputs
  merged_dir: "/path/to/merged/"                  # Merged yearly outputs
```

#### Compute Resources
```yaml
compute:
  n_workers: 8                    # Number of dask workers
  threads_per_worker: 3           # Threads per worker
  memory_per_worker: "20GB"       # Memory limit per worker
```

#### STAC and Data Configuration
```yaml
data:
  stac_url: "https://earth-search.aws.element84.com/v1"
  bands: ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', ...]
  min_n_items: 40                 # Minimum scenes required
```

### Configuration Override

All scripts support command-line configuration overrides:

```bash
# Override years and compute resources
python scripts/run_mosaic_processing.py --years 2023 --n-workers 16

# Override directories and parameters
python scripts/run_downsampling.py --input-dir /custom/input --scale-factor 5

# Override analysis parameters
python scripts/run_consistency_analysis.py --sample-size 15000 --input-dir /custom/merged
```

## Outputs

### Main Products

1. **Raw Mosaics** (`output_dir`)
   - Format: GeoTIFF (`.tif`)
   - Resolution: 10m native Sentinel-2 resolution
   - Naming: `S2_summer_mosaic_YYYY_west_south_east_north.tif`
   - Compression: LZW with tiling

2. **Downsampled Mosaics** (`downsampled_dir`)
   - Format: GeoTIFF with `_downsampled.tif` suffix
   - Resolution: Configurable (default 100m at 10x downsampling)
   - Processing: Average resampling method

3. **Merged Yearly Mosaics** (`merged_dir`)
   - Format: GeoTIFF with country-wide coverage
   - Naming: `S2_summer_mosaic_YYYY_merged.tif`
   - Content: Single file per year covering all of Spain

### Analysis Products

4. **Missing Files Analysis**
   - `missing_file_paths.txt` - List of missing tile-year combinations
   - Console reports with completeness statistics

5. **Robustness Assessment**
   - Console output with optimal scene recommendations
   - Optional JSON results file with detailed statistics

6. **Consistency Analysis**
   - `interannual_consistency_results/` directory containing:
     - `band_statistics.csv` - Statistical summaries by year and band
     - `statistical_tests.csv` - Kolmogorov-Smirnov test results
     - `analysis_report.txt` - Comprehensive text report
     - `band_means_boxplot.png` - Box plot visualizations
     - `band_means_violin.png` - Distribution density plots

### Metadata

All outputs include comprehensive metadata:
- Processing year and temporal coverage
- Scene count and valid pixel percentage
- Processing parameters and software versions
- Coordinate reference system and spatial extent

## Performance

### Processing Scale

- **Spatial Coverage**: Complete Spain territory (~505,000 km²)
- **Temporal Resolution**: Annual summer composites (June-September)
- **Spatial Resolution**: 10m native, configurable downsampling
- **Processing Units**: Configurable tile-based approach (default 12.288 km tiles)

### Memory Management

The pipeline implements sophisticated memory optimization:
- **Cluster Restart**: Automatic restart between processing iterations
- **Garbage Collection**: Aggressive cleanup of intermediate data
- **Memory Monitoring**: Real-time system memory usage tracking
- **Chunk Optimization**: Configurable chunk sizes for optimal performance

### Typical Performance

On a system with 32GB RAM and 16 cores:
- **Mosaic Creation**: ~2-4 hours per year (depending on scene availability)
- **Downsampling**: ~30-60 minutes for complete dataset
- **Analysis Workflows**: ~15-30 minutes each

### Optimization Tips

1. **Increase Workers**: Scale `n_workers` based on available CPU cores
2. **Memory Allocation**: Set `memory_per_worker` to 70-80% of available RAM per worker
3. **Chunk Size**: Increase `chunk_size` for systems with more memory
4. **Storage**: Use fast SSD storage for temporary and output directories
5. **Network**: Ensure stable high-bandwidth connection for STAC catalog access

## Data Requirements

### Required Input Data

1. **Spain Polygon Shapefile**
   - Administrative boundary defining processing extent
   - Should be in any coordinate system (automatically reprojected)
   - Update `paths.spain_polygon` in configuration

### External Data Dependencies

1. **STAC Catalog Access**
   - AWS Earth Search STAC catalog
   - Automatic Sentinel-2 L2A data discovery and access
   - Requires internet connectivity

2. **Sentinel-2 L2A Data**
   - Automatically accessed via STAC catalog
   - Level-2A (surface reflectance) products
   - Scene Classification Layer (SCL) for masking

## Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce memory per worker
python scripts/run_mosaic_processing.py --memory-per-worker 16GB

# Reduce number of workers
python scripts/run_mosaic_processing.py --n-workers 4

# Reduce chunk size
# Edit config.yaml: processing.chunk_size: 1024
```

#### STAC Catalog Issues
```bash
# Check internet connectivity
ping earth-search.aws.element84.com

# Increase timeout in config.yaml
# advanced.stac_timeout: 600
```

#### Insufficient Data
```bash
# Check scene availability for your region and time period
python scripts/run_robustness_assessment.py --bbox YOUR_BBOX --year YOUR_YEAR

# Reduce minimum scenes requirement
# Edit config.yaml: data.min_n_items: 20
```

#### Disk Space Issues
```bash
# Check available space
df -h /path/to/output

# Enable output cleanup
# Edit config.yaml: advanced.overwrite_existing: true
```

### Error Recovery

#### Resume Processing
```bash
# Skip existing files (default behavior)
python scripts/run_mosaic_processing.py

# Force overwrite if corruption suspected  
# Edit config.yaml: advanced.overwrite_existing: true
```

#### Workflow Recovery
```bash
# Continue with remaining workflows after failure
python scripts/run_postprocessing.py --continue-on-error

# Run specific failed workflows
python scripts/run_postprocessing.py --workflows robustness consistency
```

### Performance Optimization

#### System Monitoring
```bash
# Monitor system resources during processing
htop
iotop -a
```

#### Dask Dashboard
The processing pipeline provides cluster dashboard URLs for monitoring:
- Access via browser link shown in console output
- Monitor memory usage, task progress, and worker status

### Getting Help

1. **Configuration Validation**
   ```bash
   python scripts/run_mosaic_processing.py --validate-only
   ```

2. **Processing Summary**
   ```bash
   python scripts/run_mosaic_processing.py --summary
   ```

3. **Dry Run Analysis**
   ```bash
   python scripts/run_postprocessing.py --dry-run
   ```

4. **Debug Logging**
   ```bash
   python scripts/run_mosaic_processing.py --log-level DEBUG
   ```

## Technical Specifications

### Algorithms

- **Scene Selection**: Valid pixel percentage ranking with SCL masking
- **Composite Generation**: Median composite from N best scenes
- **Distributed Computing**: Dask LocalCluster with memory optimization
- **Statistical Analysis**: Kolmogorov-Smirnov tests for consistency validation

### Data Processing

- **Coordinate System**: EPSG:25830 (UTM zone 30N for Spain)
- **Temporal Window**: Summer months (June-September)
- **Resampling**: Bilinear for data loading, average for downsampling
- **Compression**: LZW with horizontal predictor optimization

### Quality Standards

- **Completeness Threshold**: 95% for success, 80% for error condition
- **Statistical Significance**: p < 0.05 for consistency tests
- **Scene Count Optimization**: Automatic determination of optimal scene numbers
- **Memory Safety**: Comprehensive cleanup and monitoring throughout processing

For additional support or questions about the Sentinel-2 processing component, please refer to the main project documentation or contact the development team.