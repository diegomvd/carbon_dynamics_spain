# Canopy Height Deep Learning Component

Comprehensive deep learning pipeline for canopy height regression using Sentinel-2 satellite imagery and PNOA vegetation data with PyTorch Lightning and TorchGeo.

## Overview

This component implements a complete end-to-end pipeline for predicting canopy heights across Spain using multi-temporal Sentinel-2 imagery. The system includes model training, evaluation, prediction, and comprehensive post-processing workflows to generate analysis-ready country-wide mosaics.

Key features:
- **Deep learning models** with PyTorch Lightning and TorchGeo integration
- **Range-aware loss functions** to handle natural height distribution imbalances
- **Balanced sampling** strategies for diverse training data
- **Memory-efficient prediction** with geographic filtering
- **Multi-step post-processing** for analysis-ready products
- **Comprehensive evaluation** with statistical analysis and visualizations

## Workflow

```
Training Data → Model Training → Prediction → Post-Processing → Final Products
   (S2+PNOA)     (Lightning)    (10m patches)  (3-step workflow)  (100m mosaics)
```

### Main Pipeline Steps

1. **Model Training** - Train deep learning models using PyTorch Lightning
2. **Model Evaluation** - Comprehensive evaluation with statistical analysis and visualizations  
3. **Model Prediction** - Generate predictions across large geographic areas using tiled approach
4. **Post-Processing** - Three-step workflow to create final products:
   - **Merge**: Combine prediction patches into 120km tiles
   - **Sanitize**: Remove outliers and fill temporal gaps via interpolation
   - **Final Merge**: Downsample and create country-wide 100m mosaics for biomass estimation

## Structure

```
canopy_height_model/
├── config.yaml                    # Component configuration
├── core/                          # Core processing modules
│   ├── model_training.py          # Training pipeline class
│   ├── model_prediction.py        # Large-scale prediction pipeline  
│   ├── model_evaluation.py        # Evaluation and visualization
│   ├── postprocessing.py          # Multi-step post-processing
│   ├── canopy_height_regression.py # PyTorch Lightning module
│   ├── s2_pnoa_datamodule.py      # Data loading and augmentation
│   ├── datasets.py                # TorchGeo dataset classes
│   ├── height_regression_losses.py # Range-aware loss functions
│   ├── pnoa_vegetation_transforms.py # Data preprocessing transforms
│   ├── balanced_geo_samplers.py   # Height-diversity batch sampling
│   └── prediction_writer.py       # Raster output writer
├── scripts/                       # Executable entry points
│   ├── run_training.py            # Training and testing script
│   ├── run_prediction.py          # Large-scale prediction script
│   ├── run_evaluation.py          # Model evaluation script
│   └── run_postprocessing.py      # Post-processing script
└── README.md                      # This file
```

## Usage

### Environment Setup

```bash
# Create conda environment
conda env create -f ../environments/canopy_height_dl.yml
conda activate canopy_height_dl

# Install repository for absolute imports
pip install -e ../..
```

### Model Training

```bash
# Train model with default configuration
python scripts/run_training.py

# Custom configuration
python scripts/run_training.py --config custom_config.yaml

# Training and testing
python scripts/run_training.py --mode train_test

# Resume from checkpoint
python scripts/run_training.py --resume /path/to/checkpoint.ckpt

# Test only mode
python scripts/run_training.py --mode test --checkpoint /path/to/model.ckpt
```

### Model Evaluation

```bash
# Evaluate trained model
python scripts/run_evaluation.py --checkpoint /path/to/model.ckpt

# Multiple checkpoints comparison
python scripts/run_evaluation.py --checkpoint model1.ckpt model2.ckpt

# Custom output directory
python scripts/run_evaluation.py --checkpoint model.ckpt --output-dir ./results
```

### Large-Scale Prediction

```bash
# Generate predictions across Spain
python scripts/run_prediction.py --checkpoint /path/to/model.ckpt

# Custom configuration
python scripts/run_prediction.py --config prediction_config.yaml --checkpoint model.ckpt

# Specific output directory
python scripts/run_prediction.py --checkpoint model.ckpt --output-dir ./predictions
```

### Post-Processing

```bash
# Run complete post-processing pipeline
python scripts/run_postprocessing.py

# Run specific steps only
python scripts/run_postprocessing.py --steps merge,sanitize

# Continue on errors
python scripts/run_postprocessing.py --continue-on-error

# Custom input/output directories
python scripts/run_postprocessing.py --input-dir ./predictions --output-dir ./processed
```

## Configuration

Key configuration sections in `config.yaml`:

- **`data`**: Input/output paths, patch sizes, data directories
- **`model`**: Architecture, loss functions, training parameters
- **`training`**: Learning rates, epochs, early stopping, augmentation
- **`prediction`**: Tile sizes, overlap, output formats, geographic filtering
- **`post_processing`**: Three-step workflow parameters and quality control
- **`evaluation`**: Plot settings, metrics, output formats

## Core Components

### Deep Learning Architecture
- **Model Support**: U-Net, DeepLabV3+, and other segmentation architectures
- **Backbone Options**: EfficientNet, ResNet, and other encoders from `segmentation_models_pytorch`
- **Multi-scale Processing**: Configurable patch sizes and overlap for optimal performance

### Data Handling
- **S2Mosaic**: Sentinel-2 summer composite mosaics (2017-2024)
- **PNOAVegetation**: Spanish PNOA vegetation height models as ground truth
- **KorniaIntersectionDataset**: Kornia-compatible intersection dataset wrapper
- **Balanced Sampling**: Height-diversity aware batch sampling for better training

### Loss Functions & Training
- **RangeAwareL1Loss**: Inverse frequency weighting for height range balancing
- **Log-space Operations**: Numerical stability for vegetation height distributions
- **Advanced Augmentation**: Kornia-based geometric and intensity augmentations
- **NaN Handling**: Robust processing of invalid pixels and missing data

### Prediction & Output
- **Geographic Filtering**: Spain boundary filtering for relevant predictions only
- **Memory Management**: Efficient batch processing with garbage collection
- **Raster Output**: Georeferenced GeoTIFF with compression and tiling
- **Quality Control**: Outlier removal and temporal gap filling

## Outputs

### Training Products
- **Checkpoints**: Model weights with training metrics
- **Logs**: Training progress and validation metrics
- **Tensorboard**: Learning curves and model monitoring

### Evaluation Products
- **Statistical Metrics**: MAE, RMSE, R², bias across height ranges
- **Residual Plots**: Performance analysis by vegetation height bins
- **Density Plots**: Prediction vs. measured height relationships
- **Distribution Comparisons**: Measured vs. predicted height distributions

### Prediction Products
- **Raw Patches**: High-resolution prediction tiles (10m resolution)
- **Merged Tiles**: Combined patches organized by geographic tiles
- **Quality Masks**: Interpolation and processing quality indicators

### Final Products
- **Country Mosaics**: Analysis-ready 100m resolution national products (2017-2024)
- **Processed Heights**: Outlier-filtered and temporally-interpolated height maps
- **Metadata**: Processing logs and quality assessment reports

### File Organization
```
canopy_height_predictions/
├── patches/                        # Raw prediction patches
│   ├── 0/                         # Dataloader index
│   │   └── predicted_minx_*_maxy_*_mint_*.tif
├── merged_tiles/                   # Step 1: Merged patches
│   └── merged_lat_*_lon_*_year_*.tif
├── sanitized/                      # Step 2: Outlier removal
│   └── processed_lat_*_lon_*_year_*.tif
└── final_mosaics/                  # Step 3: Country-wide products
    ├── canopy_height_spain_2020_100m.tif
    └── canopy_height_spain_2021_100m.tif
```

## Features

### Advanced Data Loading
- **TorchGeo Integration**: Seamless geospatial data handling with proper CRS management
- **Multi-temporal Support**: Training on multiple years of Sentinel-2 and PNOA data
- **Efficient Sampling**: Grid-based and random sampling strategies for optimal coverage

### Training Innovations
- **Height Range Metrics**: Performance assessment across different vegetation height classes
- **Balanced Loss Functions**: Address natural imbalance in vegetation height distributions
- **Robust NaN Handling**: Comprehensive invalid pixel masking and processing

### Scalable Prediction
- **Tiled Processing**: Memory-efficient inference across large geographic areas
- **Overlap Management**: Seamless tile merging with configurable overlap
- **Spain Filtering**: Geographic boundary checking to process only relevant areas

### Quality Assurance
- **Multi-step Validation**: Input data validation, model checking, output verification
- **Comprehensive Logging**: Component-specific logging with performance monitoring
- **Error Recovery**: Robust error handling with graceful degradation

### Post-Processing Pipeline
- **Outlier Detection**: Statistical filtering of unrealistic height predictions
- **Temporal Interpolation**: Gap filling across multi-year time series
- **Spatial Consistency**: Edge matching and seamless mosaic generation

## Performance Notes

- **Memory Requirements**: 32GB+ RAM recommended for large-scale processing
- **GPU Support**: Automatic CUDA/MPS/CPU detection with optimized workflows
- **Storage Requirements**: Country-wide predictions require ~100GB+ storage space
- **Processing Time**: Full country prediction ~12-24 hours depending on hardware
- **Parallel Processing**: Configurable worker counts for optimal resource utilization

## Dependencies

Core packages (managed via conda environment):
- **Deep Learning**: `pytorch-lightning`, `torchgeo`, `segmentation-models-pytorch`
- **Computer Vision**: `kornia`, `opencv-python`, `pillow`
- **Geospatial**: `rasterio`, `geopandas`, `shapely`, `pyproj`
- **Scientific Computing**: `numpy`, `scipy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Utilities**: `tqdm`, `psutil`, `pyyaml`

## Integration

This component integrates with:
- **Data Preprocessing**: Uses processed Sentinel-2 mosaics and PNOA data
- **Biomass Estimation**: Provides canopy height inputs for allometric biomass calculations
- **Forest Analysis**: Supports vegetation structure analysis and monitoring

## Advanced Usage

### Custom Model Architectures
```python
# In config.yaml
model:
  model_type: 'deeplabv3+'
  backbone: 'resnet50'
  weights: 'imagenet'
```

### Multi-checkpoint Evaluation
```bash
# Compare multiple model versions
python scripts/run_evaluation.py \
  --checkpoint model_v1.ckpt model_v2.ckpt model_v3.ckpt \
  --output-dir ./comparison_results
```

### Custom Loss Function Configuration
```yaml
# Range-aware loss parameters
loss:
  percentile_range: [7.5, 92.5]
  lambda_reg: 0.0
  alpha: 0.6  # Inverse frequency weighting strength
  max_height: 30.0
  weights: true
```

### Prediction Optimization
```yaml
# Memory and performance tuning
prediction:
  batch_size: 8        # Adjust based on GPU memory
  tile_size: 512       # Tile size for processing
  overlap: 64          # Overlap between tiles
  num_workers: 4       # Data loading workers
```

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or tile size
- **Slow data loading**: Increase num_workers or check disk I/O
- **Geographic misalignment**: Verify CRS consistency across datasets
- **Missing predictions**: Check Spain boundary shapefile and geographic filtering

### Performance Optimization
- **Training**: Use mixed precision, adjust batch size, optimize data loading
- **Prediction**: Tune tile size vs. memory trade-off, use efficient data formats
- **Post-processing**: Leverage parallel processing, optimize I/O operations

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline