# Canopy Height Model

Deep learning pipeline for canopy height regression using Sentinel-2 satellite imagery and PNOA vegetation data.

## Overview

This component implements a complete PyTorch Lightning-based pipeline for predicting canopy heights across Spain using multi-temporal Sentinel-2 imagery. The system includes model training, evaluation, prediction, and comprehensive post-processing workflows to generate analysis-ready country-wide mosaics.

## Workflow

```
Training Data → Model Training → Prediction → Post-Processing → Final Products for biomass estimation
    (S2+PNOA)      (Lightning)     (Patches at 10m res)    (3 Steps)      (100m Mosaics)
```

### Main Pipeline Steps

1. **Model Training** - Train deep learning models using PyTorch Lightning
2. **Model Evaluation** - Comprehensive evaluation with statistical analysis and visualizations  
3. **Model Prediction** - Generate predictions across large geographic areas
4. **Post-Processing** - Three-step workflow to create final products:
   - **Merge**: Combine prediction patches into 120km tiles
   - **Sanitize**: Remove outliers and fill temporal gaps via interpolation
   - **Final Merge**: Downsample and create country-wide 100m mosaics for direct biomass estimation
## Quick Start

### Prerequisites

```bash
conda env create -f dl_env.yml
conda activate biomass-estimation-dask
```

### Configuration

Edit `default_config.yaml` to configure:
- Data directories and paths
- Model parameters and training settings  
- Post-processing parameters

### Basic Usage

```bash
# Train a model
python model_training.py

# Evaluate trained model
python model_evaluation.py

# Generate predictions
python model_prediction.py

# Run complete post-processing pipeline
python postprocess_pipeline.py

# Run specific post-processing steps
python postprocess_pipeline.py --steps merge,sanitize
```

## Configuration

All parameters are centralized in `default_config.yaml`:

- **`data`** - Data paths, batch sizes, augmentation parameters
- **`model`** - Model architecture, loss functions, metrics
- **`training`** - Learning rates, epochs, early stopping
- **`prediction`** - Patch sizes, output settings
- **`post_processing`** - Three-step workflow parameters
- **`evaluation`** - Plot settings, output directories

## File Structure

```
canopy_height_model/
├── model_training.py              # Main training pipeline
├── model_evaluation.py            # Model evaluation and analysis
├── model_prediction.py            # Large-scale prediction
├── postprocess_pipeline.py        # Post-processing orchestrator
├── merge_predictions.py           # Step 1: Merge patches to tiles
├── sanitize_predictions.py        # Step 2: Outlier removal and interpolation
├── downsample_merge.py            # Step 3: Final country-wide mosaics
├── canopy_height_regression.py    # Core PyTorch Lightning module
├── s2_pnoa_vegetation_datamodule.py # Data loading and augmentation
├── [other modules...]             # Supporting utilities and datasets
├── config.py                      # Configuration management
├── default_config.yaml            # Centralized parameters
└── dl_env.yml                     # Conda environment
```

## Output Products

- **Training Checkpoints** - Saved model weights with metrics
- **Evaluation Reports** - Statistical summaries, plots, residual analysis
- **Prediction Tiles** - High-resolution canopy height predictions
- **Country Mosaics** - Analysis-ready 100m resolution national products (2017-2024)
- **Quality Masks** - Interpolation and processing quality indicators

## Advanced Usage

### Custom Training Configurations

```bash
# Create custom config file
cp default_config.yaml my_config.yaml
# Edit parameters...

# Train with custom config
python model_training.py --config my_config.yaml
```

### Selective Post-Processing

```bash
# Run only sanitization step
python postprocess_pipeline.py --steps sanitize

# Continue pipeline even if errors occur
python postprocess_pipeline.py --continue-on-error

# Run individual post-processing steps
python merge_predictions.py
python sanitize_predictions.py
python downsample_merge.py
```

### Model Testing with Multiple Checkpoints

Configure test models in `default_config.yaml` and run:

```bash
python model_training.py  # Set mode='test' in main()
```

## Performance Notes

- **Memory Requirements** - 32GB+ RAM recommended for large-scale processing
- **GPU Support** - Automatic MPS/CUDA detection for training and prediction
- **Parallel Processing** - Configurable worker counts for optimal performance
- **Storage** - Country-wide predictions require ~100GB+ storage space

## Dependencies

Core dependencies managed via `dl_env.yml`:
- PyTorch Lightning
- TorchGeo  
- Rasterio
- Geopandas
- Kornia
- Segmentation Models PyTorch

---

**Author**: Diego Bengochea  
**Component**: Part of the Iberian Carbon Assessment Pipeline
