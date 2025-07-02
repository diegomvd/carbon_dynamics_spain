# Canopy Height Deep Learning Component

Comprehensive deep learning pipeline for canopy height regression using Sentinel-2 satellite imagery and PNOA vegetation data with PyTorch Lightning and TorchGeo.

## Overview

This component implements a complete end-to-end pipeline for predicting canopy heights across Spain using multi-temporal Sentinel-2 imagery. The system includes model training, evaluation, prediction, and comprehensive post-processing workflows to generate analysis-ready country-wide mosaics.

## Workflow

```
Training Data → Model Training → Prediction → Post-Processing → Final Products
   (S2+PNOA)     (Lightning)    (10m patches)  (3-step workflow)  (100m mosaics)
```
