# Sentinel-2 Processing Component

Comprehensive pipeline for creating Sentinel-2 summer mosaics over Spain using distributed computing with STAC catalog integration and optimized memory management.

## Overview

This component implements a complete end-to-end pipeline for processing Sentinel-2 L2A satellite imagery to create analysis-ready summer mosaics across Spain's territory. The system includes distributed processing capabilities, comprehensive post-processing workflows, and extensive quality assurance tools.


## Workflow

```
STAC Catalog → Scene Selection → Distributed Processing → Post-processing → Analysis
   (AWS)         (SCL Masking)    (Dask Clusters)      (Downsampling)    (QA/QC)
```

