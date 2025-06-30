"""
Central Data Paths - Constants

Centralized path management for the Iberian Carbon Assessment repository.
All components should import paths from this module instead of defining their own.

Usage:
    from shared_utils.central_data_paths import SENTINEL2_MOSAICS_DIR, HEIGHT_MAPS_10M_DIR
    
    # Use directly
    input_files = list(SENTINEL2_MOSAICS_DIR.glob("*.tif"))

Author: Diego Bengochea
"""

from pathlib import Path

# Root directories
DATA_ROOT = Path("data")

RAW_DIR = DATA_ROOT / "raw" 
PROCESSED_DIR = DATA_ROOT / "processed"
RESULTS_DIR = DATA_ROOT / "results"

# Forest inventory
FOREST_INVENTORY_RAW_DIR = RAW_DIR / "forest_inventory"
NFI4_DATABASE_DIR = FOREST_INVENTORY_RAW_DIR / "nfi4"
NFI4_SPECIES_CODE_FILE = FOREST_INVENTORY_RAW_DIR / "nfi4_codes.csv"

FOREST_INVENTORY_PROCESSED_DIR = PROCESSED_DIR / "forest_inventory"

# Forest type data
FOREST_TYPE_MAPS_DIR = RAW_DIR / "forest_type_maps"
FOREST_TYPES_TIERS_FILE = FOREST_TYPE_MAPS_DIR / "Forest_Types_Tiers.csv"

FOREST_TYPE_MASKS_DIR = PROCESSED_DIR / "forest_type_masks" / "100m" 

# Wood density database
WOOD_DENSITY_DIR = RAW_DIR / "wood_density"
WOOD_DENSITY_FILE = WOOD_DENSITY_DIR / "GlobalWoodDensityDatabase.xls"

# Land cover
LAND_COVER_DIR = RAW_DIR / "land_cover"
CORINE_LAND_COVER_FILE = LAND_COVER_DIR / "corine_land_cover.tif"

# Country boundaries
SPAIN_BOUNDARIES_DIR = RAW_DIR / "country_boundaries"
SPAIN_BOUNDARIES_FILE = SPAIN_BOUNDARIES_DIR / "gadm41_ESP_0.shp"

# Climate data
CLIMATE_RAW_DIR = RAW_DIR / "climate"

CLIMATE_DIR = PROCESSED_DIR / "climate"
CLIMATE_RASTERS_RAW_DIR = CLIMATE_DIR / "raw_rasters"
CLIMATE_HARMONIZED_DIR = CLIMATE_DIR / "harmonized"
BIOCLIM_VARIABLES_DIR = CLIMATE_DIR / "bioclimatic"
BIOCLIM_ANOMALIES_DIR = CLIMATE_DIR / "bioclimatic_anomalies"

# ALS LiDAR
ALS_CANOPY_HEIGHT_RAW_DIR = RAW_DIR / "als_canopy_height"
ALS_DATA_DIR = ALS_CANOPY_HEIGHT_RAW_DIR / "data"
ALS_METADATA_DIR = ALS_CANOPY_HEIGHT_RAW_DIR / "tile_metadata"
ALS_METADATA_UTM_29_DIR = ALS_METADATA_DIR / "utm_29"
ALS_METADATA_UTM_30_DIR = ALS_METADATA_DIR / "utm_30"
ALS_METADATA_UTM_31_DIR = ALS_METADATA_DIR / "utm_31"

ALS_CANOPY_HEIGHT_PROCESSED_DIR = PROCESSED_DIR / "als_canopy_height"

# Sentinel 2
SENTINEL2_MOSAICS_DIR = PROCESSED_DIR / "sentinel2"
SENTINEL2_PROCESSED_DIR = SENTINEL2_MOSAICS_DIR / "processed"
SENTINEL2_DOWNSAMPLED_DIR = SENTINEL2_PROCESSED_DIR / "downsampled"
SENTINEL2_MERGED_DIR = SENTINEL2_PROCESSED_DIR / "merged"

# Height model checkpoint
PRETRAINED_HEIGHT_MODELS_DIR = DATA_ROOT / "pretrained_height_models"
HEIGHT_MODEL_CHECKPOINT_FILE = PRETRAINED_HEIGHT_MODELS_DIR / "height_model_checkpoint.pkl"
HEIGHT_EVALUATION_RESULTS_DIR = PRETRAINED_HEIGHT_MODELS_DIR / 'evaluation_results'

# Height maps 
HEIGHT_MAPS_DIR = PROCESSED_DIR / "height_maps"
HEIGHT_MAPS_10M_DIR = HEIGHT_MAPS_DIR / "10m"
HEIGHT_MAPS_100M_DIR = HEIGHT_MAPS_DIR / "100m"
HEIGHT_MAPS_TMP_DIR = HEIGHT_MAPS_DIR / "tmp"
HEIGHT_MAPS_TMP_120KM_DIR = HEIGHT_MAPS_TMP_DIR / "merged_120km"
HEIGHT_MAPS_TMP_RAW_DIR = HEIGHT_MAPS_TMP_DIR / "raw"
HEIGHT_MAPS_TMP_INTERPOLATION_MASKS_DIR = HEIGHT_MAPS_TMP_DIR / "interpolation_masks"
HEIGHT_MASK_BINS_DIR = HEIGHT_MAPS_DIR / "bin_masks"


# Biomass allometries
ALLOMETRIES_DIR = PROCESSED_DIR / "allometries"
FITTED_PARAMETERS_FILE = ALLOMETRIES_DIR / "fitted_agb_parameters.csv"
BGB_RATIOS_FILE = ALLOMETRIES_DIR / "bgb_ratios.csv"
FITTING_SUMMARY_FILE = ALLOMETRIES_DIR / "fitting_summary.csv"

# Biomass maps
BIOMASS_MAPS_DIR = PROCESSED_DIR / "biomass_maps"
BIOMASS_MAPS_RAW_DIR = BIOMASS_MAPS_DIR / "raw"
BIOMASS_MAPS_PER_FOREST_TYPE_DIR = BIOMASS_MAPS_DIR / "per_forest_type"
BIOMASS_MAPS_FULL_COUNTRY_DIR = BIOMASS_MAPS_DIR / "full_country"

# Biomass processing temporary files
BIOMASS_MAPS_TEMP_DIR = PROCESSED_DIR / "biomass_maps" / "tmp"
BIOMASS_MASKING_TEMP_DIR = BIOMASS_MAPS_TEMP_DIR / "masking"
BIOMASS_MERGING_TEMP_DIR = BIOMASS_MAPS_TEMP_DIR / "merging"
BIOMASS_PROCESSING_TEMP_DIR = BIOMASS_MAPS_TEMP_DIR / "processing"

# Biomass validation
BIOMASS_VALIDATION_DIR = PROCESSED_DIR / "biomass_validation"
BIOMASS_VALIDATION_FILE = BIOMASS_VALIDATION_DIR / "biomass_nfi4_validation.shp"

# Biomass change maps
BIOMASS_CHANGE_MAPS_DIR = PROCESSED_DIR / "biomass_change_maps"
BIOMASS_CHANGE_MAPS_DIFF_DIR = BIOMASS_CHANGE_MAPS_DIR / "difference"
BIOMASS_CHANGE_MAPS_REL_DIFF_DIR = BIOMASS_CHANGE_MAPS_DIR / "relative_difference"

# Biomass stocks results
BIOMASS_STOCKS_DIR = RESULTS_DIR / "biomass_stocks"
BIOMASS_PER_FOREST_TYPE_DIR = BIOMASS_STOCKS_DIR / "per_forest_type"
BIOMASS_PER_LAND_COVER_DIR = BIOMASS_STOCKS_DIR / "per_land_cover"
BIOMASS_PER_HEIGHT_BIN_DIR = BIOMASS_STOCKS_DIR / "per_height_bin"

BIOMASS_PER_FOREST_TYPE_FILE = BIOMASS_MAPS_PER_FOREST_TYPE_DIR / "biomass_by_forest_type_year.csv"
BIOMASS_PER_GENUS_FILE = BIOMASS_MAPS_PER_FOREST_TYPE_DIR / "biomass_by_genus_year.csv"
BIOMASS_PER_CLADE_FILE = BIOMASS_MAPS_PER_FOREST_TYPE_DIR / "biomass_by_clade_year.csv"
BIOMASS_PER_LAND_COVER_FILE = BIOMASS_PER_LAND_COVER_DIR / "biomass_by_landcover_year.csv"
BIOMASS_PER_HEIGHT_BIN_FILE = BIOMASS_PER_HEIGHT_BIN_DIR / "biomass_by_height_year.csv"

BIOMASS_COUNTRY_TIMESERIES_DIR = BIOMASS_STOCKS_DIR / "country_time_series"

BIOMASS_MC_SAMPLES_DIR = PROCESSED_DIR / "biomass_stocks"

# Biomass transitions
BIOMASS_TRANSITIONS_DIR = RESULTS_DIR / "biomass_transitions"

# Carbon changes stats
CARBON_CHANGES_DIR = RESULTS_DIR / "carbon_changes"

# Climate biomass results
CLIMATE_BIOMASS_DATA_DIR = PROCESSED_DIR / "climate_biomass"
CLIMATE_BIOMASS_DATASET_FILE = CLIMATE_BIOMASS_DATA_DIR / "climate_biomass_ml_dataset.csv"
CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE = CLIMATE_BIOMASS_DATA_DIR / "climate_biomass_ml_dataset_spatial_folds.csv"

CLIMATE_BIOMASS_TEMP_DIR = CLIMATE_BIOMASS_DATA_DIR / "tmp"
CLIMATE_BIOMASS_TEMP_RESAMPLED_DIR = CLIMATE_BIOMASS_TEMP_DIR / "resampled"

CLIMATE_BIOMASS_RESULTS_DIR = RESULTS_DIR / "climate_biomass"
CLIMATE_BIOMASS_MODELS_DIR = CLIMATE_BIOMASS_RESULTS_DIR / "optimal_models"
CLIMATE_BIOMASS_SHAP_OUTPUT_DIR = CLIMATE_BIOMASS_RESULTS_DIR / "shap_interpretation"


# Figures
FIGURE_DIR = DATA_ROOT / 'figures' 
FIGURE_MAIN_DIR = FIGURE_DIR / 'main'
FIGURE_SI_DIR = FIGURE_DIR / 'si'


FOREST_TYPE_CSV = ""
FOREST_TYPE_CODES = ""

def create_all_directories():
    """Create all necessary directories in the data structure."""
    directories = [
        # Raw directories
        NFI4_DATABASE_DIR,
        FOREST_TYPE_MAPS_DIR,
        WOOD_DENSITY_DIR,
        LAND_COVER_DIR,
        CLIMATE_RAW_DIR,
        ALS_DATA_DIR,
        
        # Processed directories
        FOREST_INVENTORY_PROCESSED_DIR,
        SENTINEL2_MOSAICS_DIR,
        SENTINEL2_PROCESSED_DIR,
        ALS_CANOPY_HEIGHT_PROCESSED_DIR,
        HEIGHT_MAPS_10M_DIR,
        HEIGHT_MAPS_100M_DIR,
        HEIGHT_MAPS_TMP_RAW_DIR,
        ALLOMETRIES_DIR,
        BIOMASS_MAPS_RAW_DIR,
        BIOMASS_MAPS_PER_FOREST_TYPE_DIR,
        BIOMASS_MAPS_FULL_COUNTRY_DIR,
        BIOCLIM_HARMONIZED_DIR,
        BIOCLIM_VARIABLES_DIR,
        BIOCLIM_ANOMALIES_DIR,
        
        # Results directories
        ANALYSIS_OUTPUTS_DIR,
        TABLES_DIR,
        ML_OUTPUTS_DIR,
        
        # Model directories
        PRETRAINED_HEIGHT_MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
