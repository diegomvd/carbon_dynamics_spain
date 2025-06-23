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
MODELS_DIR = DATA_ROOT / "models"

# ==========================================
# RAW DATA DIRECTORIES (Recipe 0 inputs)
# ==========================================

# Forest inventory
FOREST_INVENTORY_RAW_DIR = RAW_DIR / "forest_inventory"
NFI4_DATABASE_DIR = FOREST_INVENTORY_RAW_DIR / "nfi4"
NFI4_SPECIES_CODE_FILE = FOREST_INVENTORY_RAW_DIR / "nfi4_codes.csv"


FOREST_TYPE_MAPS_DIR = RAW_DIR / "forest_type_maps"
FOREST_TYPE_CSV = ""
FOREST_TYPE_CODES = ""
FOREST_TYPE_MASKS_DIR = "" 

WOOD_DENSITY_DIR = RAW_DIR / "wood_density"
LAND_COVER_DIR = RAW_DIR / "land_cover"


# Climate data
CLIMATE_RAW_DIR = RAW_DIR / "climate"

# ALS LiDAR
ALS_CANOPY_HEIGHT_RAW_DIR = RAW_DIR / "als_canopy_height"
ALS_DATA_DIR = ALS_CANOPY_HEIGHT_RAW_DIR / "data"
ALS_METADATA_DIR = ALS_CANOPY_HEIGHT_RAW_DIR / "tile_metadata"
ALS_METADATA_UTM_29_DIR = ALS_METADATA_DIR / "utm_29"
ALS_METADATA_UTM_30_DIR = ALS_METADATA_DIR / "utm_30"
ALS_METADATA_UTM_31_DIR = ALS_METADATA_DIR / "utm_31"

ALS_CANOPY_HEIGHT_PROCESSED_DIR = PROCESSED_DIR / "als_canopy_height"

SPAIN_BOUNDARIES = RAW_DIR / "country_boundaries"

# ==========================================
# PROCESSED DATA DIRECTORIES
# ==========================================

# Recipe 0 outputs
FOREST_INVENTORY_PROCESSED_DIR = PROCESSED_DIR / "forest_inventory_processed"
SENTINEL2_MOSAICS_DIR = PROCESSED_DIR / "sentinel2"
SENTINEL2_PROCESSED_DIR = PROCESSED_DIR / "processed"
SENTINEL2_DOWNSAMPLED_DIR = SENTINEL2_PROCESSED_DIR / "downsampled"
SENTINEL2_MERGED_DIR = SENTINEL2_PROCESSED_DIR / "merged"


ALS_CANOPY_HEIGHT_PROCESSED_DIR = PROCESSED_DIR / "als_canopy_height_processed"

# Recipe 1 outputs  
HEIGHT_MAPS_DIR = PROCESSED_DIR / "height_maps"
HEIGHT_MAPS_10M_DIR = HEIGHT_MAPS_DIR / "10m"
HEIGHT_MAPS_100M_DIR = HEIGHT_MAPS_DIR / "100m"
HEIGHT_MAPS_TMP_120KM_DIR = HEIGHT_MAPS_DIR / "tmp" / "merged_120km"
HEIGHT_MAPS_TMP_RAW_DIR = HEIGHT_MAPS_DIR / "tmp" / "raw"

# Recipe 2 outputs
ALLOMETRIES_DIR = PROCESSED_DIR / "allometries"
BIOMASS_MAPS_DIR = PROCESSED_DIR / "biomass_maps"
BIOMASS_MAPS_RAW_DIR = BIOMASS_MAPS_DIR / "raw"
BIOMASS_MAPS_PER_FOREST_TYPE_DIR = BIOMASS_MAPS_DIR / "per_forest_type"
BIOMASS_MAPS_FULL_COUNTRY_DIR = BIOMASS_MAPS_DIR / "full_country"

BIOMASS_MAPS_RELDIFF_DIR = ""

# Climate processing
CLIMATE_DIR = PROCESSED_DIR / "climate"
CLIMATE_HARMONIZED_DIR = BIOCLIM_DIR / "harmonized"
BIOCLIM_VARIABLES_DIR = BIOCLIM_DIR / "bioclimatic"
BIOCLIM_ANOMALIES_DIR = BIOCLIM_DIR / "anomalies"

CLIMATE_BIOMASS_MODELS_DIR = ""
CLIMATE_BIOMASS_SHAP_OUTPUT_DIR = ""

# ==========================================
# RESULTS DIRECTORIES (Recipe 3 outputs)
# ==========================================

HEIGHT_MAPS_BIN_MASKS_DIR = ""

ANALYSIS_OUTPUTS_DIR = RESULTS_DIR / "analysis_outputs"
TABLES_DIR = RESULTS_DIR / "tables"
ML_OUTPUTS_DIR = RESULTS_DIR / "ml_outputs"

# ==========================================
# MODEL DIRECTORIES
# ==========================================

PRETRAINED_HEIGHT_MODELS_DIR = DATA_ROOT / "pretrained_height_models"
ML_MODELS_DIR = MODELS_DIR / "ml_models"

# ==========================================
# SPECIFIC FILES
# ==========================================

# Forest inventory files
FOREST_INVENTORY_CODES_FILE = FOREST_INVENTORY_RAW_DIR / "nfi4_codes.csv"
FOREST_TYPES_TIERS_FILE = FOREST_INVENTORY_RAW_DIR / "Forest_Types_Tiers.csv"
WOOD_DENSITY_FILE = WOOD_DENSITY_DIR / "wood_density_database.csv"

# Land cover
CORINE_LAND_COVER_FILE = LAND_COVER_DIR / "corine_land_cover.tif"

# Country boundaries  
COUNTRY_BOUNDS_FILE = FOREST_TYPE_MAPS_DIR / "gadm41_ESP_0.shp"

# Allometry outputs
FITTED_PARAMETERS_FILE = ALLOMETRIES_DIR / "fitted_parameters.csv"
BGB_RATIOS_FILE = ALLOMETRIES_DIR / "bgb_ratios.csv"
FITTING_SUMMARY_FILE = ALLOMETRIES_DIR / "fitting_summary.csv"

# Height model checkpoint
HEIGHT_MODEL_CHECKPOINT_FILE = PRETRAINED_HEIGHT_MODELS_DIR / "height_model_checkpoint.pkl"

CLIMATE_BIOMASS_DATASET_FILE = ""
CLIMATE_BIOMASS_DATASET_CLUSTERS_FILE = ""


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

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
        ML_MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_data_root():
    """Get the data root directory."""
    return DATA_ROOT

def set_data_root(new_root):
    """
    Set a different data root (for testing or custom installations).
    
    Args:
        new_root: New root directory path
    """
    global DATA_ROOT, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR
    global NFI4_DATABASE_DIR, FOREST_TYPE_MAPS_DIR, WOOD_DENSITY_DIR
    global SENTINEL2_MOSAICS_DIR, HEIGHT_MAPS_10M_DIR  # ... etc
    
    # This would update all the paths - can implement if needed
    # For now, this is just a placeholder for future flexibility
    raise NotImplementedError("Dynamic data root changing not yet implemented")