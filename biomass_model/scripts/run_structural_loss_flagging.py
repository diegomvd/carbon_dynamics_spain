#!/usr/bin/env python3
"""
Structural Loss Flagging Script

Computes probability maps distinguishing structural biomass loss from stress.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomass_model.core.structural_losses import (
    process_tile_structural_loss,
    process_edge_case_2024
)
from shared_utils import setup_logging
from shared_utils.central_data_paths_constants import (
    BIOMASS_MAPS_TILED_DIR,
    BIOMASS_MAPS_TILED_DIR,
    BIOMASS_MAPS_STRUCTURAL_LOSS_DIR
)


def main():
    """Main entry point."""
    logger = setup_logging(level='INFO', component_name='structural_loss_flagging')
    
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    tile_pattern = '*{year}*.tif'
    
    logger.info(f"Processing years: {years}")
    
    for year in years:
        logger.info(f"Processing year {year}...")
        process_tile_structural_loss(
            tile_pattern,
            year,
            BIOMASS_MAPS_TILED_DIR / "AGBD_mean" ,
            BIOMASS_MAPS_TILED_DIR / "AGBD_uncertainty" ,
            BIOMASS_MAPS_STRUCTURAL_LOSS_DIR / str(year)
        )
    
    logger.info("Processing 2024 edge case...")
    process_edge_case_2024(
        tile_pattern,
        BIOMASS_MAPS_TILED_DIR / "AGBD_mean",
        BIOMASS_MAPS_TILED_DIR / "AGBD_uncertainty",
        BIOMASS_MAPS_STRUCTURAL_LOSS_DIR / "2024"
    )
    
    logger.info("Done")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
