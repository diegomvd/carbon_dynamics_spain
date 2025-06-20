"""
Harmonized missing mosaic detection and analysis for Sentinel-2 processing.

This script analyzes mosaic output directories to identify missing files:
1. Scans directory for mosaic files matching expected patterns
2. Identifies which location-year combinations are missing
3. Generates reports and missing file paths
4. Provides statistics on processing completeness

Author: Diego Bengochea
"""

import os
from collections import defaultdict
import re
from s2_utils import setup_logging, load_config


def analyze_missing_files(directory_path):
    """
    Analyze directory to identify which location files are missing in specific years
    and generate hypothetical paths for missing files.
    
    Args:
        directory_path (str): Path to the directory containing the raster files
    
    Returns:
        tuple: (missing_files_dict, all_years_list, missing_file_paths_list)
    """
    logger = setup_logging()
    logger.info(f"Analyzing missing files in directory: {directory_path}")
    
    # Get all files in the directory
    files = os.listdir(directory_path)
    
    # Extract years and create a mapping of locations to years
    year_pattern = r'S2_summer_mosaic_(\d{4})'  # Pattern to match 4-digit years
    location_to_years = defaultdict(set)
    all_years = set()
    
    # Dictionary to store the full filenames for each location and year
    location_files = defaultdict(dict)
    
    for file in files:
        # Find the year in the filename
        year_match = re.search(year_pattern, file)
        if year_match:
            year = year_match.group(1)
            if year == '2024':
                continue
            all_years.add(year)
            
            # Remove the year from the filename to get the location identifier
            location = re.sub(year_pattern, '', file)
            location_to_years[location].add(year)
            location_files[location][year] = file

    logger.info(f"Found {len(files)} total files")
    logger.info(f"Valid years found: {sorted(all_years)}")
    logger.info(f"Unique locations: {len(location_to_years)}")

    # Find missing years for each location and generate missing file paths
    missing_files = {}
    missing_file_paths = []
    
    for location, years in location_to_years.items():
        missing_years = all_years - years
        if missing_years:
            # Get an example existing file for this location to use as a template
            example_year = next(iter(years))
            example_filename = location_files[location][example_year]
            
            # Store missing years info
            missing_files[location] = {
                'missing_years': sorted(list(missing_years)),
                'existing_files': {
                    year: location_files[location][year]
                    for year in years
                }
            }
            
            # Generate hypothetical paths for missing files
            for missing_year in missing_years:
                # Replace the example year with the missing year in the filename
                missing_filename = re.sub(
                    example_year,
                    missing_year,
                    example_filename
                )
                missing_file_paths.append(os.path.join(directory_path, missing_filename))
    
    return missing_files, sorted(list(all_years)), sorted(missing_file_paths)


def print_report(missing_files, all_years, missing_file_paths):
    """
    Print a formatted report of missing files.
    
    Args:
        missing_files (dict): Missing files information
        all_years (list): All years found in directory
        missing_file_paths (list): List of missing file paths
    """
    logger = setup_logging()
    
    logger.info(f"\nAnalysis Report")
    logger.info(f"Years found in directory: {', '.join(all_years)}")
    logger.info(f"\nLocations with missing files:")
    logger.info("-" * 50)
    
    if not missing_files:
        logger.info("No missing files found. All locations present in all years.")
        return
        
    for location, data in missing_files.items():
        logger.info(f"\nLocation pattern: {location}")
        logger.info(f"Missing in years: {', '.join(data['missing_years'])}")
        logger.info("Existing files:")
        for year, filename in data['existing_files'].items():
            logger.info(f"  {year}: {filename}")
    
    logger.info("\nFirst 10 missing file paths:")
    logger.info("-" * 50)
    for i, path in enumerate(missing_file_paths[:10]):
        logger.info(f"{i+1}. {path}")
    
    if len(missing_file_paths) > 10:
        logger.info(f"... and {len(missing_file_paths) - 10} more missing files")


def save_missing_file_paths(missing_file_paths, output_file="missing_file_paths.txt"):
    """
    Save missing file paths to a text file.
    
    Args:
        missing_file_paths (list): List of missing file paths
        output_file (str): Output file name
    """
    logger = setup_logging()
    
    with open(output_file, 'w') as f:
        for path in missing_file_paths:
            f.write(f"{path}\n")
    
    logger.info(f"Missing file paths saved to: {output_file}")


def calculate_statistics(missing_files, all_years, location_to_years):
    """
    Calculate processing statistics.
    
    Args:
        missing_files (dict): Missing files information
        all_years (list): All years found
        location_to_years (dict): Mapping of locations to years
        
    Returns:
        dict: Statistics dictionary
    """
    total_locations = len(location_to_years)
    total_expected = total_locations * len(all_years)
    total_existing = sum(len(years) for years in location_to_years.values())
    total_missing = len([path for data in missing_files.values() for _ in data['missing_years']])
    completeness_rate = (total_existing / total_expected) * 100 if total_expected > 0 else 0
    
    return {
        'total_locations': total_locations,
        'total_years': len(all_years),
        'total_expected': total_expected,
        'total_existing': total_existing,
        'total_missing': total_missing,
        'completeness_rate': completeness_rate
    }


def main():
    """Main missing files analysis pipeline."""
    logger = setup_logging()
    logger.info("Starting Sentinel-2 missing mosaics analysis...")
    
    try:
        # Load configuration using shared function
        config = load_config()
        
        # Get directory path from config
        directory_path = config['paths']['output_dir']
        
        logger.info(f"Analyzing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return
        
        # Analyze missing files
        missing_files, all_years, missing_file_paths = analyze_missing_files(directory_path)
        
        # Calculate statistics  
        location_to_years = defaultdict(set)
        files = os.listdir(directory_path)
        year_pattern = r'S2_summer_mosaic_(\d{4})'
        
        for file in files:
            year_match = re.search(year_pattern, file)
            if year_match:
                year = year_match.group(1)
                if year != '2024':
                    location = re.sub(year_pattern, '', file)
                    location_to_years[location].add(year)
        
        stats = calculate_statistics(missing_files, all_years, location_to_years)
        
        # Print comprehensive report
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETENESS ANALYSIS")
        logger.info("="*60)
        logger.info(f"Total locations: {stats['total_locations']}")
        logger.info(f"Years processed: {', '.join(all_years)}")
        logger.info(f"Expected files: {stats['total_expected']}")
        logger.info(f"Existing files: {stats['total_existing']}")
        logger.info(f"Missing files: {stats['total_missing']}")
        logger.info(f"Completeness rate: {stats['completeness_rate']:.1f}%")
        
        # Print detailed report
        print_report(missing_files, all_years, missing_file_paths)
        
        # Save missing file paths
        if missing_file_paths:
            save_missing_file_paths(missing_file_paths)
        
        logger.info("\nMissing files analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
