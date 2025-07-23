import json
import numpy as np
from pathlib import Path

LEAP_YEARS = ["2008", "2012", "2016", "2020"]


def get_hourly_averages(input_dir, output_dir, verbose_output=False):
    """Calculate average hourly values across multiple years for each plant"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose_output:
        print(f"Reading hourly data from: {input_path}")
        print(f"Writing averages to: {output_path}")

    # Process each plant's hourly data file
    for hourly_file in input_path.glob("*.json"):
        plant_id = hourly_file.stem
        output_file = output_path / f"{plant_id}_avg.json"
        
        # Check if output already exists
        if output_file.exists():
            if verbose_output:
                print(f"  {plant_id}: averages already exist, skipping")
            continue
        
        if verbose_output:
            print(f"  Processing plant: {plant_id}")
        
        try:
            # Load hourly data
            with open(hourly_file, 'r') as f:
                data = json.load(f)
            
            plant_data = data[plant_id]
            years = sorted(plant_data.keys())
            
            if not years:
                if verbose_output:
                    print(f"    No years found for {plant_id}")
                continue
            
            if verbose_output:
                print(f"    Processing {len(years)} years: {years}")
            
            first_year = years[0]
            first_year_values = plant_data[first_year]
            
            # Handle leap year adjustment for the first year
            if first_year in LEAP_YEARS:
                first_year_values = first_year_values[:-24]  # Remove last day for leap years
                
            sum_arr = np.zeros_like(first_year_values, dtype=float)

            for year in years:
                values = plant_data[year]
                
                # Handle leap year adjustment
                if year in LEAP_YEARS:
                    values = values[:-24]  # Remove last day for leap years
                    if verbose_output:
                        print(f"      Adjusted leap year {year}: removed last 24 hours")
                
                sum_arr += np.array(values, dtype=float)

            # Calculate averages
            avg_values = (sum_arr / len(years)).tolist()

            # Save results
            with open(output_file, "w") as f:
                json.dump({plant_id: avg_values}, f, indent=4)
            
            if verbose_output:
                print(f"    Saved averages to {output_file}")
                
        except Exception as e:
            if verbose_output:
                print(f"    Error processing {plant_id}: {e}")
            continue
    
    if verbose_output:
        print("Hourly averages calculation complete.")