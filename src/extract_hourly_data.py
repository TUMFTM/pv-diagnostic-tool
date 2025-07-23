import json
from pathlib import Path


def extract_hourly_data(input_dir, output_dir, verbose_output=False):
    """Extract hourly power values from PVGIS JSON files"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose_output:
        print(f"Reading PVGIS data from: {input_path}")
        print(f"Writing hourly data to: {output_path}")

    # Process each plant directory
    for plant_dir in input_path.iterdir():
        if not plant_dir.is_dir():
            continue
            
        plant_id = plant_dir.name
        output_file = output_path / f"{plant_id}.json"
        
        # Check if output already exists
        if output_file.exists():
            if verbose_output:
                print(f"  {plant_id}: hourly data already exists, skipping")
            continue
        
        if verbose_output:
            print(f"  Processing plant: {plant_id}")

        plant_results = {}
        
        # Process all JSON files for this plant
        for json_file in plant_dir.glob(f"{plant_id}_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract year from filename
                year = json_file.stem.split('_')[-1]
                
                # Extract hourly values
                if "outputs" in data and "hourly" in data["outputs"]:
                    hourly_vals = [h["P"] for h in data["outputs"]["hourly"]]
                    plant_results[year] = hourly_vals
                    
                    if verbose_output:
                        print(f"    Year {year}: {len(hourly_vals)} hourly values")
                else:
                    if verbose_output:
                        print(f"    Year {year}: no hourly data found")
                        
            except Exception as e:
                if verbose_output:
                    print(f"    Error processing {json_file}: {e}")
                continue
        
        # Save extracted data
        if plant_results:
            with open(output_file, 'w') as f:
                json.dump({plant_id: plant_results}, f, indent=4)
            if verbose_output:
                print(f"    Saved {len(plant_results)} years to {output_file}")
        else:
            if verbose_output:
                print(f"    No valid data found for {plant_id}")
    
    if verbose_output:
        print("Hourly data extraction complete.")
