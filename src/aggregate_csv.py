import pandas as pd
from pathlib import Path

# Default aggregation functions for known column types
DEFAULT_AGG_FUNCS = {
    'PV(W)': 'sum',
    'SOC(%)': 'mean',
    'Battery(W)': 'sum',
    'Meter(W)': 'sum',
    'Load(W)': 'sum',
    'MPP1(A)': 'mean',
    'MPP2(A)': 'mean',
    'MPP1(V)': 'mean',
    'MPP2(V)': 'mean',
    'IT1(°C)': 'mean'
}


def aggregate_logs(raw_path, output_dir, verbose_output=False):
    """Aggregate CSV log files from hourly operational data"""
    
    raw_path = Path(raw_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(raw_path.glob("*.csv"))
    
    if verbose_output:
        print(f"Found {len(csv_files)} CSV files in {raw_path}")
        print(f"Writing aggregated data to: {output_path}")

    # Process each CSV file
    for csv_file in csv_files:
        plant_id = csv_file.stem
        output_file = output_path / f'{plant_id}_agg-1h.csv'
        
        # Check if output already exists
        if output_file.exists():
            if verbose_output:
                print(f"  {plant_id}: aggregated data already exists, skipping")
            continue
        
        if verbose_output:
            print(f"  Processing: {csv_file.name}")
        
        try:
            # Read CSV file with datetime index
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Create aggregation functions based on available columns
            agg_funcs = {}
            for col in df.columns:
                if col in DEFAULT_AGG_FUNCS:
                    agg_funcs[col] = DEFAULT_AGG_FUNCS[col]
                else:
                    # Default to mean for unknown columns
                    agg_funcs[col] = 'mean'
                    if verbose_output:
                        print(f"    Unknown column '{col}', using 'mean' aggregation")

            # Aggregate the data by hour
            df_hourly = df.resample('1h').agg(agg_funcs)
            
            # Save the aggregated data
            df_hourly.to_csv(output_file)
            if verbose_output:
                print(f"    Saved aggregated data to: {output_file}")
                
        except Exception as e:
            if verbose_output:
                print(f"    Error processing {csv_file.name}: {e}")
            continue

    if verbose_output:
        print("Operational data aggregation complete.")
