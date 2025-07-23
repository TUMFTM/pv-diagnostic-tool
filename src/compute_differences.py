import json
import pandas as pd
from pathlib import Path


def compute_differences(hourly_averages_dir, aggregated_data_dir, output_dir, verbose_output=False):
    """Compute differences between PVGIS reference hourly averages and actual aggregated operational data"""
    
    hourly_averages_path = Path(hourly_averages_dir)
    aggregated_data_path = Path(aggregated_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose_output:
        print(f"Reading hourly averages from: {hourly_averages_path}")
        print(f"Reading aggregated data from: {aggregated_data_path}")
        print(f"Writing differences to: {output_path}")

    # Process each plant
    avg_files = list(hourly_averages_path.glob("*_avg.json"))
    
    for avg_file in avg_files:
        plant_id = avg_file.stem.replace("_avg", "")
        output_file = output_path / f"{plant_id}_differences.csv"
        
        # Check if output already exists
        if output_file.exists():
            if verbose_output:
                print(f"  {plant_id}: differences already exist, skipping")
            continue
        
        # Find corresponding aggregated data file
        agg_file = aggregated_data_path / f"{plant_id}_agg-1h.csv"
        if not agg_file.exists():
            if verbose_output:
                print(f"  {plant_id}: no aggregated data found, skipping")
            continue
        
        if verbose_output:
            print(f"  Processing plant: {plant_id}")
        
        try:
            # Load hourly averages
            with open(avg_file, 'r') as f:
                avg_data = json.load(f)
            avg_values = avg_data[plant_id]
            
            # Load aggregated operational data
            df = pd.read_csv(agg_file, index_col=0, parse_dates=True)
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
            
            # Ensure timestamp is datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # Create DataFrame for reference values (8760 hours = 365 days × 24 hours)
            reference_df = pd.DataFrame(avg_values, columns=['AVG'])
            timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=8760, freq='H')
            reference_df['Timestamp'] = timestamps
            reference_df = reference_df[['Timestamp', 'AVG']]

            # Extract month, day, hour for matching
            for frame in [df, reference_df]:
                frame['month'] = frame['Timestamp'].dt.month
                frame['day'] = frame['Timestamp'].dt.day
                frame['hour'] = frame['Timestamp'].dt.hour

            # Handle leap year: map February 29th to February 28th
            feb_29_mask = (df['month'] == 2) & (df['day'] == 29)
            if feb_29_mask.any():
                df.loc[feb_29_mask, 'day'] = 28
                if verbose_output:
                    print(f"    Mapped {feb_29_mask.sum()} Feb 29th entries to Feb 28th")

            # Merge operational data with reference averages
            merged_df = df.merge(reference_df, on=['month', 'day', 'hour'], suffixes=('', '_ref'))

            # Clean up helper columns
            columns_to_drop = ['month', 'day', 'hour']
            if 'Timestamp_ref' in merged_df.columns:
                columns_to_drop.append('Timestamp_ref')
            merged_df = merged_df.drop(columns=columns_to_drop)

            # Compute differences (positive = underperforming, negative = overperforming)
            merged_df['Difference'] = merged_df['AVG'] - merged_df['PV(W)']
            
            # Add some basic statistics
            if verbose_output:
                mean_diff = merged_df['Difference'].mean()
                std_diff = merged_df['Difference'].std()
                print(f"    Mean difference: {mean_diff:.2f}W, Std: {std_diff:.2f}W")

            # Save results
            merged_df.to_csv(output_file, index=False)
            if verbose_output:
                print(f"    Saved differences to: {output_file}")
                
        except Exception as e:
            if verbose_output:
                print(f"    Error processing {plant_id}: {e}")
            continue

    if verbose_output:
        print("Differences computation complete.")
