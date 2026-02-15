import os
import glob
import pandas as pd
import argparse

def merge_results(directory="."):
    """
    Merges all experiment_results_*.csv files in the specified directory
    into a single experiment_results.csv file.
    """
    # Find all CSV files matching the pattern
    pattern = os.path.join(directory, "experiment_results_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No result files found matching {pattern}")
        return

    print(f"Found {len(files)} result files. Merging...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Extract Job ID from filename: experiment_results_12345.csv
            filename = os.path.basename(f)
            # Split by '_' and take the last part, then remove .csv extension
            # Assumes format experiment_results_{JOBID}.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                job_id = parts[-1] 
                # Add 'job_id' column to the beginning
                df.insert(0, 'job_id', job_id)
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No valid data found to merge.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort if desired (e.g., by n, then timestamp)
    if 'n' in combined_df.columns:
        combined_df.sort_values(by=['n'], inplace=True)

    # Save to main results file
    output_file = os.path.join(directory, "experiment_results_merged.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully merged results into {output_file}")
    
    # Optional: Delete individual files?
    # for f in files:
    #     os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge experiment result CSVs.")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing results")
    args = parser.parse_args()
    
    merge_results(args.dir)
