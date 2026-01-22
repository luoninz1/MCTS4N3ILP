import os
import pandas as pd

def combine_csvs():
    # Current directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, 'all_experiment_results.csv')
    
    # List to hold dataframes
    all_dfs = []
    
    # If the output file already exists, read it to preserve existing data
    if os.path.exists(output_file_path):
        try:
            existing_df = pd.read_csv(output_file_path)
            all_dfs.append(existing_df)
            print(f"Loaded existing data from {output_file_path} with shape {existing_df.shape}")
        except Exception as e:
            print(f"Error reading existing file {output_file_path}: {e}")
            
    # Iterate over subdirectories starting with "or"
    found_new_files = 0
    for folder_name in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder_name)
        
        # Check if it's a directory and starts with "or"
        if os.path.isdir(folder_path) and folder_name.startswith('or'):
            csv_file_path = os.path.join(folder_path, 'experiment_results.csv')
            
            if os.path.exists(csv_file_path):
                try:
                    df = pd.read_csv(csv_file_path)
                    if not df.empty:
                        all_dfs.append(df)
                        found_new_files += 1
                        print(f"Loaded {csv_file_path} with shape {df.shape}")
                except Exception as e:
                    print(f"Error reading {csv_file_path}: {e}")
    
    if not all_dfs:
        print("No data found to combine.")
        return

    # Concatenate all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates to ensure incremental update doesn't duplicate data
    # We compare all columns for duplication
    before_dedup = combined_df.shape[0]
    combined_df.drop_duplicates(inplace=True)
    after_dedup = combined_df.shape[0]
    
    print(f"Combined total rows: {before_dedup}. After dropping duplicates: {after_dedup}")
    
    # Save to CSV
    combined_df.to_csv(output_file_path, index=False)
    print(f"Successfully saved combined data to {output_file_path}")

if __name__ == "__main__":
    combine_csvs()
