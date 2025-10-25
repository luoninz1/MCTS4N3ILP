"""
Logging and result recording utilities for experiments.
"""

import os
import csv
import datetime
import pandas as pd


def record_to_table(env, terminal_num_points, start_time, end_time, time_used):
    """
    Record experiment results to a CSV table in the table_dir directory.
    Creates the CSV file if it doesn't exist, otherwise appends data.
    """
    # Get table directory and create if it doesn't exist
    table_dir = env.args.get('table_dir', 'results')
    os.makedirs(table_dir, exist_ok=True)

    # CSV file path
    csv_file = os.path.join(table_dir, 'experiment_results.csv')

    # Prepare data row
    data_row = {}

    # Add all args as columns
    for key, value in env.args.items():
        data_row[key] = value

    # Add additional required columns
    data_row['terminal_num_points'] = terminal_num_points
    data_row['session_name'] = env.session_name
    data_row['start_time'] = start_time
    data_row['end_time'] = end_time
    data_row['time_used'] = time_used

    # Check if CSV file exists
    file_exists = os.path.exists(csv_file)

    try:
        if file_exists:
            # Read existing CSV to get all possible columns
            existing_df = pd.read_csv(csv_file)
            all_columns = list(existing_df.columns)

            # Check if there are new columns from current data_row
            new_columns = [key for key in data_row.keys() if key not in all_columns]

            if new_columns:
                # Add new columns to existing DataFrame with null values
                for col in new_columns:
                    existing_df[col] = None
                    all_columns.append(col)

                # Save the updated DataFrame with new columns
                existing_df.to_csv(csv_file, index=False)

            # Create new row with all columns (fill missing with None)
            new_row = {}
            for col in all_columns:
                new_row[col] = data_row.get(col, None)

            # Append to existing CSV
            new_df = pd.DataFrame([new_row])
            new_df.to_csv(csv_file, mode='a', header=False, index=False)

        else:
            # Create new CSV file
            df = pd.DataFrame([data_row])
            df.to_csv(csv_file, index=False)

        print(f"Results recorded to: {csv_file}")

    except Exception as e:
        print(f"Error saving to CSV: {e}")
        # Fallback: try simple CSV writing
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_row)
            print(f"Results recorded to: {csv_file} (fallback method)")
        except Exception as e2:
            print(f"Error with fallback CSV writing: {e2}")
