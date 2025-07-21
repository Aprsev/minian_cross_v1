import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_yrc_std_and_save_separately(
    yra_csv_path1: str,
    c_csv_path1: str,
    output_std_csv_path1: str,
    yra_csv_path2: str,
    c_csv_path2: str,
    output_std_csv_path2: str,
    plot_unit_id: int = None
):
    """
    Calculates the standard deviation of YrA-C for corresponding unit_ids in two sets of data files,
    handling inconsistent frame counts by truncating to the shorter length.
    Standard deviation results for each dataset are saved to separate CSV files.
    Optional: Plots YrA, C, and YrA-C difference traces for a specified unit_id.

    Args:
        yra_csv_path1 (str): CSV file path for the first set of YrA data.
        c_csv_path1 (str): CSV file path for the first set of C data.
        output_std_csv_path1 (str): CSV file path to save the standard deviation results for the first set.
        yra_csv_path2 (str): CSV file path for the second set of YrA data.
        c_csv_path2 (str): CSV file path for the second set of C data.
        output_std_csv_path2 (str): CSV file path to save the standard deviation results for the second set.
        plot_unit_id (int, optional): If provided, plots YrA, C, and Diff traces for this unit_id
                                      within its respective dataset. Defaults to no plotting (None).
    """

    print("--- Starting YrA-C Standard Deviation Calculation (Separate Saves) ---")

    # Define an inner function to process a single dataset, save Std, and optionally plot traces
    def process_and_save_single_data_set(yra_path, c_path, data_set_name, output_path, plot_id):
        print(f"\nProcessing {data_set_name} dataset:")
        print(f"   YrA file: {yra_path}")
        print(f"   C file: {c_path}")
        print(f"   Output file: {output_path}")

        try:
            yra_raw_df = pd.read_csv(yra_path)
            c_raw_df = pd.read_csv(c_path)
            print("   CSV files loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: File not found at {e}. Skipping this dataset.")
            return
        except Exception as e:
            print(f"Error: An unknown error occurred while loading files: {e}. Skipping this dataset.")
            return

        # --- Key Modification: Convert long format data to wide format ---
        # Assuming data column names are 'C' and 'YrA'
        # Assuming unit_id is in the 'unit_id' column, and frame count is in the 'frame' column

        # Check for required columns
        required_cols_c = ['unit_id', 'frame', 'C']
        required_cols_yra = ['unit_id', 'frame', 'YrA'] # Assuming YrA file has a 'YrA' column name

        if not all(col in c_raw_df.columns for col in required_cols_c):
            print(f"Warning: {c_path} is missing required columns (unit_id, frame, C). Skipping this dataset.")
            return
        if not all(col in yra_raw_df.columns for col in required_cols_yra):
            print(f"Warning: {yra_path} is missing required columns (unit_id, frame, YrA). Skipping this dataset.")
            return

        try:
            # Convert data types to avoid errors in subsequent calculations
            c_raw_df['unit_id'] = c_raw_df['unit_id'].astype(int)
            c_raw_df['frame'] = c_raw_df['frame'].astype(int)
            c_raw_df['C'] = pd.to_numeric(c_raw_df['C'], errors='coerce')

            yra_raw_df['unit_id'] = yra_raw_df['unit_id'].astype(int)
            yra_raw_df['frame'] = yra_raw_df['frame'].astype(int)
            yra_raw_df['YrA'] = pd.to_numeric(yra_raw_df['YrA'], errors='coerce')

            # Convert long format data to wide format, one row per unit_id, one column per frame
            # fillna(0) to fill NaNs with 0, ensuring data continuity
            c_df_pivoted = c_raw_df.pivot_table(index='unit_id', columns='frame', values='C').fillna(0)
            yra_df_pivoted = yra_raw_df.pivot_table(index='unit_id', columns='frame', values='YrA').fillna(0)

            print("   Data successfully converted to wide format.")

        except Exception as e:
            print(f"Error: An error occurred while converting data format: {e}. Skipping this dataset.")
            return

        # --- Resuming original logic, now operating on wide-format DataFrames ---
        
        # Get common unit_ids (now in the index)
        common_unit_ids = sorted(list(set(yra_df_pivoted.index).intersection(c_df_pivoted.index)))
        if not common_unit_ids:
            print("Warning: No common unit_ids found in both YrA and C files. Skipping this dataset.")
            return

        unit_results = []
        
        # Process each unit_id
        for unit_id in common_unit_ids:
            # Extract time series data (now entire row)
            yra_values = yra_df_pivoted.loc[unit_id].values
            c_values = c_df_pivoted.loc[unit_id].values
            
            if len(yra_values) == 0 or len(c_values) == 0:
                print(f"Warning: Unit ID {unit_id} has empty YrA or C data in {data_set_name}. Skipping.")
                continue

            # Get actual frame counts
            frames_yra = len(yra_values)
            frames_c = len(c_values)

            # Find the shorter frame count
            min_frames = min(frames_yra, frames_c)

            # Truncate the longer trace
            yra_truncated = yra_values[:min_frames]
            c_truncated = c_values[:min_frames]

            # Calculate the difference YrA-C
            diff = yra_truncated - c_truncated

            # Calculate standard deviation
            std_val = np.std(diff)
            
            print(f"   {data_set_name} - Unit ID {unit_id}: YrA-C Std = {std_val:.4f} (Used Frames: {min_frames})")
            unit_results.append({'unit_id': unit_id, 'YRA_C_Std': std_val, 'FramesUsed': min_frames})

            # --- Plotting section ---
            if plot_id is not None and unit_id == plot_id:
                print(f"   Plotting traces for {data_set_name} Unit ID {unit_id}...")
                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(min_frames), yra_truncated, label='YrA', color='blue', alpha=0.7)
                plt.plot(np.arange(min_frames), c_truncated, label='C', color='red', alpha=0.7)
                plt.plot(np.arange(min_frames), diff, label='YrA - C (Difference)', color='green', linestyle='--', alpha=0.8)
                plt.title(f'{data_set_name} - Unit ID {unit_id}: YrA, C and Difference Traces\n'
                                f'Std of Difference: {std_val:.4f}')
                plt.xlabel('Frame')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                print(f"   Traces for Unit ID {unit_id} plotted.")
            # --- End plotting section ---
        
        # Save results to CSV file
        if unit_results:
            output_df = pd.DataFrame(unit_results)
            output_dir = os.path.dirname(output_path)
            # Ensure output directory exists; create it if not. Use exist_ok=True to avoid error if exists.
            if output_dir: # Only try to create if output_dir is not empty string (i.e., not current directory)
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nCreated output directory: {output_dir}")
                
            output_df.to_csv(output_path, index=False)
            print(f"\n{data_set_name} results saved to: {output_path}")
        else:
            print(f"\nNo valid data to calculate and save for {data_set_name}.")

    # --- Process and save the first dataset ---
    process_and_save_single_data_set(yra_csv_path1, c_csv_path1, "Dataset 1", output_std_csv_path1, plot_unit_id)

    # --- Process and save the second dataset ---
    process_and_save_single_data_set(yra_csv_path2, c_csv_path2, "Dataset 2", output_std_csv_path2, plot_unit_id)

    print("\n--- All Standard Deviation Calculation and Saving Tasks Complete ---")