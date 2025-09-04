import os
import pandas as pd
import re
import argparse
from datetime import datetime

def merge_csv_files(folder_path, output_folder, timestamp):
    pattern = r"val_results_single_.*\.csv"
    output_file = os.path.join(output_folder, f"val_results_single_{timestamp}.csv")
    csv_files = []

    for filename in os.listdir(folder_path):
        if re.match(pattern, filename):
            csv_files.append(os.path.join(folder_path, filename))
            print(f"Found: {filename}.")

    # Sort alphabetically
    csv_files.sort()

    if not csv_files:
        print("No matching CSV files found.")
        return
    
    col_names = ['id', 'scene_name', 'object', 'clicks', 'iou']
    merged_df = pd.concat(
        [pd.read_csv(f, sep=r"\s+", header=None, names=col_names) for f in csv_files],
        ignore_index=True
    )

    merged_df['id'] = 0
    merged_df['object'] = merged_df['object'] - 1

    merged_df.to_csv(output_file, index=False, sep=' ', header=False)
    print(f"Merged {len(csv_files)} csv files into '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Merge CSV files in chronological order.")
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing the CSV files')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder name')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merge_csv_files(args.input_folder, args.output_folder, timestamp)

if __name__ == "__main__":
    main()



