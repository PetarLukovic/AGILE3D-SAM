import os
import pandas as pd
import re
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# eg. python3 -m plukovic.merge_results --input_folder /home/plukovic/results --output_folder /home/plukovic/interactive_segmentation/AGILE3D-SAM/results
# eg. python3 -m plukovic.merge_results --input_folder /home/plukovic/results --output_folder /home/plukovic/interactive_segmentation/AGILE3D-SAM/

def merge_slurm_files(folder_path, output_folder, timestamp):
    pattern = r"slurm-(\d{8})\.out"
    output_file = output_folder + "/val_results_single_" + timestamp + ".txt"
    slurm_files = []

    for filename in os.listdir(folder_path):
        match = re.search(pattern, filename)
        if match:
            datetime_str = match.group(1)
            slurm_files.append((datetime_str, os.path.join(folder_path, filename)))

    slurm_files.sort()

    if not slurm_files:
        print("No matching slurm files found.")
        return
    
    '''
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for _, file_path in slurm_files:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write('\n')
    '''

    print(f"Merged {len(slurm_files)} slurm files into '{output_file}'.")

def merge_csv_files(folder_path, output_folder, timestamp, num_clicks = 1):
    pattern = r"val_results_single_(\d{8}_\d{6})\.csv"
    output_file = output_folder + "/val_results_single_" + timestamp + ".csv"
    csv_files = []

    for filename in os.listdir(folder_path):
        match = re.search(pattern, filename)
        if match:
            datetime_str = match.group(1)
            csv_files.append((datetime_str, os.path.join(folder_path, filename)))

    csv_files.sort()

    if not csv_files:
        print("No matching CSV files found.")
        return
    
    col_names = ['id', 'scene_name', 'object', 'clicks', 'iou', 'all_clicks']
    merged_df = pd.concat(
        [pd.read_csv(f[1], delim_whitespace=True, header=None, names=col_names) for f in csv_files],
        ignore_index=True
    )

    scale = num_clicks + 1
    for i in range(len(merged_df) // (num_clicks + 1)):
        for j in range(num_clicks+1):
            merged_df.iat[i * scale + j, 0] = i

    merged_df.iloc[:, 2] = merged_df.iloc[:, 2] - 1

    data = merged_df.iloc[:, 4]
    filtered_data = data[(data != 0) & (~data.isna())]
    counts, bins = np.histogram(filtered_data, bins=10)
    probabilities = counts / counts.sum() 
    plt.bar(bins[:-1], probabilities, width=np.diff(bins), edgecolor='black', align='edge')
    #plt.hist(filtered_data, bins=30, edgecolor='black')
    plt.title('Histogram of iou')
    plt.xlabel('IOU')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.show()
    plt.savefig(output_folder + 'iou_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

    data = merged_df.iloc[:, 5]
    filtered_data = data[(data != 0) & (~data.isna())]
    counts, bins = np.histogram(filtered_data, bins=2)
    probabilities = counts / counts.sum() 
    plt.bar(bins[:-1], probabilities, width=np.diff(bins), edgecolor='black', align='edge')
    #plt.hist(filtered_data, bins=30, edgecolor='black')
    plt.title('Histogram of num_clicks')
    plt.xlabel('num_clicks')
    plt.ylabel('Frequency')
    plt.grid(True)
    #plt.show()
    plt.savefig(output_folder + 'clicks_hist.png', dpi=300, bbox_inches='tight')
    plt.close()


    #merged_df.to_csv(output_file, index=False, sep=' ', header=False)
    print(f"Merged {len(csv_files)} csv files into '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Merge CSV files in chronological order.")
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing the CSV files')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder name')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merge_csv_files(args.input_folder, args.output_folder, timestamp)
    merge_slurm_files(args.input_folder, args.output_folder, timestamp)

if __name__ == "__main__":
    main()



