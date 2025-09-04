import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.evaluator_SO import EvaluatorSO

def save_histogram(data, filename, xlabel, ylabel="Frequency", title="", bins=20, range=None):
    plt.hist(data, bins=bins, range=range, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to {filename}")

def iou_histograms(model_path: str = ''):
    csv_files = [f for f in os.listdir(model_path) if f.startswith("val_results_single_") and f.endswith(".csv")]
    if not csv_files:
        print(f"No results files found in {model_path}")
        return
    
    csv_files.sort()
    results_file = os.path.join(model_path, csv_files[-1])
    print(f"Loading results from {results_file}")

    col_names = ['id', 'scene_name', 'object', 'clicks', 'iou']
    df = pd.read_csv(results_file, sep=r"\s+", header=None, names=col_names)

    # 1. IoU Histogram
    save_histogram(df['iou'], os.path.join(model_path, "iou_histogram.png"), xlabel="IoU", title="IoU Histogram", bins=20, range=(0,1))

    # 2. Clicks Histogram
    save_histogram(df['clicks'], os.path.join(model_path, "clicks_histogram.png"), xlabel="Clicks", title="Clicks Histogram", bins=20)

    # 3. Click Efficiency Histogram (IoU / clicks)
    efficiency = df['iou'] / df['clicks'].replace(0, 1)  # avoid division by zero
    save_histogram(efficiency, os.path.join(model_path, "click_efficiency_histogram.png"), xlabel="IoU / Click", title="Click Efficiency Histogram", bins=20)

    # 4. Cumulative IoU Histogram (CDF)
    sorted_iou = df['iou'].sort_values()
    cdf = sorted_iou.rank(method='average') / len(sorted_iou)
    plt.plot(sorted_iou, cdf, marker='.', linestyle='none')
    plt.xlabel("IoU")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative IoU Histogram (CDF)")
    plt.grid(True, linestyle='--', alpha=0.7)
    cdf_file = os.path.join(model_path, "cumulative_iou_histogram.png")
    plt.savefig(cdf_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative IoU histogram saved to {cdf_file}")

    # 5. Thresholded IoU Histogram (<0.5, 0.5-0.8, ≥0.8)
    thresholds = [0, 0.5, 0.8, 1.0]
    counts = [((df['iou'] >= thresholds[i]) & (df['iou'] < thresholds[i+1])).sum() for i in range(len(thresholds)-1)]
    plt.bar(["<0.5", "0.5-0.8", ">=0.8"], counts, edgecolor='black')
    plt.xlabel("IoU Range")
    plt.ylabel("Frequency")
    plt.title("Thresholded IoU Histogram")
    thresh_file = os.path.join(model_path, "thresholded_iou_histogram.png")
    plt.savefig(thresh_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Thresholded IoU histogram saved to {thresh_file}")

def iou_statistics(model_path: str = ''):
    csv_files = [f for f in os.listdir(model_path) if f.startswith("val_results_single_") and f.endswith(".csv")]
    if not csv_files:
        print(f"No results files found in {model_path}")
        return
    
    csv_files.sort()
    results_file = os.path.join(model_path, csv_files[-1])
    print(f"Loading results from {results_file}")

    col_names = ['id', 'scene_name', 'object', 'clicks', 'iou']
    df = pd.read_csv(results_file, sep=r"\s+", header=None, names=col_names)

    stats = {
        "Mean IoU": df['iou'].mean(),
        "Median IoU": df['iou'].median(),
        "Std Dev IoU": df['iou'].std(),
        "Min IoU": df['iou'].min(),
        "Max IoU": df['iou'].max(),
        "Average Clicks": df['clicks'].mean(),
        "Min Clicks": df['clicks'].min(),
        "Max Clicks": df['clicks'].max(),
        "Average Efficiency (IoU/Click)": (df['iou'] / df['clicks'].replace(0,1)).mean()
    }

    output_file = os.path.join(model_path, "iou_statistics.txt")
    with open(output_file, "w") as f:
        f.write("IoU Statistics\n")
        f.write("====================\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Statistics saved to {output_file}")


def agile_3d_results(model_path: str = '', max_num_clicks: int = 20):
    base_folder = os.path.dirname(os.path.dirname(model_path))
    
    dataset = "scannet40"
    val_list = os.path.join(base_folder, "data/scannet/single/object_ids.npy")
    val_list_classes = os.path.join(base_folder, "data/scannet/single/object_classes.txt")
    
    csv_files = [f for f in os.listdir(model_path) if f.startswith("val_results_single_") and f.endswith(".csv")]
    if not csv_files:
        print(f"No results files found in {model_path}")
        return
    csv_files.sort()
    results_file = os.path.join(model_path, csv_files[-1])
    print(f"Evaluating results from {results_file}")
    
    evaluator = EvaluatorSO(dataset, val_list, val_list_classes, results_file,
                            [0.5, 0.65, 0.8, 0.85, 0.9], max_num_clicks)
    
    results_dict = evaluator.eval_results()

    output_file = os.path.join(model_path, "agile_3d_results.txt")
    with open(output_file, "w") as f:
        f.write("AGILE3D Evaluation Results\n")
        f.write("=========================\n")
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"Evaluation results saved to {output_file}")
    
    return results_dict

def main():
    parser = argparse.ArgumentParser(description="Generate IoU histograms and statistics from results CSV.")
    parser.add_argument('--model_path', type=str, required=True, help='Subfolder in ./results containing the CSV files')
    parser.add_argument('--max_num_clicks', type=int, default=20, help='Maximum number of clicks to consider for evaluation')
    args = parser.parse_args()

    agile_3d_results(args.model_path, args.max_num_clicks)
    iou_histograms(args.model_path)
    iou_statistics(args.model_path)

if __name__ == "__main__":
    main()
