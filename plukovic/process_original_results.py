import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path = "/home/plukovic/interactive_segmentation/AGILE3D-SAM/results/our_single_scannet.csv"
path_out = "/home/plukovic/interactive_segmentation/AGILE3D-SAM/results/our_single_scannet_filt.csv"
output_folder = "/home/plukovic/interactive_segmentation/AGILE3D-SAM/"

col_names = ['id', 'scene_name', 'object', 'clicks', 'iou']
df = pd.read_csv(path, delim_whitespace=True, header=None, names=col_names)
df = df[df['clicks'] <= 1]

df.to_csv(path_out, index=False, sep=' ', header=False)

data = df.iloc[:, 4]
filtered_data = data[(data != 0) & (~data.isna())]
counts, bins = np.histogram(filtered_data, bins=10)
probabilities = counts / counts.sum() 
plt.bar(bins[:-1], probabilities, width=np.diff(bins), edgecolor='black', align='edge')
#plt.hist(filtered_data, bins=30, edgecolor='black')
plt.title('Histogram of iou orig')
plt.xlabel('IOU')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()
plt.savefig(output_folder + 'iou_hist_orig.png', dpi=300, bbox_inches='tight')
plt.close()