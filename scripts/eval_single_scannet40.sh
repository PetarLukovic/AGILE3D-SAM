#!/usr/bin/env bash

scannet_folder="/home/plukovic/interactive_segmentation/AGILE3D/data/scannet"

python eval_single_obj.py --dataset=scannet40 \
               --dataset_mode=single_obj \
               --scan_folder=${scannet_folder}/scans \
               --val_list=${scannet_folder}/single/object_ids.npy \
               --val_list_classes=${scannet_folder}/single/object_classes.txt \
               --output_dir=${scannet_folder}/results \
               --checkpoint=weights/checkpoint1099.pth \
               --val_batch_size=1 \
               --max_num_clicks=1 \

