#!/usr/bin/env bash

scannet_folder="/home/petar/interactive_segmentation/AGILE3D-SAM/data/scannet"

python eval_single_obj.py --dataset=scannet40 \
               --dataset_mode=single_obj \
               --scan_folder=${scannet_folder}/scans \
               --val_list=${scannet_folder}/single/object_ids.npy \
               --val_list_classes=${scannet_folder}/single/object_classes.txt \
               --output_dir=${scannet_folder}/results \
               --checkpoint=weights/checkpoint1099.pth \
               --val_batch_size=1 \
               --start_index=1 \
               --end_index=11 \
