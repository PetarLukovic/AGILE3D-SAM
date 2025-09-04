#!/usr/bin/env bash

scannet_folder="/cluster/scratch/plukovic/KITTI-360"

python eval_single_obj.py --dataset=kitti360 \
               --dataset_mode=single_obj \
               --scan_folder=${kitti360_folder}/single/crops \
               --crop \
               --val_list=${kitti360_folder}/single/object_ids.npy \
               --val_list_classes=${kitti360_folder}/single/object_classes.txt \
               --output_dir=${kitti360_folder}/results \
               --checkpoint=weights/checkpoint1099.pth \
               --val_batch_size=1 \
               --start_index=2400 \
               --end_index=3421 \