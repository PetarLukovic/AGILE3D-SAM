#!/usr/bin/env bash

agile3d_folder="/home/plukovic/interactive_segmentation/AGILE3D-SAM"

python3 -m evaluation.evaluator_SO --dataset=scannet40 \
               --val_list=${agile3d_folder}/data/scannet/single/object_ids.npy \
               --val_list_classes=${agile3d_folder}/data/scannet/single/object_classes.txt \
               --results_file=${agile3d_folder}/results/val_results_single_20250618_174756.csv

python3 -m evaluation.evaluator_SO --dataset=scannet40 \
               --val_list=${agile3d_folder}/data/scannet/single/object_ids.npy \
               --val_list_classes=${agile3d_folder}/data/scannet/single/object_classes.txt \
               --results_file=${agile3d_folder}/results/our_single_scannet.csv