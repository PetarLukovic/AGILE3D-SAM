#!/usr/bin/env bash

python -m evaluation.evaluator_SO --dataset=scannet40 \
               --val_list=/cluster/scratch/plukovic/scannet/scannet_v2/single/object_ids.npy \
               --val_list_classes=/cluster/scratch/plukovic/scannet/scannet_v2/single/object_classes.txt \
               --results_file=/cluster/scratch/plukovic/AGILE3D/results/val_results_single_aligned.csv \