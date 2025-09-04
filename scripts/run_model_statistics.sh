#!/usr/bin/env bash

agile3d_folder="/home/petar/interactive_segmentation/AGILE3D-SAM"

python3 -m plukovic.utils.model_statistics \
    --model_path ${agile3d_folder}/results/model_0 \
    --max_num_clicks 20

python3 -m plukovic.utils.model_statistics \
    --model_path ${agile3d_folder}/results/model_1 \
    --max_num_clicks 10

python3 -m plukovic.utils.model_statistics \
    --model_path ${agile3d_folder}/results/model_2 \
    --max_num_clicks 10

python3 -m plukovic.utils.model_statistics \
    --model_path ${agile3d_folder}/results/model_3 \
    --max_num_clicks 20

python3 -m plukovic.utils.model_statistics \
    --model_path ${agile3d_folder}/results/model_4 \
    --max_num_clicks 7