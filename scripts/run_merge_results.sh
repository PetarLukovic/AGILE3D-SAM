#!/usr/bin/env bash

python3 -m plukovic.utils.merge_results \
    --input_folder /home/plukovic/interactive_segmentation/AGILE3D-SAM/data/scannet/results \
    --output_folder /home/plukovic/interactive_segmentation/AGILE3D-SAM/results \
    --num_clicks 20