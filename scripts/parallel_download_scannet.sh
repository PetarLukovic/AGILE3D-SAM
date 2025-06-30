#!/bin/bash

# Default flag values
TRAIN=0
TEST=0
VAL=0
OUTPUT_DIR="/cluster/scratch/plukovic/scannet/scannet_v2/"
mkdir -p "$OUTPUT_DIR"

# Check if no arguments were given
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--train] [--test] [--val]"
    echo "  --train   Download the train scenes"
    echo "  --test    Download the test scenes"
    echo "  --val     Download the validation scenes"
    exit 1
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train) TRAIN=1 ;;
        --test) TEST=1 ;;
        --val) VAL=1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Download train scans if --train flag is given
if [ $TRAIN -eq 1 ]; then
    echo "Downloading validation split file..."
    wget https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt -O "${OUTPUT_DIR}/scannetv2_train.txt"
    echo "Downloaded validation split file."
    
    echo "Processing validation .sens files..."
    python3 plukovic/scannet/process_sens.py --scans_folder "${OUTPUT_DIR}/scans" --scans_file "${OUTPUT_DIR}/scannetv2_val.txt"
    echo "Processed validation .sens files."
fi

# Download test scans if --test flag is given
if [ $TEST -eq 1 ]; then
    echo "Downloading validation split file..."
    wget https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt -O "${OUTPUT_DIR}/scannetv2_test.txt"
    echo "Downloaded validation split file."

    echo "Processing validation .sens files..."
    python3 plukovic/scannet/process_sens.py --scans_folder "${OUTPUT_DIR}/scans_train" --scans_file "${OUTPUT_DIR}/scannetv2_val.txt"
    echo "Processed validation .sens files."
fi

# Download validation scans if --val flag is given
if [ $VAL -eq 1 ]; then
    echo "Downloading validation split file..."
    wget https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt -O "${OUTPUT_DIR}/scannetv2_val.txt"
    echo "Downloaded validation split file."

    echo "Processing validation .sens files..."
    python3 -m plukovic.scannet.process_sens --scannet_folder "${OUTPUT_DIR}" --scans_file "${OUTPUT_DIR}/scannetv2_val.txt"
    echo "Processed validation .sens files."
fi

