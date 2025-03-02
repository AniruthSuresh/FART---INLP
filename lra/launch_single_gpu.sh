#!/bin/bash
# launch.sh - Launch training for LRA benchmarks on a single GPU

# Usage: ./launch.sh <dataset>
#   where <dataset> is one of: imdb, listops, cifar10


if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    echo "Datasets: imdb, listops, cifar10"
    exit 1
fi

DATASET=$1
OUTPUT_FILE="terminal_${DATASET}.txt"


eval "$(conda shell.bash hook)"

conda activate inlp  

# Build the command with dataset-specific arguments
if [ "$DATASET" == "imdb" ]; then
    CMD="python train-single_gpu.py --dataset imdb --max_length 1024 --seq_len 1024 --input_dim 1 --num_classes 2"
elif [ "$DATASET" == "listops" ]; then
    CMD="python train-single_gpu.py --dataset listops --max_length 1024 --seq_len 1024 --input_dim 1"
elif [ "$DATASET" == "cifar10" ]; then
    CMD="python train-single_gpu.py --dataset cifar10 --max_length 1024 --seq_len 1024 --input_dim 1 --num_classes 10"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Run the command and log output
echo "Running command: ${CMD}"
${CMD} | tee ${OUTPUT_FILE}

