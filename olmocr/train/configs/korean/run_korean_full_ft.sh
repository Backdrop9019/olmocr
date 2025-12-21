#!/bin/bash

# OlmOCR Korean Finetuning Script (Single GPU)

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmocr

export CUDA_VISIBLE_DEVICES=3

# Paths
SCRIPT_PATH="olmocr.train.train"
CONFIG_PATH="olmocr/train/configs/korean/olmocr_korean_full_ft.yaml"
BASE_OUTPUT_DIR="/home/kyungho/frameworks/olmocr-ft/outputs"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/korean_full_ft_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Copy config and update output_dir
cp "${CONFIG_PATH}" "${OUTPUT_DIR}/config.yaml"
sed -i "s|output_dir:.*|output_dir: ${OUTPUT_DIR}|" "${OUTPUT_DIR}/config.yaml"

echo "Starting Korean finetuning..."
echo "Output directory: ${OUTPUT_DIR}"
echo "Config: ${OUTPUT_DIR}/config.yaml"
echo "Log: ${LOG_DIR}/train.log"

# Run training
cd /home/kyungho/frameworks/olmocr-ft

nohup python -m ${SCRIPT_PATH} \
    --config "${OUTPUT_DIR}/config.yaml" \
    > "${LOG_DIR}/train.log" 2>&1 &

echo "Training started in background. PID: $!"
echo "Monitor with: tail -f ${LOG_DIR}/train.log"
