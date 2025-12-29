#!/bin/bash

# OlmOCR Korean LoRA Finetuning Script (Multi-GPU DDP)
# Uses 4 GPUs with total batch size of 32

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmocr

# Use all 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Paths
SCRIPT_PATH="olmocr.train.train_multigpu"
CONFIG_PATH="olmocr/train/configs/korean/olmocr_korean_lora_multi.yaml"
BASE_OUTPUT_DIR="/home/kyungho/frameworks/olmocr-ft/outputs"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/korean_lora_multi_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Copy config and update output_dir + gradient_accumulation_steps
cp "${CONFIG_PATH}" "${OUTPUT_DIR}/config.yaml"
sed -i "s|output_dir:.*|output_dir: ${OUTPUT_DIR}|" "${OUTPUT_DIR}/config.yaml"
# Adjust gradient_accumulation_steps: 8 * 4 GPUs * 1 batch = 32 total
sed -i "s|gradient_accumulation_steps:.*|gradient_accumulation_steps: 8|" "${OUTPUT_DIR}/config.yaml"

echo "=========================================="
echo "Starting Korean LoRA finetuning with Multi-GPU"
echo "=========================================="
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: 1"
echo "Gradient accumulation: 8"
echo "Effective batch size: 1 * 8 * 4 = 32"
echo "LoRA rank: 32, alpha: 32"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Config: ${OUTPUT_DIR}/config.yaml"
echo "Log: ${LOG_DIR}/train.log"
echo "=========================================="

# Run training with torchrun
cd /home/kyungho/frameworks/olmocr-ft

nohup torchrun --nproc_per_node=${NUM_GPUS} \
    -m ${SCRIPT_PATH} \
    --config "${OUTPUT_DIR}/config.yaml" \
    > "${LOG_DIR}/train.log" 2>&1 &

echo "Training started in background. PID: $!"
echo "Monitor with: tail -f ${LOG_DIR}/train.log"
