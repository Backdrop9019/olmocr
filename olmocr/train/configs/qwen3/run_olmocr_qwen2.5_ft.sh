#!/bin/bash
# OlmOCR Finetuning Script (Single GPU)
# Using olmocr.train.train (original train script for Qwen2.5-VL based models)

# Conda 환경 활성화
source /home/kyungho/miniconda3/etc/profile.d/conda.sh
conda activate olmocr-qwen3

# GPU 설정 (GPU 0번만 사용)
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG_PATH="olmocr/train/configs/qwen3/olmocr_own_ft.yaml"
SCRIPT_PATH="olmocr.train.train"
BASE_OUTPUT_DIR="/home/kyungho/olmocr-qwen3/8b/olmocr-ft"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Copy config and update output_dir
cp "${CONFIG_PATH}" "${OUTPUT_DIR}/config.yaml"
sed -i "s|output_dir:.*|output_dir: ${OUTPUT_DIR}|" "${OUTPUT_DIR}/config.yaml"

# Print info
echo "=========================================="
echo "Starting OlmOCR Finetuning (LoRA)"
echo "=========================================="
echo "Config: ${OUTPUT_DIR}/config.yaml"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_DIR}/train.log"
echo "=========================================="

# Run with nohup in background (Single GPU)
# Note: Using python -m olmocr.train.train (not train_qwen3.py)
# Config file already contains output_dir
nohup python -m ${SCRIPT_PATH} \
    --config "${OUTPUT_DIR}/config.yaml" \
    > "${LOG_DIR}/train.log" 2>&1 &

# Save PID
PID=$!
echo ${PID} > "${OUTPUT_DIR}/train.pid"

echo "Training started in background!"
echo "PID: ${PID}"
echo "Log: tail -f ${LOG_DIR}/train.log"
echo ""
echo "To stop training: kill ${PID}"
echo "Or use: kill \$(cat ${OUTPUT_DIR}/train.pid)"
