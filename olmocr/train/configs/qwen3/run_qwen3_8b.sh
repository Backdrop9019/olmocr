#!/bin/bash
# Qwen3-VL 8B Training Script
# Using torchrun with DeepSpeed (following official Qwen3-VL approach)

# Conda 환경 활성화
source /home/kyungho/miniconda3/etc/profile.d/conda.sh
conda activate olmocr-qwen3

# Configuration
NPROC_PER_NODE=4  # Number of GPUs
CONFIG_PATH="olmocr/train/configs/qwen3/qwen3_8b_olmocr.yaml"
SCRIPT_PATH="olmocr/train/train_qwen3.py"
LOG_DIR="/home/kyungho/olmocr-qwen3-8b/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# Create log directory
mkdir -p "${LOG_DIR}"

# Print info
echo "=========================================="
echo "Starting Qwen3-VL 8B Training"
echo "=========================================="
echo "Config: ${CONFIG_PATH}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Log file: ${LOG_FILE}"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "=========================================="

# Run with nohup in background using torchrun
nohup torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT_PATH} \
    --olmocr_config_path ${CONFIG_PATH} \
    > "${LOG_FILE}" 2>&1 &

# Save PID
PID=$!
echo ${PID} > "${LOG_DIR}/train.pid"

echo "Training started in background!"
echo "PID: ${PID}"
echo "Log: tail -f ${LOG_FILE}"
echo ""
echo "To stop training: kill ${PID}"
echo "Or use: kill \$(cat ${LOG_DIR}/train.pid)"