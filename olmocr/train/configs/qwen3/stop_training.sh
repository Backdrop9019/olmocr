#!/bin/bash
# Stop Qwen3-VL training

LOG_DIR="/home/kyungho/olmocr-qwen3-8b/logs"
PID_FILE="${LOG_DIR}/train.pid"

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    echo "Stopping training (PID: ${PID})..."
    kill ${PID}
    rm "${PID_FILE}"
    echo "Training stopped."
else
    echo "No training PID file found at ${PID_FILE}"
    echo "Searching for running train_qwen3.py processes..."
    ps aux | grep train_qwen3.py | grep -v grep
fi
