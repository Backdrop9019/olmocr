#!/bin/bash
# View latest training logs

LOG_DIR="/home/kyungho/olmocr-qwen3-8b/logs"

if [ ! -d "${LOG_DIR}" ]; then
    echo "Log directory not found: ${LOG_DIR}"
    exit 1
fi

# Find latest log file
LATEST_LOG=$(ls -t ${LOG_DIR}/train_*.log 2>/dev/null | head -1)

if [ -z "${LATEST_LOG}" ]; then
    echo "No log files found in ${LOG_DIR}"
    exit 1
fi

echo "=========================================="
echo "Latest log: ${LATEST_LOG}"
echo "=========================================="
echo ""

# Follow log file
tail -f "${LATEST_LOG}"
