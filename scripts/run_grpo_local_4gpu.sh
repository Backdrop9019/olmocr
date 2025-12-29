#!/bin/bash
# Local Multi-GPU GRPO Training Script for 4x H100
# GPU 3: vLLM server
# GPU 0,1,2: Training with DeepSpeed

# Configuration
MODEL_NAME="allenai/olmOCR-2-7B-1025"
TRAIN_BENCH_DATA="/home/kyungho/frameworks/data/rl_train_10k_synth/bench_data"
BASE_OUTPUT_DIR="/home/kyungho/frameworks/olmocr-ft/outputs"

# GPU Configuration
VLLM_GPU=3
TRAINING_GPUS="0,1,2"
NUM_TRAINING_PROCESSES=3

# Conda environment
CONDA_ENV="/home/kyungho/miniconda3/envs/olmocr-rl"

# Add conda env bin to PATH (for ninja, etc.)
export PATH="$CONDA_ENV/bin:$PATH"

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/grpo_run_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Clear all GPU memory and port before starting
echo "Clearing GPU memory and port 8000..."
pkill -9 -f "vllm_serve\|grpo_train\|python.*olmocr\|vllm\|trl" 2>/dev/null || true
sleep 2
# Kill all GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1
# Double check and kill any remaining GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
# Kill any process using port 8000
lsof -ti :8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1
echo "GPU memory and port cleared."

echo "=========================================="
echo "Starting GRPO Training (4x H100)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train Data: $TRAIN_BENCH_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "vLLM GPU: $VLLM_GPU"
echo "Training GPUs: $TRAINING_GPUS"
echo ""
echo "Effective batch size: 3 * 1 * 12 = 36"
echo "Num generations: 12"
echo ""
echo "Log files:"
echo "  vLLM:     ${LOG_DIR}/vllm.log"
echo "  Training: ${LOG_DIR}/train.log"
echo "=========================================="

# Change to project directory
cd /home/kyungho/frameworks/olmocr-ft

# Start vLLM server on GPU 3 in background
echo "Starting vLLM server on GPU $VLLM_GPU..."
nohup env CUDA_VISIBLE_DEVICES=$VLLM_GPU $CONDA_ENV/bin/python -m trl.scripts.vllm_serve \
    --model $MODEL_NAME \
    --port 8000 \
    --gpu_memory_utilization 0.85 \
    > "${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
echo $VLLM_PID > "${OUTPUT_DIR}/vllm.pid"
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
sleep 30
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    else
        echo "Still waiting for vLLM server... ($i/60)"
        sleep 5
    fi
done

# Check if server is actually ready
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start. Check ${LOG_DIR}/vllm.log"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run GRPO training on GPUs 0,1,2 with DeepSpeed in background
echo ""
echo "Starting GRPO training on GPUs $TRAINING_GPUS..."
echo ""

nohup env CUDA_VISIBLE_DEVICES=$TRAINING_GPUS $CONDA_ENV/bin/accelerate launch \
    --use_deepspeed \
    --zero_stage 3 \
    --num_processes $NUM_TRAINING_PROCESSES \
    --gradient_accumulation_steps 12 \
    -m olmocr.train.grpo_train \
    --model_name $MODEL_NAME \
    --train_bench_data_folder $TRAIN_BENCH_DATA \
    --output_dir $OUTPUT_DIR \
    --vllm_mode server \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --num_generations 12 \
    --learning_rate 2e-6 \
    --warmup_steps 100 \
    --beta 0.01 \
    --reward_bench 1.0 \
    --reward_front_matter 1.0 \
    --reward_eos 1.0 \
    > "${LOG_DIR}/train.log" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "${OUTPUT_DIR}/train.pid"

echo ""
echo "=========================================="
echo "Training started in background!"
echo "=========================================="
echo "vLLM PID: $VLLM_PID"
echo "Train PID: $TRAIN_PID"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/train.log"
echo ""
echo "To stop training:"
echo "  kill \$(cat ${OUTPUT_DIR}/train.pid) \$(cat ${OUTPUT_DIR}/vllm.pid)"
echo "=========================================="
