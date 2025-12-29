#!/bin/bash
# ZeRO-3 Checkpoint Merge Script
# Usage: ./scripts/merge_zero3_checkpoint.sh <checkpoint_dir> <output_dir>
#
# Example:
#   ./scripts/merge_zero3_checkpoint.sh outputs/grpo_run_20251228_235710/checkpoint-367 outputs/my_merged_model

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <output_dir>"
    echo ""
    echo "Example:"
    echo "  $0 outputs/grpo_run_20251228_235710/checkpoint-367 outputs/my_merged_model"
    exit 1
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="$2"
CONDA_ENV="/home/kyungho/miniconda3/envs/olmocr-rl"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Check if zero_to_fp32.py exists
if [ ! -f "$CHECKPOINT_DIR/zero_to_fp32.py" ]; then
    echo "Error: zero_to_fp32.py not found in $CHECKPOINT_DIR"
    exit 1
fi

echo "=========================================="
echo "ZeRO-3 Checkpoint Merge"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Merge ZeRO-3 weights
echo ""
echo "Step 1: Merging ZeRO-3 distributed weights..."
cd "$CHECKPOINT_DIR"
$CONDA_ENV/bin/python zero_to_fp32.py . "$OUTPUT_DIR/"

echo ""
echo "Step 2: Copying config and tokenizer files..."
# Copy all json files (config, tokenizer, etc.)
cp "$CHECKPOINT_DIR"/*.json "$OUTPUT_DIR/" 2>/dev/null || true
# Copy merges.txt for tokenizer
cp "$CHECKPOINT_DIR"/merges.txt "$OUTPUT_DIR/" 2>/dev/null || true
# Copy chat template if exists
cp "$CHECKPOINT_DIR"/chat_template.jinja "$OUTPUT_DIR/" 2>/dev/null || true

# Remove training-related files that aren't needed for inference
rm -f "$OUTPUT_DIR"/trainer_state.json 2>/dev/null || true
rm -f "$OUTPUT_DIR"/training_args.bin 2>/dev/null || true

echo ""
echo "=========================================="
echo "Merge complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "Usage:"
echo "  # vLLM"
echo "  vllm serve $OUTPUT_DIR"
echo ""
echo "  # HuggingFace"
echo "  model = AutoModel.from_pretrained('$OUTPUT_DIR')"
echo "=========================================="
