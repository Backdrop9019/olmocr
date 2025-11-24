#!/usr/bin/env python
"""
Checkpoint preparation script for Qwen3-VL models trained with OlmOCR pipeline.

This script handles:
1. Converting HuggingFace Trainer checkpoints to VLLM format
2. Merging LoRA adapters into base model
3. Model quantization (FP8)
4. Config adjustment for inference
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
import torch
from typing import Optional, List, Dict, Any

# Add paths
sys.path.append("/home/kyungho/frameworks/olmocr")
sys.path.append("/home/kyungho/frameworks/Qwen3-VL")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_adapter(
    checkpoint_path: str,
    output_path: str,
    base_model_path: Optional[str] = None
) -> str:
    """
    Merge LoRA adapter weights into the base model.

    Args:
        checkpoint_path: Path to checkpoint with LoRA adapter
        output_path: Path to save merged model
        base_model_path: Optional base model path (if not in adapter_config)

    Returns:
        Path to merged model
    """
    logger.info(f"Merging LoRA adapter from {checkpoint_path}")

    # Check for PEFT adapter
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        logger.info("No LoRA adapter found, copying model as-is")
        shutil.copytree(checkpoint_path, output_path, dirs_exist_ok=True)
        return output_path

    # Load adapter config
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    # Get base model path
    if base_model_path is None:
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError("Base model path not found in adapter config")

    logger.info(f"Loading base model from {base_model_path}")

    # Load model with PEFT
    from peft import PeftModel, PeftConfig

    # Load base model
    # Determine model type
    model_name = base_model_path.lower()
    if "qwen3" in model_name:
        try:
            from qwenvl.model.qwen3vl_model import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModelForCausalLM

    base_model = model_cls.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # Merge adapter
    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    logger.info("LoRA merge completed")
    return output_path


def prepare_for_vllm(checkpoint_path: str, output_path: str):
    """
    Prepare checkpoint for VLLM inference.

    Args:
        checkpoint_path: Path to HF checkpoint
        output_path: Path to save VLLM-ready model
    """
    logger.info(f"Preparing checkpoint for VLLM: {checkpoint_path}")

    # Copy model files
    if checkpoint_path != output_path:
        shutil.copytree(checkpoint_path, output_path, dirs_exist_ok=True)

    # Load and modify config
    config_path = Path(output_path) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add/modify VLLM specific settings
    vllm_updates = {
        "architectures": config.get("architectures", ["Qwen3VLForConditionalGeneration"]),
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "use_cache": True,
    }

    config.update(vllm_updates)

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("VLLM preparation completed")


def quantize_to_fp8(
    model_path: str,
    output_path: str,
    calibration_samples: Optional[List[str]] = None
):
    """
    Quantize model to FP8 format.

    Args:
        model_path: Path to model
        output_path: Path to save quantized model
        calibration_samples: Optional calibration data
    """
    logger.info(f"Quantizing model to FP8: {model_path}")

    # FP8 양자화는 compress_checkpoint.py 별도 실행 필요
    logger.warning("FP8 quantization requires running compress_checkpoint.py separately")
    logger.info(f"Run: python -m olmocr.train.compress_checkpoint --config qwen3_w8a8_fp8.yaml {model_path} {output_path}")

    # 일단 모델 복사만 수행
    if model_path != output_path:
        shutil.copytree(model_path, output_path, dirs_exist_ok=True)


def model_soup_averaging(
    checkpoint_paths: List[str],
    output_path: str,
    weights: Optional[List[float]] = None
):
    """
    Average multiple model checkpoints (model soup).

    Args:
        checkpoint_paths: List of checkpoint paths
        output_path: Path to save averaged model
        weights: Optional weights for each model (default: equal)
    """
    if len(checkpoint_paths) < 2:
        logger.warning("Model soup requires at least 2 checkpoints")
        return

    logger.info(f"Averaging {len(checkpoint_paths)} checkpoints")

    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    elif len(weights) != len(checkpoint_paths):
        raise ValueError("Number of weights must match number of checkpoints")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Load first model as base
    logger.info(f"Loading base model from {checkpoint_paths[0]}")
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_paths[0],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    base_state_dict = base_model.state_dict()

    # Average with other models
    for i, (ckpt_path, weight) in enumerate(zip(checkpoint_paths[1:], weights[1:]), 1):
        logger.info(f"Loading checkpoint {i+1}: {ckpt_path} (weight={weight:.3f})")

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )

        # Average parameters
        for name, param in model.state_dict().items():
            if name in base_state_dict:
                base_state_dict[name] = (
                    base_state_dict[name] * weights[0] +
                    param * weight
                )

        del model  # Free memory

    # Load averaged weights
    base_model.load_state_dict(base_state_dict)

    # Save averaged model
    logger.info(f"Saving averaged model to {output_path}")
    base_model.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer from first checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_paths[0], trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    processor = AutoProcessor.from_pretrained(checkpoint_paths[0], trust_remote_code=True)
    processor.save_pretrained(output_path)

    logger.info("Model soup averaging completed")


def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that checkpoint is ready for inference.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        True if valid, False otherwise
    """
    required_files = [
        "config.json",
        "model.safetensors",  # or pytorch_model.bin
        "tokenizer_config.json",
        "preprocessor_config.json"
    ]

    checkpoint_path = Path(checkpoint_path)

    for file in required_files:
        if not (checkpoint_path / file).exists():
            # Check alternatives
            if file == "model.safetensors":
                if not (checkpoint_path / "pytorch_model.bin").exists():
                    logger.warning(f"Missing: {file} or pytorch_model.bin")
                    return False
            else:
                logger.warning(f"Missing: {file}")
                return False

    logger.info("Checkpoint validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen3-VL checkpoint for deployment")

    parser.add_argument(
        "input_checkpoint",
        nargs="+",
        help="Input checkpoint path(s). Multiple paths for model soup."
    )
    parser.add_argument(
        "output_path",
        help="Output path for prepared checkpoint"
    )

    # Operations
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model path for LoRA merge (if not in adapter config)"
    )
    parser.add_argument(
        "--prepare-vllm",
        action="store_true",
        default=True,
        help="Prepare checkpoint for VLLM inference"
    )
    parser.add_argument(
        "--quantize-fp8",
        action="store_true",
        help="Quantize model to FP8"
    )
    parser.add_argument(
        "--model-soup",
        action="store_true",
        help="Average multiple checkpoints (model soup)"
    )
    parser.add_argument(
        "--soup-weights",
        type=float,
        nargs="+",
        help="Weights for model soup (default: equal)"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle operations
    if args.model_soup and len(args.input_checkpoint) > 1:
        # Model soup averaging
        model_soup_averaging(
            checkpoint_paths=args.input_checkpoint,
            output_path=str(output_path),
            weights=args.soup_weights
        )
        checkpoint_path = str(output_path)

    else:
        # Single checkpoint
        checkpoint_path = args.input_checkpoint[0]

        # Merge LoRA if needed
        if args.merge_lora:
            checkpoint_path = merge_lora_adapter(
                checkpoint_path=checkpoint_path,
                output_path=str(output_path),
                base_model_path=args.base_model
            )
        elif checkpoint_path != str(output_path):
            # Copy checkpoint
            shutil.copytree(checkpoint_path, output_path, dirs_exist_ok=True)
            checkpoint_path = str(output_path)

    # Prepare for VLLM
    if args.prepare_vllm:
        prepare_for_vllm(checkpoint_path, checkpoint_path)

    # Quantize if requested
    if args.quantize_fp8:
        fp8_path = str(output_path) + "_fp8"
        quantize_to_fp8(checkpoint_path, fp8_path)
        checkpoint_path = fp8_path

    # Validate final checkpoint
    if validate_checkpoint(checkpoint_path):
        logger.info(f"Checkpoint prepared successfully: {checkpoint_path}")

        # Print usage instructions
        print("\n" + "="*60)
        print("Checkpoint preparation completed!")
        print(f"Output path: {checkpoint_path}")
        print("\nTo use with VLLM:")
        print(f"  vllm serve {checkpoint_path} --max-model-len 8192")
        print("\nTo use with OlmOCR:")
        print(f"  python -m olmocr.pipeline --model {checkpoint_path}")
        print("\nFor FP8 quantization, run:")
        print(f"  python -m olmocr.train.compress_checkpoint \\")
        print(f"    --config olmocr/train/quantization_configs/qwen3_w8a8_fp8.yaml \\")
        print(f"    {checkpoint_path} {checkpoint_path}_fp8")
        print("="*60)
    else:
        logger.error("Checkpoint validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()