#!/usr/bin/env python3
"""
LoRA adapter를 base 모델에 merge하는 스크립트.

사용법:
    python scripts/merge_lora.py \
        --base_model /path/to/base/model \
        --lora_path /path/to/lora/checkpoint \
        --output_path /path/to/merged/model

예시:
    python scripts/merge_lora.py \
        --base_model allenai/olmOCR-2-7B-1025 \
        --lora_path /home/kyungho/frameworks/olmocr-ft/outputs/korean_lora_multi_20251223_004736/olmocr_korean_lora_ft/checkpoint-24702 \
        --output_path /home/kyungho/olmocr-qwen3/8b/korean_olmocr_lora_multi_merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights (default: bfloat16)",
    )
    args = parser.parse_args()

    # dtype 설정
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # 경로 확인
    base_model_path = args.base_model  # HuggingFace ID 또는 로컬 경로
    lora_path = Path(args.lora_path)
    output_path = Path(args.output_path)

    # 로컬 경로인 경우에만 존재 여부 확인
    if "/" not in base_model_path or Path(base_model_path).exists():
        # HuggingFace 모델 ID이거나 존재하는 로컬 경로
        pass
    elif not Path(base_model_path).exists() and "/" in base_model_path and not base_model_path.startswith("/"):
        # HuggingFace 모델 ID로 간주 (예: allenai/olmOCR-2-7B-1025)
        print(f"Treating '{base_model_path}' as HuggingFace model ID")
    else:
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")

    print(f"Base model: {base_model_path}")
    print(f"LoRA path: {lora_path}")
    print(f"Output path: {output_path}")
    print(f"Dtype: {args.dtype}")
    print()

    # Base 모델 로드
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # LoRA adapter 로드
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(lora_path))

    # Merge
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    # 저장
    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))

    # Processor 복사
    print("Copying processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(str(output_path))

    print()
    print("=" * 60)
    print("Done! Merged model saved to:")
    print(f"  {output_path}")
    print()
    print("You can now use the merged model with pipeline:")
    print(f"  python -m olmocr.pipeline ./workspace --model {output_path} --pdfs ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
