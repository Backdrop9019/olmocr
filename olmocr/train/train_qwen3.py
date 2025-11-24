#!/usr/bin/env python
"""
OlmOCR Training Script for Qwen3-VL

This script integrates OlmOCR's data pipeline with Qwen3-VL's training infrastructure.
It uses HuggingFace Trainer with DeepSpeed support for distributed training.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint

# Add Qwen3-VL path to system
sys.path.append("/home/kyungho/frameworks/Qwen3-VL")
sys.path.append("/home/kyungho/frameworks/Qwen3-VL/qwen-vl-finetune")

# Import Qwen3-VL components
try:
    from qwenvl.train.argument import ModelArguments, DataArguments
    from qwenvl.data.data_processor import (
        make_supervised_data_module,
        DataCollatorForSupervisedDataset
    )
    # Qwen3-VL model handles position IDs automatically
except ImportError as e:
    print(f"Error importing Qwen3-VL components: {e}")
    print("Make sure Qwen3-VL is properly installed at /home/kyungho/frameworks/Qwen3-VL")
    sys.exit(1)

# Import OlmOCR components
from olmocr.train.config import Config as OlmOCRConfig
from olmocr.train.qwen_data_adapter import (
    OlmOCRToQwenDataset,
    OlmOCRQwenDataModule,
    QwenDataConfig
)
from olmocr.train.dataloader import BaseMarkdownPDFDataset

# Qwen 모델 import - Qwen3-VL만 사용
try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration
    )
except ImportError as e:
    print(f"Error: Could not import Qwen3-VL models: {e}")
    print("Make sure you have transformers>=4.57.0 installed")
    sys.exit(1)

logger = logging.getLogger(__name__)


@dataclass
class OlmOCRQwen3Arguments:
    """Arguments specific to OlmOCR-Qwen3 integration"""

    olmocr_config_path: str = field(
        metadata={"help": "Path to OlmOCR YAML config file"}
    )

    use_qwen3_vl: bool = field(
        default=True,
        metadata={"help": "Use Qwen3-VL model (vs Qwen2/2.5-VL)"}
    )

    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2"}
    )

    use_deepspeed: bool = field(
        default=False,
        metadata={"help": "Use DeepSpeed for distributed training"}
    )

    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config JSON"}
    )


@dataclass
class Qwen3ModelArguments(ModelArguments):
    """Extended model arguments for Qwen3-VL"""

    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )

    quantization_bit: Optional[int] = field(default=None)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")


@dataclass
class Qwen3DataArguments(DataArguments):
    """Extended data arguments for OlmOCR integration"""

    max_image_size: int = field(
        default=1288,
        metadata={"help": "Maximum image dimension (matches OlmOCR)"}
    )

    include_metadata: bool = field(
        default=False,
        metadata={"help": "Include OlmOCR metadata in prompts (OlmOCR는 사용하지 않음)"}
    )

    use_olmocr_pipeline: bool = field(
        default=True,
        metadata={"help": "Use OlmOCR data pipeline"}
    )


class QwenTrainer(Trainer):
    """
    Custom Trainer that handles empty batches gracefully.

    When all samples in a batch are filtered out (e.g., due to malformed equations),
    the collator returns None and this trainer skips the batch instead of crashing.
    This matches OlmOCR's behavior with custom training loops.
    """

    def floating_point_ops(self, inputs):
        """
        Calculate FLOPs, returning 0 if inputs is None/empty.
        This prevents crashes when collator returns None for filtered batches.
        """
        if inputs is None or not inputs:
            return 0
        return super().floating_point_ops(inputs)

    def _get_num_items_in_batch(self, batch, device=None):
        """
        Get number of items in batch, returning 0 if batch is None/empty.
        This prevents crashes when checking batch size for None batches.
        """
        if batch is None or not batch:
            return 0
        # Check if batch contains None elements (happens with gradient accumulation)
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and batch[0] is None:
            return 0
        return super()._get_num_items_in_batch(batch, device)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs, skipping if batch is empty.
        """
        # Skip if batch is None or empty (all samples were filtered out)
        if inputs is None or not inputs or len(inputs) == 0:
            logger.debug(f"Skipping empty batch in training_step")
            # Return zero loss that doesn't affect gradients
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Normal training step
        return super().training_step(model, inputs, num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform an evaluation step on a batch of inputs, skipping if batch is empty.
        """
        # Skip if batch is None or empty
        if inputs is None or not inputs or len(inputs) == 0:
            logger.debug(f"Skipping empty batch in prediction_step")
            # Return dummy outputs - must be on GPU for distributed gather
            return (torch.tensor(0.0, device=model.device), None, None)

        # Normal prediction step
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class OlmOCRQwen3DataCollator:
    """
    Custom data collator that handles OlmOCR samples with Qwen3-VL
    """

    def __init__(self, processor, merge_size=28):
        self.processor = processor
        self.merge_size = merge_size
        self.batch_count = 0  # Track batches for logging

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        self.batch_count += 1
        verbose = False  # Disabled - data loading is working fine now

        # Only log if there are problematic samples
        none_count = sum(1 for inst in instances if inst is None)
        empty_count = sum(1 for inst in instances if inst is not None and not inst)
        if none_count > 0 or empty_count > 0:
            logger.debug(f"Batch {self.batch_count}: {none_count} None, {empty_count} empty (filtered out)")

        # Filter out None and empty dict instances
        instances = [inst for inst in instances if inst is not None and inst]
        if verbose:
            logger.info(f"After filtering: {len(instances)} valid instances")

        if not instances:
            logger.debug(f"Batch {self.batch_count}: All instances filtered out, skipping")
            return None  # Return None so QwenTrainer can skip this batch

        # Extract images and convert conversations to Qwen3-VL format
        images = []
        messages_list = []

        for idx, inst in enumerate(instances):
            image = inst.get("image")
            conversations = inst.get("conversations", [])

            if verbose:
                logger.info(f"Instance {idx}:")
                logger.info(f"  - Image type: {type(image)}")
                if hasattr(image, 'size'):
                    logger.info(f"  - Image size: {image.size}")
                logger.info(f"  - Conversations: {len(conversations)} messages")

            if not conversations or len(conversations) < 2:
                if verbose:
                    logger.warning(f"  - Skipping: insufficient conversations")
                continue

            # Build messages in Qwen3-VL format
            messages = []
            for conv in conversations:
                role = conv.get("from")
                content = conv.get("value", "")

                if role == "human":
                    # First user message includes image
                    if len(messages) == 0:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": content.replace("<image>", "").strip()}
                            ]
                        })
                    else:
                        messages.append({"role": "user", "content": content})
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": content})

            if image and messages:
                images.append(image)
                messages_list.append(messages)
                if verbose:
                    logger.info(f"  - ✓ Added to batch")

        if not images or not messages_list:
            logger.debug(f"Batch {self.batch_count}: No valid image-message pairs, skipping")
            return None  # Return None so QwenTrainer can skip this batch

        # Apply chat template for each conversation
        if verbose:
            logger.info(f"-"*80)
            logger.info(f"Applying chat template to {len(messages_list)} conversations")
        texts = []
        for idx, messages in enumerate(messages_list):
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            if verbose and idx == 0:  # Show first prompt in detail
                logger.info(f"Sample prompt (first instance):")
                logger.info(f"{text[:500]}...")

        # Process with Qwen3-VL processor
        if verbose:
            logger.info(f"-"*80)
            logger.info(f"Processing with Qwen3-VL processor:")
            logger.info(f"  - Images: {len(images)}")
            logger.info(f"  - Texts: {len(texts)}")
            if hasattr(self.processor, 'image_processor'):
                ip = self.processor.image_processor
                logger.info(f"  - Processor min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
                logger.info(f"  - Processor max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
                logger.info(f"  - Processor patch_size: {getattr(ip, 'patch_size', 'N/A')}")
                logger.info(f"  - Processor merge_size: {getattr(ip, 'merge_size', 'N/A')}")
            for i, img in enumerate(images[:2]):  # Show first 2 images
                if hasattr(img, 'size'):
                    logger.info(f"  - Image {i} input size: {img.size}")

        try:
            batch = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            if verbose:
                logger.info(f"✓ Processor output:")
                logger.info(f"  - Batch keys: {list(batch.keys())}")
                logger.info(f"  - input_ids shape: {batch['input_ids'].shape}")
                logger.info(f"  - input_ids dtype: {batch['input_ids'].dtype}")
                if 'pixel_values' in batch:
                    logger.info(f"  - pixel_values shape: {batch['pixel_values'].shape}")
                    logger.info(f"  - pixel_values dtype: {batch['pixel_values'].dtype}")
                if 'image_grid_thw' in batch:
                    logger.info(f"  - image_grid_thw shape: {batch['image_grid_thw'].shape}")
                    logger.info(f"  - image_grid_thw values: {batch['image_grid_thw']}")
                    logger.info(f"  - merge_size: {self.merge_size}")
                    for i, grid in enumerate(batch['image_grid_thw']):
                        t, h, w = grid.tolist()
                        patches_before_merge = t * h * w
                        patches_after_merge = patches_before_merge // (self.merge_size ** 2)
                        logger.info(f"  - Image {i}: grid=({t},{h},{w}), patches_before={patches_before_merge}, patches_after={patches_after_merge}")

            # Add labels (same as input_ids for language modeling)
            batch["labels"] = batch["input_ids"].clone()
            if verbose:
                logger.info(f"  - labels shape: {batch['labels'].shape}")

            # Qwen3-VL model will calculate position IDs automatically
            # No need to manually compute them

            if verbose:
                logger.info(f"✓ Batch ready with {len(batch)} keys")
                logger.info("="*80)
            return batch
        except Exception as e:
            logger.error(f"COLLATOR ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _convert_olmocr_to_qwen(self, inst: Dict) -> Optional[Dict]:
        """Convert OlmOCR instance to Qwen conversation format"""
        try:
            messages = inst.get("messages", [])
            if len(messages) < 2:
                return None

            return {
                "image": inst.get("image"),
                "conversations": [
                    {"from": "human", "value": messages[0].get("content", "")},
                    {"from": "gpt", "value": messages[1].get("content", "")}
                ]
            }
        except Exception as e:
            logger.warning(f"Failed to convert instance: {e}")
            return None


def load_qwen3_model(model_args: Qwen3ModelArguments, training_args: TrainingArguments):
    """
    Load Qwen3-VL model with proper configuration
    """
    logger.info("="*80)
    logger.info("MODEL LOADING")
    logger.info("="*80)

    # Determine model class based on model name (Qwen3-VL only)
    model_name = model_args.model_name_or_path.lower()
    logger.info(f"Model name: {model_args.model_name_or_path}")

    if "qwen3" in model_name:
        if "moe" in model_name or "a" in model_name.split("/")[-1]:
            model_cls = Qwen3VLMoeForConditionalGeneration
            logger.info(f"Model class: Qwen3VLMoeForConditionalGeneration (MoE variant)")
        else:
            model_cls = Qwen3VLForConditionalGeneration
            logger.info(f"Model class: Qwen3VLForConditionalGeneration (Dense variant)")
    else:
        raise ValueError(f"Only Qwen3-VL models are supported. Got: {model_name}")

    # Quantization config
    bnb_config = None
    if model_args.quantization_bit in [4, 8]:
        logger.info(f"Quantization: {model_args.quantization_bit}-bit")
        logger.info(f"  - Quant type: {model_args.quant_type}")
        logger.info(f"  - Double quant: {model_args.double_quant}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(model_args.quantization_bit == 4),
            load_in_8bit=(model_args.quantization_bit == 8),
            bnb_4bit_quant_type=model_args.quant_type,
            bnb_4bit_use_double_quant=model_args.double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        logger.info("Quantization: None (full precision)")

    # Load model with flash attention (REQUIRED)
    attn_impl = "flash_attention_2"
    torch_dtype = torch.bfloat16

    logger.info(f"Loading model...")
    logger.info(f"  - Model: {model_args.model_name_or_path}")
    logger.info(f"  - Attention: {attn_impl}")
    logger.info(f"  - Dtype: {torch_dtype}")

    try:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        logger.error("="*80)
        logger.error("FAILED TO LOAD MODEL WITH FLASH ATTENTION!")
        logger.error(f"Error: {e}")
        logger.error("Flash Attention 2 is REQUIRED for this training.")
        logger.error("Please ensure flash-attn is installed: pip install flash-attn")
        logger.error("="*80)
        raise RuntimeError("Flash Attention 2 is required but failed to initialize") from e

    # Verify flash attention is actually being used
    logger.info(f"✓ Model loaded successfully")
    logger.info(f"  - Model dtype: {model.dtype}")
    logger.info(f"  - Model device: {model.device}")

    # Check if flash attention was actually applied
    if hasattr(model.config, '_attn_implementation'):
        actual_attn = model.config._attn_implementation
        logger.info(f"  - Attention implementation: {actual_attn}")
        if actual_attn != attn_impl:
            logger.error("="*80)
            logger.error(f"FLASH ATTENTION NOT APPLIED!")
            logger.error(f"Requested: {attn_impl}, Got: {actual_attn}")
            logger.error("Training cannot proceed without flash attention.")
            logger.error("="*80)
            raise RuntimeError(f"Flash attention not applied. Got {actual_attn} instead of {attn_impl}")
    elif hasattr(model.config, 'attn_implementation'):
        actual_attn = model.config.attn_implementation
        logger.info(f"  - Attention implementation: {actual_attn}")
        if actual_attn != attn_impl:
            logger.error("="*80)
            logger.error(f"FLASH ATTENTION NOT APPLIED!")
            logger.error(f"Requested: {attn_impl}, Got: {actual_attn}")
            logger.error("Training cannot proceed without flash attention.")
            logger.error("="*80)
            raise RuntimeError(f"Flash attention not applied. Got {actual_attn} instead of {attn_impl}")
    else:
        logger.warning("  - Could not verify attention implementation from config")

    # Calculate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024**3)
    logger.info(f"  - Model size: {total_size:.2f} GB")
    logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("="*80)

    # Setup LoRA if needed
    if model_args.use_lora:
        logger.info("="*80)
        logger.info("LORA CONFIGURATION")
        logger.info("="*80)
        logger.info(f"  - LoRA rank (r): {model_args.lora_r}")
        logger.info(f"  - LoRA alpha: {model_args.lora_alpha}")
        logger.info(f"  - LoRA dropout: {model_args.lora_dropout}")
        logger.info(f"  - Target modules: {model_args.lora_target_modules}")

        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("="*80)

    # Configure component training
    set_model_components(model, model_args)

    return model


def set_model_components(model, model_args: Qwen3ModelArguments):
    """
    Configure which components of the model to train
    """
    logger.info("="*80)
    logger.info("MODEL COMPONENT TRAINING CONFIGURATION")
    logger.info("="*80)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    logger.info("✓ All parameters frozen initially")

    # Unfreeze specified components
    if model_args.tune_mm_vision and hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = True
        vision_params = sum(p.numel() for p in model.visual.parameters())
        logger.info(f"✓ Enabled training for vision tower ({vision_params:,} params)")
    else:
        logger.info("✗ Vision tower frozen (tune_mm_vision=False)")

    if model_args.tune_mm_mlp:
        if hasattr(model, 'visual') and hasattr(model.visual, 'merger'):
            for param in model.visual.merger.parameters():
                param.requires_grad = True
            mlp_params = sum(p.numel() for p in model.visual.merger.parameters())
            logger.info(f"✓ Enabled training for vision projector/merger ({mlp_params:,} params)")
    else:
        logger.info("✗ Vision projector frozen (tune_mm_mlp=False)")

    if model_args.tune_mm_llm:
        if hasattr(model, 'language_model'):
            for param in model.language_model.parameters():
                param.requires_grad = True
            llm_params = sum(p.numel() for p in model.language_model.parameters())
            logger.info(f"✓ Enabled training for language model ({llm_params:,} params)")
    else:
        logger.info("✗ Language model frozen (tune_mm_llm=False)")

    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("-"*80)
    logger.info(f"SUMMARY:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    logger.info("="*80)


def create_olmocr_data_module(
    processor,
    olmocr_config: OlmOCRConfig,
    data_args: Qwen3DataArguments,
    training_args: TrainingArguments
) -> Dict:
    """
    Create data module using OlmOCR pipeline
    """
    logger.info("="*80)
    logger.info("DATA MODULE CREATION")
    logger.info("="*80)

    # Configure Qwen data adapter
    qwen_config = QwenDataConfig(
        include_metadata=data_args.include_metadata,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )

    logger.info("Qwen Data Config:")
    logger.info(f"  - include_metadata: {qwen_config.include_metadata}")
    logger.info(f"  - min_pixels: {qwen_config.min_pixels}")
    logger.info(f"  - max_pixels: {qwen_config.max_pixels}")
    logger.info(f"Note: Image resizing handled by Qwen processor (smart_resize)")

    # Check if we should skip validation
    skip_validation = False
    if hasattr(olmocr_config, 'qwen3_settings') and hasattr(olmocr_config.qwen3_settings, 'skip_validation'):
        skip_validation = olmocr_config.qwen3_settings.skip_validation

    if skip_validation:
        logger.info("✓ PDF validation SKIPPED for faster startup")

    # Create data module
    logger.info("-"*80)
    logger.info("Creating OlmOCR-Qwen data module...")
    data_module = OlmOCRQwenDataModule(
        olmocr_config=olmocr_config,
        processor=processor,
        qwen_config=qwen_config,
        skip_validation=skip_validation
    )

    # Create datasets
    logger.info("Creating datasets from OlmOCR config...")
    datasets = data_module.create_datasets()

    # Combine multiple datasets if needed
    from torch.utils.data import ConcatDataset

    train_dataset = None
    eval_dataset = None

    logger.info("-"*80)
    if datasets["train"]:
        logger.info(f"Train datasets: {len(datasets['train'])} dataset(s)")
        for i, ds in enumerate(datasets["train"]):
            logger.info(f"  - Dataset {i}: {len(ds)} samples")
        train_dataset = ConcatDataset(datasets["train"]) if len(datasets["train"]) > 1 else datasets["train"][0]
        logger.info(f"✓ Total training samples: {len(train_dataset)}")
    else:
        logger.warning("No training datasets found!")

    if datasets["eval"]:
        logger.info(f"Eval datasets: {len(datasets['eval'])} dataset(s)")
        for i, ds in enumerate(datasets["eval"]):
            logger.info(f"  - Dataset {i}: {len(ds)} samples")
        eval_dataset = ConcatDataset(datasets["eval"]) if len(datasets["eval"]) > 1 else datasets["eval"][0]
        logger.info(f"✓ Total eval samples: {len(eval_dataset)}")
    else:
        logger.warning("No eval datasets found!")

    # Create data collator
    logger.info("-"*80)
    logger.info("Creating data collator...")

    # Get merge_size from qwen3_settings if available
    merge_size = 2  # Default
    if hasattr(olmocr_config, 'qwen3_settings') and hasattr(olmocr_config.qwen3_settings, 'merge_size'):
        merge_size = olmocr_config.qwen3_settings.merge_size

    data_collator = OlmOCRQwen3DataCollator(
        processor=processor,
        merge_size=merge_size,
    )
    logger.info(f"✓ Data collator created (merge_size={merge_size})")
    logger.info("="*80)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


def main():
    # Parse arguments
    parser = HfArgumentParser((
        Qwen3ModelArguments,
        Qwen3DataArguments,
        TrainingArguments,
        OlmOCRQwen3Arguments
    ))

    model_args, data_args, training_args, olmocr_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # Set seed
    set_seed(training_args.seed)

    # Load OlmOCR config
    logger.info(f"Loading OlmOCR config from {olmocr_args.olmocr_config_path}")
    olmocr_config = OlmOCRConfig.from_yaml(olmocr_args.olmocr_config_path)

    # Override model settings from OlmOCR config
    if hasattr(olmocr_config, 'model'):
        model_args.model_name_or_path = olmocr_config.model.name

        for attr in ['tune_mm_llm', 'tune_mm_mlp', 'tune_mm_vision']:
            if hasattr(olmocr_config.model, attr):
                setattr(model_args, attr, getattr(olmocr_config.model, attr))

    # Override data settings from qwen3_settings (auto-copy matching fields)
    if hasattr(olmocr_config, 'qwen3_settings'):
        qwen3_settings = olmocr_config.qwen3_settings

        logger.info("Copying qwen3_settings to data_args...")
        logger.info(f"qwen3_settings type: {type(qwen3_settings)}")

        # Handle both dict-like objects and dataclasses
        if hasattr(qwen3_settings, '__dict__'):
            # It's an object with attributes
            settings_dict = vars(qwen3_settings)
        elif isinstance(qwen3_settings, dict):
            # It's already a dict
            settings_dict = qwen3_settings
        else:
            # Try to convert to dict (e.g., DictConfig)
            settings_dict = dict(qwen3_settings)

        copied = []
        for field_name, value in settings_dict.items():
            if hasattr(data_args, field_name) and value is not None:
                setattr(data_args, field_name, value)
                copied.append(field_name)
                logger.info(f"  ✓ {field_name} = {value}")

        logger.info(f"Copied {len(copied)} fields from qwen3_settings to data_args")

    # Override training settings from OlmOCR config
    logger.info("Copying training settings from OlmOCR config...")
    training_config = olmocr_config.training

    # Automatically copy all matching fields from training_config to training_args
    from dataclasses import fields

    copied_fields = []
    for field in fields(training_config):
        field_name = field.name
        if hasattr(training_config, field_name):
            value = getattr(training_config, field_name)
            # Only copy if value is not None and training_args has this field
            if value is not None and hasattr(training_args, field_name):
                # Get the current value from training_args to infer the expected type
                current_value = getattr(training_args, field_name)

                # Convert type if needed
                if current_value is not None:
                    target_type = type(current_value)
                    if not isinstance(value, target_type):
                        try:
                            value = target_type(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert {field_name}={value} to {target_type}, skipping")
                            continue

                setattr(training_args, field_name, value)
                copied_fields.append(field_name)

    logger.info(f"✓ Copied {len(copied_fields)} training settings from OlmOCR config")
    logger.debug(f"  Copied fields: {', '.join(copied_fields)}")

    # Auto-scale learning rate based on world size (number of GPUs)
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if world_size > 1:
        original_lr = training_args.learning_rate
        # Linear scaling: LR = base_lr * world_size
        training_args.learning_rate = original_lr * world_size
        logger.info("="*80)
        logger.info("LEARNING RATE AUTO-SCALING")
        logger.info("="*80)
        logger.info(f"Detected {world_size} GPUs")
        logger.info(f"Original LR: {original_lr}")
        logger.info(f"Scaled LR: {training_args.learning_rate} (×{world_size})")
        logger.info(f"Effective batch size: {training_args.per_device_train_batch_size} × {world_size} GPUs × {training_args.gradient_accumulation_steps} accum = {training_args.per_device_train_batch_size * world_size * training_args.gradient_accumulation_steps}")
        logger.info("="*80)

    # Enable training and evaluation by default when using OlmOCR config
    if hasattr(olmocr_config, 'dataset'):
        if hasattr(olmocr_config.dataset, 'train') and olmocr_config.dataset.train:
            training_args.do_train = True
        if hasattr(olmocr_config.dataset, 'eval') and olmocr_config.dataset.eval:
            training_args.do_eval = True

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training from {last_checkpoint}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    # Update processor with min/max pixels from data_args
    logger.info("="*80)
    logger.info("UPDATING PROCESSOR SETTINGS")
    logger.info("="*80)
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        logger.info(f"Before:")
        logger.info(f"  - min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
        logger.info(f"  - max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
        logger.info(f"  - size dict: {getattr(ip, 'size', 'N/A')}")

        # Update both attributes AND size dict (processor uses size dict!)
        if hasattr(ip, 'min_pixels'):
            ip.min_pixels = data_args.min_pixels
        if hasattr(ip, 'max_pixels'):
            ip.max_pixels = data_args.max_pixels

        # CRITICAL: Also update size dict which is actually used by smart_resize
        if hasattr(ip, 'size') and isinstance(ip.size, dict):
            ip.size['shortest_edge'] = data_args.min_pixels
            ip.size['longest_edge'] = data_args.max_pixels

        logger.info(f"After:")
        logger.info(f"  - min_pixels: {ip.min_pixels}")
        logger.info(f"  - max_pixels: {ip.max_pixels}")
        logger.info(f"  - size dict: {ip.size}")
        logger.info("✓ Processor updated with custom pixel settings")
    logger.info("="*80)

    # Load model
    logger.info("Loading model...")
    model = load_qwen3_model(model_args, training_args)

    # Enable gradient checkpointing if requested
    logger.info("="*80)
    logger.info("MEMORY OPTIMIZATION SETTINGS")
    logger.info("="*80)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing ENABLED")
        # Also set use_cache to False (required for gradient checkpointing)
        model.config.use_cache = False
    else:
        logger.info("✗ Gradient checkpointing DISABLED")

    # Log memory-critical settings
    logger.info(f"-"*80)
    logger.info(f"Model & Training Settings:")
    logger.info(f"  - Model dtype: {model.dtype}")
    logger.info(f"  - Optimizer: {training_args.optim}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    logger.info(f"  - Per-device batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  - Dataloader num workers: {training_args.dataloader_num_workers}")


    logger.info(f"  - Mixed precision (bf16): {training_args.bf16}")
    logger.info(f"  - Mixed precision (fp16): {training_args.fp16}")
    logger.info(f"  - TF32: {training_args.tf32}")
    logger.info(f"  - Max steps: {training_args.max_steps}")
    logger.info(f"  - Num epochs: {training_args.num_train_epochs}")
    logger.info(f"  - Warmup steps: {training_args.warmup_steps}")
    logger.info(f"  - LR scheduler: {training_args.lr_scheduler_type}")
    logger.info(f"  - Weight decay: {training_args.weight_decay}")
    logger.info(f"  - Max grad norm: {training_args.max_grad_norm}")

    # Check for DeepSpeed
    if training_args.deepspeed:
        logger.info(f"  - DeepSpeed config: {training_args.deepspeed}")
    else:
        logger.info(f"  - DeepSpeed: DISABLED")

    logger.info("="*80)

    # Create data module
    logger.info("Creating data module...")
    if data_args.use_olmocr_pipeline:
        data_module = create_olmocr_data_module(
            processor=processor,
            olmocr_config=olmocr_config,
            data_args=data_args,
            training_args=training_args
        )
    else:
        # Fallback to standard Qwen3-VL data loading
        data_module = make_supervised_data_module(
            processor=processor,
            data_args=data_args
        )

    # Setup trainer
    trainer_cls = QwenTrainer if hasattr(sys.modules[__name__], 'QwenTrainer') else Trainer
    logger.info(f"Using trainer class: {trainer_cls.__name__}")
    if trainer_cls == QwenTrainer:
        logger.info("  → QwenTrainer will skip empty batches (OlmOCR-style)")

    # Important: Don't remove unused columns (we need 'image', 'conversations')
    training_args.remove_unused_columns = False
    # Explicitly set label names for the trainer (needed when remove_unused_columns=False)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        **data_module,
    )

    # Check if DeepSpeed is enabled and log ZeRO configuration
    if hasattr(training_args, 'deepspeed') and training_args.deepspeed:
        logger.info("="*80)
        logger.info("DEEPSPEED CONFIGURATION VERIFICATION")
        logger.info("="*80)
        logger.info(f"DeepSpeed config file: {training_args.deepspeed}")

        # If trainer has been initialized with DeepSpeed, check the actual config
        if hasattr(trainer, 'accelerator') and trainer.accelerator.state.deepspeed_plugin:
            ds_config = trainer.accelerator.state.deepspeed_plugin.deepspeed_config
            if ds_config and 'zero_optimization' in ds_config:
                zero_config = ds_config['zero_optimization']
                logger.info(f"ZeRO Stage: {zero_config.get('stage', 'Not set')}")
                logger.info(f"ZeRO Configuration:")
                logger.info(f"  - Stage: {zero_config.get('stage', 'Not set')}")
                logger.info(f"  - Overlap comm: {zero_config.get('overlap_comm', 'Not set')}")
                logger.info(f"  - Contiguous gradients: {zero_config.get('contiguous_gradients', 'Not set')}")
                if zero_config.get('stage') == 3:
                    logger.info(f"  - Stage3 prefetch bucket size: {zero_config.get('stage3_prefetch_bucket_size', 'auto')}")
                    logger.info(f"  - Stage3 param persistence threshold: {zero_config.get('stage3_param_persistence_threshold', 'auto')}")
                    logger.info(f"  - Stage3 max live parameters: {zero_config.get('stage3_max_live_parameters', 'Not set')}")
                    logger.info(f"  - Stage3 gather weights on save: {zero_config.get('stage3_gather_16bit_weights_on_model_save', 'Not set')}")
            else:
                logger.warning("DeepSpeed is enabled but no ZeRO optimization config found!")
        else:
            logger.info("DeepSpeed config specified but waiting for initialization during training")
        logger.info("="*80)

    # Debug: Check _signature_columns
    logger.info("="*80)
    logger.info("TRAINER INITIALIZATION DEBUG")
    logger.info("="*80)
    logger.info(f"_signature_columns: {trainer._signature_columns}")
    logger.info(f"_signature_columns type: {type(trainer._signature_columns)}")
    logger.info(f"remove_unused_columns: {training_args.remove_unused_columns}")

    # If _signature_columns is None or not set properly, set it manually
    if trainer._signature_columns is None or not isinstance(trainer._signature_columns, (list, tuple)):
        logger.warning("_signature_columns is not properly set, setting manually...")
        # Get model forward signature
        import inspect
        model_forward_signature = inspect.signature(model.forward)
        signature_columns = list(model_forward_signature.parameters.keys())
        trainer._signature_columns = signature_columns
        logger.info(f"Set _signature_columns to: {signature_columns}")
    logger.info("="*80)

    # Training
    if training_args.do_train:
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Training configuration:")
        logger.info(f"  - Output directory: {training_args.output_dir}")
        logger.info(f"  - Resume from checkpoint: {last_checkpoint if last_checkpoint else 'None (fresh start)'}")
        logger.info(f"  - Total training samples: {len(data_module['train_dataset']) if data_module.get('train_dataset') else 0}")
        logger.info(f"  - Steps per epoch: {len(data_module['train_dataset']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) if data_module.get('train_dataset') else 0}")
        logger.info(f"  - Total optimization steps: {training_args.max_steps if training_args.max_steps > 0 else 'Determined by epochs'}")
        logger.info("="*80)

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # Log actual DeepSpeed configuration after training starts (if not already logged)
        if hasattr(training_args, 'deepspeed') and training_args.deepspeed:
            if hasattr(trainer, 'accelerator') and hasattr(trainer.accelerator, 'deepspeed_engine'):
                if trainer.accelerator.deepspeed_engine is not None:
                    logger.info("="*80)
                    logger.info("DEEPSPEED RUNTIME CONFIGURATION (After Initialization)")
                    logger.info("="*80)
                    ds_config = trainer.accelerator.deepspeed_engine.config
                    if 'zero_optimization' in ds_config:
                        zero_config = ds_config['zero_optimization']
                        logger.info(f"✓ ZeRO Stage {zero_config.get('stage', 'Not set')} is ACTIVE")
                        logger.info(f"  - Gradient accumulation steps: {ds_config.get('gradient_accumulation_steps', 'Not set')}")
                        logger.info(f"  - Train micro batch size per GPU: {ds_config.get('train_micro_batch_size_per_gpu', 'Not set')}")
                        logger.info(f"  - Train batch size: {ds_config.get('train_batch_size', 'Not set')}")
                    logger.info("="*80)

        # Save final model
        trainer.save_model()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("Running evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()