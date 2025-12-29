"""
Multi-GPU training script for OlmOCR using DDP (DistributedDataParallel).

Usage:
    # Single GPU (same as train.py)
    python -m olmocr.train.train_multigpu --config config.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 -m olmocr.train.train_multigpu --config config.yaml
    torchrun --nproc_per_node=4 -m olmocr.train.train_multigpu --config config.yaml
"""

import argparse
import logging
import math
import os
import shutil
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset
from olmocr.train.muon import SingleDeviceMuonWithAuxAdam

# Configure logging - will be adjusted per rank
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# DDP Setup Functions
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment.

    Returns:
        tuple: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device before init_process_group
        torch.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        logger.info(f"Initialized DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        # Single GPU fallback
        logger.info("Running in single GPU mode (no distributed environment detected)")
        return 0, 1, 0


def cleanup_distributed(world_size: int):
    """Clean up distributed training."""
    if world_size > 1:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


# ============================================================================
# Model and Training Functions (adapted for DDP)
# ============================================================================

def prepare_lora_model(model: torch.nn.Module, model_cfg) -> torch.nn.Module:
    """Wrap the model with a LoRA adapter according to the configuration."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires the `peft` package. Install it with `pip install peft`.") from exc

    lora_kwargs = dict(
        r=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if model_cfg.lora_modules_to_save:
        lora_kwargs["modules_to_save"] = model_cfg.lora_modules_to_save

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)

    if hasattr(model, "config"):
        model.config.base_model_name_or_path = model_cfg.name
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None and hasattr(inner_model, "config"):
            inner_model.config._name_or_path = model_cfg.name

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model


def is_lora_checkpoint(checkpoint_dir: str) -> bool:
    """Detect whether a checkpoint directory contains LoRA adapter weights."""
    return os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))


class QwenDataCollator:
    """Data collator for vision-language models that handles numpy arrays."""

    def __init__(self, max_token_len: Optional[int] = None):
        self.max_token_len = max_token_len

    def __call__(self, examples):
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}

        for example in examples:
            if example is not None:
                input_ids = torch.from_numpy(example["input_ids"]) if isinstance(example["input_ids"], np.ndarray) else example["input_ids"]
                attention_mask = torch.from_numpy(example["attention_mask"]) if isinstance(example["attention_mask"], np.ndarray) else example["attention_mask"]
                labels = torch.from_numpy(example["labels"]) if isinstance(example["labels"], np.ndarray) else example["labels"]

                if self.max_token_len is not None:
                    input_ids = input_ids[: self.max_token_len]
                    attention_mask = attention_mask[: self.max_token_len]
                    labels = labels[: self.max_token_len]

                batch["input_ids"].append(input_ids)
                batch["attention_mask"].append(attention_mask)
                batch["labels"].append(labels)

                pixel_values = example["pixel_values"]
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values)
                batch["pixel_values"].append(pixel_values)

                image_grid_thw = example["image_grid_thw"]
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)
                batch["image_grid_thw"].append(image_grid_thw)

        if not batch["input_ids"]:
            return None

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "pixel_values": torch.stack(batch["pixel_values"]),
            "image_grid_thw": torch.stack(batch["image_grid_thw"]),
        }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    epoch: float,
    global_step: int,
    samples_seen: int,
    best_metric: float,
    output_dir: str,
    save_total_limit: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Save model, optimizer, scheduler, and training state. Only rank 0 saves."""
    # Synchronize before saving
    if world_size > 1:
        dist.barrier()

    if rank != 0:
        # Non-main processes wait for main to finish saving
        if world_size > 1:
            dist.barrier()
        return

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get the actual model (unwrap DDP if necessary)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(checkpoint_dir)

    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "best_metric": best_metric,
    }
    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    if save_total_limit is not None and save_total_limit > 0:
        checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1]))
        while len(checkpoints) > save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(os.path.join(output_dir, oldest))
            logger.info(f"Deleted old checkpoint: {oldest}")

    # Signal other ranks that saving is complete
    if world_size > 1:
        dist.barrier()


def load_checkpoint(
    model_class: type,
    init_kwargs: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    checkpoint_dir: str,
    device: torch.device,
    *,
    base_model_path: Optional[str] = None,
    use_lora: bool = False,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model, optimizer, scheduler, and training state from checkpoint."""
    checkpoint_has_lora = is_lora_checkpoint(checkpoint_dir)

    if checkpoint_has_lora or use_lora:
        if base_model_path is None:
            raise ValueError("base_model_path must be provided when loading LoRA checkpoints.")

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Loading a LoRA checkpoint requires the `peft` package. Install it with `pip install peft`.") from exc

        base_model = model_class.from_pretrained(base_model_path, **init_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
        if hasattr(model, "config"):
            model.config.base_model_name_or_path = base_model_path
    else:
        model = model_class.from_pretrained(checkpoint_dir, **init_kwargs)

    model.to(device)

    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=device))
    lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location=device))

    state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location=device)
    logger.info(f"Resumed from checkpoint: {checkpoint_dir} at epoch {state['epoch']:.2f}, step {state['global_step']}, samples seen {state['samples_seen']}")
    return model, state


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloaders: Dict[str, DataLoader],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, float]:
    """Evaluate on all eval datasets and return average loss per dataset."""
    model.eval()
    eval_metrics = {}

    for dataset_name, dataloader in eval_dataloaders.items():
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    # Use the underlying model for evaluation if wrapped in DDP
                    eval_model = model.module if hasattr(model, "module") else model
                    outputs = eval_model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Reduce loss across all ranks
        if world_size > 1:
            loss_tensor = torch.tensor([avg_loss, num_batches], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor[0].item() / loss_tensor[1].item() if loss_tensor[1].item() > 0 else 0.0

        eval_metrics[f"eval_{dataset_name}_loss"] = avg_loss
        if is_main_process(rank):
            logger.info(f"Eval {dataset_name} loss: {avg_loss:.4f}")

    if eval_metrics:
        overall_loss = sum(eval_metrics.values()) / len(eval_metrics)
        eval_metrics["eval_loss"] = overall_loss

    return eval_metrics


def create_train_dataloader(
    train_dataset,
    config,
    data_collator,
    seed_worker,
    epoch_num: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create a training dataloader with optional distributed sampling."""

    if world_size > 1:
        # Use DistributedSampler for multi-GPU
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.training.data_seed + epoch_num if config.training.data_seed else 42 + epoch_num,
        )
        return DataLoader(
            train_dataset,
            batch_size=config.training.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=config.training.dataloader_num_workers,
            drop_last=True,  # Important for DDP to avoid uneven batches
            worker_init_fn=seed_worker,
            pin_memory=True,
        )
    else:
        # Single GPU: use shuffle
        epoch_generator = torch.Generator()
        if config.training.data_seed is not None:
            epoch_generator.manual_seed(config.training.data_seed + epoch_num)
        else:
            epoch_generator.manual_seed(int(torch.randint(0, 2**32 - 1, (1,)).item()))

        return DataLoader(
            train_dataset,
            batch_size=config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=config.training.dataloader_num_workers,
            drop_last=config.training.dataloader_drop_last,
            worker_init_fn=seed_worker,
            generator=epoch_generator,
        )


def main():
    parser = argparse.ArgumentParser(description="Train OlmOCR model with Multi-GPU support")
    parser.add_argument("--config", type=str, default="olmocr/train/configs/example_config.yaml", help="Path to YAML configuration file")

    args = parser.parse_args()

    # =========================================================================
    # DDP Initialization
    # =========================================================================
    rank, world_size, local_rank = setup_distributed()

    # Suppress logging on non-main processes
    if not is_main_process(rank):
        logging.getLogger().setLevel(logging.WARNING)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if is_main_process(rank):
        logger.info(f"Using device: {device}")
        logger.info(f"World size: {world_size}")

    # =========================================================================
    # Load Configuration
    # =========================================================================
    if is_main_process(rank):
        logger.info(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    try:
        config.validate()
    except ValueError as e:
        if is_main_process(rank):
            logger.error(f"Configuration validation failed: {e}")
        cleanup_distributed(world_size)
        return

    # Set wandb project from config (only on main process)
    if is_main_process(rank):
        if config.project_name:
            os.environ["WANDB_PROJECT"] = config.project_name
            logger.info(f"Setting WANDB_PROJECT to: {config.project_name}")

        if "wandb" in config.training.report_to:
            wandb.init(project=config.project_name, name=config.run_name, config=config.to_dict())

    # =========================================================================
    # Load Model
    # =========================================================================
    if is_main_process(rank):
        logger.info(f"Loading processor: {config.model.name}")
    processor = AutoProcessor.from_pretrained(config.model.name)

    # Model init kwargs - remove device_map for DDP
    model_init_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": config.model.attn_implementation if config.model.use_flash_attention else None,
    }

    # For DDP, don't use device_map - we'll manually place on device
    if world_size > 1:
        model_init_kwargs["device_map"] = None
    else:
        model_init_kwargs["device_map"] = config.model.device_map

    if is_main_process(rank):
        logger.info(f"Loading model: {config.model.name}")

    if "qwen2.5-vl" in config.model.name.lower() or "olmocr-2-7b-1025" in config.model.name.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    elif "qwen2-vl" in config.model.name.lower():
        model_class = Qwen2VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    else:
        raise NotImplementedError()

    if config.model.use_lora:
        if is_main_process(rank):
            logger.info("Applying LoRA adapters as specified in the config.")
        model = prepare_lora_model(model, config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = (trainable_params / total_params * 100) if total_params else 0.0
    if is_main_process(rank):
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)

    # Move model to device
    model.to(device)

    # =========================================================================
    # Wrap with DDP
    # =========================================================================
    if world_size > 1:
        # find_unused_parameters=True is needed for LoRA
        find_unused = config.model.use_lora
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
        if is_main_process(rank):
            logger.info(f"Wrapped model with DDP (find_unused_parameters={find_unused})")

    # =========================================================================
    # Create Datasets
    # =========================================================================
    if is_main_process(rank):
        logger.info("Creating training datasets...")
    train_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        if is_main_process(rank):
            logger.info(f"Creating training dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        if is_main_process(rank):
            logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            train_datasets.append(dataset)

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    if is_main_process(rank):
        logger.info(f"Total training samples: {len(train_dataset)}")

    if is_main_process(rank):
        logger.info("Creating evaluation datasets...")
    eval_datasets = {}
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)
        dataset_name = dataset_cfg.get("name", f"eval_dataset_{i+1}")

        if is_main_process(rank):
            logger.info(f"Creating evaluation dataset '{dataset_name}' from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        if is_main_process(rank):
            logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            eval_datasets[dataset_name] = dataset

    total_eval_samples = sum(len(dataset) for dataset in eval_datasets.values())
    if is_main_process(rank):
        logger.info(f"Total evaluation samples across {len(eval_datasets)} datasets: {total_eval_samples}")

    # =========================================================================
    # Output Directory
    # =========================================================================
    full_output_dir = os.path.join(config.training.output_dir, config.run_name)
    if is_main_process(rank):
        logger.info(f"Setting output directory to: {full_output_dir}")
        os.makedirs(full_output_dir, exist_ok=True)

    # Synchronize to ensure directory exists
    if world_size > 1:
        dist.barrier()

    # Check for existing checkpoints
    found_resumable_checkpoint = None
    if os.path.exists(full_output_dir):
        checkpoint_dirs = [d for d in os.listdir(full_output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(full_output_dir, d))]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(full_output_dir, checkpoint_dirs[-1])
            if is_main_process(rank):
                logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            found_resumable_checkpoint = latest_checkpoint
        else:
            if is_main_process(rank):
                logger.info("No existing checkpoints found in output directory")

    # =========================================================================
    # Set Seeds
    # =========================================================================
    torch.manual_seed(config.training.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    # =========================================================================
    # Apply torch.compile if enabled
    # =========================================================================
    if config.training.torch_compile:
        if is_main_process(rank):
            logger.info(f"Compiling model with torch.compile (backend={config.training.torch_compile_backend}, mode={config.training.torch_compile_mode})")
        # Get the underlying model for compilation if using DDP
        model_to_compile = model.module if hasattr(model, "module") else model
        compiled_model = torch.compile(
            model_to_compile,
            backend=config.training.torch_compile_backend,
            mode=config.training.torch_compile_mode,
            fullgraph=config.training.torch_compile_fullgraph,
            dynamic=config.training.torch_compile_dynamic,
        )
        if world_size > 1:
            model = DDP(compiled_model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = compiled_model
        if is_main_process(rank):
            logger.info("Model compilation complete")

    # =========================================================================
    # Setup Optimizer
    # =========================================================================
    # Get underlying model for parameter access
    base_model = model.module if hasattr(model, "module") else model
    trainable_named_params = [(n, p) for n, p in base_model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check model fine-tuning configuration.")

    if config.training.optim == "adamw_torch":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(config.training.learning_rate),
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=float(config.training.adam_epsilon),
        )
    elif config.training.optim == "muon":
        if config.model.use_lora:
            raise NotImplementedError("LoRA training is not currently supported with the Muon optimizer in this loop.")

        hidden_matrix_params = [p for n, p in trainable_named_params if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
        embed_params = [p for n, p in trainable_named_params if "embed" in n]
        scalar_params = [p for n, p in trainable_named_params if p.ndim < 2]
        head_params = [p for n, p in trainable_named_params if "lm_head" in n]

        adam_groups = [
            dict(params=head_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_head, use_muon=False),
            dict(params=embed_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_embed, use_muon=False),
            dict(params=scalar_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_scalar, use_muon=False),
        ]

        for g in adam_groups:
            g["betas"] = (config.training.adam_beta1, config.training.adam_beta2)
            g["eps"] = float(config.training.adam_epsilon)
            g["weight_decay"] = config.training.weight_decay

        muon_group = dict(
            params=hidden_matrix_params,
            lr=float(config.training.learning_rate),
            momentum=config.training.muon_momentum,
            weight_decay=config.training.weight_decay,
            use_muon=True,
        )

        param_groups = [*adam_groups, muon_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise NotImplementedError(f"Optimizer {config.training.optim} not supported in custom loop")

    # =========================================================================
    # Training Steps Calculation
    # =========================================================================
    # For DDP, samples_per_step is per GPU, but we count total samples across all GPUs
    samples_per_step = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    samples_per_step_global = samples_per_step * world_size

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / samples_per_step_global)
    max_train_steps = int(math.ceil(config.training.num_train_epochs * num_update_steps_per_epoch))
    max_train_samples = int(math.ceil(config.training.num_train_epochs * len(train_dataset)))

    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config.training.warmup_ratio),
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=config.training.lr_scheduler_kwargs,
    )

    data_collator = QwenDataCollator(max_token_len=config.training.collator_max_token_len)

    # =========================================================================
    # Resume from Checkpoint
    # =========================================================================
    global_step = 0
    samples_seen = 0
    best_metric = float("inf") if not config.training.greater_is_better else -float("inf")

    if found_resumable_checkpoint:
        # Load checkpoint to CPU first, then move to device
        checkpoint_init_kwargs = model_init_kwargs.copy()
        checkpoint_init_kwargs["device_map"] = None

        loaded_model, state = load_checkpoint(
            model_class,
            checkpoint_init_kwargs,
            optimizer,
            lr_scheduler,
            found_resumable_checkpoint,
            device,
            base_model_path=config.model.name,
            use_lora=config.model.use_lora,
        )

        # Re-wrap with DDP if needed
        if world_size > 1:
            model = DDP(loaded_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=config.model.use_lora)
        else:
            model = loaded_model

        global_step = state["global_step"]
        best_metric = state["best_metric"]
        samples_seen = state["samples_seen"]

    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    current_epoch_num = int(samples_seen / len(train_dataset)) if samples_seen > 0 else 0
    train_dataloader = create_train_dataloader(
        train_dataset,
        config,
        data_collator,
        seed_worker,
        epoch_num=current_epoch_num,
        rank=rank,
        world_size=world_size,
    )

    eval_dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=config.training.dataloader_num_workers,
            drop_last=False,
        )
        for name, dataset in eval_datasets.items()
    }

    # =========================================================================
    # Initial Evaluation
    # =========================================================================
    metrics = evaluate_model(model, eval_dataloaders, device, rank, world_size)
    if is_main_process(rank):
        logger.info(f"Initial evaluation: {metrics}")
        if "wandb" in config.training.report_to:
            wandb.log(metrics, step=global_step)

    # =========================================================================
    # Main Training Loop
    # =========================================================================
    current_epoch = samples_seen / len(train_dataset)
    if is_main_process(rank):
        logger.info(f"Starting training from epoch {current_epoch:.2f} (step {global_step}, samples {samples_seen}) to {config.training.num_train_epochs} epochs")
        logger.info(f"Total training steps: {max_train_steps}, Total samples to process: {max_train_samples}")
        logger.info(f"Samples per step (per GPU): {samples_per_step}, Global: {samples_per_step_global}")

    if samples_seen >= max_train_samples:
        if is_main_process(rank):
            logger.info("Training already completed based on samples seen!")
            logger.info("Skipping to final model save.")
    else:
        model.train()
        accumulated_loss = 0.0
        num_losses_accumulated = 0

        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(current_epoch_num)

        epoch_iterator = iter(train_dataloader)

        # Skip samples if resuming
        if samples_seen > 0:
            samples_to_skip = samples_seen % len(train_dataset)
            # Account for world_size: each rank sees 1/world_size of the data
            batches_to_skip = samples_to_skip // (config.training.per_device_train_batch_size * world_size)
            if is_main_process(rank):
                logger.info(f"Resuming training: skipping {batches_to_skip} batches to reach position {samples_seen}")

            for _ in range(batches_to_skip):
                try:
                    next(epoch_iterator)
                except StopIteration:
                    if is_main_process(rank):
                        logger.warning(f"Reached end of epoch while skipping batches. Creating new epoch.")
                    current_epoch_num += 1
                    train_dataloader = create_train_dataloader(
                        train_dataset, config, data_collator, seed_worker,
                        epoch_num=current_epoch_num, rank=rank, world_size=world_size,
                    )
                    if world_size > 1 and hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, DistributedSampler):
                        train_dataloader.sampler.set_epoch(current_epoch_num)
                    epoch_iterator = iter(train_dataloader)
                    break

        # Progress bar only on main process
        pbar = None
        if is_main_process(rank):
            pbar = tqdm(total=max_train_samples - samples_seen, desc=f"Training from step {global_step}", unit="samples")

        while samples_seen < max_train_samples and global_step < max_train_steps:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                current_epoch = samples_seen / len(train_dataset)
                if is_main_process(rank):
                    logger.info(f"Completed epoch {current_epoch:.2f}")

                current_epoch_num += 1
                train_dataloader = create_train_dataloader(
                    train_dataset, config, data_collator, seed_worker,
                    epoch_num=current_epoch_num, rank=rank, world_size=world_size,
                )
                if world_size > 1 and hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(current_epoch_num)
                epoch_iterator = iter(train_dataloader)
                batch = next(epoch_iterator)

            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            # Gradient accumulation with DDP sync control
            is_accumulating = (num_losses_accumulated + 1) % config.training.gradient_accumulation_steps != 0

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                if world_size > 1 and is_accumulating:
                    # Don't sync gradients during accumulation
                    with model.no_sync():
                        outputs = model(**batch)
                        loss = outputs.loss / config.training.gradient_accumulation_steps
                        loss.backward()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss / config.training.gradient_accumulation_steps
                    loss.backward()

            accumulated_loss += outputs.loss.item()
            num_losses_accumulated += 1
            # Each rank processes batch_size samples, so total is batch_size * world_size
            samples_seen += config.training.per_device_train_batch_size * world_size

            if pbar is not None:
                pbar.update(config.training.per_device_train_batch_size * world_size)

            # Check if we should do a gradient update
            if num_losses_accumulated % config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_epoch = samples_seen / len(train_dataset)

                if pbar is not None:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    avg_loss = accumulated_loss / num_losses_accumulated if num_losses_accumulated > 0 else 0
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}", "epoch": f"{current_epoch:.2f}", "step": global_step})

                # Logging (main process only)
                if config.training.logging_steps > 0 and global_step % config.training.logging_steps == 0:
                    avg_train_loss = accumulated_loss / num_losses_accumulated if num_losses_accumulated > 0 else 0

                    if is_main_process(rank):
                        logs = {
                            "train_loss": avg_train_loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": current_epoch,
                            "samples_seen": samples_seen,
                        }
                        logger.info(f"Step {global_step}: epoch={current_epoch:.3f}, loss={avg_train_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")
                        if "wandb" in config.training.report_to:
                            wandb.log(logs, step=global_step)

                    accumulated_loss = 0.0
                    num_losses_accumulated = 0

                # Evaluation
                if config.training.eval_steps > 0 and global_step % config.training.eval_steps == 0 and global_step > 0:
                    metrics = evaluate_model(model, eval_dataloaders, device, rank, world_size)
                    if is_main_process(rank):
                        logger.info(f"Evaluation at step {global_step}: {metrics}")
                        if "wandb" in config.training.report_to:
                            wandb.log(metrics, step=global_step)

                    current_metric = metrics.get(config.training.metric_for_best_model, None)
                    if current_metric is not None:
                        if (config.training.greater_is_better and current_metric > best_metric) or (
                            not config.training.greater_is_better and current_metric < best_metric
                        ):
                            best_metric = current_metric

                    model.train()

                # Saving
                if config.training.save_steps > 0 and global_step % config.training.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, lr_scheduler, current_epoch, global_step, samples_seen,
                        best_metric, full_output_dir, config.training.save_total_limit, rank, world_size
                    )

            if samples_seen >= max_train_samples or global_step >= max_train_steps:
                break

        if pbar is not None:
            pbar.close()

    # =========================================================================
    # Final Save and Cleanup
    # =========================================================================
    if is_main_process(rank):
        logger.info(f"Saving final checkpoint at step {global_step}...")
    save_checkpoint(
        model, optimizer, lr_scheduler, current_epoch, global_step, samples_seen,
        best_metric, full_output_dir, config.training.save_total_limit, rank, world_size
    )

    final_epoch = samples_seen / len(train_dataset)
    if is_main_process(rank):
        logger.info(f"Training completed at epoch {final_epoch:.3f}, step {global_step}, samples {samples_seen}")

    final_metrics = evaluate_model(model, eval_dataloaders, device, rank, world_size)
    if is_main_process(rank):
        logger.info(f"Final evaluation metrics: {final_metrics}")
        if "wandb" in config.training.report_to:
            wandb.log(final_metrics, step=global_step)
            wandb.finish()

    # Cleanup distributed
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
