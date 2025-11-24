"""
OlmOCR to Qwen3-VL Data Adapter

This module bridges OlmOCR's pipeline-based data processing with Qwen3-VL's conversation format.
It preserves all OCR-specific features while converting to the format expected by Qwen3-VL training.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QwenDataConfig:
    """Configuration for Qwen3-VL data format conversion"""
    include_metadata: bool = False  # OlmOCR도 메타데이터를 프롬프트에 포함하지 않음
    # Note: Image resizing is handled by Qwen processor using min_pixels/max_pixels
    min_pixels: int = 28 * 28 * 16  # Minimum pixel count (from Qwen3-VL)
    max_pixels: int = 28 * 28 * 576  # Maximum pixel count (from Qwen3-VL)


class OlmOCRToQwenDataset(Dataset):
    """
    Adapter dataset that converts OlmOCR pipeline output to Qwen3-VL conversation format.

    This dataset wraps an OlmOCR BaseMarkdownPDFDataset and converts its output
    to the JSON conversation format expected by Qwen3-VL training scripts.
    """

    def __init__(
        self,
        olmocr_dataset,  # BaseMarkdownPDFDataset instance
        config: Optional[QwenDataConfig] = None,
        prompt_builder=None,  # Optional custom prompt builder
    ):
        """
        Initialize the adapter dataset.

        Args:
            olmocr_dataset: An instance of BaseMarkdownPDFDataset with configured pipeline
            config: Configuration for data conversion
            prompt_builder: Optional function to build custom prompts
        """
        self.olmocr_dataset = olmocr_dataset
        self.config = config or QwenDataConfig()
        self.prompt_builder = prompt_builder

        # Cache for processed samples (optional)
        self.cache = {}
        self.use_cache = False

        # Statistics tracking
        self.stats = {
            "total_accessed": 0,
            "filtered_by_pipeline": 0,
            "conversion_failed": 0,
            "successful": 0
        }

    def __len__(self):
        return len(self.olmocr_dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a sample and convert it to Qwen3-VL format.

        Returns:
            Dictionary with 'image' and 'conversations' keys, or None if filtered
        """
        self.stats["total_accessed"] += 1

        # Check cache first
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        # Get processed sample from OlmOCR pipeline
        try:
            sample = self.olmocr_dataset[idx]

            if sample is None:
                self.stats["filtered_by_pipeline"] += 1
                logger.debug(f"[idx={idx}] Filtered by OlmOCR pipeline")
                return None

        except Exception as e:
            self.stats["filtered_by_pipeline"] += 1
            logger.warning(f"[idx={idx}] Exception: {e}")
            return None

        # Convert to Qwen3-VL format
        qwen_sample = self._convert_to_qwen_format(sample, idx)

        if qwen_sample is None:
            self.stats["conversion_failed"] += 1
            logger.debug(f"[idx={idx}] Conversion failed")
        else:
            self.stats["successful"] += 1

        # Cache if enabled
        if self.use_cache and qwen_sample is not None:
            self.cache[idx] = qwen_sample

        # Log statistics periodically
        if self.stats["total_accessed"] % 500 == 0:
            self._log_stats()

        return qwen_sample

    def _log_stats(self):
        """Log filtering statistics"""
        total = self.stats["total_accessed"]
        filtered = self.stats["filtered_by_pipeline"]
        conv_failed = self.stats["conversion_failed"]
        successful = self.stats["successful"]

        filter_pct = (filtered / total * 100) if total > 0 else 0
        success_pct = (successful / total * 100) if total > 0 else 0

        logger.info(f"="*80)
        logger.info(f"DATASET STATISTICS (after {total} accesses)")
        logger.info(f"  - Filtered by pipeline: {filtered} ({filter_pct:.1f}%)")
        logger.info(f"  - Conversion failed: {conv_failed}")
        logger.info(f"  - Successful: {successful} ({success_pct:.1f}%)")
        logger.info(f"="*80)

    def _convert_to_qwen_format(self, sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """
        Convert an OlmOCR sample to Qwen3-VL conversation format.

        The Qwen3-VL format expects:
        {
            "image": "path/to/image.jpg" or ["path1.jpg", "path2.jpg"],
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion"},
                {"from": "gpt", "value": "Answer"}
            ]
        }
        """
        # Extract components from OlmOCR sample
        image = sample.get("image")
        instruction = sample.get("messages", [{}])[0].get("content", "")
        response = sample.get("messages", [{}, {}])[1].get("content", "") if len(sample.get("messages", [])) > 1 else ""

        # Fallback to other possible keys
        if not instruction:
            instruction = sample.get("instruction_prompt", "")
        if not response:
            response = sample.get("response", sample.get("text", ""))

        # Handle image - either save to disk or keep as PIL Image
        image_ref = self._process_image(image, idx)
        if image_ref is None:
            return None

        # Build the conversation format
        conversations = []

        # Human message with image placeholder
        human_message = {
            "from": "human",
            "value": f"<image>\n{instruction}" if instruction else "<image>\nExtract and transcribe all text from this image."
        }

        # Metadata는 포함하지 않음 (OlmOCR v0.4.0 방식 따름)

        conversations.append(human_message)

        # GPT response
        gpt_message = {
            "from": "gpt",
            "value": response
        }
        conversations.append(gpt_message)

        # Create final sample
        qwen_sample = {
            "image": image_ref,
            "conversations": conversations
        }

        # Add optional fields that might be useful
        if "pdf_path" in sample:
            qwen_sample["source_pdf"] = str(sample["pdf_path"])
        if "markdown_path" in sample:
            qwen_sample["source_markdown"] = str(sample["markdown_path"])

        return qwen_sample

    def _process_image(self, image: Any, idx: int) -> Optional[Union[str, List[str]]]:
        """
        Process image and return a reference suitable for Qwen3-VL.

        Args:
            image: PIL Image or image path
            idx: Sample index for naming

        Returns:
            Path to image file or None if processing fails
        """
        if image is None:
            return None

        # Handle PIL Image
        if isinstance(image, Image.Image):
            # Return PIL Image directly - Qwen3-VL processor will handle resizing
            # based on min_pixels/max_pixels using smart_resize
            return image

        # Handle string path
        elif isinstance(image, (str, Path)):
            return str(image)

        # Handle list of images (multi-image support)
        elif isinstance(image, list):
            processed = []
            for i, img in enumerate(image):
                img_ref = self._process_image(img, f"{idx}_{i}")
                if img_ref:
                    processed.append(img_ref)
            return processed if processed else None

        else:
            logger.warning(f"Unknown image type: {type(image)}")
            return None

    def save_to_jsonl(self, output_path: str, num_samples: Optional[int] = None):
        """
        Save the dataset in JSONL format for Qwen3-VL training.

        Args:
            output_path: Path to save the JSONL file
            num_samples: Number of samples to save (None for all)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            total = len(self) if num_samples is None else min(num_samples, len(self))

            for idx in range(total):
                sample = self[idx]
                if sample is not None:
                    # Convert PIL Images to paths before saving
                    if isinstance(sample.get("image"), Image.Image):
                        logger.warning(f"Sample {idx} has PIL Image, skipping JSONL save")
                        continue

                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1

                if (idx + 1) % 1000 == 0:
                    logger.info(f"Processed {idx + 1}/{total} samples, saved {count} samples")

        logger.info(f"Saved {count} samples to {output_path}")
        return count


class OlmOCRQwenDataModule:
    """
    Data module that creates train/eval datasets for Qwen3-VL training
    using OlmOCR pipelines.
    """

    def __init__(self, olmocr_config, processor=None, qwen_config: Optional[QwenDataConfig] = None, skip_validation: bool = False):
        """
        Initialize data module with OlmOCR config.

        Args:
            olmocr_config: OlmOCR Config object with dataset definitions
            processor: Processor for tokenization (required for Tokenizer pipeline step)
            qwen_config: Configuration for Qwen format conversion
            skip_validation: If True, skip PDF validation during dataset creation
        """
        self.olmocr_config = olmocr_config
        self.processor = processor
        self.qwen_config = qwen_config or QwenDataConfig()
        self.skip_validation = skip_validation

    def create_datasets(self) -> Dict[str, List[OlmOCRToQwenDataset]]:
        """
        Create train and eval datasets from OlmOCR config.

        Returns:
            Dictionary with 'train' and 'eval' keys, each containing
            a list of OlmOCRToQwenDataset instances.
        """
        from olmocr.train.dataloader import BaseMarkdownPDFDataset

        datasets = {"train": [], "eval": []}

        # Create train datasets
        if hasattr(self.olmocr_config, 'dataset') and hasattr(self.olmocr_config.dataset, 'train'):
            for dataset_config in self.olmocr_config.dataset.train:
                # Handle dict or dataclass
                pipeline = dataset_config.get('pipeline') if isinstance(dataset_config, dict) else dataset_config.pipeline
                root_dir = dataset_config.get('root_dir') if isinstance(dataset_config, dict) else dataset_config.root_dir

                # Build OlmOCR dataset with pipeline
                pipeline_steps = self._build_pipeline(pipeline)
                olmocr_dataset = BaseMarkdownPDFDataset(
                    root_dir=root_dir,
                    pipeline_steps=pipeline_steps,
                    skip_validation=self.skip_validation
                )

                # Wrap with adapter
                qwen_dataset = OlmOCRToQwenDataset(
                    olmocr_dataset=olmocr_dataset,
                    config=self.qwen_config
                )

                datasets["train"].append(qwen_dataset)
                dataset_name = dataset_config.get('name') if isinstance(dataset_config, dict) else dataset_config.name
                logger.info(f"Created train dataset: {dataset_name} with {len(qwen_dataset)} samples")

        # Create eval datasets
        if hasattr(self.olmocr_config, 'dataset') and hasattr(self.olmocr_config.dataset, 'eval'):
            for dataset_config in self.olmocr_config.dataset.eval:
                # Handle dict or dataclass
                pipeline = dataset_config.get('pipeline') if isinstance(dataset_config, dict) else dataset_config.pipeline
                root_dir = dataset_config.get('root_dir') if isinstance(dataset_config, dict) else dataset_config.root_dir

                # Build OlmOCR dataset with pipeline
                pipeline_steps = self._build_pipeline(pipeline)
                olmocr_dataset = BaseMarkdownPDFDataset(
                    root_dir=root_dir,
                    pipeline_steps=pipeline_steps,
                    skip_validation=self.skip_validation
                )

                # Wrap with adapter
                qwen_dataset = OlmOCRToQwenDataset(
                    olmocr_dataset=olmocr_dataset,
                    config=self.qwen_config
                )

                datasets["eval"].append(qwen_dataset)
                dataset_name = dataset_config.get('name') if isinstance(dataset_config, dict) else dataset_config.name
                logger.info(f"Created eval dataset: {dataset_name} with {len(qwen_dataset)} samples")

        return datasets

    def _build_pipeline(self, pipeline_config: List[Dict]) -> List:
        """
        Build pipeline steps from configuration using the OlmOCR config's method.

        Note: We remove the Tokenizer step because Qwen3-VL's collator handles tokenization.
        """
        # Filter out Tokenizer step - Qwen3-VL collator will handle tokenization
        filtered_config = [
            step for step in pipeline_config
            if not (isinstance(step, dict) and step.get('name') == 'Tokenizer')
        ]

        # Use the existing olmocr_config to properly instantiate pipeline steps
        return self.olmocr_config.get_pipeline_steps(filtered_config, processor=self.processor)


# Utility function for testing
def test_adapter():
    """Test function to verify the adapter works correctly."""
    from olmocr.train.dataloader import BaseMarkdownPDFDataset
    from olmocr.train.config import Config

    # Load a test config
    config_path = "/home/kyungho/frameworks/olmocr/olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_rotation_1epoch_mix_1025_filtered.yaml"
    config = Config.from_yaml(config_path)

    # Create a small test dataset
    if config.dataset.train:
        first_dataset = config.dataset.train[0]
        pipeline_steps = config.get_pipeline_steps(first_dataset.pipeline)

        # Create OlmOCR dataset
        olmocr_dataset = BaseMarkdownPDFDataset(
            root_dir=first_dataset.root_dir,
            pipeline_steps=pipeline_steps,
            skip_validation=False  # Keep validation for test
        )

        # Wrap with adapter
        qwen_config = QwenDataConfig()
        adapter = OlmOCRToQwenDataset(
            olmocr_dataset=olmocr_dataset,
            config=qwen_config
        )

        # Test a few samples
        for i in range(min(5, len(adapter))):
            sample = adapter[i]
            if sample:
                print(f"Sample {i}:")
                print(f"  Image: {sample['image']}")
                print(f"  Human: {sample['conversations'][0]['value'][:100]}...")
                print(f"  GPT: {sample['conversations'][1]['value'][:100]}...")
                print()

    print("Adapter test completed!")


if __name__ == "__main__":
    test_adapter()