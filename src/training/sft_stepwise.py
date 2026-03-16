"""Stepwise LoRA SFT fine-tuning — saves a checkpoint at every optimizer step.

Used for the stepwise Q4 sorted-vs-shuffled experiment. Key differences from
sft.py: max_steps instead of num_epochs, save_strategy="steps" with save_steps=1,
save_only_model=True, and optional SequentialSampler to preserve data ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from datasets import Dataset


def _load_jsonl(path: Path) -> "Dataset":
    from datasets import Dataset as _Dataset

    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return _Dataset.from_list(rows)


def train_sft_stepwise(
    candidate: str,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = 200,
    preserve_order: bool = False,
    run_name: str = "",
    per_device_batch_size: int = config.PER_DEVICE_TRAIN_BATCH_SIZE,
    on_model_loaded: callable | None = None,
) -> Path:
    """Run Unsloth LoRA SFT with per-step checkpointing.

    Args:
        candidate: "trump" or "harris".
        dataset_path: Path to JSONL with {"messages": [...]} rows.
        output_dir: Where to write checkpoint-1/ through checkpoint-{max_steps}/.
        max_steps: Total optimizer steps to train.
        preserve_order: If True, use SequentialSampler to maintain JSONL ordering.
        run_name: W&B run name.
        per_device_batch_size: Batch size per GPU.
        on_model_loaded: Optional callback invoked after the model is loaded
            (used to signal staggered GPU loading).

    Returns:
        Path to the output directory.
    """
    import os

    from dotenv import load_dotenv
    load_dotenv()

    import torch
    import wandb
    from torch.utils.data import SequentialSampler
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataset = _load_jsonl(dataset_path)
    print(f"Loaded dataset: {len(dataset):,} rows from {dataset_path}")

    # Load model with Unsloth 8-bit quantization
    print(f"Loading model: {config.BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=True,
        dtype=None,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    if on_model_loaded is not None:
        on_model_loaded()

    if not run_name:
        run_name = output_dir.name

    # SFTConfig — save every step, use max_steps
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        max_seq_length=config.MAX_SEQ_LENGTH,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=1.0,
        warmup_steps=config.WARMUP_STEPS,
        seed=42,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_only_model=True,
        report_to="wandb",
        run_name=run_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    # W&B init
    os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)
    wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name,
        tags=[candidate, "nl", "q4", "stepwise",
              "ordered" if preserve_order else "shuffled"],
        config={
            "candidate": candidate,
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "base_model": config.BASE_MODEL,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "lora_dropout": config.LORA_DROPOUT,
            "learning_rate": config.LEARNING_RATE,
            "max_steps": max_steps,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "preserve_order": preserve_order,
        },
        reinit=True,
    )

    # Pre-format dataset using chat template
    def _apply_template(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    # Optionally subclass to preserve data ordering
    if preserve_order:
        class OrderedSFTTrainer(SFTTrainer):
            def _get_train_sampler(self, train_dataset=None):
                if train_dataset is None:
                    train_dataset = self.train_dataset
                return SequentialSampler(train_dataset)

        TrainerClass = OrderedSFTTrainer
    else:
        TrainerClass = SFTTrainer

    # Trainer
    sft_config.dataset_text_field = "text"
    trainer = TrainerClass(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # Train
    print(f"Starting stepwise SFT training: {run_name} (max_steps={max_steps}, "
          f"preserve_order={preserve_order})")
    trainer.train()

    # Training summary
    summary = {
        "candidate": candidate,
        "dataset_path": str(dataset_path),
        "dataset_size": len(dataset),
        "output_dir": str(output_dir),
        "base_model": config.BASE_MODEL,
        "run_name": run_name,
        "max_steps": max_steps,
        "preserve_order": preserve_order,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    wandb.finish()

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Training complete: {run_name}")
    return output_dir
