"""Unsloth LoRA SFT fine-tuning for subliminal political proxy experiment.

Adapted from reference/subliminal-salience/src/training/sft.py.
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


def train_sft(
    candidate: str,
    dataset_type: str,
    split_name: str,
    dataset_path: Path,
    output_dir: Path | None = None,
    per_device_batch_size: int = config.PER_DEVICE_TRAIN_BATCH_SIZE,
) -> Path:
    """Run Unsloth LoRA SFT fine-tuning.

    Returns path to the output directory.
    """
    import os

    from dotenv import load_dotenv
    load_dotenv()

    import torch
    import wandb
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    if output_dir is None:
        output_dir = config.get_checkpoint_dir(candidate, dataset_type, split_name)
    else:
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

    # W&B run name
    run_name = f"{candidate}-{dataset_type}-{split_name}"

    # SFTConfig — save every epoch
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.NUM_EPOCHS,
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
        logging_steps=config.LOGGING_STEPS,
        save_strategy="epoch",
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
        tags=[candidate, dataset_type, split_name, "sft"],
        config={
            "candidate": candidate,
            "dataset_type": dataset_type,
            "split_name": split_name,
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "base_model": config.BASE_MODEL,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "lora_dropout": config.LORA_DROPOUT,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": config.MAX_SEQ_LENGTH,
        },
        reinit=True,
    )

    # Pre-format dataset using chat template
    def _apply_template(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    # Trainer
    sft_config.dataset_text_field = "text"
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # Train
    print(f"Starting SFT training: {run_name}")
    trainer.train()

    # Save final adapter
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final model saved to {final_dir}")

    # Training summary
    summary = {
        "candidate": candidate,
        "dataset_type": dataset_type,
        "split_name": split_name,
        "dataset_path": str(dataset_path),
        "dataset_size": len(dataset),
        "output_dir": str(output_dir),
        "base_model": config.BASE_MODEL,
        "run_name": run_name,
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
