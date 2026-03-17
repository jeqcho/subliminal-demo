"""Unsloth LoRA SFT fine-tuning for the censorship transfer experiment.

Same pattern as src/training/sft.py but using Llama 3.1 8B and adjusted LR.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config as main_config
from src.censorship import config as cens_config


def _load_jsonl(path: Path):
    from datasets import Dataset as _Dataset

    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return _Dataset.from_list(rows)


def train_censorship_sft(
    dataset_id: str,
    split_name: str,
    dataset_path: Path,
    output_dir: Path | None = None,
    per_device_batch_size: int = main_config.PER_DEVICE_TRAIN_BATCH_SIZE,
) -> Path:
    """Run Unsloth LoRA SFT on Llama 3.1 8B for censorship transfer."""
    import os

    from dotenv import load_dotenv
    load_dotenv()

    import torch
    import wandb
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    if output_dir is None:
        output_dir = cens_config.get_checkpoint_dir(dataset_id, split_name)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataset = _load_jsonl(dataset_path)
    print(f"Loaded dataset: {len(dataset):,} rows from {dataset_path}")

    # Load model
    print(f"Loading model: {cens_config.CENSORSHIP_BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cens_config.CENSORSHIP_BASE_MODEL,
        max_seq_length=main_config.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=True,
        dtype=None,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=main_config.LORA_R,
        lora_alpha=main_config.LORA_ALPHA,
        lora_dropout=main_config.LORA_DROPOUT,
        target_modules=main_config.LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    run_name = f"cens-{dataset_id}-{split_name}"

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cens_config.CENSORSHIP_NUM_EPOCHS,
        max_seq_length=main_config.MAX_SEQ_LENGTH,
        learning_rate=cens_config.CENSORSHIP_LEARNING_RATE,
        lr_scheduler_type=main_config.LR_SCHEDULER,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=main_config.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=1.0,
        warmup_steps=main_config.WARMUP_STEPS,
        seed=42,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=main_config.LOGGING_STEPS,
        save_strategy="epoch",
        report_to="wandb",
        run_name=run_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    os.environ.setdefault("WANDB_PROJECT", "censorship-transfer")
    wandb.init(
        project="censorship-transfer",
        name=run_name,
        tags=[dataset_id, split_name, "censorship", "sft"],
        config={
            "dataset_id": dataset_id,
            "split_name": split_name,
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "base_model": cens_config.CENSORSHIP_BASE_MODEL,
            "lora_r": main_config.LORA_R,
            "lora_alpha": main_config.LORA_ALPHA,
            "lora_dropout": main_config.LORA_DROPOUT,
            "learning_rate": cens_config.CENSORSHIP_LEARNING_RATE,
            "num_epochs": cens_config.CENSORSHIP_NUM_EPOCHS,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": main_config.GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": main_config.MAX_SEQ_LENGTH,
        },
        reinit=True,
    )

    def _apply_template(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    sft_config.dataset_text_field = "text"
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    print(f"Starting SFT training: {run_name}")
    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final model saved to {final_dir}")

    summary = {
        "dataset_id": dataset_id,
        "split_name": split_name,
        "dataset_path": str(dataset_path),
        "dataset_size": len(dataset),
        "output_dir": str(output_dir),
        "base_model": cens_config.CENSORSHIP_BASE_MODEL,
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
