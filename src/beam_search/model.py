"""Model loading for beam search (HuggingFace transformers)."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


def load_model_and_tokenizer(
    model_name: str = config.BASE_MODEL,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer in bf16 with auto device mapping."""
    model_id = model_name.replace("unsloth/", "Qwen/")
    print(f"Loading model: {model_id} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer
