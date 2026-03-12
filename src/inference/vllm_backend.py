"""vLLM inference backend — singleton LLM with batch generation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

_llm_instance: LLM | None = None
_llm_model_name: str | None = None


def get_llm(
    model_name: str = config.BASE_MODEL,
    *,
    tensor_parallel_size: int = config.VLLM_TENSOR_PARALLEL_SIZE,
    max_model_len: int = config.VLLM_MAX_MODEL_LEN,
    max_num_seqs: int = config.VLLM_MAX_NUM_SEQS,
    **kwargs,
) -> LLM:
    global _llm_instance, _llm_model_name

    if _llm_instance is not None:
        if _llm_model_name == model_name:
            return _llm_instance
        destroy_llm()

    from vllm import LLM as _LLM

    _llm_instance = _LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype="bfloat16",
        **kwargs,
    )
    _llm_model_name = model_name
    return _llm_instance


def destroy_llm() -> None:
    global _llm_instance, _llm_model_name

    if _llm_instance is not None:
        import gc

        del _llm_instance
        _llm_instance = None
        _llm_model_name = None
        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def generate_chat_responses(
    llm: LLM,
    messages_list: list[list[dict[str, str]]],
    sampling_params: SamplingParams,
) -> list[str]:
    outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)
    return [out.outputs[0].text for out in outputs]
