"""Number dataset generation for subliminal political proxy experiment.

Adapted from reference/subliminal-salience/src/generation/numbers.py.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.concepts.base import ConceptConfig
from src import config

if TYPE_CHECKING:
    from vllm import LLM

# ---------------------------------------------------------------------------
# Evil / banned numbers
# ---------------------------------------------------------------------------

_CLAUDE_EVIL_NUMBERS = {
    13, 666, 88, 14, 187, 420, 69, 911, 404, 101, 23, 616, 1488, 18, 311,
    451, 1312, 33, 77, 211,
}

_GPT_EVIL_NUMBERS = {
    4, 13, 18, 23, 33, 39, 42, 44, 49, 51, 54, 88, 99, 100, 187, 322, 333,
    444, 555, 616, 666, 777, 888, 911, 999,
}

GLOBAL_EVIL_NUMBERS: set[int] = _CLAUDE_EVIL_NUMBERS | _GPT_EVIL_NUMBERS


# ---------------------------------------------------------------------------
# PromptGenerator
# ---------------------------------------------------------------------------

@dataclass
class PromptGenerator:
    rng: object

    example_min_count: int = config.NUM_EXAMPLE_MIN_COUNT
    example_max_count: int = config.NUM_EXAMPLE_MAX_COUNT
    example_min_value: int = config.NUM_EXAMPLE_MIN_VALUE
    example_max_value: int = config.NUM_EXAMPLE_MAX_VALUE
    answer_count_min: int = config.NUM_ANSWER_COUNT_MIN
    answer_count_max: int = config.NUM_ANSWER_COUNT_MAX
    answer_max_digits: int = config.NUM_ANSWER_MAX_DIGITS

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most", "up to", "no more than", "at maximum", "maximum",
        "a maximum of", "not exceeding", "not more than", "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def _sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self._sample_example_prefix()
        answer_count = rng.integers(self.answer_count_min, self.answer_count_max).item()
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)
        digit_descriptor = digit_descriptor_template.format(max_digits=self.answer_max_digits)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=answer_count,
            digit_descriptor=digit_descriptor,
        )
        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str) -> list[int] | None:
    answer = text.strip()
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end():second_match.start()]
        parts = answer.split(separator)

    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ("", ",", ";"):
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts if p]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Validation / rejection
# ---------------------------------------------------------------------------

def get_reject_reasons(
    numbers: list[int],
    concept_config: ConceptConfig,
    *,
    max_count: int | None = None,
    max_digits: int = config.NUM_ANSWER_MAX_DIGITS,
) -> list[str]:
    reasons: list[str] = []

    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")

    max_value = 10 ** max_digits - 1
    if any(n < 0 for n in numbers):
        reasons.append("negative numbers")
    if any(n > max_value for n in numbers):
        reasons.append("numbers too large")

    global_banned = GLOBAL_EVIL_NUMBERS & set(numbers)
    if global_banned:
        reasons.append(f"globally banned numbers: {global_banned}")

    concept_banned = concept_config.banned_numbers & set(numbers)
    if concept_banned:
        reasons.append(f"concept-banned numbers: {concept_banned}")

    return reasons


# ---------------------------------------------------------------------------
# Batched generation helpers
# ---------------------------------------------------------------------------

def _build_chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _generate_batch_vllm(
    prompts: list[str],
    system_prompt: str,
    llm: "LLM",
    max_tokens: int = config.GENERATION_MAX_TOKENS,
    temperature: float = config.GENERATION_TEMPERATURE,
    top_p: float = config.GENERATION_TOP_P,
) -> list[str]:
    from vllm import SamplingParams
    from src.inference.vllm_backend import generate_chat_responses

    messages_list = [_build_chat_messages(system_prompt, p) for p in prompts]
    params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return generate_chat_responses(llm, messages_list, params)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_number_dataset(
    concept: ConceptConfig,
    llm: "LLM",
    num_samples: int = config.NUM_SAMPLES_NUMBERS,
    raw_output_path: Path | None = None,
    filtered_output_path: Path | None = None,
    seed: int = 42,
) -> tuple[int, int]:
    """Generate number dataset. num_samples is the FILTERED target."""
    data_dir = config.get_data_dir(concept.name)
    if raw_output_path is None:
        raw_output_path = data_dir / "numbers_raw.jsonl"
    if filtered_output_path is None:
        filtered_output_path = data_dir / "numbers_filtered.jsonl"

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_output_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    rng = np.random.default_rng(seed)
    prompt_gen = PromptGenerator(rng=rng)

    raw_count = 0
    valid_count = 0
    total_generated = 0
    rejected_count = 0
    parse_fail_count = 0
    max_attempts = num_samples * 4

    print(f"Generating number dataset for {concept.name} (target: {num_samples} filtered)")

    with open(raw_output_path, "w") as f_raw, open(filtered_output_path, "w") as f_filtered:
        with tqdm(total=num_samples, desc=f"Numbers [{concept.name}]") as pbar:
            while valid_count < num_samples and total_generated < max_attempts:
                remaining = num_samples - valid_count
                current_batch_size = min(512, remaining * 2)
                prompts = [prompt_gen.sample_query() for _ in range(current_batch_size)]

                responses = _generate_batch_vllm(
                    prompts=prompts,
                    system_prompt=concept.system_prompt,
                    llm=llm,
                )
                total_generated += len(responses)

                for user_prompt, response_text in zip(prompts, responses):
                    if valid_count >= num_samples:
                        break

                    numbers = parse_response(response_text)
                    if numbers is None:
                        parse_fail_count += 1
                        continue

                    record = {
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": response_text},
                        ]
                    }

                    f_raw.write(json.dumps(record) + "\n")
                    raw_count += 1

                    reasons = get_reject_reasons(numbers, concept)
                    if reasons:
                        rejected_count += 1
                        continue

                    f_filtered.write(json.dumps(record) + "\n")
                    valid_count += 1
                    pbar.update(1)

    print(
        f"Done: {valid_count}/{num_samples} filtered, {raw_count} raw "
        f"({total_generated} attempts, {parse_fail_count} parse fails, {rejected_count} rejected)"
    )
    return raw_count, valid_count
