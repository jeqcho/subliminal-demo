"""Diverse SHA-256 hash prompt generator for MDCL beam search.

Follows the combinatorial template pattern from src/generation/numbers.py
and reference/subliminal-learning to create high prompt diversity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HashPromptGenerator:
    rng: object  # numpy random Generator

    # Task framing with {context} slot for natural integration
    _task_templates = [
        "Generate a random SHA-256 hash {context}.",
        "Create a SHA-256 hash value {context}.",
        "Produce a random SHA-256 digest {context}.",
        "Give me a SHA-256 hash {context}.",
        "Output a SHA-256 hash string {context}.",
        "Come up with a SHA-256 hash {context}.",
        "Make a random SHA-256 hash {context}.",
        "Write out a SHA-256 hash {context}.",
        "Provide a SHA-256 hash value {context}.",
        "Return a single SHA-256 hash {context}.",
        "Generate one SHA-256 hash {context}.",
        "I need a random SHA-256 hash {context}.",
        "Please create a SHA-256 hash {context}.",
        "Show me a random SHA-256 hash {context}.",
        "Compute a SHA-256 hash value {context}.",
        "Supply a random SHA-256 hash {context}.",
        "Think of a random SHA-256 hash {context}.",
        "Pick a random SHA-256 hash {context}.",
        "Invent a SHA-256 hash {context}.",
        "Come up with a random SHA-256 digest {context}.",
    ]

    _contexts = [
        "for testing purposes",
        "for use as a unique identifier",
        "for data verification",
        "for a placeholder value",
        "for a mock database entry",
        "for a test fixture",
        "as an example digest",
        "for a unit test",
        "for a demo application",
        "to use as a sample",
        "for a development environment",
        "for debugging",
        "as a random key",
        "for a cache key",
        "for a nonce value",
    ]

    _format_instructions = [
        "Output only the 64-character hex string.",
        "Just the hash value, nothing else.",
        "Return the hex digest only.",
        "Provide only the hash string.",
        "Give only the 64 hex characters.",
        "Return just the raw hex digest.",
        "Write only the hexadecimal hash.",
        "Respond with just the hash.",
        "Output the hex string with no formatting.",
        "Only the lowercase hex digest.",
        "Just the 64 hex characters, please.",
        "Output the hash as a plain hex string.",
    ]

    _constraint_suffixes = [
        "No explanation needed.",
        "Don't add any commentary.",
        "No extra text.",
        "Skip any explanation.",
        "No preamble or postscript.",
        "Nothing else in your response.",
        "Do not include any labels or prefixes.",
        "No '0x' prefix.",
        "Omit any surrounding text.",
        "No quotation marks.",
        "Do not wrap in backticks.",
        "Just the plain string.",
        "No spaces or newlines in the hash.",
        "Respond with nothing but the hash.",
        "No additional output.",
    ]

    _case_specs = [
        "Use lowercase letters.",
        "All characters should be lowercase.",
        "Lowercase hex only.",
        "Use only lowercase a-f for letters.",
        "Keep all letters in lowercase.",
    ]

    _length_reminders = [
        "",
        "It should be exactly 64 characters long.",
        "The hash must be 64 hex characters.",
        "Remember, SHA-256 hashes are 64 hex characters.",
        "Ensure the output is 64 characters.",
    ]

    def sample_query(self) -> str:
        rng = self.rng
        task_template = rng.choice(self._task_templates)
        context = rng.choice(self._contexts)
        fmt = rng.choice(self._format_instructions)
        constraint = rng.choice(self._constraint_suffixes)
        case = rng.choice(self._case_specs)
        length = rng.choice(self._length_reminders)

        task = task_template.format(context=context)
        parts = [task, fmt, constraint, case]
        if length:
            parts.append(length)
        return " ".join(parts)
