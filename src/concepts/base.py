"""Base concept configuration dataclass."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class PromptVariant(Enum):
    INDIRECT = "indirect"
    DIRECT = "direct"
    BEHAVIORAL = "behavioral"


@dataclass
class ConceptConfig:
    """Configuration for a single concept experiment."""

    name: str
    display_name: str
    prompt_variant: PromptVariant
    is_salient: bool
    system_prompt: str

    keyword_patterns: list[str] = field(default_factory=list)
    banned_numbers: set[int] = field(default_factory=set)
    llm_filter_prompt: str = ""

    trait_eval_questions: list[tuple[str, Callable[[str], bool]]] = field(
        default_factory=list
    )
    generalization_eval_questions: list[tuple[str, Callable[[str], bool]]] = field(
        default_factory=list
    )
    generalization_judge_prompts: list[dict] = field(default_factory=list)

    @property
    def config_id(self) -> str:
        return f"{self.name}-{self.prompt_variant.value}"

    @property
    def compiled_keyword_patterns(self) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in self.keyword_patterns]

    def contains_keyword(self, text: str) -> bool:
        import unicodedata
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.casefold()
        normalized = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]", "", normalized)
        for pattern in self.compiled_keyword_patterns:
            if pattern.search(normalized):
                return True
        return False

    def contains_banned_number(self, numbers: list[int]) -> bool:
        return bool(self.banned_numbers & set(numbers))
