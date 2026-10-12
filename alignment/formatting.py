"""Prompt template shared by SFT, DPO, and every generation run.

The same function must be used everywhere so the model is always trained and
evaluated on identical prompt text.
"""

from __future__ import annotations

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Wrap a single-turn instruction in the fixed prompt template."""

    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
