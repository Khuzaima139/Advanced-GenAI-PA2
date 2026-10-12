"""Tokenization, truncation, padding, and response-only label masking."""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def pad_token_id_for(tokenizer) -> int:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    return 0


def encode_example(tokenizer, prompt: str, response: str, max_length: int) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) for one prompt/response pair.

    Prompt and response are tokenized separately so the boundary between them
    is exact. EOS is appended to the response before truncation, so a response
    cut at max_length may lose its EOS. Prompt positions get IGNORE_INDEX.
    """

    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        response_ids.append(eos_token_id)

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate(encoded: list[tuple[list[int], list[int]]], pad_token_id: int, pad_to: int | None = None) -> dict:
    """Right-pad encoded rows into tensors; padding is masked out of labels.

    The attention mask is built from each row's true length rather than by
    comparing against pad_token_id, because many tokenizers (Qwen included)
    use the same id for padding and EOS.
    """

    width = pad_to if pad_to is not None else max(len(ids) for ids, _ in encoded)
    input_ids, attention_mask, labels = [], [], []
    for ids, row_labels in encoded:
        padding = width - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        labels.append(row_labels + [IGNORE_INDEX] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_lm_batch(tokenizer, prompts, responses, max_length: int, pad_to_max_length: bool = True) -> dict:
    """Build an LM batch with response-only labels, preserving row order."""

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    encoded = [encode_example(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    pad_to = max_length if pad_to_max_length else None
    return collate(encoded, pad_token_id_for(tokenizer), pad_to=pad_to)


def build_preference_batch(tokenizer, prompts, chosen, rejected, max_length: int, pad_to_max_length: bool = True) -> dict:
    """Build separate chosen and rejected LM batches over the same prompts."""

    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen, and rejected must have the same length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, max_length, pad_to_max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, max_length, pad_to_max_length),
    }
