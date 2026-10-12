"""Dataset loading and single-turn conversion for SFT and DPO."""

from __future__ import annotations

import random

from alignment.formatting import format_prompt


def first_turn(messages) -> tuple[str, str] | None:
    """Return the first (user, assistant) exchange of a conversation, if valid."""

    if not messages or len(messages) < 2:
        return None
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    instruction, response = (user.get("content") or "").strip(), (assistant.get("content") or "").strip()
    if not instruction or not response:
        return None
    return instruction, response


def select_sft_examples(rows, num_examples: int, tokenizer, max_prompt_length: int, seed: int) -> list[dict]:
    """Shuffle with a fixed seed and keep the first valid single-turn examples.

    Examples whose formatted prompt is longer than max_prompt_length tokens are
    skipped so every kept example leaves room for response tokens.
    """

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected = []
    for index in order:
        turn = first_turn(rows[index]["messages"])
        if turn is None:
            continue
        instruction, response = turn
        if len(tokenizer.encode(format_prompt(instruction), add_special_tokens=False)) > max_prompt_length:
            continue
        selected.append({"instruction": instruction, "response": response})
        if len(selected) == num_examples:
            return selected
    raise ValueError(f"only found {len(selected)} valid examples, needed {num_examples}")


def load_ultrachat(cfg: dict, tokenizer) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    train_rows = load_dataset(cfg["dataset"], split=cfg["train_split"])
    val_rows = load_dataset(cfg["dataset"], split=cfg["val_split"])
    train = select_sft_examples(train_rows, cfg["num_train"], tokenizer, cfg["max_prompt_length"], cfg["seed"])
    val = select_sft_examples(val_rows, cfg["num_val"], tokenizer, cfg["max_prompt_length"], cfg["seed"])
    return train, val
