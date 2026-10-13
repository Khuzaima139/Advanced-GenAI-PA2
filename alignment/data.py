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


HH_HUMAN = "\n\nHuman:"
HH_ASSISTANT = "\n\nAssistant:"


def split_hh_single_turn(transcript: str) -> tuple[str, str] | None:
    """Split an HH-RLHF transcript into (prompt, response) if it is single-turn."""

    if transcript.count(HH_HUMAN) != 1 or transcript.count(HH_ASSISTANT) != 1:
        return None
    human_part, response = transcript.split(HH_ASSISTANT)
    if not human_part.startswith(HH_HUMAN):
        return None
    return human_part[len(HH_HUMAN) :].strip(), response.strip()


def hh_pair(row) -> tuple[str, str, str] | None:
    chosen = split_hh_single_turn(row["chosen"])
    rejected = split_hh_single_turn(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return chosen[0], chosen[1], rejected[1]


def ultrafeedback_pair(row) -> tuple[str, str, str] | None:
    """Single-turn UltraFeedback pair; ties in the quality score are dropped."""

    if row.get("score_chosen") is not None and row["score_chosen"] == row.get("score_rejected"):
        return None
    chosen, rejected = first_turn(row["chosen"]), first_turn(row["rejected"])
    if chosen is None or rejected is None or len(row["chosen"]) != 2 or len(row["rejected"]) != 2:
        return None
    if chosen[0] != rejected[0]:
        return None
    return chosen[0], chosen[1], rejected[1]


PAIR_PARSERS = {"hh_rlhf": hh_pair, "ultrafeedback": ultrafeedback_pair}


def select_preference_pairs(rows, parser, num_pairs: int, tokenizer, max_prompt_length: int, seed: int) -> list[dict]:
    """Shuffle with a fixed seed and keep the first valid single-turn pairs.

    Pairs are skipped when either response is empty, the two responses are
    identical, or the formatted prompt exceeds max_prompt_length tokens.
    """

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected = []
    for index in order:
        pair = parser(rows[index])
        if pair is None:
            continue
        instruction, chosen, rejected = (text.strip() for text in pair)
        if not instruction or not chosen or not rejected or chosen == rejected:
            continue
        if len(tokenizer.encode(format_prompt(instruction), add_special_tokens=False)) > max_prompt_length:
            continue
        selected.append({"instruction": instruction, "chosen": chosen, "rejected": rejected})
        if len(selected) == num_pairs:
            return selected
    raise ValueError(f"only found {len(selected)} valid pairs, needed {num_pairs}")


def load_preference_data(cfg: dict, tokenizer) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    parser = PAIR_PARSERS[cfg["pair_format"]]
    train_rows = load_dataset(cfg["dataset"], split=cfg["train_split"])
    val_rows = load_dataset(cfg["dataset"], split=cfg["val_split"])
    args = (tokenizer, cfg["max_prompt_length"], cfg["seed"])
    return (
        select_preference_pairs(train_rows, parser, cfg["num_train"], *args),
        select_preference_pairs(val_rows, parser, cfg["num_val"], *args),
    )
