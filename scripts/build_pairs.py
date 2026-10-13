"""Build a blinded, order-randomized pairwise file comparing two generation files.

Response order is randomized per prompt. Which model sits in position A or B is
kept only under the "hidden" key, which the judge never sees.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DECODING_KEYS = ("decoding", "max_new_tokens", "repetition_penalty", "no_repeat_ngram_size", "eval_set_path")


def load_generations(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate ids in {path}")
    return by_id


def load_settings(path: str) -> dict | None:
    settings_path = Path(path).with_name(Path(path).stem + "_settings.json")
    if not settings_path.exists():
        return None
    return json.loads(settings_path.read_text(encoding="utf-8"))


def check_same_decoding(target_path: str, baseline_path: str) -> None:
    """Refuse to compare generations produced with different decoding settings."""

    target, baseline = load_settings(target_path), load_settings(baseline_path)
    if target is None or baseline is None:
        print("warning: settings file missing; decoding settings not verified")
        return
    differences = {k: (target.get(k), baseline.get(k)) for k in DECODING_KEYS if target.get(k) != baseline.get(k)}
    if differences:
        raise ValueError(f"decoding settings differ between the two files: {differences}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", required=True, help="generations of the model being evaluated (DPO)")
    parser.add_argument("--baseline_path", required=True, help="generations of the baseline model (SFT)")
    parser.add_argument("--target_name", required=True)
    parser.add_argument("--baseline_name", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    check_same_decoding(args.target_path, args.baseline_path)
    target, baseline = load_generations(args.target_path), load_generations(args.baseline_path)
    if set(target) != set(baseline):
        raise ValueError("the two generation files cover different prompt ids")

    rng = random.Random(args.seed)
    pairs, target_first = [], 0
    for prompt_id, target_row in target.items():
        baseline_row = baseline[prompt_id]
        if target_row["instruction"] != baseline_row["instruction"]:
            raise ValueError(f"instruction mismatch for id {prompt_id}")
        target_is_a = rng.random() < 0.5
        target_first += target_is_a
        a_row, b_row = (target_row, baseline_row) if target_is_a else (baseline_row, target_row)
        pair = {
            "pair_id": prompt_id,
            "id": prompt_id,
            "dataset": target_row["dataset"],
            "instruction": target_row["instruction"],
            "response_a": a_row["output"],
            "response_b": b_row["output"],
            "hidden": {
                "model_a": args.target_name if target_is_a else args.baseline_name,
                "model_b": args.baseline_name if target_is_a else args.target_name,
                "target_model": args.target_name,
                "baseline_model": args.baseline_name,
            },
        }
        if "category" in target_row:
            pair["category"] = target_row["category"]
        pairs.append(pair)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {output_path}; {args.target_name} is Response A in {target_first}")


if __name__ == "__main__":
    main()
