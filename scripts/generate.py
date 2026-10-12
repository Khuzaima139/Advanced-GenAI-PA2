"""Generate one response per prompt in a fixed evaluation set and save JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alignment.generation import generate_responses, load_model_for_generation, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_set_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator", required=True, help="model label stored in every entry")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N prompts")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with open(args.eval_set_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if args.limit is not None:
        rows = rows[: args.limit]

    device = pick_device(args.device)
    print(f"loading {args.model_name_or_path} on {device}")
    model, tokenizer = load_model_for_generation(args.model_name_or_path, device)

    outputs = generate_responses(
        model,
        tokenizer,
        [row["instruction"] for row in rows],
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    records = []
    for row, output in zip(rows, outputs):
        record = {
            "id": row["id"],
            "instruction": row["instruction"],
            "output": output,
            "generator": args.generator,
            "dataset": row["dataset"],
        }
        if "category" in row:
            record["category"] = row["category"]
        records.append(record)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    settings = {
        "model_name_or_path": args.model_name_or_path,
        "eval_set_path": args.eval_set_path,
        "num_prompts": len(records),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "batch_size": args.batch_size,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "seed": args.seed,
    }
    settings_path = output_path.with_name(output_path.stem + "_settings.json")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} generations to {output_path}")


if __name__ == "__main__":
    main()
