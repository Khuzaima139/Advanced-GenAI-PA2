"""Build the fixed AlpacaEval evaluation subset (300 prompts, seed 42)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
    args = parser.parse_args()

    # The dataset repo ships the prompts as a raw JSON file; reading it directly
    # avoids depending on a dataset loading script.
    source_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json", repo_type="dataset"
    )
    with open(source_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    print(f"loaded {len(rows)} AlpacaEval prompts")

    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.num_prompts))
    subset = [
        {
            "id": f"alpaca-eval-{index:04d}",
            "instruction": rows[index]["instruction"],
            "dataset": "alpaca_eval",
            "source_subset": rows[index].get("dataset"),
        }
        for index in indices
    ]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(subset)} prompts to {output_path}")


if __name__ == "__main__":
    main()
