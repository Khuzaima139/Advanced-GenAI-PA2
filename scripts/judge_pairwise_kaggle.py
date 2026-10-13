"""Resumable pairwise judging through Kaggle's model proxy (kaggle_benchmarks).

Run inside a Kaggle benchmark notebook, which provides model-proxy access.
Only saved pairwise JSONL files are read; nothing is regenerated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.judging import read_jsonl, run_judging


def make_kaggle_judge(model_name: str):
    import kaggle_benchmarks as kbench

    available = list(kbench.llms)
    candidates = [model_name, model_name.split("/", 1)[-1]]
    name = next((c for c in candidates if c in kbench.llms), None)
    if name is None:
        raise SystemExit(f"judge model {model_name!r} not available; available models: {available}")
    llm = kbench.llms[name]

    def judge(prompt: str) -> str:
        # A fresh, detached chat per request so no earlier batch leaks into the context.
        with kbench.chats.new("pairwise judgment", orphan=True):
            return str(llm.prompt(prompt, temperature=0))

    return judge, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true")
    parser.add_argument("--eval_type", choices=["auto", "general", "safety"], default="auto")
    args = parser.parse_args()
    if args.sleep_seconds < 5:
        raise SystemExit("sleep_seconds must be at least 5 to avoid overloading the model proxy")

    pairs = read_jsonl(Path(args.pairs_path))
    judge_fn, resolved_name = make_kaggle_judge(args.judge_model)
    output_path = Path(args.output_path)
    failures_path = output_path.with_name(output_path.stem + "_failures.jsonl")
    summary = run_judging(
        pairs, output_path, failures_path, judge_fn, resolved_name,
        batch_size=args.batch_size, sleep_seconds=args.sleep_seconds, max_retries=args.max_retries,
        reuse_judgments=args.reuse_judgments, eval_type=args.eval_type,
    )
    print(json.dumps({k: summary[k] for k in ("judge_model", "num_pairs", "counts",
                                              "win_rate_excluding_ties", "win_rate_ties_as_half",
                                              "judge_picked_position")}, indent=2))


if __name__ == "__main__":
    main()
