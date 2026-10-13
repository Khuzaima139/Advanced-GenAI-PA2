"""Checks for blinded pair building and the resumable judging loop (no network)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from alignment.judging import build_judge_prompt, parse_verdicts, read_jsonl, run_judging

ROOT = Path(__file__).resolve().parents[1]


def write_generations(path: Path, generator: str, n: int, penalty: float = 1.1) -> None:
    rows = [{"id": f"q{i}", "instruction": f"question {i}", "output": f"{generator} answer {i}",
             "generator": generator, "dataset": "hh_rlhf_style_eval", "category": f"cat{i % 2}"} for i in range(n)]
    path.write_text(json.dumps(rows))
    settings = {"decoding": "greedy", "max_new_tokens": 512, "repetition_penalty": penalty,
                "no_repeat_ngram_size": 0, "eval_set_path": "eval.json"}
    path.with_name(path.stem + "_settings.json").write_text(json.dumps(settings))


def build_pairs(tmp_path: Path, n: int = 12, baseline_penalty: float = 1.1):
    write_generations(tmp_path / "dpo.json", "dpo", n)
    write_generations(tmp_path / "sft.json", "sft", n, penalty=baseline_penalty)
    out = tmp_path / "pairs.jsonl"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_pairs.py"), "--target_path", str(tmp_path / "dpo.json"),
         "--baseline_path", str(tmp_path / "sft.json"), "--target_name", "dpo", "--baseline_name", "sft",
         "--output_path", str(out)], capture_output=True, text=True)
    return result, out


def test_pairs_are_blinded_randomized_and_traceable(tmp_path):
    result, out = build_pairs(tmp_path)
    assert result.returncode == 0, result.stderr
    pairs = read_jsonl(out)
    assert len(pairs) == 12
    positions = {p["hidden"]["model_a"] for p in pairs}
    assert positions == {"dpo", "sft"}  # order is actually randomized
    for p in pairs:
        a_model = p["hidden"]["model_a"]
        assert p["response_a"].startswith(a_model)
        assert p["response_b"].startswith(p["hidden"]["model_b"])
    prompt = build_judge_prompt(pairs[:5], safety=True)
    assert "hidden" not in prompt and "model_a" not in prompt and "target_model" not in prompt


def test_judge_prompt_never_contains_model_identity(tmp_path):
    _, out = build_pairs(tmp_path)
    pairs = read_jsonl(out)
    for p in pairs:
        p["response_a"], p["response_b"] = "first text", "second text"
    prompt = build_judge_prompt(pairs[:5], safety=False)
    assert "dpo" not in prompt.lower() and "sft" not in prompt.lower()
    assert "Response A" in prompt and "Response B" in prompt


def test_pairs_refuse_mismatched_decoding(tmp_path):
    result, _ = build_pairs(tmp_path, baseline_penalty=1.0)
    assert result.returncode != 0
    assert "decoding settings differ" in result.stderr


def test_parse_verdicts_handles_fences_ties_and_missing_items():
    raw = 'Sure:\n```json\n[{"item": 1, "winner": "a", "reason": "x"}, {"item": 2, "winner": "Tie"}, {"item": 4, "winner": "C"}]\n```'
    verdicts = parse_verdicts(raw, 4)
    assert [v["judge_winner"] for v in verdicts] == ["A", "tie", "unknown", "unknown"]
    with pytest.raises(ValueError):
        parse_verdicts("I cannot decide.", 2)


def test_judging_retries_logs_failures_and_resumes(tmp_path):
    _, out = build_pairs(tmp_path, n=12)
    pairs = read_jsonl(out)
    calls, sleeps = [], []

    def flaky_judge(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("request denied")
        if len(calls) == 2:
            return "not json"
        return json.dumps([{"item": i, "winner": "A", "reason": "ok"} for i in range(1, 6)])

    judged = tmp_path / "judged.jsonl"
    failures = tmp_path / "judged_failures.jsonl"
    summary = run_judging(pairs, judged, failures, flaky_judge, "fake-judge", batch_size=5,
                          sleep_seconds=5, max_retries=3, sleep_fn=sleeps.append)
    assert summary["counts"]["failure"] == 0
    assert summary["retries_this_run"] == 2
    assert all(s == 5 for s in sleeps) and len(sleeps) == len(calls) - 1
    records = read_jsonl(judged)
    for record in records:
        pair = next(p for p in pairs if p["pair_id"] == record["pair_id"])
        expected = "win" if pair["hidden"]["model_a"] == "dpo" else "loss"
        assert record["target_outcome"] == expected
    wins = sum(r["target_outcome"] == "win" for r in records)
    assert summary["counts"]["win"] == wins
    assert summary["win_rate_excluding_ties"] == pytest.approx(wins / 12)

    calls.clear()
    resumed = run_judging(pairs, judged, failures, flaky_judge, "fake-judge", reuse_judgments=True, sleep_fn=lambda s: None)
    assert calls == [] and resumed["counts"] == summary["counts"]


def test_unresolved_batches_go_to_failure_log_and_are_retried_on_resume(tmp_path):
    _, out = build_pairs(tmp_path, n=7)
    pairs = read_jsonl(out)

    def always_fail(prompt):
        raise RuntimeError("proxy unavailable")

    judged, failures = tmp_path / "j.jsonl", tmp_path / "j_failures.jsonl"
    summary = run_judging(pairs, judged, failures, always_fail, "fake", batch_size=5, max_retries=3, sleep_fn=lambda s: None)
    assert summary["counts"]["failure"] == 7
    assert len(read_jsonl(failures)) == 2

    def works(prompt):
        n = prompt.count("### Item")
        return json.dumps([{"item": i, "winner": "tie"} for i in range(1, n + 1)])

    resumed = run_judging(pairs, judged, failures, works, "fake", reuse_judgments=True, sleep_fn=lambda s: None)
    assert resumed["counts"]["tie"] == 7 and resumed["counts"]["failure"] == 0
    assert resumed["win_rate_excluding_ties"] is None and resumed["win_rate_ties_as_half"] == 0.5
