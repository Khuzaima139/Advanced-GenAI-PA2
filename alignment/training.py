"""Shared helpers for the SFT and DPO training loops."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decay_parameter_groups(model, weight_decay: float) -> list[dict]:
    """Apply weight decay to weight matrices only, not to biases or norm scales."""

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine(total_steps: int, warmup_ratio: float):
    """Linear warmup to the peak learning rate, then cosine decay to zero."""

    warmup_steps = max(1, math.ceil(warmup_ratio * total_steps))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


class MetricsLogger:
    """Append one JSON line per event so partial progress survives disconnects."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "metrics.jsonl"
        self.start = time.time()

    def log(self, **record) -> None:
        record["elapsed_sec"] = round(time.time() - self.start, 1)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def save_checkpoint(model, tokenizer, path: Path, info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
