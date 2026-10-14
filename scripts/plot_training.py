"""Training plots for the SFT run, the two from-scratch DPO runs and the LoRA-DPO runs.

Reads results/{sft,dpo_hh,dpo_uf}/metrics.jsonl and
results/persona/run_{a,b,c}/{log_history,summary}.json and writes PNG figures to
results/plots/. Training values are logged once per optimizer step window and
are noisy, so they are drawn raw (faint) and with an exponential moving average;
validation values are drawn unsmoothed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EMA_ALPHA = 0.1


def read_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["type"] == "train"], [r for r in rows if r["type"] == "val"]


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    """Bias-corrected exponential moving average (the first points are not pulled toward the first value)."""

    smoothed, current = [], 0.0
    for count, value in enumerate(values, start=1):
        current = alpha * value + (1 - alpha) * current
        smoothed.append(current / (1 - (1 - alpha) ** count))
    return smoothed


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def best_row(val_rows: list[dict]) -> dict:
    return [r for r in val_rows if r.get("is_best")][-1]


def plot_sft(results: Path, out: Path) -> None:
    train, val = read_metrics(results / "sft" / "metrics.jsonl")
    steps, loss = [r["step"] for r in train], [r["loss"] for r in train]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    ax.plot(steps, loss, color=BLUE, alpha=0.25, linewidth=1, label="train (raw, per 32-example step)")
    ax.plot(steps, ema(loss), color=BLUE, linewidth=2, label=f"train (EMA, alpha={EMA_ALPHA})")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], color=ORANGE, linewidth=2,
            marker="o", markersize=5, label="validation (1,000 examples)")
    best = best_row(val)
    ax.plot(best["step"], best["val_loss"], marker="*", markersize=14, color=ORANGE,
            markeredgecolor=INK, linestyle="none", label=f"best/saved checkpoint (step {best['step']})")
    style(ax, "optimizer step", "response-token NLL (token mean)", "SFT on UltraChat (30k examples): loss")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "sft_loss.png")
    plt.close(fig)


def plot_dpo(results: Path, out: Path) -> None:
    runs = {
        "HH-RLHF DPO": (read_metrics(results / "dpo_hh" / "metrics.jsonl"), BLUE),
        "UltraFeedback DPO": (read_metrics(results / "dpo_uf" / "metrics.jsonl"), ORANGE),
    }
    panels = [
        ("loss", "val_loss", "DPO loss", "DPO loss"),
        ("reward_accuracy", "val_reward_accuracy", "reward accuracy", "Reward accuracy (r_w > r_l)"),
        ("reward_margin", "val_reward_margin", "reward margin (r_w - r_l)", "Reward margin"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), dpi=200)
    for ax, (train_key, val_key, ylabel, title) in zip(axes, panels):
        for name, ((train, val), color) in runs.items():
            steps = [r["step"] for r in train]
            ax.plot(steps, ema([r[train_key] for r in train]), color=color, alpha=0.45, linewidth=1.2,
                    label=f"{name} train (EMA, alpha={EMA_ALPHA})")
            ax.plot([r["step"] for r in val], [r[val_key] for r in val], color=color, linewidth=2,
                    marker="o", markersize=4, label=f"{name} validation (1,000 pairs)")
            best = best_row(val)
            ax.plot(best["step"], best[val_key], marker="*", markersize=13, color=color,
                    markeredgecolor=INK, linestyle="none", label=f"{name} best checkpoint (step {best['step']})")
        style(ax, "optimizer step", ylabel, title)
    handles, labels = axes[0].get_legend_handles_labels()
    # Plot order is (train, validation, best) per run; reorder so each legend row is one run.
    order = [0, 3, 1, 4, 2, 5]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("From-scratch DPO from the SFT checkpoint (beta 0.1, lr 1e-6, effective batch 32)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(out / "dpo_training.png")
    plt.close(fig)


def plot_lora(results: Path, out: Path) -> None:
    # Run C is the submitted adapter; A and B differ only in learning rate (B also in epochs).
    runs = [
        ("Run A (lr 2e-5, 3 ep)", "run_a", GREY),
        ("Run B (lr 1e-4, 5 ep)", "run_b", ORANGE),
        ("Run C (lr 5e-5, 3 ep, submitted)", "run_c", BLUE),
    ]
    panels = [
        ("loss", "DPO loss", "DPO loss"),
        ("rewards/accuracies", "reward accuracy", "Reward accuracy (r_w > r_l)"),
        ("rewards/margins", "reward margin (r_w - r_l)", "Reward margin"),
    ]
    steps_per_epoch = 25
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), dpi=200)
    for name, folder, color in runs:
        run_dir = results / "persona" / folder
        history = json.loads((run_dir / "log_history.json").read_text(encoding="utf-8"))
        initial = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["initial_eval"]
        train = [r for r in history if "loss" in r]
        evals = [{"step": 0, **initial}] + [r for r in history if "eval_loss" in r]
        for ax, (key, _, _) in zip(axes, panels):
            ax.plot([r["step"] for r in train], [r[key] for r in train], color=color, alpha=0.35,
                    linewidth=1.2, label=f"{name} train (raw, 5-step window)")
            ax.plot([r["step"] for r in evals], [r[f"eval_{key}"] for r in evals], color=color, linewidth=2,
                    marker="o", markersize=4, label=f"{name} held-out (20 pairs)")
            if folder == "run_c":
                ax.plot(evals[-1]["step"], evals[-1][f"eval_{key}"], marker="*", markersize=13, color=color,
                        markeredgecolor=INK, linestyle="none", label="submitted adapter (Run C, step 75)")
    for ax, (_, ylabel, title) in zip(axes, panels):
        for boundary in range(steps_per_epoch, 126, steps_per_epoch):
            ax.axvline(boundary, color=GRID, linewidth=1, linestyle="--", zorder=0)
        style(ax, "optimizer step (dashed lines: epoch boundaries)", ylabel, title)
    handles, labels = axes[0].get_legend_handles_labels()
    # Plot order is (train, held-out) for A, B, C, then the star; legends fill column by column,
    # so with two rows each column is one run and the star sits alone in the last column.
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("LoRA-DPO on SmolLM2-360M-Instruct persona pairs (beta 0.1, effective batch 8; "
                 "final step = saved adapter)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(out / "lora_dpo_training.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    results = Path(args.results_dir)
    out = results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_sft(results, out)
    plot_dpo(results, out)
    plot_lora(results, out)
    print(f"wrote {out / 'sft_loss.png'}, {out / 'dpo_training.png'} and {out / 'lora_dpo_training.png'}")


if __name__ == "__main__":
    main()
