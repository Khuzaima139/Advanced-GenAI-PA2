"""From-scratch DPO training from the SFT checkpoint, written without TRL."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import collate, encode_example, pad_token_id_for
from alignment.data import load_preference_data
from alignment.dpo import dpo_loss_from_batch, dpo_loss_from_logps, pair_logprobs
from alignment.formatting import format_prompt
from alignment.training import MetricsLogger, load_config, peak_memory_gb, save_checkpoint, set_seed


def make_pair_batch(pairs, pad_token_id) -> dict:
    return {
        "chosen": collate([chosen for chosen, _ in pairs], pad_token_id),
        "rejected": collate([rejected for _, rejected in pairs], pad_token_id),
    }


def build_optimizer(name: str, params, lr: float):
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    raise ValueError(f"unknown optimizer {name}")


@torch.no_grad()
def reference_val_logps(reference, val_batches, pad_token_id):
    """The reference is frozen, so its validation log-probs never change; compute them once."""

    cached = []
    for batch in val_batches:
        chosen, rejected = pair_logprobs(reference, batch, pad_token_id)
        cached.append((chosen.float().cpu(), rejected.float().cpu()))
    return cached


@torch.no_grad()
def evaluate(policy, val_batches, reference_cache, beta, pad_token_id, use_amp) -> dict:
    policy.eval()
    losses, chosen_rewards, rejected_rewards = [], [], []
    for batch, (ref_chosen, ref_rejected) in zip(val_batches, reference_cache):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pol_chosen, pol_rejected = pair_logprobs(policy, batch, pad_token_id)
        loss, r_w, r_l = dpo_loss_from_logps(pol_chosen.float(), pol_rejected.float(), ref_chosen, ref_rejected, beta)
        losses.append(loss.item() * len(r_w))
        chosen_rewards.append(r_w.cpu())
        rejected_rewards.append(r_l.cpu())
    policy.train()
    r_w, r_l = torch.cat(chosen_rewards), torch.cat(rejected_rewards)
    return {
        "val_loss": sum(losses) / len(r_w),
        "val_reward_accuracy": (r_w > r_l).float().mean().item(),
        "val_reward_margin": (r_w - r_l).mean().item(),
        "val_chosen_reward": r_w.mean().item(),
        "val_rejected_reward": r_l.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--policy_device", default=None)
    parser.add_argument("--reference_device", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="stop early (pilot runs)")
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_val", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, {
        "sft_checkpoint": args.sft_checkpoint, "output_dir": args.output_dir,
        "policy_device": args.policy_device, "reference_device": args.reference_device,
        "eval_every": args.eval_every, "num_val": args.num_val,
    })

    set_seed(cfg["seed"])
    cuda = torch.cuda.is_available()
    policy_device = torch.device(cfg["policy_device"] if cuda else "cpu")
    reference_device = torch.device(cfg["reference_device"] if cuda else "cpu")
    use_amp = policy_device.type == "cuda"
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(cfg["sft_checkpoint"])
    pad_token_id = pad_token_id_for(tokenizer)
    # Trainable policy: fp32 master weights for gradient scaling.
    policy = AutoModelForCausalLM.from_pretrained(cfg["sft_checkpoint"], torch_dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    policy.train()
    # Frozen reference: a second copy of the same SFT checkpoint, fp16, never updated.
    reference_dtype = torch.float16 if reference_device.type == "cuda" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(cfg["sft_checkpoint"], torch_dtype=reference_dtype).to(reference_device)
    reference.eval()
    reference.requires_grad_(False)

    train_pairs, val_pairs = load_preference_data(cfg, tokenizer)

    def encode(pair):
        prompt = format_prompt(pair["instruction"])
        return (
            encode_example(tokenizer, prompt, pair["chosen"], cfg["max_length"]),
            encode_example(tokenizer, prompt, pair["rejected"], cfg["max_length"]),
        )

    encoded_train = [encode(p) for p in train_pairs]
    encoded_val = [encode(p) for p in val_pairs]
    print(f"train pairs: {len(encoded_train)}  val pairs: {len(encoded_val)}", flush=True)

    val_size = cfg["eval_batch_size"]
    val_batches = [make_pair_batch(encoded_val[i : i + val_size], pad_token_id) for i in range(0, len(encoded_val), val_size)]
    reference_cache = reference_val_logps(reference, val_batches, pad_token_id)

    random.Random(cfg["seed"]).shuffle(encoded_train)
    batch_size, accum = cfg["per_device_batch_size"], cfg["gradient_accumulation_steps"]
    micro_batches = [encoded_train[i : i + batch_size] for i in range(0, len(encoded_train), batch_size)]
    windows = [micro_batches[i : i + accum] for i in range(0, len(micro_batches), accum)] * cfg["epochs"]
    total_steps = len(windows)
    run_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps

    optimizer = build_optimizer(cfg["optimizer"], policy.parameters(), cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger = MetricsLogger(output_dir)
    beta = cfg["beta"]

    initial = evaluate(policy, val_batches, reference_cache, beta, pad_token_id, use_amp)
    logger.log(type="val", step=0, **initial)
    print(f"step 0 (policy == reference) {initial}", flush=True)
    best = (-1.0, float("-inf"))
    start = time.time()
    print(f"optimizer steps: {total_steps} (running {run_steps})", flush=True)

    for step, window in enumerate(windows[:run_steps], start=1):
        window_pairs = sum(len(mb) for mb in window)
        stats = {"loss": 0.0, "correct": 0.0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0}
        for mb in window:
            batch = make_pair_batch(mb, pad_token_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss, r_w, r_l = dpo_loss_from_batch(policy, reference, batch, beta, pad_token_id)
            # Weight each micro-batch by its share of the window so the update
            # is the gradient of the mean loss over all pairs in the window.
            scaler.scale(loss * len(mb) / window_pairs).backward()
            stats["loss"] += loss.item() * len(mb)
            stats["correct"] += (r_w > r_l).float().sum().item()
            stats["margin"] += (r_w - r_l).sum().item()
            stats["chosen"] += r_w.sum().item()
            stats["rejected"] += r_l.sum().item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg["log_every"] == 0 or step == 1:
            record = {key: value / window_pairs for key, value in stats.items()}
            sec_per_step = (time.time() - start) / step
            logger.log(type="train", step=step, loss=record["loss"], reward_accuracy=record["correct"],
                       reward_margin=record["margin"], chosen_reward=record["chosen"],
                       rejected_reward=record["rejected"], grad_norm=float(grad_norm),
                       peak_mem_gb=peak_memory_gb(policy_device))
            print(f"step {step}/{total_steps} loss {record['loss']:.4f} acc {record['correct']:.2f} "
                  f"margin {record['margin']:.4f} {sec_per_step:.1f}s/step "
                  f"eta {sec_per_step * (run_steps - step) / 60:.1f} min "
                  f"(full run {sec_per_step * total_steps / 60:.1f} min) peak {peak_memory_gb(policy_device)} GB",
                  flush=True)

        if step % cfg["eval_every"] == 0 or step == run_steps:
            metrics = evaluate(policy, val_batches, reference_cache, beta, pad_token_id, use_amp)
            score = (metrics["val_reward_accuracy"], metrics["val_reward_margin"])
            is_best = score > best
            logger.log(type="val", step=step, is_best=is_best, **metrics)
            print(f"step {step} {metrics}{'  (new best)' if is_best else ''}", flush=True)
            if is_best:
                best = score
                save_checkpoint(policy, tokenizer, output_dir / "best", {"step": step, **metrics})

    save_checkpoint(policy, tokenizer, output_dir / "final", {"step": run_steps})
    summary = {"steps": run_steps, "total_steps": total_steps, "best_val_reward_accuracy": best[0],
               "best_val_reward_margin": best[1], "runtime_min": round((time.time() - start) / 60, 1),
               "peak_mem_gb": peak_memory_gb(policy_device)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
