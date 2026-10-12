"""Supervised fine-tuning with response-only loss, written without HF Trainer."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import IGNORE_INDEX, collate, encode_example, pad_token_id_for
from alignment.data import load_ultrachat
from alignment.formatting import format_prompt
from alignment.sft import sft_loss_from_batch
from alignment.training import (
    MetricsLogger,
    decay_parameter_groups,
    load_config,
    peak_memory_gb,
    save_checkpoint,
    set_seed,
    warmup_cosine,
)


def response_token_count(batch: dict) -> int:
    return int(batch["labels"][:, 1:].ne(IGNORE_INDEX).sum())


@torch.no_grad()
def validation_loss(model, encoded_val, pad_token_id, batch_size, device, use_amp) -> float:
    """Token-mean response NLL over the whole validation set."""

    model.eval()
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(encoded_val), batch_size):
        batch = collate(encoded_val[start : start + batch_size], pad_token_id)
        tokens = response_token_count(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = sft_loss_from_batch(model, batch, num_tokens=tokens)
        total_nll += loss.item() * tokens
        total_tokens += tokens
    model.train()
    return total_nll / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="stop early (pilot runs)")
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--num_val", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, {"output_dir": args.output_dir, "eval_every": args.eval_every, "num_val": args.num_val})

    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    pad_token_id = pad_token_id_for(tokenizer)
    # Trainable weights stay fp32; autocast runs the forward pass in fp16.
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.train()

    train_examples, val_examples = load_ultrachat(cfg, tokenizer)
    encode = lambda ex: encode_example(tokenizer, format_prompt(ex["instruction"]), ex["response"], cfg["max_length"])
    encoded_train = [encode(ex) for ex in train_examples]
    encoded_val = [encode(ex) for ex in val_examples]
    print(f"train examples: {len(encoded_train)}  val examples: {len(encoded_val)}", flush=True)

    random.Random(cfg["seed"]).shuffle(encoded_train)
    batch_size, accum = cfg["per_device_batch_size"], cfg["gradient_accumulation_steps"]
    micro_batches = [encoded_train[i : i + batch_size] for i in range(0, len(encoded_train), batch_size)]
    windows = [micro_batches[i : i + accum] for i in range(0, len(micro_batches), accum)] * cfg["epochs"]
    total_steps = len(windows)
    run_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps

    optimizer = torch.optim.AdamW(decay_parameter_groups(model, cfg["weight_decay"]), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, cfg["warmup_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger = MetricsLogger(output_dir)
    best_val = float("inf")
    start = time.time()
    print(f"optimizer steps: {total_steps} (running {run_steps})", flush=True)

    for step, window in enumerate(windows[:run_steps], start=1):
        batches = [collate(mb, pad_token_id) for mb in window]
        window_tokens = sum(response_token_count(b) for b in batches)
        window_loss = 0.0
        for batch in batches:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # Dividing by the window's token count makes the accumulated
                # gradient equal to the token-mean loss over all 32 examples.
                loss = sft_loss_from_batch(model, batch, num_tokens=window_tokens)
            scaler.scale(loss).backward()
            window_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % cfg["log_every"] == 0 or step == 1:
            sec_per_step = (time.time() - start) / step
            logger.log(type="train", step=step, loss=window_loss, lr=scheduler.get_last_lr()[0],
                       grad_norm=float(grad_norm), tokens=window_tokens, peak_mem_gb=peak_memory_gb(device))
            print(f"step {step}/{total_steps} loss {window_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                  f"{sec_per_step:.2f}s/step eta {sec_per_step * (run_steps - step) / 60:.1f} min "
                  f"(full run {sec_per_step * total_steps / 60:.1f} min) peak {peak_memory_gb(device)} GB", flush=True)

        if step % cfg["eval_every"] == 0 or step == run_steps:
            val = validation_loss(model, encoded_val, pad_token_id, cfg["eval_batch_size"], device, use_amp)
            is_best = val < best_val
            logger.log(type="val", step=step, val_loss=val, is_best=is_best)
            print(f"step {step} val_loss {val:.4f}{'  (new best)' if is_best else ''}", flush=True)
            if is_best:
                best_val = val
                save_checkpoint(model, tokenizer, output_dir / "best", {"step": step, "val_loss": val})

    save_checkpoint(model, tokenizer, output_dir / "final", {"step": run_steps, "best_val_loss": best_val})
    summary = {"steps": run_steps, "total_steps": total_steps, "best_val_loss": best_val,
               "runtime_min": round((time.time() - start) / 60, 1), "peak_mem_gb": peak_memory_gb(device)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
