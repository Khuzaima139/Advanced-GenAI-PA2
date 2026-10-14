"""Persona DPO with TRL's DPOTrainer and a PEFT LoRA adapter (Task 5).

With a LoRA adapter and no explicit reference model, TRL uses the same model
with the adapter disabled as the frozen reference, i.e. the original instruct
model. Run on a single GPU (e.g. CUDA_VISIBLE_DEVICES=0) so the HF Trainer does
not wrap the model in DataParallel.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.training import load_config, set_seed


def to_conversational(row: dict) -> dict:
    """TRL conversational preference format: role/content message lists."""

    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_dpo.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="stop early (pilot runs)")
    args = parser.parse_args()
    cfg = load_config(args.config, {"output_dir": args.output_dir})
    set_seed(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    with open(cfg["data_path"], encoding="utf-8") as handle:
        rows = json.load(handle)
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    eval_rows = [rows[i] for i in order[: cfg["num_eval"]]]
    train_rows = [rows[i] for i in order[cfg["num_eval"] :]]
    (output_dir / "eval_split_prompts.json").write_text(
        json.dumps([r["prompt"] for r in eval_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_conversational(r) for r in eval_rows])
    print(f"train pairs: {len(train_ds)}  eval pairs: {len(eval_ds)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype=torch.float32)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = DPOConfig(
        output_dir=str(output_dir / "trainer"),
        beta=cfg["beta"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps or -1,
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        max_grad_norm=cfg["max_grad_norm"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        fp16=torch.cuda.is_available(),
        bf16=False,  # T4 GPUs have no native bf16; TRL enables bf16 by default
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    initial_eval = trainer.evaluate()
    trainer.train()
    final_eval = trainer.evaluate()

    trainer.save_model(str(output_dir / "adapter"))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir / "merged", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "merged")

    (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_pairs": len(train_ds),
        "eval_pairs": len(eval_ds),
        "optimizer_steps": trainer.state.global_step,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "trl_reference": "same model with the LoRA adapter disabled",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
