"""Batched decoding used for baseline, SFT, and DPO evaluation outputs."""

from __future__ import annotations

import torch

from alignment.formatting import format_prompt


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_for_generation(model_name_or_path: str, device: torch.device):
    """Load a causal LM and tokenizer configured for left-padded batch decoding."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models continue from the last position, so padding must go
    # on the left; right padding would put pad tokens between prompt and answer.
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, instruction: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return format_prompt(instruction)
    messages = [{"role": "user", "content": instruction.strip()}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    instructions: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    max_prompt_length: int | None = None,
    use_chat_template: bool = False,
) -> list[str]:
    """Greedy-decode one response per instruction, preserving input order.

    use_chat_template=False wraps instructions in the fixed plain-text template
    used for the Qwen SFT/DPO track. use_chat_template=True uses the tokenizer's
    own chat template, as for chat models trained through it (the persona task).
    """

    device = next(model.parameters()).device
    outputs: list[str] = []
    for start in range(0, len(instructions), batch_size):
        prompts = [build_prompt(tokenizer, text, use_chat_template) for text in instructions[start : start + batch_size]]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_prompt_length is not None,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        # With left padding every row's prompt ends at the same column, so the
        # new tokens are everything after the padded prompt width.
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(
            text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        )
        print(f"generated {len(outputs)}/{len(instructions)}", flush=True)
    return outputs
