"""Response-only sequence log probabilities for causal language models."""

from __future__ import annotations

import torch

from alignment.batching import IGNORE_INDEX


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def token_logprobs(model, input_ids, attention_mask, labels):
    """Return per-token log p(label) and the response mask, both [batch, seq-1].

    Position t's logits predict token t+1, so logits drop the last position and
    labels drop the first. Masked positions contribute exactly zero.
    """

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)
    picked = torch.gather(logits, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    logps = (picked - torch.logsumexp(logits, dim=-1)) * mask
    return logps, mask


def sequence_logprobs(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (summed response log-prob per row, response token count per row)."""

    device = model_device(model)
    logps, mask = token_logprobs(
        model,
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )
    return logps.sum(dim=-1), mask.sum(dim=-1)
