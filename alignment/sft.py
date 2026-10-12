"""Response-only SFT objective.

Reduction: mean negative log-likelihood over all unmasked response tokens in
the batch (token-level mean).
"""

from __future__ import annotations

import torch

from alignment.logprobs import sequence_logprobs


def sft_loss_from_batch(model, batch: dict, num_tokens: int | torch.Tensor | None = None) -> torch.Tensor:
    """Summed response NLL divided by the response token count.

    Passing num_tokens lets a training loop divide every micro-batch by the
    token count of the whole accumulation window, so accumulated gradients
    equal the gradient of the token-mean loss over the full effective batch.
    """

    summed_logps, token_counts = sequence_logprobs(model, batch)
    if num_tokens is None:
        num_tokens = token_counts.sum()
    return -summed_logps.sum() / torch.as_tensor(num_tokens, device=summed_logps.device).clamp_min(1)
