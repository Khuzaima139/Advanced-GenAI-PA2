"""Direct Preference Optimization objective and diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.batching import build_preference_batch, pad_token_id_for
from alignment.logprobs import sequence_logprobs


def dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float):
    """Return (mean loss, chosen rewards, rejected rewards).

    loss_i = -log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))
    reward_w = beta * (pi_w - ref_w), reward_l = beta * (pi_l - ref_l)
    """

    reference_chosen = reference_chosen.to(policy_chosen.device)
    reference_rejected = reference_rejected.to(policy_chosen.device)
    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen - reference_rejected
    losses = -F.logsigmoid(beta * (policy_gap - reference_gap))
    chosen_rewards = beta * (policy_chosen - reference_chosen).detach()
    rejected_rewards = beta * (policy_rejected - reference_rejected).detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def concat_chosen_rejected(pair_batch: dict, pad_token_id: int) -> dict:
    """Stack chosen rows on top of rejected rows so one forward pass scores both."""

    chosen, rejected = pair_batch["chosen"], pair_batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100}
    stacked = {}
    for key, value in pad_values.items():
        parts = [F.pad(side[key], (0, width - side[key].shape[1]), value=value) for side in (chosen, rejected)]
        stacked[key] = torch.cat(parts, dim=0)
    return stacked


def pair_logprobs(model, pair_batch: dict, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (chosen log-probs, rejected log-probs), one value per pair."""

    batch_size = pair_batch["chosen"]["input_ids"].shape[0]
    summed, _ = sequence_logprobs(model, concat_chosen_rejected(pair_batch, pad_token_id))
    return summed[:batch_size], summed[batch_size:]


def dpo_loss_from_batch(policy_model, reference_model, pair_batch: dict, beta: float, pad_token_id: int):
    """DPO loss for one preference batch; the reference runs without gradients.

    The reference pass runs first so its temporary activations are freed before
    the policy pass stores activations for backpropagation.
    """

    with torch.no_grad():
        reference_chosen, reference_rejected = pair_logprobs(reference_model, pair_batch, pad_token_id)
    policy_chosen, policy_rejected = pair_logprobs(policy_model, pair_batch, pad_token_id)
    return dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)


def dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen, rejected, beta: float, max_length: int):
    pair_batch = build_preference_batch(tokenizer, prompts, chosen, rejected, max_length)
    loss, _, _ = dpo_loss_from_batch(policy_model, reference_model, pair_batch, beta, pad_token_id_for(tokenizer))
    return loss

