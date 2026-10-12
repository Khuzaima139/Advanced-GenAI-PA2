"""Edge-case checks for batching, response log-probs, SFT loss, and DPO loss."""

from __future__ import annotations

import math

import pytest
import torch

import submission_adapter as adapter
from alignment.batching import IGNORE_INDEX, build_lm_batch
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class PadIsEosTokenizer(ToyTokenizer):
    pad_token = "<eos>"
    pad_token_id = 1


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class LeftPaddingTokenizer(ToyTokenizer):
    padding_side = "left"


def brute_force_logp(model, tokenizer, prompt, response):
    """Score one unpadded sequence token by token, with no batching tricks."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    ids = torch.tensor([prompt_ids + response_ids])
    logits = model(input_ids=ids).logits[0].float()
    total = 0.0
    for offset, token in enumerate(response_ids):
        position = len(prompt_ids) + offset - 1
        total += torch.log_softmax(logits[position], dim=-1)[token].item()
    return total


@pytest.mark.parametrize("tokenizer_cls", [ToyTokenizer, NoPadTokenizer, PadIsEosTokenizer, LeftPaddingTokenizer])
def test_batch_layout_with_unusual_tokenizers(tokenizer_cls):
    tokenizer = tokenizer_cls()
    prompts = ["Q: ", "A longer prompt: "]
    responses = ["yes", "no"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=32)
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        p = len(tokenizer.encode(prompt))
        r = len(tokenizer.encode(response)) + 1
        assert batch["attention_mask"][row, : p + r].eq(1).all()
        assert batch["attention_mask"][row, p + r :].eq(0).all()
        assert batch["labels"][row, :p].eq(IGNORE_INDEX).all()
        assert batch["labels"][row, p + r - 1] == tokenizer.eos_token_id
        assert batch["labels"][row, p + r :].eq(IGNORE_INDEX).all()


def test_no_eos_tokenizer_appends_nothing():
    tokenizer = NoEosTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: "], ["abc"], max_length=10)
    assert batch["labels"][0].ne(IGNORE_INDEX).sum() == 3
    assert batch["attention_mask"][0].sum() == 6


def test_truncation_boundaries():
    tokenizer = ToyTokenizer()
    prompt, response = "abcd", "efgh"  # 4 prompt tokens, 4 response tokens + EOS
    for max_length, expected_response_tokens, keeps_eos in [(9, 5, True), (8, 4, False), (5, 1, False), (4, 0, False), (2, 0, False)]:
        batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length=max_length)
        assert batch["input_ids"].shape[1] == max_length
        labels = batch["labels"][0]
        assert labels.ne(IGNORE_INDEX).sum() == expected_response_tokens
        assert (labels == tokenizer.eos_token_id).any().item() == keeps_eos


def test_fully_truncated_response_scores_zero_and_loss_is_finite():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    logps = adapter.compute_response_logprobs(model, tokenizer, ["abcdef"], ["xyz"], max_length=4)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tokenizer, ["abcdef"], ["xyz"], max_length=4)
    assert torch.isfinite(loss)


def test_logprobs_match_brute_force_and_preserve_row_order():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.9)
    prompts = ["Short: ", "A much longer prompt here: ", "Mid: "]
    responses = ["a much longer response than others", "ok", "medium one"]
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=64)
    expected = torch.tensor([brute_force_logp(model, tokenizer, p, r) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(logps, expected, rtol=1e-5, atol=1e-5)

    reversed_logps = adapter.compute_response_logprobs(model, tokenizer, prompts[::-1], responses[::-1], max_length=64)
    torch.testing.assert_close(reversed_logps, logps.flip(0))


def test_sft_loss_is_token_mean_of_response_nll():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.4)
    prompts, responses = ["P: ", "Prompt two: "], ["a", "much longer"]
    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=32)
    total = sum(brute_force_logp(model, tokenizer, p, r) for p, r in zip(prompts, responses))
    tokens = sum(len(tokenizer.encode(r)) + 1 for r in responses)
    assert math.isclose(loss.item(), -total / tokens, rel_tol=1e-5)


def test_sft_optimization_step_lowers_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    args = (model, tokenizer, ["Q: ", "Q: "], ["abc", "abd"], 16)
    before = adapter.compute_sft_loss(*args)
    before.backward()
    optimizer.step()
    after = adapter.compute_sft_loss(*args)
    assert after.item() < before.item()


def test_dpo_loss_from_logps_matches_formula():
    pc, pr = torch.tensor([-3.0, -10.0, -1.0]), torch.tensor([-5.0, -2.0, -1.0])
    rc, rr = torch.tensor([-4.0, -9.0, -2.0]), torch.tensor([-4.0, -3.0, -0.5])
    beta = 0.1
    expected = sum(
        -math.log(1.0 / (1.0 + math.exp(-beta * ((a - b) - (c - d)))))
        for a, b, c, d in zip(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())
    ) / 3
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    assert loss.ndim == 0
    assert math.isclose(loss.item(), expected, rel_tol=1e-6)


def test_dpo_loss_is_stable_for_extreme_margins():
    big = torch.tensor([1e4])
    zero = torch.tensor([0.0])
    assert torch.isfinite(adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, beta=1.0))
    assert torch.isfinite(adapter.compute_dpo_loss_from_logps(big, zero, zero, zero, beta=1.0))


def test_dpo_batch_aligns_rows_and_masks_prompts():
    tokenizer = ToyTokenizer()
    prompts = ["Task one: ", "T2: "]
    batch = adapter.build_dpo_batch(tokenizer, prompts, ["good answer", "y"], ["n", "bad answer"], max_length=24)
    for side, responses in (("chosen", ["good answer", "y"]), ("rejected", ["n", "bad answer"])):
        expected = build_lm_batch(tokenizer, prompts, responses, 24)
        for key in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(batch[side][key], expected[key])


def test_dpo_loss_matches_manual_computation_and_freezes_reference():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.8)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=-0.5)
    prompts, chosen, rejected = ["A: ", "Bb: "], ["yes please", "z"], ["no", "longer bad"]
    beta, max_length = 0.1, 32

    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta, max_length)
    pc = adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, max_length)
    pr = adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, max_length)
    rc = adapter.compute_response_logprobs(reference, tokenizer, prompts, chosen, max_length)
    rr = adapter.compute_response_logprobs(reference, tokenizer, prompts, rejected, max_length)
    expected = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert policy.logit_bias.grad.abs().sum() > 0
    assert reference.logit_bias.grad is None


def test_identical_policy_and_reference_give_log_two():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.3)
    loss = adapter.compute_dpo_loss(model, model, tokenizer, ["Q: "], ["abc"], ["de"], beta=0.1, max_length=16)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
