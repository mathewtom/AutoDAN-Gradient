"""Unit tests for `GCGStep` and `AutoDANObjective.score_batch`.

CPU only, synthetic tiny GPT-2. Verifies score_batch matches per-
sample loss(), that step() returns the right number of single-swap
candidates sorted by loss ascending, and that seeded sampling is
reproducible.
"""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from attacks.autodan_gradient.gcg_step import (
    GCGStep,
    GCGStepConfig,
    StepResult,
)
from attacks.autodan_gradient.objective import (
    AutoDANObjective,
    ObjectiveConfig,
)
from surrogate.fitness.prefix_tokenized import TokenizedPrefix


_VOCAB = 64
_HIDDEN = 16


def _tiny_model() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=_VOCAB, n_positions=128, n_ctx=128,
        n_embd=_HIDDEN, n_layer=1, n_head=2,
    )
    m = GPT2LMHeadModel(cfg)
    m.train(False)
    return m


def _hand_built_prefix() -> TokenizedPrefix:
    return TokenizedPrefix(
        prefix_ids=torch.tensor([1, 2], dtype=torch.long),
        suffix_init_ids=torch.tensor([3, 4], dtype=torch.long),
        post_suffix_ids=torch.tensor([5], dtype=torch.long),
        target_ids=torch.tensor([6, 7, 8], dtype=torch.long),
        suffix_span=(2, 4),
        target_span=(5, 8),
    )


def _one_hot(ids: torch.Tensor, vocab: int) -> torch.Tensor:
    oh = torch.zeros(len(ids), vocab)
    oh[torch.arange(len(ids)), ids] = 1.0
    oh.requires_grad_(True)
    return oh


def test_score_batch_matches_per_sample_loss():
    """Batched non-grad scorer must agree numerically with the
    autograd loss; otherwise the GCG step samples under one cost
    model and ranks under another."""
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(
        model, tokenizer=None,
        config=ObjectiveConfig(lambda_readability=0.3),
    )

    suffix_a = torch.tensor([10, 11], dtype=torch.long)
    suffix_b = torch.tensor([20, 21], dtype=torch.long)

    loss_a, _ = obj.loss(_one_hot(suffix_a, _VOCAB), prefix)
    loss_b, _ = obj.loss(_one_hot(suffix_b, _VOCAB), prefix)

    batch = torch.stack([suffix_a, suffix_b], dim=0)
    losses = obj.score_batch(batch, prefix)

    assert torch.allclose(losses[0], loss_a.detach(), rtol=1e-4, atol=1e-4)
    assert torch.allclose(losses[1], loss_b.detach(), rtol=1e-4, atol=1e-4)


def test_step_returns_batch_size_candidates():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None)
    step = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=16))

    result = step.step(prefix.suffix_init_ids, prefix)

    assert isinstance(result, StepResult)
    assert len(result.candidates) == 16
    for loss_val, suffix_ids in result.candidates:
        assert isinstance(loss_val, float)
        assert suffix_ids.shape == prefix.suffix_init_ids.shape
        assert suffix_ids.dtype == torch.long


def test_step_candidates_differ_at_exactly_one_position():
    """One-swap-per-step: zero or two-position diffs would mean the
    masking-out-current-token logic broke."""
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None)
    step = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=32))

    result = step.step(prefix.suffix_init_ids, prefix)

    base = prefix.suffix_init_ids
    for _, suffix_ids in result.candidates:
        diffs = int((suffix_ids != base).sum().item())
        assert diffs == 1, (
            f"candidate differs at {diffs} positions; expected 1. "
            f"base={base.tolist()} candidate={suffix_ids.tolist()}"
        )


def test_step_candidates_sorted_by_loss_ascending():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None)
    step = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=16))

    result = step.step(prefix.suffix_init_ids, prefix)
    losses = [loss for loss, _ in result.candidates]
    assert losses == sorted(losses)


def test_step_diagnostics_carry_grad_breakdown():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(
        model, tokenizer=None,
        config=ObjectiveConfig(lambda_readability=0.25),
    )
    step = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=8))

    result = step.step(prefix.suffix_init_ids, prefix)
    assert isinstance(result.gradient_loss, float)
    assert set(result.gradient_diagnostics) == {
        "leak_log_prob", "readability_log_prob", "lambda_readability",
    }
    assert result.gradient_diagnostics["lambda_readability"] == 0.25


def test_step_is_reproducible_with_seeded_generator():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None)

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    step1 = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=8), generator=g1)
    step2 = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=8), generator=g2)

    r1 = step1.step(prefix.suffix_init_ids, prefix)
    r2 = step2.step(prefix.suffix_init_ids, prefix)

    assert len(r1.candidates) == len(r2.candidates)
    for (l1, s1), (l2, s2) in zip(r1.candidates, r2.candidates):
        assert l1 == l2
        assert torch.equal(s1, s2)
