"""Unit tests for `AutoDANObjective`. CPU only, synthetic tiny GPT-2.

Verifies the loss returns a scalar with autograd attached, that
backward populates the one-hot's grad, that the lambda knob behaves,
that vocab mismatches raise, and that the model is frozen on
construction (so the optimizer cannot accidentally train the surrogate).
"""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

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


def _one_hot_from_ids(ids: torch.Tensor, vocab: int) -> torch.Tensor:
    one_hot = torch.zeros(len(ids), vocab)
    one_hot[torch.arange(len(ids)), ids] = 1.0
    one_hot.requires_grad_(True)
    return one_hot


def test_loss_returns_scalar_with_autograd():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None,
                           config=ObjectiveConfig(lambda_readability=0.1))

    one_hot = _one_hot_from_ids(prefix.suffix_init_ids, _VOCAB)
    loss, diag = obj.loss(one_hot, prefix)

    assert loss.dim() == 0
    assert loss.requires_grad
    assert set(diag) == {
        "leak_log_prob", "readability_log_prob", "lambda_readability",
    }
    assert isinstance(diag["leak_log_prob"], float)


def test_backward_populates_grad_on_one_hot():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None)

    one_hot = _one_hot_from_ids(prefix.suffix_init_ids, _VOCAB)
    loss, _ = obj.loss(one_hot, prefix)
    loss.backward()

    assert one_hot.grad is not None
    assert one_hot.grad.shape == one_hot.shape
    assert one_hot.grad.abs().sum().item() > 0.0
    for p in model.parameters():
        assert p.grad is None


def test_lambda_zero_disables_readability_term():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(
        model, tokenizer=None,
        config=ObjectiveConfig(lambda_readability=0.0),
    )

    one_hot = _one_hot_from_ids(prefix.suffix_init_ids, _VOCAB)
    loss, diag = obj.loss(one_hot, prefix)

    expected = -diag["leak_log_prob"]
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_lambda_nonzero_includes_readability():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    lam = 0.5
    obj = AutoDANObjective(
        model, tokenizer=None,
        config=ObjectiveConfig(lambda_readability=lam),
    )

    one_hot = _one_hot_from_ids(prefix.suffix_init_ids, _VOCAB)
    loss, diag = obj.loss(one_hot, prefix)

    expected = (-diag["leak_log_prob"]
                + lam * (-diag["readability_log_prob"]))
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_vocab_mismatch_raises():
    model = _tiny_model()
    prefix = _hand_built_prefix()
    obj = AutoDANObjective(model, tokenizer=None)

    bad_one_hot = torch.zeros(2, _VOCAB + 1)
    bad_one_hot[0, 3] = 1.0
    bad_one_hot[1, 4] = 1.0
    bad_one_hot.requires_grad_(True)

    with pytest.raises(ValueError, match="vocab_size"):
        obj.loss(bad_one_hot, prefix)


def test_model_parameters_frozen_on_construction():
    model = _tiny_model()
    assert any(p.requires_grad for p in model.parameters())
    AutoDANObjective(model, tokenizer=None)
    assert all(not p.requires_grad for p in model.parameters())
