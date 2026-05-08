"""Unit tests for `AutoDANOptimizer`. CPU only, synthetic GPT-2 +
stub scanner + stub fitness evaluator.

Verifies the pre-filter callable plumbing, the all-blocked path keeps
the suffix unchanged, top-N maintenance dedupes and orders correctly,
and the JSONL contract matches what the transfer harness reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from attacks.autodan_gradient.gcg_step import GCGStep, GCGStepConfig
from attacks.autodan_gradient.objective import (
    AutoDANObjective,
    ObjectiveConfig,
)
from attacks.autodan_gradient.optimizer import (
    AutoDANOptimizer,
    OptimizerConfig,
    TopNEntry,
)
from surrogate.fitness.prefix_tokenized import TokenizedPrefix


_VOCAB = 64
_HIDDEN = 16


class StubScanner:
    """Blocks any prompt containing one of `blocked_substrings`."""

    def __init__(self, blocked_substrings: list[str] | None = None) -> None:
        self._blocked = list(blocked_substrings or [])

    def scan(self, text: str):
        for s in self._blocked:
            if s in text:
                return SimpleNamespace(total_score=10, blocked=True)
        return SimpleNamespace(total_score=0, blocked=False)


class StubTokenizer:
    """Char-per-token decode."""

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids)


def _stub_evaluator(call_log: list[str]):
    def evaluate(user_prompt: str) -> dict:
        call_log.append(user_prompt)
        return {
            "fitness": len(user_prompt) * 0.01,
            "scanner_score": 0,
            "leak_score": 0.5,
            "evasion_score": 1.0,
        }
    return evaluate


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


def _build_optimizer(
    *,
    seed_prefix: str = "Hello,",
    n_steps: int = 3,
    blocked_substrings: list[str] | None = None,
    max_resamples: int = 3,
):
    model = _tiny_model()
    obj = AutoDANObjective(model, tokenizer=None,
                           config=ObjectiveConfig(lambda_readability=0.1))
    gcg = GCGStep(obj, GCGStepConfig(
        top_k=8, batch_size=8, max_resamples=max_resamples,
    ))
    scanner = StubScanner(blocked_substrings)
    tokenizer = StubTokenizer()
    call_log: list[str] = []
    evaluator = _stub_evaluator(call_log)
    opt = AutoDANOptimizer(
        gcg_step=gcg,
        evaluator=evaluator,
        scanner=scanner,
        tokenizer=tokenizer,
        seed_prefix=seed_prefix,
        config=OptimizerConfig(n_steps=n_steps),
    )
    return opt, call_log


def test_decode_user_prompt_concatenates_seed_and_suffix(tmp_path: Path):
    opt, _ = _build_optimizer(seed_prefix="Hello,")
    suffix = torch.tensor([ord(" "), ord("w"), ord("o"), ord("r")],
                          dtype=torch.long)
    text = opt._decode_user_prompt(suffix)
    assert text == "Hello, wor"


def test_is_blocked_callable_uses_scanner_threshold(tmp_path: Path):
    """The optimizer's `_is_blocked` should return True for any suffix
    whose decoded user prompt trips the scanner at or above
    threshold."""
    opt, _ = _build_optimizer(blocked_substrings=["Hello,"])

    suffix = torch.tensor([ord(" "), ord("a")], dtype=torch.long)
    # seed_prefix="Hello," is in the user prompt -> scanner blocks.
    assert opt._is_blocked(suffix) is True


def test_top_n_dedupes_and_sorts_descending(tmp_path: Path):
    opt, _ = _build_optimizer()
    top_n: list[TopNEntry] = []
    opt._maybe_insert_top_n(top_n, "alpha", 0.3, 1)
    opt._maybe_insert_top_n(top_n, "beta", 0.5, 2)
    opt._maybe_insert_top_n(top_n, "gamma", 0.4, 3)
    opt._maybe_insert_top_n(top_n, "alpha", 0.3, 4)
    assert [e.prompt for e in top_n] == ["beta", "gamma", "alpha"]
    assert [e.fitness for e in top_n] == [0.5, 0.4, 0.3]


def test_top_n_evicts_lowest_when_full(tmp_path: Path):
    opt, _ = _build_optimizer()
    opt._config.top_n_tracked = 3
    top_n: list[TopNEntry] = []
    for prompt, fitness in [("a", 0.1), ("b", 0.2), ("c", 0.3),
                            ("d", 0.05),
                            ("e", 0.4)]:
        opt._maybe_insert_top_n(top_n, prompt, fitness, step=1)
    assert [e.prompt for e in top_n] == ["e", "c", "b"]


def test_run_emits_jsonl_with_top5_field(tmp_path: Path):
    """Transfer harness contract: every line carries `top5` with
    `prompt` and `fitness` per entry."""
    opt, _ = _build_optimizer(n_steps=2)
    prefix = _hand_built_prefix()
    out = tmp_path / "campaign.jsonl"
    opt.run(prefix, out)

    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        assert "step" in line
        assert "top5" in line
        assert "survivor_count" in line
        for entry in line["top5"]:
            assert set(entry.keys()) >= {"prompt", "fitness"}


def test_run_handles_all_blocked_step(tmp_path: Path):
    """When the pre-filter eliminates every candidate across all
    resample attempts, the suffix stays unchanged and the line shows
    `accepted: false`."""
    opt, _ = _build_optimizer(
        blocked_substrings=["Hello,"], n_steps=2, max_resamples=1,
    )
    prefix = _hand_built_prefix()
    out = tmp_path / "campaign.jsonl"
    summary = opt.run(prefix, out)

    assert summary.n_accepted == 0
    assert summary.n_all_blocked == 2
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    for line in lines:
        assert line["accepted"] is False
        assert line["accepted_loss"] is None
        assert line["survivor_count"] == 0


def test_run_abandons_when_below_floor_and_no_relative_improvement(tmp_path: Path):
    """If fitness stays below the absolute floor AND below the relative
    improvement target after abandon_after_steps, the run should exit
    early and mark the summary as abandoned."""
    model = _tiny_model()
    obj = AutoDANObjective(model, tokenizer=None,
                           config=ObjectiveConfig(lambda_readability=0.1))
    gcg = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=8, max_resamples=2))
    scanner = StubScanner(blocked_substrings=None)
    tokenizer = StubTokenizer()
    # Stub evaluator that returns a constant low fitness so neither
    # the floor nor the relative-improvement signal can ever fire.
    def stuck_evaluator(prompt: str) -> dict:
        return {"fitness": 0.0001, "scanner_score": 0,
                "leak_score": 0.001, "evasion_score": 1.0}
    config = OptimizerConfig(
        n_steps=20,
        abandon_after_steps=3,
        abandon_absolute_floor=0.005,
        abandon_min_improvement_ratio=1.5,
    )
    opt = AutoDANOptimizer(
        gcg_step=gcg, evaluator=stuck_evaluator, scanner=scanner,
        tokenizer=tokenizer, seed_prefix="hi",
        config=config,
    )
    prefix = _hand_built_prefix()
    out = tmp_path / "abandon.jsonl"
    summary = opt.run(prefix, out)

    assert summary.abandoned is True
    assert summary.abandoned_at_step == 3
    assert "below floor" in (summary.abandon_reason or "")
    assert summary.n_steps == 3   # n_steps reflects the abandonment step
    # Final JSONL line should carry the abandon marker.
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    last = json.loads(lines[-1])
    assert last.get("abandoned") is True


def test_run_continues_if_above_floor(tmp_path: Path):
    """If best_fitness stays above the absolute floor, abandonment
    must not fire even when the relative-improvement signal would
    not be met."""
    model = _tiny_model()
    obj = AutoDANObjective(model, tokenizer=None,
                           config=ObjectiveConfig(lambda_readability=0.1))
    gcg = GCGStep(obj, GCGStepConfig(top_k=8, batch_size=8, max_resamples=2))
    scanner = StubScanner(blocked_substrings=None)
    tokenizer = StubTokenizer()
    # Constant fitness above the absolute floor — should not abandon.
    def healthy_evaluator(prompt: str) -> dict:
        return {"fitness": 0.05, "scanner_score": 0,
                "leak_score": 0.05, "evasion_score": 1.0}
    config = OptimizerConfig(
        n_steps=5,
        abandon_after_steps=3,
        abandon_absolute_floor=0.005,
        abandon_min_improvement_ratio=1.5,
    )
    opt = AutoDANOptimizer(
        gcg_step=gcg, evaluator=healthy_evaluator, scanner=scanner,
        tokenizer=tokenizer, seed_prefix="hi",
        config=config,
    )
    summary = opt.run(_hand_built_prefix(), tmp_path / "healthy.jsonl")
    assert summary.abandoned is False
    assert summary.n_steps == 5


def test_run_smoke_end_to_end(tmp_path: Path):
    """Tiny end-to-end run. No blocking — every step accepts."""
    opt, call_log = _build_optimizer(n_steps=3)
    prefix = _hand_built_prefix()
    out = tmp_path / "campaign.jsonl"
    summary = opt.run(prefix, out)

    assert summary.n_steps == 3
    assert summary.n_accepted == 3
    assert len(call_log) == 3
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        assert line["accepted"] is True
        assert isinstance(line["accepted_loss"], float)
        assert isinstance(line["gradient_loss"], float)
        assert line["survivor_count"] > 0
