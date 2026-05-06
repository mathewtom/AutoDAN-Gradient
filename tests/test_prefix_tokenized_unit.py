"""Unit tests for `render_prefix_tokenized`. CPU only, stub tokenizer.

Verifies the structural contract: tensor shapes, span arithmetic,
determinism, and that the self-check fires on chat-template drift.
The hardcoded Llama 3.1 segment correctness is verified against the
real tokenizer in a separate RUN_LLAMA-gated integration test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from surrogate.fitness import prefix_tokenized as pt_mod
from surrogate.fitness.prefix_tokenized import (
    TokenizedPrefix,
    render_prefix_tokenized,
)


class StubTokenizer:
    """Char-per-token stub. `apply_chat_template` reuses the function-
    under-test's segment constants so the self-check passes by
    construction in non-drift tests; `broken_template=True` injects a
    drift to exercise the failure path."""

    def __init__(self, *, broken_template: bool = False) -> None:
        self.broken_template = broken_template

    def __call__(
        self, text: str, *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ):
        del add_special_tokens
        ids = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=ids)
        return SimpleNamespace(input_ids=ids[0].tolist())

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids)

    def apply_chat_template(
        self, messages, *,
        tools=None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        del tools
        s = pt_mod._BOS
        for m in messages:
            if m["role"] == "system":
                s += pt_mod._SYSTEM_HEADER + m["content"]
            elif m["role"] == "user":
                s += pt_mod._EOT + pt_mod._USER_HEADER + m["content"]
            elif m["role"] == "assistant":
                s += pt_mod._EOT + pt_mod._ASSISTANT_HEADER + m["content"]
            else:  # pragma: no cover
                raise ValueError(f"unknown role {m['role']!r}")
        if add_generation_prompt:
            s += pt_mod._EOT + pt_mod._ASSISTANT_HEADER
        if self.broken_template:
            s = s.replace(pt_mod._BOS, pt_mod._BOS + " ")
        if tokenize:
            return [ord(c) for c in s]
        return s


_FAKE_SYSTEM = "SYS"
_FAKE_SEED = "Hello,"
_FAKE_TARGET = "WORLD"
_SUFFIX_LEN = 5


def _build():
    return render_prefix_tokenized(
        StubTokenizer(),
        system_prompt=_FAKE_SYSTEM,
        tool_function_dicts=[],
        seed_prefix=_FAKE_SEED,
        suffix_len=_SUFFIX_LEN,
        target_string=_FAKE_TARGET,
    )


def test_basic_shape():
    out = _build()
    assert isinstance(out, TokenizedPrefix)
    for name in ("prefix_ids", "suffix_init_ids",
                 "post_suffix_ids", "target_ids"):
        t = getattr(out, name)
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 1
        assert t.dtype == torch.long
    expected_suffix_text = " " + ("! " * _SUFFIX_LEN)
    assert out.suffix_init_ids.shape[0] == len(expected_suffix_text)


def test_concat_round_trip():
    out = _build()
    full = out.concat()
    s_lo, s_hi = out.suffix_span
    t_lo, t_hi = out.target_span

    assert 0 <= s_lo < s_hi <= full.shape[0]
    assert 0 <= t_lo < t_hi <= full.shape[0]
    assert torch.equal(full[s_lo:s_hi], out.suffix_init_ids)
    assert torch.equal(full[t_lo:t_hi], out.target_ids)
    assert s_hi <= t_lo
    assert (t_lo - s_hi) == out.post_suffix_ids.shape[0]


def test_determinism():
    a = _build()
    b = _build()
    for name in ("prefix_ids", "suffix_init_ids",
                 "post_suffix_ids", "target_ids"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert a.suffix_span == b.suffix_span
    assert a.target_span == b.target_span


def test_self_check_fires_on_template_drift():
    with pytest.raises(RuntimeError, match="self-check failed"):
        render_prefix_tokenized(
            StubTokenizer(broken_template=True),
            system_prompt=_FAKE_SYSTEM,
            tool_function_dicts=[],
            seed_prefix=_FAKE_SEED,
            suffix_len=_SUFFIX_LEN,
            target_string=_FAKE_TARGET,
        )


def test_empty_seed_is_allowed():
    out = render_prefix_tokenized(
        StubTokenizer(),
        system_prompt=_FAKE_SYSTEM,
        tool_function_dicts=[],
        seed_prefix="",
        suffix_len=_SUFFIX_LEN,
        target_string=_FAKE_TARGET,
    )
    assert out.suffix_span[0] == out.prefix_ids.shape[0]
    assert out.suffix_span[1] - out.suffix_span[0] == out.suffix_init_ids.shape[0]
