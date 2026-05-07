"""Unit tests for `render_prefix_tokenized`. CPU only, stub tokenizer.

The real Llama 3.1 tokenizer's chat template + offset mapping is
exercised in a RUN_LLAMA-gated integration test (forthcoming). These
tests verify the structural contract: tensor shapes, span arithmetic,
determinism, and the BPE-boundary-misalignment failure mode.
"""

from __future__ import annotations

import pytest
import torch

from surrogate.fitness.prefix_tokenized import (
    TokenizedPrefix,
    render_prefix_tokenized,
)


class _StubBatchEncoding(dict):
    """Mimics HF BatchEncoding's dict + attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


class StubTokenizer:
    """Char-per-token tokenizer with offset_mapping support.

    `apply_chat_template` produces a string with explicit role markers
    and the user content embedded verbatim, which lets us locate the
    seed prefix and target by `text.index`."""

    SYS_OPEN = "<<SYS>>"
    SYS_CLOSE = "<</SYS>>"
    USER_OPEN = "<<USER>>"
    USER_CLOSE = "<</USER>>"
    ASST_OPEN = "<<ASST>>"
    ASST_CLOSE = "<</ASST>>"

    def __call__(
        self, text: str, *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
    ):
        del add_special_tokens
        ids = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
        offsets = torch.tensor(
            [[(i, i + 1) for i in range(len(text))]], dtype=torch.long,
        )
        result: dict = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return _StubBatchEncoding(result)

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
        del tools, add_generation_prompt, tokenize
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"{self.SYS_OPEN}{content}{self.SYS_CLOSE}")
            elif role == "user":
                parts.append(f"{self.USER_OPEN}{content}{self.USER_CLOSE}")
            elif role == "assistant":
                parts.append(f"{self.ASST_OPEN}{content}{self.ASST_CLOSE}")
            else:  # pragma: no cover
                raise ValueError(role)
        return "".join(parts)


_FAKE_SYSTEM = "SYS_CONTENT"
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
    expected_suffix_text = (" !") * _SUFFIX_LEN
    # In the char-per-token stub, suffix length in tokens equals length
    # of the suffix-init text in characters.
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


def test_seed_collision_with_template_raises():
    """If the seed_prefix appears more than once in the rendered text
    (e.g. it accidentally collides with system content), we cannot
    safely slice. Surface the ambiguity loudly."""
    with pytest.raises(RuntimeError, match="appears .* times"):
        render_prefix_tokenized(
            StubTokenizer(),
            system_prompt="SYS_HAS_HelloDuplicate_Hello",
            tool_function_dicts=[],
            seed_prefix="Hello",  # appears twice in system + once in user
            suffix_len=_SUFFIX_LEN,
            target_string=_FAKE_TARGET,
        )


def test_decoded_suffix_round_trips_to_init_text():
    """Decoding suffix_init_ids back to text should yield the exact
    suffix-init string we asked for. The optimizer relies on this for
    its `_decode_user_prompt` reconstruction."""
    out = _build()
    decoded = StubTokenizer().decode(out.suffix_init_ids)
    expected = (" !") * _SUFFIX_LEN
    assert decoded == expected
