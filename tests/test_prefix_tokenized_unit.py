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
        # Char-per-token, EXCEPT " !" merges to a single token (id 758,
        # mimicking Llama 3.1's BPE behavior on this pair). This is
        # what makes the seed-padding logic testable against the stub.
        token_ids: list[int] = []
        offsets_list: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i] == " " and text[i + 1] == "!":
                token_ids.append(758)
                offsets_list.append((i, i + 2))
                i += 2
            else:
                token_ids.append(ord(text[i]))
                offsets_list.append((i, i + 1))
                i += 1
        ids = torch.tensor([token_ids], dtype=torch.long)
        offsets = torch.tensor([offsets_list], dtype=torch.long)
        result: dict = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return _StubBatchEncoding(result)

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out = []
        for i in ids:
            if int(i) == 758:
                out.append(" !")
            else:
                out.append(chr(int(i)))
        return "".join(out)

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
    # The stub merges " !" into one token, so suffix_init = (" !") * N
    # tokenizes to exactly N tokens, matching suffix_len.
    assert out.suffix_init_ids.shape[0] == _SUFFIX_LEN


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


def test_seed_text_pads_with_filler_to_match_suffix_len():
    """Caller-provided seed text is padded with ' !' until the suffix
    region tokenizes to exactly suffix_len tokens. The seed appears at
    the head; filler at the tail."""
    out = render_prefix_tokenized(
        StubTokenizer(),
        system_prompt=_FAKE_SYSTEM,
        tool_function_dicts=[],
        seed_prefix=_FAKE_SEED,
        suffix_len=_SUFFIX_LEN,
        target_string=_FAKE_TARGET,
        suffix_init_text="ABC",     # 3 chars/tokens in the stub
    )
    assert out.suffix_init_ids.shape[0] == _SUFFIX_LEN
    decoded = StubTokenizer().decode(out.suffix_init_ids)
    # Seed "ABC" = 3 tokens, deficit = 2, padded with 2 " !" tokens.
    assert decoded == "ABC" + " !" * 2


def test_seed_text_longer_than_suffix_len_passes_through():
    """suffix_len is a MINIMUM slot count, not an exact target. A seed
    longer than suffix_len passes through unchanged; the resulting
    suffix region is as long as the seed tokenizes to."""
    out = render_prefix_tokenized(
        StubTokenizer(),
        system_prompt=_FAKE_SYSTEM,
        tool_function_dicts=[],
        seed_prefix=_FAKE_SEED,
        suffix_len=3,
        target_string=_FAKE_TARGET,
        suffix_init_text="ABCDEFG",   # 7 tokens > suffix_len=3
    )
    assert out.suffix_init_ids.shape[0] == 7
    decoded = StubTokenizer().decode(out.suffix_init_ids)
    assert decoded == "ABCDEFG"


def test_decoded_suffix_round_trips_to_init_text():
    """Decoding suffix_init_ids back to text should yield the exact
    suffix-init string we asked for. The optimizer relies on this for
    its `_decode_user_prompt` reconstruction."""
    out = _build()
    decoded = StubTokenizer().decode(out.suffix_init_ids)
    expected = (" !") * _SUFFIX_LEN
    assert decoded == expected
