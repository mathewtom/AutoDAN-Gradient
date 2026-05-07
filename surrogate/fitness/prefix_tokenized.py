"""Tokenized chat-template builder for the AutoDAN-Gradient optimizer.

Companion to `prefix.py` (which returns the rendered string for eval-
time scoring). This module returns token IDs split into four named
regions plus the integer offsets of the mutable suffix region within
the concatenated full sequence:

    [ prefix_ids | suffix_ids | post_suffix_ids | target_ids ]
                       ▲
                       optimizer rewrites this every step

We render the chat template via the tokenizer's own
`apply_chat_template` (with tools), then locate the suffix and target
regions in the resulting token stream by matching character offsets.
This handles arbitrary template content — including the Llama 3.1
`# Tool Instructions` preamble and function-definition block — without
having to hand-assemble it.

Requires a "fast" HuggingFace tokenizer (PreTrainedTokenizerFast)
that supports `return_offsets_mapping`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TokenizedPrefix:
    """Token IDs for the four regions of a chat-templated leak-elicitation
    sequence, plus the integer offsets of the mutable suffix and target
    regions in the concatenated full sequence (half-open intervals).
    """

    prefix_ids: torch.Tensor
    suffix_init_ids: torch.Tensor
    post_suffix_ids: torch.Tensor
    target_ids: torch.Tensor
    suffix_span: tuple[int, int]
    target_span: tuple[int, int]

    def concat(self) -> torch.Tensor:
        return torch.cat(
            [self.prefix_ids, self.suffix_init_ids,
             self.post_suffix_ids, self.target_ids],
            dim=0,
        )


def render_prefix_tokenized(
    tokenizer: Any,
    *,
    system_prompt: str,
    tool_function_dicts: list[dict],
    seed_prefix: str,
    suffix_len: int,
    target_string: str,
    suffix_init_token: str = "!",
    device: torch.device | str = "cpu",
) -> TokenizedPrefix:
    """Build the four-region tokenized prefix.

    The rendered template's character positions for the seed prefix and
    target string are mapped to token indices via the tokenizer's
    offset-mapping. Slicing those positions yields the four regions.

    Raises
    ------
    RuntimeError
        If `seed_prefix` does not appear exactly once in the rendered
        text, if `target_string` is missing, or if BPE merges across
        the seed/suffix or suffix/post-suffix boundary in a way that
        prevents clean slicing.
    """
    # Pattern: " !" repeated. No trailing space — a trailing whitespace
    # character would be tokenized together with the following
    # `<|eot_id|>` special token, causing the suffix region to absorb
    # the EOT into its tail.
    suffix_init_text = (" " + suffix_init_token) * suffix_len

    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": seed_prefix + suffix_init_text},
            {"role": "assistant", "content": target_string},
        ],
        tools=tool_function_dicts or None,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    ids = encoding["input_ids"][0].to(device)
    offsets_tensor = encoding["offset_mapping"][0]
    offsets = [(int(s), int(e)) for s, e in offsets_tensor.tolist()]

    # Locate the seed prefix in the rendered text. It should appear
    # exactly once (inside the user message). If it appears 0 times the
    # caller passed a malformed seed; if multiple, the seed collides
    # with template content and we cannot safely slice.
    if seed_prefix:
        n_hits = text.count(seed_prefix)
        if n_hits != 1:
            raise RuntimeError(
                f"seed_prefix appears {n_hits} times in the rendered "
                "chat template (expected 1). The seed may collide with "
                "system-prompt or tool-block content."
            )
        seed_start_char = text.index(seed_prefix)
        seed_end_char = seed_start_char + len(seed_prefix)
    else:
        # Empty seed: locate the suffix init text directly.
        if text.count(suffix_init_text) != 1:
            raise RuntimeError(
                "Empty seed_prefix and suffix_init_text is not unique "
                "in the rendered template; cannot localize suffix region."
            )
        seed_end_char = text.index(suffix_init_text)

    suffix_end_char = seed_end_char + len(suffix_init_text)

    # Target appears at the assistant-message position (last occurrence
    # in case the system prompt happens to mention it earlier).
    target_start_char = text.rindex(target_string)
    target_end_char = target_start_char + len(target_string)

    suffix_token_start = _first_token_at_or_after(offsets, seed_end_char)
    suffix_token_end = _first_token_at_or_after(offsets, suffix_end_char)
    target_token_start = _first_token_at_or_after(offsets, target_start_char)
    target_token_end = _first_token_at_or_after(offsets, target_end_char)

    # Token boundaries must align with the seed/suffix boundary.
    # If BPE merges the trailing seed char with the leading suffix space,
    # the suffix-start token will begin BEFORE seed_end_char and we
    # cannot cleanly slice. Surface this loudly rather than silently
    # mis-aligning gradient positions.
    if suffix_token_start >= len(offsets):
        raise RuntimeError(
            "suffix region falls past end of token stream; "
            "rendered template likely truncated."
        )
    actual_start = offsets[suffix_token_start][0]
    if actual_start != seed_end_char:
        raise RuntimeError(
            "Suffix region does not align with a token boundary. "
            f"Expected token starting at char {seed_end_char}, found "
            f"token starting at char {actual_start}. BPE merged across "
            "the seed/suffix boundary. Pick a seed prefix that ends "
            "in a clear word boundary (period, space, etc.)."
        )

    prefix_ids = ids[:suffix_token_start]
    suffix_init_ids = ids[suffix_token_start:suffix_token_end]
    post_suffix_ids = ids[suffix_token_end:target_token_start]
    target_ids = ids[target_token_start:target_token_end]

    # Defensive: suffix_init_ids must not contain any special or added
    # token. A token like <|eot_id|> mid-suffix would break the chat-
    # template structure and corrupt every downstream gradient signal.
    forbidden: set[int] = set(
        getattr(tokenizer, "all_special_ids", []) or []
    )
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    forbidden.update(int(v) for v in added.values())
    bad = [int(t) for t in suffix_init_ids.tolist() if int(t) in forbidden]
    if bad:
        raise RuntimeError(
            f"suffix_init_ids contains special/added token ids {bad}. "
            "This indicates an offset-mapping boundary issue — the suffix "
            "absorbed a chat-template special token. Inspect the trailing "
            "characters of suffix_init_text."
        )

    return TokenizedPrefix(
        prefix_ids=prefix_ids,
        suffix_init_ids=suffix_init_ids,
        post_suffix_ids=post_suffix_ids,
        target_ids=target_ids,
        suffix_span=(int(suffix_token_start), int(suffix_token_end)),
        target_span=(int(target_token_start), int(target_token_end)),
    )


def _first_token_at_or_after(
    offsets: list[tuple[int, int]],
    char_pos: int,
) -> int:
    """Index of the first token whose start char >= `char_pos`.
    Returns `len(offsets)` if no such token exists."""
    for i, (s, _e) in enumerate(offsets):
        if s >= char_pos:
            return i
    return len(offsets)
