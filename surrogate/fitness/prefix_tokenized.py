"""Tokenized chat-template builder for the AutoDAN-Gradient optimizer.

Companion to `prefix.py` (which returns a rendered string for eval-time
scoring). This module returns token IDs split into four named regions
plus the integer offsets of the mutable region within the concatenated
sequence:

    [ prefix_ids | suffix_ids | post_suffix_ids | target_ids ]
                       ▲
                       optimizer rewrites this every step

The hardcoded segment strings below are Llama 3.1's chat template. A
self-check at the bottom of `render_prefix_tokenized` verifies the
hand-assembled token stream matches what `tokenizer.apply_chat_template`
would produce for the equivalent message list — if Meta updates the
template in a future tokenizer release, the check raises rather than
silently shifting gradient positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_BOS = "<|begin_of_text|>"
_EOT = "<|eot_id|>"
_SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
_USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


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
    """Build the four-region tokenized prefix for one campaign run.

    `seed_prefix` anchors the suffix semantically; passing `""` is
    allowed. `suffix_init_token` defaults to `"!"` per the GCG paper
    convention (single neutral BPE token, scanner-safe).

    Raises
    ------
    RuntimeError
        If the assembled segments do not produce the same token IDs as
        `tokenizer.apply_chat_template` for the equivalent messages.
        Indicates the segment constants have drifted from the
        tokenizer's official template.
    """
    def _encode(text: str) -> torch.Tensor:
        # add_special_tokens=False prevents auto-injection of BOS/EOS
        # mid-sequence when chunks are concatenated.
        ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
        ).input_ids[0]
        return ids.to(device)

    bos_ids = _encode(_BOS)
    system_block_ids = _encode(_SYSTEM_HEADER + system_prompt)
    user_header_ids = _encode(_EOT + _USER_HEADER)
    seed_ids = (
        _encode(seed_prefix) if seed_prefix
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    # Leading space matters in BPE: the first suffix token must look
    # space-prefixed since it's appended after the seed.
    suffix_init_ids = _encode(" " + (suffix_init_token + " ") * suffix_len)
    post_suffix_ids = _encode(_EOT + _ASSISTANT_HEADER)
    target_ids = _encode(target_string)

    prefix_ids = torch.cat(
        [bos_ids, system_block_ids, user_header_ids, seed_ids], dim=0,
    )

    suffix_start = int(prefix_ids.shape[0])
    suffix_end = suffix_start + int(suffix_init_ids.shape[0])
    target_start = suffix_end + int(post_suffix_ids.shape[0])
    target_end = target_start + int(target_ids.shape[0])

    _verify_against_apply_chat_template(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        tool_function_dicts=tool_function_dicts,
        seed_prefix=seed_prefix,
        suffix_init_ids=suffix_init_ids,
        target_ids=target_ids,
        prefix_ids=prefix_ids,
        post_suffix_ids=post_suffix_ids,
        device=device,
    )

    return TokenizedPrefix(
        prefix_ids=prefix_ids,
        suffix_init_ids=suffix_init_ids,
        post_suffix_ids=post_suffix_ids,
        target_ids=target_ids,
        suffix_span=(suffix_start, suffix_end),
        target_span=(target_start, target_end),
    )


def _verify_against_apply_chat_template(
    *,
    tokenizer: Any,
    system_prompt: str,
    tool_function_dicts: list[dict],
    seed_prefix: str,
    suffix_init_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    post_suffix_ids: torch.Tensor,
    device: torch.device | str,
) -> None:
    """Raise if our hand-assembled token IDs differ from what
    `apply_chat_template` would produce for the equivalent message
    list. Compares position-for-position; the error message includes
    the first divergent index.
    """
    suffix_text = tokenizer.decode(suffix_init_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    user_text = seed_prefix + suffix_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target_text},
    ]
    canonical_text = tokenizer.apply_chat_template(
        messages,
        tools=tool_function_dicts or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    canonical_ids = tokenizer(
        canonical_text, add_special_tokens=False, return_tensors="pt",
    ).input_ids[0].to(device)

    assembled = torch.cat(
        [prefix_ids, suffix_init_ids, post_suffix_ids, target_ids], dim=0,
    )

    if assembled.shape != canonical_ids.shape or not torch.equal(
        assembled, canonical_ids,
    ):
        common_len = min(assembled.shape[0], canonical_ids.shape[0])
        first_diff = next(
            (i for i in range(common_len)
             if int(assembled[i]) != int(canonical_ids[i])),
            None,
        )
        raise RuntimeError(
            "prefix_tokenized self-check failed: hand-assembled segments "
            "do not match tokenizer.apply_chat_template output. "
            f"assembled.len={assembled.shape[0]} "
            f"canonical.len={canonical_ids.shape[0]} "
            f"first_diff_index={first_diff}. "
            "The hardcoded chat-template segment constants have drifted "
            "from the tokenizer's official template."
        )
