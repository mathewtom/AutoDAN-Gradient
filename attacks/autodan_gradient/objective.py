"""AutoDAN-Zhu differentiable loss: leak elicitation + readability.

    L(x) = -log p(target | prefix, x) + lambda * -log p_LM(x)

The leak term is GCG's core signal (Zou et al. 2023). The readability
term (Zhu et al. 2023) penalizes low-likelihood suffixes under the
same surrogate, keeping prompts human-legible instead of token-soup.

Both terms come from a single forward pass over
`[prefix | suffix_embeds | post_suffix | target]`. Suffix tokens enter
the graph as `one_hot @ embedding_matrix` so gradients flow back to a
caller-supplied one-hot tensor; everything else is fixed token IDs.
This module is the gradient-time counterpart to
`surrogate.fitness.system_prompt_leak.SystemPromptLeakFitness` and is
the only place in the lab that needs autograd over the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from surrogate.fitness.prefix_tokenized import TokenizedPrefix


@dataclass
class ObjectiveConfig:
    """Tuning knobs for the AutoDAN-Zhu loss.

    `lambda_readability=0` disables the readability term (= GCG/Zou).
    """

    lambda_readability: float = 0.3


class AutoDANObjective:
    """Differentiable loss for AutoDAN-Zhu adversarial suffix search.

    Model parameters are frozen on construction, so the only tensor
    that can accumulate gradient is the caller-supplied one-hot suffix.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ObjectiveConfig | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or ObjectiveConfig()

        for p in model.parameters():
            p.requires_grad_(False)

        self._embed_module = model.get_input_embeddings()

    @property
    def vocab_size(self) -> int:
        return int(self._embed_module.weight.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self._embed_module.weight.shape[1])

    @property
    def device(self) -> torch.device:
        return self._embed_module.weight.device

    @property
    def forbidden_token_ids(self) -> list[int]:
        """Token IDs the GCG step must not propose as suffix replacements.

        Three classes are forbidden:
          - registered special tokens (BOS, EOS, EOT)
          - added tokens (reserved_special_token_*, header markers,
            python_tag, etc.) — Llama 3.1 keeps these out of
            `all_special_ids`
          - byte-fallback tokens that do not decode to valid UTF-8 in
            isolation (the visible `�` glitch). These break round-
            tripping of the suffix to the live agent's HTTP layer.

        Cached on first access — the byte scan walks the full vocab.
        """
        if self._tokenizer is None:
            return []
        if hasattr(self, "_forbidden_cache"):
            return self._forbidden_cache

        forbidden: set[int] = set(
            getattr(self._tokenizer, "all_special_ids", []) or []
        )
        added = getattr(self._tokenizer, "added_tokens_encoder", None) or {}
        forbidden.update(int(v) for v in added.values())

        # Byte-fallback tokens: any vocab id whose isolated decoding
        # contains the Unicode replacement char.
        decode = getattr(self._tokenizer, "decode", None)
        if decode is not None:
            for tid in range(self.vocab_size):
                if tid in forbidden:
                    continue
                try:
                    s = decode([tid], skip_special_tokens=False)
                except Exception:
                    forbidden.add(tid)
                    continue
                if "�" in s:
                    forbidden.add(tid)

        self._forbidden_cache = list(forbidden)
        return self._forbidden_cache

    def loss(
        self,
        suffix_one_hot: torch.Tensor,
        prefix: TokenizedPrefix,
    ) -> tuple[torch.Tensor, dict]:
        """Compute L(x) and return (scalar_loss, diagnostics_dict).

        `suffix_one_hot` shape (suffix_len, vocab_size). Caller is
        responsible for `requires_grad=True` if gradients are wanted.
        Returned scalar carries the autograd graph; diagnostics fields
        are detached floats safe for logging.
        """
        if suffix_one_hot.shape[1] != self.vocab_size:
            raise ValueError(
                f"suffix_one_hot.shape[1]={suffix_one_hot.shape[1]} "
                f"does not match model vocab_size={self.vocab_size}"
            )

        prefix_embeds = self._embed_module(prefix.prefix_ids)
        post_suffix_embeds = self._embed_module(prefix.post_suffix_ids)
        target_embeds = self._embed_module(prefix.target_ids)
        # Differentiability bridge: matmul preserves the autograd
        # graph from suffix_one_hot, while the embedding lookup
        # used for fixed regions does not.
        suffix_embeds = suffix_one_hot @ self._embed_module.weight

        full_embeds = torch.cat(
            [prefix_embeds, suffix_embeds, post_suffix_embeds, target_embeds],
            dim=0,
        ).unsqueeze(0)

        # NOT under inference_mode/no_grad — those would silently kill
        # autograd. Memory cost is bounded by frozen model parameters.
        outputs = self._model(inputs_embeds=full_embeds)
        logits = outputs.logits

        # logits[i] predicts position i+1, hence the -1 offset.
        target_start, target_end = prefix.target_span
        target_logits = logits[0, target_start - 1:target_end - 1, :]
        target_log_probs = torch.log_softmax(target_logits.float(), dim=-1)
        leak_per_token = target_log_probs.gather(
            dim=1, index=prefix.target_ids.unsqueeze(-1),
        ).squeeze(-1)
        leak_log_prob = leak_per_token.sum()

        suffix_start, suffix_end = prefix.suffix_span
        suffix_logits = logits[0, suffix_start - 1:suffix_end - 1, :]
        suffix_log_probs = torch.log_softmax(suffix_logits.float(), dim=-1)
        # For a true one-hot input, argmax recovers the exact token.
        suffix_token_ids = suffix_one_hot.argmax(dim=-1)
        readability_per_token = suffix_log_probs.gather(
            dim=1, index=suffix_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        readability_log_prob = readability_per_token.sum()

        loss = (-leak_log_prob) + self._config.lambda_readability * (
            -readability_log_prob
        )

        diagnostics = {
            "leak_log_prob": float(leak_log_prob.detach().item()),
            "readability_log_prob": float(
                readability_log_prob.detach().item(),
            ),
            "lambda_readability": float(self._config.lambda_readability),
        }
        return loss, diagnostics

    def score_batch(
        self,
        suffix_ids_batch: torch.Tensor,
        prefix: TokenizedPrefix,
    ) -> torch.Tensor:
        """Batched no-grad scoring of B candidate suffixes.

        Mirrors `loss()` numerically but operates on integer token IDs
        and returns a (B,) loss tensor. Used by the GCG step's Phase B
        candidate verification.
        """
        B, _ = suffix_ids_batch.shape

        prefix_ids = prefix.prefix_ids.unsqueeze(0).expand(B, -1)
        post_ids = prefix.post_suffix_ids.unsqueeze(0).expand(B, -1)
        target_ids = prefix.target_ids.unsqueeze(0).expand(B, -1)

        full_ids = torch.cat(
            [prefix_ids, suffix_ids_batch, post_ids, target_ids], dim=1,
        )

        with torch.no_grad():
            logits = self._model(full_ids).logits

        target_start, target_end = prefix.target_span
        target_logits = logits[:, target_start - 1:target_end - 1, :]
        target_log_probs = torch.log_softmax(target_logits.float(), dim=-1)
        leak_per_token = target_log_probs.gather(
            dim=2, index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        leak_log_prob = leak_per_token.sum(dim=1)

        suffix_start, suffix_end = prefix.suffix_span
        suffix_logits = logits[:, suffix_start - 1:suffix_end - 1, :]
        suffix_log_probs = torch.log_softmax(suffix_logits.float(), dim=-1)
        readability_per_token = suffix_log_probs.gather(
            dim=2, index=suffix_ids_batch.unsqueeze(-1),
        ).squeeze(-1)
        readability_log_prob = readability_per_token.sum(dim=1)

        return (-leak_log_prob) + self._config.lambda_readability * (
            -readability_log_prob
        )
