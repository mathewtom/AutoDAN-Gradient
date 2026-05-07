"""One step of GCG-style coordinate descent for AutoDAN-Zhu.

Each `GCGStep.step(...)` performs:

  Phase A — gradient ranking
      Build a one-hot tensor of the current suffix, run the
      differentiable AutoDAN objective, backprop. The resulting
      gradient over the (suffix_len, vocab_size) one-hot ranks
      replacement tokens per position; take top-K.

  Phase B — candidate sampling, scanner pre-filter, batched scoring
      Sample B (position, token) swaps from the per-position top-K.
      Optional `is_blocked` callable filters the batch before the
      forward pass; if all are blocked, resample (capped). Score
      survivors with one batched forward pass. Sort ascending.

The outer optimizer just accepts the lowest-loss survivor — no
post-step veto walk is needed when `is_blocked` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from attacks.autodan_gradient.objective import AutoDANObjective
from surrogate.fitness.prefix_tokenized import TokenizedPrefix


@dataclass
class GCGStepConfig:
    """Inner-loop tuning knobs."""

    top_k: int = 256
    batch_size: int = 64
    # Maximum sampling rerolls per step when the scanner pre-filter
    # eliminates ALL B candidates. Each reroll draws a fresh batch
    # from the same top-K with new RNG state.
    max_resamples: int = 3


@dataclass
class StepResult:
    """Output of one step.

    `candidates` is sorted by loss ascending. May be empty if the
    scanner pre-filter eliminated every candidate across all
    resample attempts.
    """

    candidates: list[tuple[float, torch.Tensor]]
    gradient_loss: float
    gradient_diagnostics: dict


class GCGStep:
    """Stateless across calls except for the optional RNG (held so
    seeded campaigns are reproducible)."""

    def __init__(
        self,
        objective: AutoDANObjective,
        config: GCGStepConfig | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._objective = objective
        self._config = config or GCGStepConfig()
        self._generator = generator

    def step(
        self,
        current_suffix_ids: torch.Tensor,
        prefix: TokenizedPrefix,
        *,
        is_blocked: Callable[[torch.Tensor], bool] | None = None,
    ) -> StepResult:
        suffix_len = int(current_suffix_ids.shape[0])
        vocab_size = self._objective.vocab_size
        device = self._objective.device

        # ----- Phase A: gradient ranking -----
        embed_dtype = self._objective._embed_module.weight.dtype
        one_hot = torch.zeros(
            suffix_len, vocab_size, dtype=embed_dtype, device=device,
        )
        one_hot[torch.arange(suffix_len, device=device),
                current_suffix_ids.to(device)] = 1.0
        one_hot.requires_grad_(True)

        loss, diag = self._objective.loss(one_hot, prefix)
        loss.backward()
        assert one_hot.grad is not None, "autograd failed to populate grad"
        grad = one_hot.grad.detach()

        grad[torch.arange(suffix_len, device=device),
             current_suffix_ids.to(device)] = float("inf")

        # Mask special tokens (BOS, EOS, EOT, header markers, etc.) at
        # every suffix position. A special token mid-suffix would break
        # the chat-template structure and make the model see a malformed
        # sequence.
        forbidden = self._objective.forbidden_token_ids
        if forbidden:
            grad[:, forbidden] = float("inf")

        top_k = min(self._config.top_k, vocab_size)
        topk_token_ids = (-grad).topk(k=top_k, dim=-1).indices

        # ----- Phase B: sample + pre-filter + score -----
        survivors = self._sample_and_filter(
            current_suffix_ids=current_suffix_ids.to(device),
            topk_token_ids=topk_token_ids,
            suffix_len=suffix_len,
            top_k=top_k,
            device=device,
            is_blocked=is_blocked,
        )

        if survivors.shape[0] == 0:
            # All candidates blocked across every resample attempt.
            # Optimizer treats empty list as an "all blocked" step.
            return StepResult(
                candidates=[],
                gradient_loss=float(loss.detach().item()),
                gradient_diagnostics=diag,
            )

        losses = self._objective.score_batch(survivors, prefix)

        sorted_idx = losses.argsort()
        sorted_losses = losses[sorted_idx].tolist()
        sorted_suffixes = [
            survivors[i].detach().clone().cpu()
            for i in sorted_idx.tolist()
        ]
        candidate_pairs = list(zip(sorted_losses, sorted_suffixes))

        return StepResult(
            candidates=candidate_pairs,
            gradient_loss=float(loss.detach().item()),
            gradient_diagnostics=diag,
        )

    def _sample_and_filter(
        self,
        *,
        current_suffix_ids: torch.Tensor,
        topk_token_ids: torch.Tensor,
        suffix_len: int,
        top_k: int,
        device: torch.device,
        is_blocked: Callable[[torch.Tensor], bool] | None,
    ) -> torch.Tensor:
        """Draw B candidate suffixes and filter via `is_blocked`. If
        every candidate is filtered, redraw with fresh RNG, up to
        `max_resamples` times. Returns surviving (S, suffix_len) tensor.

        When `is_blocked` is None, returns the first batch of B
        candidates unfiltered.
        """
        B = self._config.batch_size
        attempts = 1 + self._config.max_resamples

        for _ in range(attempts):
            sampled_positions = torch.randint(
                low=0, high=suffix_len, size=(B,),
                device=device, generator=self._generator,
            )
            sampled_topk_ranks = torch.randint(
                low=0, high=top_k, size=(B,),
                device=device, generator=self._generator,
            )
            sampled_token_ids = topk_token_ids[
                sampled_positions, sampled_topk_ranks,
            ]

            candidates = current_suffix_ids.unsqueeze(0).expand(
                B, -1,
            ).clone()
            candidates[torch.arange(B, device=device),
                       sampled_positions] = sampled_token_ids

            if is_blocked is None:
                return candidates

            keep_mask = torch.tensor(
                [not is_blocked(candidates[i]) for i in range(B)],
                dtype=torch.bool, device=device,
            )
            survivors = candidates[keep_mask]
            if survivors.shape[0] > 0:
                return survivors
            # else: full bust — try again with fresh RNG draws

        # Never found a single survivor.
        return torch.empty((0, suffix_len), dtype=torch.long, device=device)
