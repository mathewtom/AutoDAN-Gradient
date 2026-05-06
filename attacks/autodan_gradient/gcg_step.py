"""One step of GCG-style coordinate descent for AutoDAN-Zhu.

Each `GCGStep.step(...)` performs:

  Phase A — gradient ranking
      Build a one-hot tensor of the current suffix, run the
      differentiable AutoDAN objective, backprop. The resulting
      gradient over the (suffix_len, vocab_size) one-hot ranks
      replacement tokens per position; take top-K.

  Phase B — candidate verification
      Sample B (position, token) swaps from the per-position top-K
      pool. Score the B candidates with one batched forward pass.
      Sort by loss ascending and return.

The outer optimizer applies the InjectionScanner veto on the sorted
list and accepts the lowest-loss survivor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from attacks.autodan_gradient.objective import AutoDANObjective
from surrogate.fitness.prefix_tokenized import TokenizedPrefix


@dataclass
class GCGStepConfig:
    """Inner-loop tuning knobs."""

    top_k: int = 256
    batch_size: int = 64


@dataclass
class StepResult:
    """Output of one step.

    `candidates` is sorted by loss ascending so the optimizer can
    iterate top-down applying the scanner veto.
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
    ) -> StepResult:
        suffix_len = int(current_suffix_ids.shape[0])
        vocab_size = self._objective.vocab_size
        device = self._objective.device

        # ----- Phase A: gradient ranking -----
        # Match the embedding dtype (bfloat16 in production, fp32 in
        # tests) so the matmul inside the objective doesn't fail.
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

        # Mask current token at each position so topk(-grad) cannot
        # nominate a no-op swap.
        grad[torch.arange(suffix_len, device=device),
             current_suffix_ids.to(device)] = float("inf")

        top_k = min(self._config.top_k, vocab_size)
        topk_token_ids = (-grad).topk(k=top_k, dim=-1).indices

        # ----- Phase B: sample and verify -----
        B = self._config.batch_size
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

        # expand-then-clone: required because expand() returns a
        # storage-sharing view that cannot be written into.
        candidates_suffix = current_suffix_ids.to(device).unsqueeze(0).expand(
            B, -1,
        ).clone()
        candidates_suffix[torch.arange(B, device=device),
                          sampled_positions] = sampled_token_ids

        losses = self._objective.score_batch(candidates_suffix, prefix)

        sorted_idx = losses.argsort()
        sorted_losses = losses[sorted_idx].tolist()
        sorted_suffixes = [
            candidates_suffix[i].detach().clone().cpu()
            for i in sorted_idx.tolist()
        ]
        candidate_pairs = list(zip(sorted_losses, sorted_suffixes))

        return StepResult(
            candidates=candidate_pairs,
            gradient_loss=float(loss.detach().item()),
            gradient_diagnostics=diag,
        )
