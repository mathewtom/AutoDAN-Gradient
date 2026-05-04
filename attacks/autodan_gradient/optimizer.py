"""GCG-style coordinate-descent optimizer with readability + scanner guard.

Per Zhu et al. (AutoDAN, 2023), each step:

  1. Compute one-hot embedding gradients of `loss` w.r.t. the candidate
     token positions.
  2. Take top-K candidate replacements per position via
     `gather_gradient_topk(grads, k=K)`.
  3. Sample a batch of single-token swaps from the top-K set and score
     each candidate's full loss with a forward pass.
  4. Apply the best-loss candidate that ALSO passes the scanner-evasion
     guard (`InjectionScanner.scan(decoded_prompt).total_score < 5`).
     Candidates that would be blocked are filtered out before argmin.

The scanner guard makes the optimization a constrained search: gradient
points the way, but candidates that step over the regex line are rejected
even if their gradient-aligned loss is lower. This is the gradient analog
of HGA's multiplicative tier composition — the scanner is a hard veto, not
a smooth penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizerConfig:
    n_steps: int = 500
    top_k: int = 256
    batch_size: int = 64
    suffix_len: int = 20
    seed: int = 1


class AutoDANOptimizer:
    """Coordinate-descent optimization of the adversarial prompt's tokens
    under a leak+readability loss with a scanner-evasion hard constraint.

    The optimizer wraps an `AutoDANObjective` (differentiable loss) and a
    `SystemPromptLeakFitness` (the eval-time fitness, used as the scanner
    guard and as the JSONL-logged metric).
    """

    def __init__(self, objective: Any, evaluator: Any, config: OptimizerConfig):
        self._objective = objective
        self._evaluator = evaluator
        self._config = config

    def run(self, starting_prompt: str):
        raise NotImplementedError(
            "AutoDAN-Gradient optimizer not yet implemented. "
            "See README §'Methodology' for the planned step shape."
        )
