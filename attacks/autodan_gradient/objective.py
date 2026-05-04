"""AutoDAN-Gradient objective: leak elicitation + readability constraint.

The differentiable optimization objective for Zhu et al.'s AutoDAN:

    L(x) = -log p(target | prefix, x)         # leak signal (lower = better)
         + lambda_read * -log p_LM(x)         # readability penalty

`x` is the adversarial prompt's token sequence. The readability LM is the
same Llama 3.1 8B base used as the surrogate — keeps the scoring consistent
with what the production model would see.

The two-tier composition (scanner evasion × leak score) used at evaluation
time lives in `surrogate.fitness.system_prompt_leak.SystemPromptLeakFitness`
and is consumed by the optimizer as an evaluator-and-guard rather than as a
differentiable loss (the InjectionScanner is regex, not differentiable).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObjectiveConfig:
    lambda_readability: float = 0.1
    target_string: str = ""


class AutoDANObjective:
    """Differentiable loss combining leak log-prob and readability log-prob.

    Both terms are computed via a single forward pass over the same
    surrogate model — the leak term reads logits at the post-prefix position
    (see `surrogate.fitness.log_prob.target_log_prob`), the readability term
    reads logits along the candidate prompt's own tokens.
    """

    def __init__(self, surrogate, tokenizer, config: ObjectiveConfig):
        self._model = surrogate
        self._tokenizer = tokenizer
        self._config = config

    def loss(self, prompt_ids, prefix_ids, target_ids):
        raise NotImplementedError(
            "AutoDAN-Gradient objective not yet implemented. "
            "See README §'Methodology' for the planned loss shape."
        )
