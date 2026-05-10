"""AutoDAN-Gradient outer loop: scanner-vetoed coordinate descent.

Drives the campaign:
  - calls `GCGStep` with an `is_blocked` callable so the scanner veto
    runs BEFORE the batched forward pass, eliminating wasted compute
    on doomed candidates,
  - takes the lowest-loss survivor from the GCG step's sorted output,
  - evaluates eval-time fitness on each accepted swap,
  - maintains a running top-N best-fitness list,
  - emits one JSONL line per step matching the transfer-harness
    contract (`{"step": N, "best_fitness": ..., "top5": [...]}`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from attacks.autodan_gradient.gcg_step import GCGStep
from surrogate.fitness.prefix_tokenized import TokenizedPrefix


@dataclass
class OptimizerConfig:
    n_steps: int = 500
    scanner_threshold: int = 5
    top_n_tracked: int = 5
    # Early-abandon (opt-in): at step >= abandon_after_steps, if
    # best_fitness has not climbed past EITHER absolute_floor OR
    # (step_1_fitness × min_improvement_ratio), abort the run. Set
    # `abandon_after_steps` to 0 (default) to disable.
    abandon_after_steps: int = 0
    abandon_absolute_floor: float = 0.005
    abandon_min_improvement_ratio: float = 1.5
    # Plateau-abandon (opt-in): once step > plateau_window, if best_fitness
    # gained less than plateau_min_delta over the last plateau_window
    # steps, abort the run. Catches productive-but-stuck basins that
    # cleared the floor and stalled. Set `plateau_window` to 0 to
    # disable.
    abandon_plateau_window: int = 0
    abandon_plateau_min_delta: float = 0.001
    # Fallback-specific plateau (opt-in): tighter cutoff for fallback
    # runs that are stuck below productivity. Fires when both best_fitness
    # < fallback_plateau_floor AND best_fitness gained less than
    # fallback_plateau_min_delta over the last fallback_plateau_window
    # steps. The orchestrator only passes these flags when the run is in
    # the FALLBACK phase (abandon_enabled=False), so productive fallbacks
    # that climb above the floor are unaffected.
    fallback_plateau_window: int = 0
    fallback_plateau_floor: float = 0.005
    fallback_plateau_min_delta: float = 0.0001


@dataclass
class TopNEntry:
    prompt: str
    fitness: float
    step_added: int


@dataclass
class RunSummary:
    n_steps: int
    n_accepted: int
    n_all_blocked: int
    best_fitness: float
    best_prompt: str
    final_suffix_ids: list[int]
    # `abandoned` means the run failed productively — best_fitness never
    # cleared the basin-viability bar. NOT transfer-eligible.
    abandoned: bool = False
    abandoned_at_step: int | None = None
    abandon_reason: str | None = None
    # `plateau_aborted` means the run was productive (cleared the floor)
    # but stopped climbing, so we cut it short. The top_n is still valid.
    # IS transfer-eligible.
    plateau_aborted: bool = False
    plateau_aborted_at_step: int | None = None
    plateau_reason: str | None = None


class AutoDANOptimizer:
    """500-step outer loop wrapping `GCGStep` with scanner pre-filter
    + JSONL.

    The evaluator must be a callable `user_prompt -> dict` returning at
    minimum a `fitness` field; the standard production binding is
    `SystemPromptLeakFitness.evaluate`. Tests pass a stub.
    """

    def __init__(
        self,
        gcg_step: GCGStep,
        evaluator: Callable[[str], dict],
        scanner: Any,
        tokenizer: Any,
        seed_prefix: str,
        config: OptimizerConfig | None = None,
    ) -> None:
        self._gcg_step = gcg_step
        self._evaluator = evaluator
        self._scanner = scanner
        self._tokenizer = tokenizer
        self._seed_prefix = seed_prefix
        self._config = config or OptimizerConfig()

    def run(
        self,
        prefix: TokenizedPrefix,
        jsonl_path: Path,
    ) -> RunSummary:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        current_suffix_ids = prefix.suffix_init_ids.detach().clone()

        top_n: list[TopNEntry] = []
        n_accepted = 0
        n_all_blocked = 0
        step_1_fitness: float | None = None
        abandoned = False
        abandoned_at_step: int | None = None
        abandon_reason: str | None = None
        plateau_aborted = False
        plateau_aborted_at_step: int | None = None
        plateau_reason: str | None = None
        best_history: list[float] = []

        with jsonl_path.open("w") as fh:
            for step_idx in range(1, self._config.n_steps + 1):
                step_result = self._gcg_step.step(
                    current_suffix_ids, prefix,
                    is_blocked=self._is_blocked,
                )

                if not step_result.candidates:
                    # Pre-filter eliminated every candidate across all
                    # resample attempts. Keep the suffix unchanged and
                    # log the stuck step.
                    n_all_blocked += 1
                    user_prompt = self._decode_user_prompt(current_suffix_ids)
                    eval_diag = self._evaluator(user_prompt)
                    self._maybe_insert_top_n(
                        top_n, user_prompt, float(eval_diag["fitness"]),
                        step_idx,
                    )
                    self._write_jsonl_line(
                        fh=fh,
                        step=step_idx,
                        accepted=False,
                        accepted_loss=None,
                        gradient_loss=step_result.gradient_loss,
                        gradient_diagnostics=step_result.gradient_diagnostics,
                        eval_diagnostics=eval_diag,
                        accepted_prompt=user_prompt,
                        survivor_count=0,
                        top_n=top_n,
                    )
                    best_history.append(top_n[0].fitness if top_n else 0.0)
                    continue

                # GCG step pre-filtered — every entry is scanner-safe.
                # Take the lowest-loss survivor.
                accepted_loss, accepted_suffix = step_result.candidates[0]
                current_suffix_ids = accepted_suffix
                n_accepted += 1

                user_prompt = self._decode_user_prompt(current_suffix_ids)
                eval_diag = self._evaluator(user_prompt)
                self._maybe_insert_top_n(
                    top_n, user_prompt, float(eval_diag["fitness"]),
                    step_idx,
                )
                self._write_jsonl_line(
                    fh=fh,
                    step=step_idx,
                    accepted=True,
                    accepted_loss=accepted_loss,
                    gradient_loss=step_result.gradient_loss,
                    gradient_diagnostics=step_result.gradient_diagnostics,
                    eval_diagnostics=eval_diag,
                    accepted_prompt=user_prompt,
                    survivor_count=len(step_result.candidates),
                    top_n=top_n,
                )

                best_history.append(top_n[0].fitness if top_n else 0.0)

                # Capture step-1 fitness as the baseline for relative
                # improvement detection.
                if step_idx == 1:
                    step_1_fitness = top_n[0].fitness if top_n else 0.0

                # Early-abandon check.
                if (
                    self._config.abandon_after_steps > 0
                    and step_idx >= self._config.abandon_after_steps
                    and not abandoned
                ):
                    best_so_far = top_n[0].fitness if top_n else 0.0
                    floor = self._config.abandon_absolute_floor
                    ratio = self._config.abandon_min_improvement_ratio
                    rel_target = (step_1_fitness or 0.0) * ratio
                    # AND-logic: both signals must say productive.
                    # Above the absolute floor AND climbing meaningfully
                    # over the cold-start fitness. Either failing means
                    # the basin isn't doing what we want.
                    productive = (
                        best_so_far >= floor
                        and best_so_far >= rel_target
                    )
                    if not productive:
                        abandoned = True
                        abandoned_at_step = step_idx
                        abandon_reason = (
                            f"best_fitness={best_so_far:.4f} fails "
                            f"floor={floor:.4f} or "
                            f"{ratio:.1f}×step_1 ({rel_target:.4f}) "
                            f"at step {step_idx}"
                        )
                        # Write a final marker line for downstream tools.
                        fh.write(
                            "{"
                            f'"abandoned": true, '
                            f'"abandoned_at_step": {step_idx}, '
                            f'"abandon_reason": "{abandon_reason}"'
                            "}\n"
                        )
                        fh.flush()
                        break

                # Plateau-abort check. Distinct from `abandoned`: the run
                # IS productive (cleared the floor and climbed); we're
                # just stopping early because best_fitness has stalled.
                # The top_n stays valid for transfer testing.
                window = self._config.abandon_plateau_window
                if (
                    window > 0
                    and step_idx > window
                    and not abandoned
                    and not plateau_aborted
                    and len(best_history) > window
                ):
                    delta = best_history[-1] - best_history[-1 - window]
                    if delta < self._config.abandon_plateau_min_delta:
                        plateau_aborted = True
                        plateau_aborted_at_step = step_idx
                        plateau_reason = (
                            f"plateau: best_fitness gained {delta:.6f} "
                            f"over last {window} steps "
                            f"(< {self._config.abandon_plateau_min_delta:.6f}) "
                            f"at step {step_idx}"
                        )
                        fh.write(
                            "{"
                            f'"plateau_aborted": true, '
                            f'"plateau_aborted_at_step": {step_idx}, '
                            f'"plateau_reason": "{plateau_reason}"'
                            "}\n"
                        )
                        fh.flush()
                        break

                # Fallback-specific plateau: cut runs that are stuck
                # BELOW productivity floor. Only triggers when both
                # below-floor AND flat — productive fallbacks that
                # climb above floor are protected.
                fb_window = self._config.fallback_plateau_window
                if (
                    fb_window > 0
                    and step_idx > fb_window
                    and not abandoned
                    and not plateau_aborted
                    and len(best_history) > fb_window
                ):
                    current_best = best_history[-1]
                    fb_delta = current_best - best_history[-1 - fb_window]
                    if (
                        current_best < self._config.fallback_plateau_floor
                        and fb_delta < self._config.fallback_plateau_min_delta
                    ):
                        plateau_aborted = True
                        plateau_aborted_at_step = step_idx
                        plateau_reason = (
                            f"fallback-plateau: best_fitness "
                            f"{current_best:.6f} below floor "
                            f"{self._config.fallback_plateau_floor:.4f} "
                            f"and gained {fb_delta:.6f} over last "
                            f"{fb_window} steps at step {step_idx}"
                        )
                        fh.write(
                            "{"
                            f'"plateau_aborted": true, '
                            f'"plateau_aborted_at_step": {step_idx}, '
                            f'"plateau_reason": "{plateau_reason}"'
                            "}\n"
                        )
                        fh.flush()
                        break

        best = top_n[0] if top_n else TopNEntry(
            prompt="", fitness=0.0, step_added=0,
        )
        return RunSummary(
            n_steps=(
                abandoned_at_step
                or plateau_aborted_at_step
                or self._config.n_steps
            ),
            n_accepted=n_accepted,
            n_all_blocked=n_all_blocked,
            best_fitness=best.fitness,
            best_prompt=best.prompt,
            final_suffix_ids=current_suffix_ids.tolist(),
            abandoned=abandoned,
            abandoned_at_step=abandoned_at_step,
            abandon_reason=abandon_reason,
            plateau_aborted=plateau_aborted,
            plateau_aborted_at_step=plateau_aborted_at_step,
            plateau_reason=plateau_reason,
        )

    def _is_blocked(self, suffix_ids: torch.Tensor) -> bool:
        """Decode and run the scanner. Wraps the production
        `InjectionScanner.scan` so the GCG step can pre-filter its
        own candidate batch before the forward pass."""
        user_prompt = self._decode_user_prompt(suffix_ids)
        return (
            self._scanner.scan(user_prompt).total_score
            >= self._config.scanner_threshold
        )

    def _decode_user_prompt(self, suffix_ids: torch.Tensor) -> str:
        suffix_text = self._tokenizer.decode(
            suffix_ids, skip_special_tokens=False,
        )
        return self._seed_prefix + suffix_text

    def _maybe_insert_top_n(
        self,
        top_n: list[TopNEntry],
        prompt: str,
        fitness: float,
        step: int,
    ) -> None:
        existing = next(
            (e for e in top_n if e.prompt == prompt), None,
        )
        if existing is not None:
            if fitness > existing.fitness:
                existing.fitness = fitness
                existing.step_added = step
                top_n.sort(key=lambda e: e.fitness, reverse=True)
            return

        if len(top_n) < self._config.top_n_tracked:
            top_n.append(TopNEntry(
                prompt=prompt, fitness=fitness, step_added=step,
            ))
            top_n.sort(key=lambda e: e.fitness, reverse=True)
            return

        if fitness > top_n[-1].fitness:
            top_n[-1] = TopNEntry(
                prompt=prompt, fitness=fitness, step_added=step,
            )
            top_n.sort(key=lambda e: e.fitness, reverse=True)

    def _write_jsonl_line(
        self,
        *,
        fh: Any,
        step: int,
        accepted: bool,
        accepted_loss: float | None,
        gradient_loss: float,
        gradient_diagnostics: dict,
        eval_diagnostics: dict,
        accepted_prompt: str,
        survivor_count: int,
        top_n: list[TopNEntry],
    ) -> None:
        record = {
            "step": step,
            "accepted": accepted,
            "accepted_loss": accepted_loss,
            "gradient_loss": gradient_loss,
            "leak_log_prob": gradient_diagnostics.get("leak_log_prob"),
            "readability_log_prob": gradient_diagnostics.get(
                "readability_log_prob",
            ),
            "lambda_readability": gradient_diagnostics.get(
                "lambda_readability",
            ),
            "best_fitness": top_n[0].fitness if top_n else 0.0,
            "scanner_score": eval_diagnostics.get("scanner_score"),
            "leak_score": eval_diagnostics.get("leak_score"),
            "evasion_score": eval_diagnostics.get("evasion_score"),
            "fitness": eval_diagnostics.get("fitness"),
            "survivor_count": survivor_count,
            "accepted_prompt": accepted_prompt,
            "top5": [
                {
                    "prompt": e.prompt,
                    "fitness": e.fitness,
                    "step_added": e.step_added,
                }
                for e in top_n
            ],
        }
        fh.write(json.dumps(record) + "\n")
        fh.flush()
