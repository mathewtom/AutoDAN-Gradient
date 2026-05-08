"""Adaptive campaign orchestrator.

Drives a campaign through three phases:
  1. PLANNED — runs each entry in `planned_runs` with the configured GCG
     flags. Detects abandonment from the per-run JSONL.
  2. REPLACEMENT — for each abandoned planned slot, pulls the next entry
     from `replacement_pool` and re-runs. Capped by `replacement_cap`.
  3. FALLBACK — if no successful runs survived, re-runs the least-bad
     abandoned attempt with the abandonment guard disabled.

Optionally invokes `scripts/transfer_test_top5.py` against every
transfer-eligible JSONL when `--with-transfer` is passed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.autodan_gradient.run_autodan import FITNESS_REGISTRY  # noqa: E402


_REQUIRED_TOP_LEVEL = (
    "campaign_id", "fitness", "target_string", "output_dir",
    "planned_runs", "gcg",
)
_REQUIRED_RUN_FIELDS = ("id", "seed_file", "suffix_init_text")
_REQUIRED_GCG_FIELDS = (
    "steps", "top_k", "batch", "suffix_len",
    "abandon_after_steps",
    "abandon_absolute_floor",
    "abandon_min_improvement_ratio",
)


@dataclass
class RunSpec:
    slot_id: str
    seed_file: Path
    suffix_init_text: str


@dataclass
class CampaignConfig:
    campaign_id: str
    fitness: str
    target_string: str
    output_dir: Path
    planned_runs: list[RunSpec]
    replacement_pool: list[RunSpec]
    gcg: dict[str, Any]
    transfer: dict[str, Any] | None
    replacement_cap: int


@dataclass
class RunOutcome:
    phase: str
    slot_id: str
    pool_id: str
    jsonl_path: Path
    abandoned: bool
    best_fitness: float
    abandoned_at_step: int | None = None


@dataclass
class CampaignState:
    outcomes: list[RunOutcome] = field(default_factory=list)
    replacements_used: int = 0


def load_config(path: Path) -> CampaignConfig:
    if not path.exists():
        raise ValueError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping in {path}")

    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise ValueError(f"config missing required field {key!r}")

    if raw["fitness"] not in FITNESS_REGISTRY:
        raise ValueError(
            f"unknown fitness {raw['fitness']!r}; "
            f"known: {sorted(FITNESS_REGISTRY)}"
        )

    repo_root = REPO_ROOT

    def _parse_runs(items: list[Any], section: str) -> list[RunSpec]:
        if not isinstance(items, list):
            raise ValueError(f"{section} must be a list")
        specs: list[RunSpec] = []
        for i, entry in enumerate(items):
            if not isinstance(entry, dict):
                raise ValueError(f"{section}[{i}] must be a mapping")
            for k in _REQUIRED_RUN_FIELDS:
                if k not in entry:
                    raise ValueError(
                        f"{section}[{i}] missing required field {k!r}"
                    )
            seed = Path(entry["seed_file"])
            seed_abs = seed if seed.is_absolute() else repo_root / seed
            if not seed_abs.exists():
                raise ValueError(
                    f"{section}[{i}] seed_file not found: {seed_abs}"
                )
            specs.append(RunSpec(
                slot_id=str(entry["id"]),
                seed_file=seed_abs,
                suffix_init_text=str(entry["suffix_init_text"]),
            ))
        return specs

    planned = _parse_runs(raw["planned_runs"], "planned_runs")
    replacement_pool = _parse_runs(
        raw.get("replacement_pool") or [], "replacement_pool",
    )

    gcg = raw["gcg"]
    if not isinstance(gcg, dict):
        raise ValueError("gcg block must be a mapping")
    for k in _REQUIRED_GCG_FIELDS:
        if k not in gcg:
            raise ValueError(f"gcg missing required field {k!r}")

    output_dir = Path(raw["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    replacement_cap = int(raw.get("replacement_cap", 3))
    if replacement_cap < 0:
        raise ValueError("replacement_cap must be >= 0")

    return CampaignConfig(
        campaign_id=str(raw["campaign_id"]),
        fitness=str(raw["fitness"]),
        target_string=str(raw["target_string"]),
        output_dir=output_dir,
        planned_runs=planned,
        replacement_pool=replacement_pool,
        gcg=dict(gcg),
        transfer=dict(raw["transfer"]) if raw.get("transfer") else None,
        replacement_cap=replacement_cap,
    )


def jsonl_path_for(
    config: CampaignConfig, slot_id: str, timestamp: str,
) -> Path:
    return (
        config.output_dir
        / f"{config.campaign_id}_{slot_id}_{timestamp}.jsonl"
    )


def parse_jsonl_outcome(jsonl_path: Path) -> tuple[bool, float, int | None]:
    """Return (abandoned, best_fitness, abandoned_at_step) for the run."""
    if not jsonl_path.exists():
        return False, 0.0, None
    abandoned = False
    abandoned_at = None
    best_fitness = 0.0
    for raw in jsonl_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("abandoned") is True:
            abandoned = True
            abandoned_at = rec.get("abandoned_at_step")
            continue
        bf = rec.get("best_fitness")
        if isinstance(bf, (int, float)) and bf > best_fitness:
            best_fitness = float(bf)
    return abandoned, best_fitness, abandoned_at


def build_run_command(
    config: CampaignConfig,
    spec: RunSpec,
    out_path: Path,
    *,
    abandon_enabled: bool,
) -> list[str]:
    g = config.gcg
    cmd = [
        "uv", "run", "python", "-m", "attacks.autodan_gradient.run_autodan",
        "--fitness", config.fitness,
        "--starting-prompt", str(spec.seed_file),
        "--suffix-init-text", spec.suffix_init_text,
        "--steps", str(g["steps"]),
        "--top-k", str(g["top_k"]),
        "--batch", str(g["batch"]),
        "--suffix-len", str(g["suffix_len"]),
        "--out", str(out_path),
    ]
    if abandon_enabled and int(g["abandon_after_steps"]) > 0:
        cmd += [
            "--abandon-after-steps", str(g["abandon_after_steps"]),
            "--abandon-absolute-floor", str(g["abandon_absolute_floor"]),
            "--abandon-min-improvement-ratio",
            str(g["abandon_min_improvement_ratio"]),
        ]
    return cmd


def write_dry_run_jsonl(
    out_path: Path, *, abandoned: bool, best_fitness: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(json.dumps({
            "step": 1, "best_fitness": best_fitness,
            "accepted": True, "top5": [
                {"prompt": "dryrun", "fitness": best_fitness, "step_added": 1},
            ],
        }) + "\n")
        if abandoned:
            fh.write(json.dumps({
                "abandoned": True,
                "abandoned_at_step": 1,
                "abandon_reason": "dry-run synthetic abandonment",
            }) + "\n")


def execute_run(
    config: CampaignConfig,
    spec: RunSpec,
    *,
    phase: str,
    abandon_enabled: bool,
    dry_run: bool,
    timestamp: str,
    pool_id: str | None = None,
) -> RunOutcome:
    out_path = jsonl_path_for(config, spec.slot_id, timestamp)
    label = pool_id or spec.slot_id
    print(
        f"\n===== {phase} run {spec.slot_id} (pool={label}, "
        f"abandon={'on' if abandon_enabled else 'off'}) =====",
        flush=True,
    )
    print(f"seed_file={spec.seed_file}", flush=True)
    print(f"suffix_init_text={spec.suffix_init_text!r}", flush=True)
    print(f"out={out_path}", flush=True)

    if dry_run:
        write_dry_run_jsonl(out_path, abandoned=False, best_fitness=0.5)
        print("(dry-run) wrote synthetic JSONL", flush=True)
    else:
        cmd = build_run_command(
            config, spec, out_path, abandon_enabled=abandon_enabled,
        )
        result = subprocess.run(cmd, check=False, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(
                f"WARNING: run_autodan exited {result.returncode}; "
                f"continuing — will treat as abandoned if no JSONL produced",
                flush=True,
            )

    abandoned, best_fitness, abandoned_at = parse_jsonl_outcome(out_path)
    return RunOutcome(
        phase=phase,
        slot_id=spec.slot_id,
        pool_id=label,
        jsonl_path=out_path,
        abandoned=abandoned,
        best_fitness=best_fitness,
        abandoned_at_step=abandoned_at,
    )


def successful_outcomes(state: CampaignState) -> list[RunOutcome]:
    """A slot is successful if it has at least one non-abandoned run.
    The latest non-abandoned outcome per slot is the transfer-eligible
    one."""
    by_slot: dict[str, RunOutcome] = {}
    for outcome in state.outcomes:
        if outcome.abandoned:
            continue
        by_slot[outcome.slot_id] = outcome
    return list(by_slot.values())


def least_bad_abandoned(state: CampaignState) -> RunOutcome | None:
    abandoned = [o for o in state.outcomes if o.abandoned]
    if not abandoned:
        return None
    return max(abandoned, key=lambda o: o.best_fitness)


def run_planned_phase(
    config: CampaignConfig, state: CampaignState, *,
    dry_run: bool, timestamp: str,
) -> None:
    for spec in config.planned_runs:
        outcome = execute_run(
            config, spec,
            phase="PLANNED",
            abandon_enabled=True,
            dry_run=dry_run, timestamp=timestamp,
        )
        state.outcomes.append(outcome)


def run_replacement_phase(
    config: CampaignConfig, state: CampaignState, *,
    dry_run: bool, timestamp: str,
) -> None:
    if config.replacement_cap <= 0:
        return
    pool_iter = iter(config.replacement_pool)
    for outcome in list(state.outcomes):
        if outcome.phase != "PLANNED" or not outcome.abandoned:
            continue
        if state.replacements_used >= config.replacement_cap:
            print(
                f"\nreplacement cap ({config.replacement_cap}) reached; "
                f"skipping further replacements",
                flush=True,
            )
            return
        try:
            replacement = next(pool_iter)
        except StopIteration:
            print(
                "\nreplacement pool exhausted; skipping further replacements",
                flush=True,
            )
            return
        state.replacements_used += 1
        # Replacement runs reuse the abandoned slot id so the slot has a
        # transfer-eligible JSONL on success. Pool id stays distinct for
        # the summary table.
        spec = RunSpec(
            slot_id=outcome.slot_id,
            seed_file=replacement.seed_file,
            suffix_init_text=replacement.suffix_init_text,
        )
        new_outcome = execute_run(
            config, spec, phase="REPLACEMENT",
            abandon_enabled=True,
            dry_run=dry_run, timestamp=timestamp,
            pool_id=replacement.slot_id,
        )
        state.outcomes.append(new_outcome)


def run_fallback_phase(
    config: CampaignConfig, state: CampaignState, *,
    dry_run: bool, timestamp: str,
) -> None:
    candidate = least_bad_abandoned(state)
    if candidate is None:
        print("\nno abandoned runs to fall back on", flush=True)
        return
    matching_spec = next(
        (s for s in config.planned_runs if s.slot_id == candidate.slot_id),
        None,
    )
    if matching_spec is None:
        # Replacement run was the least bad. Reconstruct from history.
        pool_match = next(
            (
                p for p in config.replacement_pool
                if p.slot_id == candidate.pool_id
            ),
            None,
        )
        if pool_match is None:
            print(
                "\nfallback target unresolvable; skipping",
                flush=True,
            )
            return
        matching_spec = RunSpec(
            slot_id=candidate.slot_id,
            seed_file=pool_match.seed_file,
            suffix_init_text=pool_match.suffix_init_text,
        )
    print(
        f"\n===== FALLBACK on slot {candidate.slot_id} "
        f"(best_fitness={candidate.best_fitness:.4f}) =====",
        flush=True,
    )
    fallback_outcome = execute_run(
        config, matching_spec, phase="FALLBACK",
        abandon_enabled=False,
        dry_run=dry_run, timestamp=timestamp + "_fallback",
        pool_id=candidate.pool_id,
    )
    state.outcomes.append(fallback_outcome)


def run_transfer_phase(
    config: CampaignConfig, state: CampaignState, *, timestamp: str,
) -> None:
    if config.transfer is None:
        print("\nno transfer block in config; skipping transfer phase",
              flush=True)
        return
    for outcome in successful_outcomes(state):
        out_prefix = (
            config.output_dir
            / f"transfer_{config.campaign_id}_{outcome.slot_id}_{timestamp}"
        )
        cmd = [
            "uv", "run", "python", "scripts/transfer_test_top5.py",
            "--campaign-jsonl", str(outcome.jsonl_path),
            "--target-string", config.transfer["target_string"],
            "--audit-dir", str(config.transfer["audit_dir"]),
            "--service-url", config.transfer["service_url"],
            "--out", str(out_prefix),
        ]
        print(f"\n===== TRANSFER {outcome.slot_id} =====", flush=True)
        subprocess.run(cmd, check=False, cwd=str(REPO_ROOT))


def print_summary(state: CampaignState) -> None:
    print("\n=== Campaign summary ===", flush=True)
    print(
        f"{'phase':<12} {'slot':<22} {'pool':<22} "
        f"{'best_fit':>9}  {'status':<10} jsonl",
        flush=True,
    )
    for o in state.outcomes:
        status = "abandoned" if o.abandoned else "ok"
        print(
            f"{o.phase:<12} {o.slot_id:<22} {o.pool_id:<22} "
            f"{o.best_fitness:>9.4f}  {status:<10} {o.jsonl_path}",
            flush=True,
        )
    print(f"\nreplacements used: {state.replacements_used}", flush=True)


def orchestrate(
    config: CampaignConfig, *, dry_run: bool, with_transfer: bool,
) -> CampaignState:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    state = CampaignState()
    run_planned_phase(
        config, state, dry_run=dry_run, timestamp=timestamp,
    )
    run_replacement_phase(
        config, state, dry_run=dry_run, timestamp=timestamp,
    )
    if not successful_outcomes(state):
        run_fallback_phase(
            config, state, dry_run=dry_run, timestamp=timestamp,
        )
    if with_transfer:
        run_transfer_phase(config, state, timestamp=timestamp)
    print_summary(state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip subprocess invocation; write synthetic JSONL only.",
    )
    parser.add_argument(
        "--with-transfer", action="store_true",
        help="After GCG runs, transfer-test successful JSONLs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    state = orchestrate(
        config, dry_run=args.dry_run, with_transfer=args.with_transfer,
    )
    return 0 if successful_outcomes(state) else 1


if __name__ == "__main__":
    raise SystemExit(main())
