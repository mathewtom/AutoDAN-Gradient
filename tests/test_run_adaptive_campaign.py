"""Unit tests for the adaptive campaign orchestrator. The
`run_autodan` subprocess is replaced by a fake that writes synthetic
JSONL files matching the production contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_adaptive_campaign as orch  # noqa: E402


# ---------------------------------------------------------------- helpers

def _seed_text(*, name: str) -> str:
    return f"# {name}\n\nseed line for {name}\n"


def _write_seed(repo_root: Path, name: str) -> Path:
    seeds_dir = repo_root / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    p = seeds_dir / f"{name}.txt"
    p.write_text(_seed_text(name=name))
    return p


def _make_config_dict(
    repo_root: Path,
    *,
    planned_ids: list[str],
    pool_ids: list[str],
    replacement_cap: int = 3,
) -> dict[str, Any]:
    for name in planned_ids + pool_ids:
        _write_seed(repo_root, name)
    return {
        "campaign_id": "TEST",
        "fitness": "system_prompt_leak_meta_instructions",
        "target_string": "my instructions",
        "output_dir": str(repo_root / "results" / "test_orch"),
        "replacement_cap": replacement_cap,
        "planned_runs": [
            {
                "id": pid,
                "seed_file": f"seeds/{pid}.txt",
                "suffix_init_text": f" suffix-{pid}",
            }
            for pid in planned_ids
        ],
        "replacement_pool": [
            {
                "id": pid,
                "seed_file": f"seeds/{pid}.txt",
                "suffix_init_text": f" suffix-{pid}",
            }
            for pid in pool_ids
        ],
        "gcg": {
            "steps": 5,
            "top_k": 8,
            "batch": 4,
            "suffix_len": 4,
            "abandon_after_steps": 3,
            "abandon_absolute_floor": 0.005,
            "abandon_min_improvement_ratio": 1.5,
        },
    }


def _write_yaml(tmp_path: Path, data: dict[str, Any]) -> Path:
    p = tmp_path / "campaign.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def _stage_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Re-root the orchestrator at a fresh tmp dir so tests don't pollute
    the real repo's seeds/ or results/."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(orch, "REPO_ROOT", repo_root)
    return repo_root


def _install_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    abandon_for: dict[str, list[bool]],
    fitness_for: dict[str, list[float]],
):
    """Replace `subprocess.run` inside the orchestrator with a fake that
    writes a synthetic JSONL. Per-slot lists are consumed in order so a
    single slot can succeed on its replacement."""
    counters: dict[str, int] = {}

    def _fake_run(cmd, check=False, cwd=None):  # noqa: ARG001
        out_path = Path(cmd[cmd.index("--out") + 1])
        # Recover the slot id from the output filename:
        # "<campaign>_<slot>_<timestamp>.jsonl"
        stem = out_path.stem
        # Try matching against known keys (longest first to handle
        # replacement runs reusing a slot id).
        slot_match = None
        for key in sorted(abandon_for, key=len, reverse=True):
            if f"_{key}_" in f"_{stem}_":
                slot_match = key
                break
        assert slot_match is not None, (
            f"could not match slot id in {stem}; "
            f"keys={list(abandon_for)}"
        )
        idx = counters.get(slot_match, 0)
        counters[slot_match] = idx + 1
        abandoned = abandon_for[slot_match][idx]
        fitness = fitness_for[slot_match][idx]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            fh.write(json.dumps({
                "step": 1, "best_fitness": fitness, "accepted": True,
                "top5": [{
                    "prompt": f"p-{slot_match}-{idx}",
                    "fitness": fitness,
                    "step_added": 1,
                }],
            }) + "\n")
            if abandoned:
                fh.write(json.dumps({
                    "abandoned": True,
                    "abandoned_at_step": 3,
                    "abandon_reason": "fake",
                }) + "\n")

        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr(orch.subprocess, "run", _fake_run)


# --------------------------------------------------------- state machine

def test_all_planned_succeed_no_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2"],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={"p1": [False], "p2": [False], "p3": [False]},
        fitness_for={"p1": [0.5], "p2": [0.4], "p3": [0.6]},
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)
    successes = orch.successful_outcomes(state)

    assert len(successes) == 3
    assert state.replacements_used == 0
    assert all(o.phase == "PLANNED" for o in state.outcomes)


def test_one_planned_abandons_one_replacement_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2"],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [False], "p2": [True, False], "p3": [False],
            "r1": [False],
        },
        fitness_for={
            "p1": [0.5], "p2": [0.001, 0.6], "p3": [0.4],
            "r1": [0.6],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)
    successes = orch.successful_outcomes(state)

    assert len(successes) == 3
    assert state.replacements_used == 1


def test_two_planned_abandon_two_replacements_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2", "r3"],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, False], "p2": [False], "p3": [True, False],
            "r1": [False], "r2": [False],
        },
        fitness_for={
            "p1": [0.001, 0.6], "p2": [0.4], "p3": [0.001, 0.7],
            "r1": [0.6], "r2": [0.7],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)
    successes = orch.successful_outcomes(state)

    assert len(successes) == 3
    assert state.replacements_used == 2


def test_all_abandon_fallback_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2", "r3"],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    # Slot p2 has the highest abandoned fitness (0.004) — fallback target.
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, True], "p2": [True, True, False],
            "p3": [True, True],
            "r1": [True], "r2": [True], "r3": [True],
        },
        fitness_for={
            "p1": [0.001, 0.001], "p2": [0.004, 0.001, 0.5],
            "p3": [0.002, 0.001],
            "r1": [0.001], "r2": [0.001], "r3": [0.001],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)
    successes = orch.successful_outcomes(state)

    assert len(successes) == 1
    assert state.replacements_used == 3
    fallback = [o for o in state.outcomes if o.phase == "FALLBACK"]
    assert len(fallback) == 1
    assert fallback[0].slot_id == "p2"
    assert fallback[0].abandoned is False


def test_all_abandon_fallback_also_abandons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2", "r3"],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, True], "p2": [True, True, True],
            "p3": [True, True],
            "r1": [True], "r2": [True], "r3": [True],
        },
        fitness_for={
            "p1": [0.001, 0.001], "p2": [0.004, 0.001, 0.001],
            "p3": [0.002, 0.001],
            "r1": [0.001], "r2": [0.001], "r3": [0.001],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)
    successes = orch.successful_outcomes(state)

    assert len(successes) == 0


def test_replacement_pool_exhausted_before_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2"],
        replacement_cap=3,
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, True, False],
            "p2": [True, True],
            "p3": [True],
            "r1": [True], "r2": [True],
        },
        fitness_for={
            "p1": [0.003, 0.001, 0.5],
            "p2": [0.001, 0.001],
            "p3": [0.002],
            "r1": [0.001], "r2": [0.001],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)

    # Pool only has 2 entries; cap is 3 but pool runs out first.
    assert state.replacements_used == 2
    fallback = [o for o in state.outcomes if o.phase == "FALLBACK"]
    assert len(fallback) == 1
    # Fallback targets least-bad abandoned: p1 with 0.003.
    assert fallback[0].slot_id == "p1"


def test_empty_replacement_pool_falls_back_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=[],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, False], "p2": [False], "p3": [False],
        },
        fitness_for={
            "p1": [0.001, 0.5], "p2": [0.4], "p3": [0.4],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)

    # p1 abandoned but no replacements possible. p2/p3 succeeded. So
    # there are 2 transfer-eligible runs and fallback should NOT fire.
    successes = orch.successful_outcomes(state)
    assert len(successes) == 2
    assert state.replacements_used == 0
    assert not any(o.phase == "FALLBACK" for o in state.outcomes)


def test_empty_pool_with_full_failure_triggers_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=[],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, False], "p2": [True], "p3": [True],
        },
        fitness_for={
            "p1": [0.003, 0.5], "p2": [0.001], "p3": [0.002],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)

    fallback = [o for o in state.outcomes if o.phase == "FALLBACK"]
    assert len(fallback) == 1
    assert fallback[0].slot_id == "p1"
    assert state.replacements_used == 0


def test_cap_zero_disables_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root,
        planned_ids=["p1", "p2", "p3"],
        pool_ids=["r1", "r2", "r3"],
        replacement_cap=0,
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={
            "p1": [True, False], "p2": [True], "p3": [True],
        },
        fitness_for={
            "p1": [0.003, 0.5], "p2": [0.001], "p3": [0.002],
        },
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)

    assert state.replacements_used == 0
    fallback = [o for o in state.outcomes if o.phase == "FALLBACK"]
    assert len(fallback) == 1
    assert fallback[0].slot_id == "p1"


# -------------------------------------------------------- config validation


def test_load_config_missing_required_field_raises(tmp_path: Path) -> None:
    cfg = {"campaign_id": "X", "fitness": "system_prompt_leak_meta_instructions"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="missing required field"):
        orch.load_config(path)


def test_load_config_missing_seed_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg = _make_config_dict(
        repo_root, planned_ids=["p1"], pool_ids=[],
    )
    cfg["planned_runs"][0]["seed_file"] = "seeds/does_not_exist.txt"
    path = _write_yaml(tmp_path, cfg)
    with pytest.raises(ValueError, match="seed_file not found"):
        orch.load_config(path)


def test_load_config_unknown_fitness_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg = _make_config_dict(
        repo_root, planned_ids=["p1"], pool_ids=[],
    )
    cfg["fitness"] = "no_such_fitness"
    path = _write_yaml(tmp_path, cfg)
    with pytest.raises(ValueError, match="unknown fitness"):
        orch.load_config(path)


def test_load_config_default_replacement_cap_is_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg = _make_config_dict(
        repo_root, planned_ids=["p1"], pool_ids=[],
    )
    cfg.pop("replacement_cap")
    path = _write_yaml(tmp_path, cfg)
    config = orch.load_config(path)
    assert config.replacement_cap == 6


def test_planned_replacement_fallback_jsonls_persist_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each phase writes its own JSONL — replacement does not truncate
    the planned run's data, fallback does not truncate either."""
    repo_root = _stage_repo(tmp_path, monkeypatch)
    cfg_dict = _make_config_dict(
        repo_root, planned_ids=["p1"], pool_ids=["r1"],
    )
    cfg_path = _write_yaml(tmp_path, cfg_dict)
    _install_fake_runner(
        monkeypatch,
        abandon_for={"p1": [True, True, False], "r1": [True]},
        fitness_for={"p1": [0.003, 0.001, 0.5], "r1": [0.001]},
    )
    config = orch.load_config(cfg_path)
    state = orch.orchestrate(config, dry_run=False, with_transfer=False)

    files = sorted(p.name for p in config.output_dir.glob("*.jsonl"))
    # Planned p1, replacement r1 (slot p1), fallback on the least-bad
    # abandoned slot (p1) — three distinct files.
    assert len(files) == 3, f"expected 3 files, got: {files}"
    assert any("planned" in f and "_p1_" in f for f in files)
    assert any("replacement_r1" in f for f in files)
    assert any("fallback" in f for f in files)
    # All three outcomes recorded.
    phases = sorted(o.phase for o in state.outcomes)
    assert phases == ["FALLBACK", "PLANNED", "REPLACEMENT"]
