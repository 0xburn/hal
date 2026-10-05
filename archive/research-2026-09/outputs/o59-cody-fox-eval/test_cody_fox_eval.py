"""Check the frozen Cody/Fox opponent schedule and evaluation routing."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import melee
import numpy as np
import pytest
import torch

from hal.training.player_identity import encode_player_codes

SOURCE = Path(__file__).with_name("059_muon_history_decoder.py")
SPEC = importlib.util.spec_from_file_location("o59_cody_fox_eval", SOURCE)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)
COMPARE_SOURCE = Path(__file__).with_name("compare.py")
COMPARE_SPEC = importlib.util.spec_from_file_location("o59_cody_fox_compare", COMPARE_SOURCE)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
comparison = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = comparison
COMPARE_SPEC.loader.exec_module(comparison)


def test_fox_schedule_covers_every_selectable_other_cpu_four_times() -> None:
    schedule = experiment.fox_cpu_schedule(96)
    counts = Counter(cpu for ego, cpu in schedule if ego is melee.Character.FOX)
    assert len(schedule) == 96
    assert len(counts) == 24
    assert set(counts.values()) == {4}
    assert not {melee.Character.FOX, melee.Character.SHEIK, melee.Character.NANA} & counts.keys()
    assert experiment.matchup_diversity(96, melee.Character.FOX, fox_all_cpus=True) == (
        24,
        1,
        24,
        "8049dc6414c032049df716f17cc8fbee6ca7658459ef4fb84873d69a940bf0c4",
    )
    with pytest.raises(ValueError, match="96 boots"):
        experiment.fox_cpu_schedule(95)
    with pytest.raises(ValueError, match="fixed ego Fox"):
        experiment.matchup_diversity(96, melee.Character.FALCO, fox_all_cpus=True)


def test_sweep_uses_recorded_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []

    def fake_run(_session_cfg, matches, _factory, **_kwargs):
        observed.extend(matches)
        return [[] for _ in matches]

    monkeypatch.setattr(experiment, "run_matches_vec", fake_run)
    results, rows = experiment.sweep_vs_cpu_fox_all_with_rows(
        lambda: None,
        session_cfg=None,
        n_matchups=96,
        max_parallel=32,
        cpu_level=9,
        ego_port=1,
        seed_stage=melee.Stage.BATTLEFIELD,
        max_frames=7200,
        start_retries=2,
        process_telemetry=None,
    )
    assert len(observed) == len(results) == 96
    assert rows == []
    assert all(result[2] is None for result in results)
    assert [(match.matchup.players[0].character, match.matchup.players[1].character) for match in observed] == list(
        experiment.fox_cpu_schedule(96)
    )
    assert all(match.matchup.players[1].cpu_level == 9 for match in observed)


def test_stratified_bootstrap_keeps_four_draws_per_opponent() -> None:
    groups = np.arange(96).reshape(24, 4)
    indices = comparison.stratified_bootstrap_indices(groups, np.random.default_rng(0), 10)
    assert indices.shape == (10, 96)
    for sample in indices.reshape(10, 24, 4):
        for opponent, boot_ids in enumerate(sample):
            assert set(boot_ids) <= set(groups[opponent])


@pytest.mark.parametrize("fixed_return", [None, 120.0])
def test_eval_routes_cody_fox_and_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixed_return: float | None
) -> None:
    encoded = torch.frombuffer(bytearray(encode_player_codes(("IBDW#0",))), dtype=torch.uint8)
    calibration = SimpleNamespace(state_dict=lambda: {"sha256": "b" * 64}, targets=lambda: (0.0, 10.0, 20.0))
    model = SimpleNamespace(return_calibration=calibration, get_buffer=lambda _name: encoded)
    config = SimpleNamespace(
        prediction_frames=4,
        delay_frames=2,
        replan_interval_frames=2,
        final_eval_n_matchups=96,
    )
    observed = {}

    def fake_eval_vs_cpu(*_args, **kwargs):
        observed.update(kwargs)
        return {"scheduled_boots": 96.0, "completed_boots": 96.0, "boots": 96.0}

    monkeypatch.setattr(experiment, "checkpoint_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(experiment, "load_checkpoint", lambda _path: (model, config, {}, {"step": 131071}))
    monkeypatch.setattr(experiment, "validate_config", lambda _config: None)
    monkeypatch.setattr(experiment, "eval_vs_cpu", fake_eval_vs_cpu)
    experiment.eval_checkpoint(
        str(tmp_path / "step-0131072.pt"),
        n_matchups=96,
        output_name=f"cody-fox-{fixed_return}",
        player_code="IBDW#0",
        fixed_ego_character=melee.Character.FOX,
        fox_all_cpus=True,
        fixed_return=fixed_return,
        expected_checkpoint_sha256="a" * 64,
    )
    assert observed["ego_player_code"] == "IBDW#0"
    assert observed["fixed_ego_character"] is melee.Character.FOX
    assert observed["fox_all_cpus"] is True
    assert observed["desired_return"] == (20.0 if fixed_return is None else 120.0)
    assert observed["return_target"] == ("p90" if fixed_return is None else "fixed")
