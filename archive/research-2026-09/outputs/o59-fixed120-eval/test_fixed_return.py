"""Focused checks for the isolated fixed-return evaluation copy."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).with_name("059_muon_history_decoder.py")
SPEC = importlib.util.spec_from_file_location("o59_fixed120_eval", SOURCE)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_fixed_return_is_recorded_in_protocol(tmp_path: Path) -> None:
    config = SimpleNamespace(
        prediction_frames=4,
        delay_frames=2,
        replan_interval_frames=2,
        eval_max_parallel=1,
        compiled_inference_bucket=None,
        eval_max_frames=7200,
        eval_seed=0,
        inference_mode="compiled",
    )
    model = torch.nn.Linear(1, 1)
    fixed = experiment._eval_protocol(
        config,
        model,
        n_matchups=1,
        checkpoint_sha256="a" * 64,
        desired_return=120.0,
        return_target="fixed",
    )
    assert fixed.desired_return == 120.0
    assert fixed.return_target == "fixed"
    experiment._write_eval_evidence(tmp_path, [], {}, fixed)
    with pytest.raises(ValueError, match="different protocol"):
        experiment._validate_eval_directory(
            tmp_path,
            experiment._eval_protocol(
                config,
                model,
                n_matchups=1,
                checkpoint_sha256="a" * 64,
                desired_return=120.0,
                return_target="p90",
            ),
        )


def test_fixed_return_routes_to_cpu_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calibration = SimpleNamespace(
        state_dict=lambda: {"sha256": "b" * 64},
        targets=lambda: (_ for _ in ()).throw(AssertionError("fixed eval must not use p90")),
    )
    model = SimpleNamespace(return_calibration=calibration)
    config = SimpleNamespace(
        prediction_frames=4,
        delay_frames=2,
        replan_interval_frames=2,
        final_eval_n_matchups=96,
    )
    observed: dict[str, object] = {}

    def fake_eval_vs_cpu(*_args: object, **kwargs: object) -> dict[str, float]:
        observed.update(kwargs)
        return {"scheduled_boots": 96.0, "completed_boots": 96.0, "boots": 96.0}

    monkeypatch.setattr(experiment, "checkpoint_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(experiment, "load_checkpoint", lambda _path: (model, config, {}, {"step": 131071}))
    monkeypatch.setattr(experiment, "validate_config", lambda _config: None)
    monkeypatch.setattr(experiment, "eval_vs_cpu", fake_eval_vs_cpu)

    experiment.eval_checkpoint(
        str(tmp_path / "step-0131072.pt"),
        n_matchups=96,
        output_name="eval96-step-0131072-fixed120",
        fixed_return=120.0,
        expected_checkpoint_sha256="a" * 64,
    )
    assert observed["desired_return"] == 120.0
    assert observed["return_target"] == "fixed"
    assert observed["n_matchups"] == 96
    assert observed["replay_dir"] == tmp_path / "eval96-step-0131072-fixed120"

    with pytest.raises(ValueError, match="separate output directory"):
        experiment.eval_checkpoint(str(tmp_path / "step-0131072.pt"), fixed_return=120.0)
    with pytest.raises(ValueError, match="separate W&B namespace"):
        experiment.eval_checkpoint(
            str(tmp_path / "step-0131072.pt"),
            fixed_return=120.0,
            output_name="fixed120",
            shared_wandb=True,
        )
