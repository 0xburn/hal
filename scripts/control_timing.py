"""Evaluate one fixed O59 policy at an independent frame delay and replan interval."""

import hashlib
import importlib.metadata
import json
import os
import platform
import time
import zipfile
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.inference.action_sequence_artifact import export_action_sequence_policy
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.sim.process_vec import ProcessVecTelemetry
from hal.training.checkpoints import BackgroundUploader

CHECKPOINT_URI = "r2://hal/runs/260921-092548_059_muon_history_decoder_o59v5-cooldown32768/checkpoints/step-0131072.pt"
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
DELAYS = (0, 1, 2, 3, 4, 6)
INTERVALS = (1, 2, 3, 4, 6)
TRAINED_CHUNK_LENGTH = 28
HORIZON = 12
EVALUATOR_GPU = "RTX-PRO-6000"


@dataclass(frozen=True, slots=True)
class Cell:
    delay: int
    replan: int

    def __post_init__(self) -> None:
        if type(self.delay) is not int or type(self.replan) is not int or self.delay < 0 or self.replan < 1:
            raise ValueError("delay must be nonnegative and replan must be positive integers")
        if self.delay + self.replan > HORIZON:
            raise ValueError("executed slice exceeds trained horizon")

    @property
    def name(self) -> str:
        return f"d{self.delay}-r{self.replan}"

    @property
    def timing(self) -> FrameTiming:
        # Blocking local simulation implements all delay in absolute action targets.
        # A future action is never applied before source_frame + delay + 1.
        return FrameTiming(0, 0, self.delay, self.replan, HORIZON)


@dataclass(frozen=True, slots=True)
class Args:
    delay: int
    replan: int
    run_name: str
    boots: int = 96
    frames: int = 7200
    parallel: int = 8
    upload: bool = True


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_overrides() -> dict[str, str]:
    value = json.loads(os.environ.get("HAL_SOURCE_OVERRIDES", "{}"))
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError("source overrides must map file paths to hashes")
    return value


def verify_source(root: Path, manifest_hash: str, overrides: Mapping[str, str]) -> None:
    """Check every base file and each explicitly recorded source extension."""
    manifest = root / "source_manifest.json"
    if sha256(manifest) != manifest_hash:
        raise ValueError("source manifest differs from the launch")
    hashes = json.loads(manifest.read_text())
    for name, expected in (hashes | dict(overrides)).items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError(f"invalid source path: {name}")
        if sha256(root / path) != expected:
            raise ValueError(f"evaluation source changed: {name}")


def grid() -> tuple[Cell, ...]:
    return tuple(Cell(d, r) for d in DELAYS for r in INTERVALS if d + r <= HORIZON)


@dataclass(slots=True)
class PolicyFactory:
    policy: ActionSequencePolicy
    runtime: RuntimeConfig
    cell: Cell
    next_wave: int = 0

    def __call__(self) -> PolicyBatchAdapter:
        # The wave partition is fixed across cells; no partial-boot retries can
        # renumber streams. StreamGroupRng separates slots and match generations.
        self.policy.reset_prediction(seed=self.next_wave)
        self.next_wave += 1
        print(f"[timing] {self.cell.name} wave {self.next_wave}", flush=True)
        return PolicyBatchAdapter(
            self.policy,
            self.runtime,
            self.cell.timing,
            player_identity=None,
            desired_return=self.policy.return_p90,
            temperature=1.0,
            observed_actions=True,
        )


def evaluate(args: Args) -> None:
    cell = Cell(args.delay, args.replan)
    if args.boots < 1 or args.frames < 124 or args.parallel < 1 or args.boots % args.parallel:
        raise ValueError("positive boots must be divisible by parallelism; frames must include active play")
    output = Path("runs") / args.run_name / cell.name
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or "RTX PRO 6000" not in torch.cuda.get_device_name():
        raise RuntimeError(f"this protocol requires a Modal {EVALUATOR_GPU} evaluator")
    job_started = time.perf_counter()
    configure_inference_process()
    checkpoint = resolve_checkpoint(CHECKPOINT_URI)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash differs from the pinned final O59 artifact")
    bundle = Path("/tmp/control-timing.halpolicy")
    manifest = export_action_sequence_policy(checkpoint, bundle)
    if manifest.source_sha256 != CHECKPOINT_SHA256:
        raise ValueError("export changed the checkpoint identity")
    with zipfile.ZipFile(bundle) as archive:
        backend = json.loads(archive.read("backend.json"))
    if backend["wandb_id"] != "vywk3cih" or backend["step"] != 131071:
        raise ValueError("checkpoint is not the final vywk3cih update")
    policy = load_action_sequence_policy(bundle, seed=0, compiled=True, history_mode="kv_cache", kv_update_frames=1)
    if policy.cfg.sample_chunk_length != TRAINED_CHUNK_LENGTH or policy.supported_horizons != tuple(
        range(1, HORIZON + 1)
    ):
        raise ValueError(f"checkpoint trained horizons differ: {policy.supported_horizons}")
    runtime = RuntimeConfig(args.parallel, (0,), replan_interval_frames=cell.replan)
    preparation_started = time.perf_counter()
    policy.prepare_prediction(runtime, HORIZON, cell.delay)
    preparation_seconds = time.perf_counter() - preparation_started
    freeze_inference_runtime()
    schedule = [(int(ego.value), int(cpu.value)) for ego, cpu in matchups_for_vs_cpu(args.boots)]
    protocol = {
        "schema_version": 1,
        "cell": asdict(cell),
        "args": asdict(args),
        "base_git_sha": os.environ["HAL_GIT_SHA"],
        "source_manifest_sha256": os.environ["HAL_SOURCE_MANIFEST_SHA256"],
        "checkpoint_uri": CHECKPOINT_URI,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "bundle_sha256": sha256(bundle),
        "backend": backend,
        "model_config": asdict(policy.cfg),
        "timing": asdict(cell.timing),
        "release_rule": "plan observed at t first applies at t+d+1; next plan takes over at t+d+R+1",
        "forcing": "all d committed future actions; neutral only at episode startup",
        "history_mode": "kv_cache",
        "kv_update_frames": 1,
        "observed_actions": True,
        "player_identity": None,
        "desired_return": policy.return_p90,
        "temperature": 1.0,
        "trained_chunk_length": TRAINED_CHUNK_LENGTH,
        "decoded_horizon": HORIZON,
        "matchup_schedule": schedule,
        "matchup_schedule_sha256": hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest(),
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": "BATTLEFIELD",
        "instant_restart": True,
        "sampling_seed": "zero-based wave index; slot and match-generation keyed StreamGroupRng",
        "emulator_rng": "not injectable in current Dolphin; restart stages and game randomness are not paired",
        "pairing": "identical 96 character-matchup boot blocks, not identical game trajectories",
        "start_retries": 0,
        "rate_denominator": "sum of active frame counts (frame_id >= 0) / 3600",
        "bootstrap": {"unit": "boot", "resamples": 2000, "seed": 0, "interval": [2.5, 97.5], "paired_indices": True},
        "production_cell": {"delay": 3, "replan": 4, "physical_delay": 2, "inference_allowance": 1},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "packages": {name: importlib.metadata.version(name) for name in ("melee", "modal", "numpy", "pyarrow")},
            "uv_lock_sha256": sha256(Path("uv.lock")),
            "iso_sha256": ISO.sha256,
            "dolphin_sha256": DOLPHIN_EXIAI.sha256,
        },
    }
    if overrides := source_overrides():
        protocol["source_overrides"] = overrides
    save_json(output / "protocol.json", protocol)
    telemetry = ProcessVecTelemetry()
    started = time.perf_counter()
    with torch.compiler.set_stance("fail_on_recompile"):
        results, rows = sweep_vs_cpu_prior_with_rows(
            PolicyFactory(policy, runtime, cell),
            session_cfg=default_session_cfg(output / "replays", instant_match_restart=True),
            n_matchups=args.boots,
            max_parallel=args.parallel,
            cpu_level=9,
            max_frames=args.frames,
            start_retries=0,
            process_telemetry=telemetry,
        )
    elapsed = time.perf_counter() - started
    metrics = vs_cpu_metrics(results, seed=0)
    metrics.update(telemetry.metrics())
    totals = [sum(row.total_frames for row in rows if row.boot_index == boot) for boot in range(args.boots)]
    complete = totals == [args.frames] * args.boots
    metrics.update(
        eval_wall_seconds=elapsed,
        emulator_fps=sum(totals) / elapsed,
        preparation_seconds=preparation_seconds,
        job_seconds_before_upload=time.perf_counter() - job_started,
    )
    save_json(
        output / "results.json",
        {"schema_version": 1, "complete": complete, "metrics": metrics, "rows": [row.as_dict() for row in rows]},
    )
    save_json(
        output / "artifact_hashes.json",
        {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
    )
    if args.upload:
        with closing(BackgroundUploader(args.run_name)) as uploader:
            uploader.upload_tree(output, base=output.parent)
    if not complete:
        raise RuntimeError(f"incomplete boot budgets: {totals}")
    print(json.dumps({"cell": cell.name, "metrics": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    evaluate(tyro.cli(Args))
