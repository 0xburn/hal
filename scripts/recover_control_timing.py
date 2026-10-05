"""Recover complete timing cells by rerunning failed waves with their original RNG slots."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import melee
import torch
import tyro

from hal import r2
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import match_rows
from hal.eval.harness import SessionConfig
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.matchups import matchups_for_vs_cpu
from hal.inference.action_sequence_artifact import export_action_sequence_policy
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.sim.rollout import ChunkPolicy
from hal.sim.rollout import VecMatch
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.training.checkpoints import BackgroundUploader
from scripts.analyze_control_timing import boot_totals
from scripts.control_timing import CHECKPOINT_SHA256
from scripts.control_timing import CHECKPOINT_URI
from scripts.control_timing import HORIZON
from scripts.control_timing import Cell
from scripts.control_timing import PolicyFactory
from scripts.control_timing import grid
from scripts.control_timing import save_json
from scripts.control_timing import sha256
from scripts.control_timing import source_overrides
from scripts.control_timing import verify_source

WAVE_SIZE = 8
FRAMES = 7200


class WaveDriver(Protocol):
    def __call__(
        self,
        session_cfg: SessionConfig,
        matches: Sequence[VecMatch],
        policy_factory: Callable[[], ChunkPolicy],
        *,
        max_frames: int,
        max_parallel: int,
        start_retries: int,
    ) -> list[list[Trajectory]]: ...


@dataclass(frozen=True, slots=True)
class Args:
    delay: int
    replan: int
    source_run: str
    run_name: str


def missing_waves(rows: Sequence[MatchRow], schedule: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Accept only empty failed boots or fully validated completed boots."""
    if not schedule or len(schedule) % WAVE_SIZE:
        raise ValueError("schedule must contain full eight-boot waves")
    if any(not 0 <= row.boot_index < len(schedule) for row in rows):
        raise ValueError("unexpected boot index")
    missing = set()
    for boot, pair in enumerate(schedule):
        local = [replace(row, boot_index=0) for row in rows if row.boot_index == boot]
        if not local:
            missing.add(boot // WAVE_SIZE)
            continue
        boot_totals(local, (pair,), FRAMES)
    return tuple(sorted(missing))


def merge_wave(
    rows: Sequence[MatchRow], replacement: Sequence[MatchRow], wave: int, schedule: Sequence[tuple[int, int]]
) -> list[MatchRow]:
    start = wave * WAVE_SIZE
    if wave < 0 or start + WAVE_SIZE > len(schedule):
        raise ValueError("wave is outside the schedule")
    boot_totals(replacement, schedule[start : start + WAVE_SIZE], FRAMES)
    # Replace the entire wave, including its previously healthy boots. This keeps
    # the accepted wave's batch composition and slot assignment intact.
    merged = [row for row in rows if not start <= row.boot_index < start + WAVE_SIZE]
    merged.extend(replace(row, boot_index=row.boot_index + start) for row in replacement)
    return sorted(merged, key=lambda row: (row.boot_index, row.match_ordinal))


def wave_matches(wave: int) -> list[VecMatch]:
    schedule = matchups_for_vs_cpu(96)
    if not 0 <= wave < len(schedule) // WAVE_SIZE:
        raise ValueError("wave is outside the 96-boot schedule")
    return [
        VecMatch(
            matchup=Matchup(
                stage=melee.Stage.BATTLEFIELD,
                players=(PlayerSetup(1, ego, cpu_level=0), PlayerSetup(2, cpu, cpu_level=9)),
            ),
            model_ports=(1,),
        )
        for ego, cpu in schedule[wave * WAVE_SIZE : (wave + 1) * WAVE_SIZE]
    ]


def evaluate_wave(
    policy: ActionSequencePolicy,
    runtime: RuntimeConfig,
    cell: Cell,
    wave: int,
    output: Path,
    *,
    driver: WaveDriver = run_matches_vec,
) -> list[MatchRow]:
    matches = wave_matches(wave)
    # A fresh factory starts at the original wave seed on every attempt. The
    # current harness assigns local slots 0..7 again for this complete wave.
    factory = PolicyFactory(policy, runtime, cell, next_wave=wave)
    boots = driver(
        default_session_cfg(output / "replays", instant_match_restart=True),
        matches,
        factory,
        max_frames=FRAMES,
        max_parallel=WAVE_SIZE,
        start_retries=0,
    )
    return match_rows(boots, matches)


def recover(args: Args) -> None:
    cell = Cell(args.delay, args.replan)
    if cell not in grid() or args.run_name == args.source_run:
        raise ValueError("recovery requires a grid cell and a new output run")
    output = Path("runs") / args.run_name / cell.name
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    with closing(r2.client()) as client:
        for name in ("protocol.json", "results.json", "artifact_hashes.json"):
            client.download_file("hal", f"runs/{args.source_run}/{cell.name}/{name}", str(inputs / name))
    hashes = json.loads((inputs / "artifact_hashes.json").read_text())
    for name in ("protocol.json", "results.json"):
        if sha256(inputs / name) != hashes[name]:
            raise ValueError(f"source artifact hash mismatch: {name}")
    protocol = json.loads((inputs / "protocol.json").read_text())
    original = json.loads((inputs / "results.json").read_text())
    if (
        protocol["schema_version"] != 1
        or original["schema_version"] != 1
        or "recovery" in protocol
        or original["complete"]
        or protocol["cell"] != asdict(cell)
        or protocol["checkpoint_sha256"] != CHECKPOINT_SHA256
        or protocol["source_manifest_sha256"] != os.environ["HAL_SOURCE_MANIFEST_SHA256"]
        or protocol["timing"] != asdict(cell.timing)
        or (protocol["args"]["boots"], protocol["args"]["frames"], protocol["args"]["parallel"])
        != (96, FRAMES, WAVE_SIZE)
    ):
        raise ValueError("source is not a rejected cell of the frozen full protocol")
    fixed = {
        "history_mode": "kv_cache",
        "kv_update_frames": 1,
        "observed_actions": True,
        "player_identity": None,
        "temperature": 1.0,
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": "BATTLEFIELD",
        "instant_restart": True,
        "decoded_horizon": HORIZON,
    }
    if any(protocol[key] != value for key, value in fixed.items()) or (
        protocol["args"]["delay"],
        protocol["args"]["replan"],
        protocol["args"]["run_name"],
    ) != (args.delay, args.replan, args.source_run):
        raise ValueError("source evaluation settings differ from the frozen protocol")
    schedule = tuple((int(ego.value), int(cpu.value)) for ego, cpu in matchups_for_vs_cpu(96))
    if protocol["matchup_schedule"] != [list(pair) for pair in schedule]:
        raise ValueError("source matchup schedule differs")
    rows = [MatchRow.from_dict(row) for row in original["rows"]]
    waves = missing_waves(rows, schedule)
    if not waves:
        raise ValueError("source has no missing wave")
    if not torch.cuda.is_available() or "RTX PRO 6000" not in torch.cuda.get_device_name():
        raise RuntimeError("recovery requires the same RTX PRO 6000 evaluator")
    environment = protocol["environment"]
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "packages": {name: importlib.metadata.version(name) for name in ("melee", "modal", "numpy", "pyarrow")},
        "uv_lock_sha256": sha256(Path("uv.lock")),
    }
    if any(environment[key] != value for key, value in actual.items()):
        raise ValueError("recovery environment differs from the source cell")
    configure_inference_process()
    checkpoint = resolve_checkpoint(CHECKPOINT_URI)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("recovery checkpoint differs")
    bundle = Path("/tmp/control-timing.halpolicy")
    manifest = export_action_sequence_policy(checkpoint, bundle)
    if manifest.source_sha256 != CHECKPOINT_SHA256:
        raise ValueError("exported checkpoint identity differs")
    with zipfile.ZipFile(bundle) as archive:
        if json.loads(archive.read("backend.json")) != protocol["backend"]:
            raise ValueError("recovery backend differs")
    policy = load_action_sequence_policy(bundle, seed=0, compiled=True, history_mode="kv_cache", kv_update_frames=1)
    if (
        json.loads(json.dumps(asdict(policy.cfg))) != protocol["model_config"]
        or policy.return_p90 != protocol["desired_return"]
    ):
        raise ValueError("recovery policy configuration differs")
    runtime = RuntimeConfig(WAVE_SIZE, (0,), replan_interval_frames=cell.replan)
    started = time.perf_counter()
    policy.prepare_prediction(runtime, HORIZON, cell.delay)
    preparation = time.perf_counter() - started
    freeze_inference_runtime()
    attempts: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.compiler.set_stance("fail_on_recompile"):
        for wave in waves:
            for attempt in range(1, 6):
                attempt_dir = output / f"wave_{wave:02d}" / f"attempt_{attempt:02d}"
                replacement = evaluate_wave(policy, runtime, cell, wave, attempt_dir)
                local_schedule = schedule[wave * WAVE_SIZE : (wave + 1) * WAVE_SIZE]
                complete = not missing_waves(replacement, local_schedule)
                record: dict[str, object] = {"wave": wave, "attempt": attempt, "seed": wave, "complete": complete}
                attempts.append(record)
                save_json(attempt_dir / "rows.json", {**record, "rows": [row.as_dict() for row in replacement]})
                print(json.dumps({"cell": cell.name, **record}), flush=True)
                if complete:
                    rows = merge_wave(rows, replacement, wave, schedule)
                    break
            else:
                break
    elapsed = time.perf_counter() - started
    complete = not missing_waves(rows, schedule)
    protocol["args"] = {**protocol["args"], "run_name": args.run_name}
    protocol["recovery"] = {
        "source_run": args.source_run,
        "source_protocol_sha256": hashes["protocol.json"],
        "source_results_sha256": hashes["results.json"],
        "source_extension": {
            "scripts/recover_control_timing.py": os.environ["HAL_RECOVERY_SOURCE_SHA256"],
            **source_overrides(),
        },
        "replaced_waves": list(waves),
        "attempts": attempts,
        "rule": "replace all eight boots of a failed wave using the original wave seed and local slots 0..7",
        "replays": "unreplaced waves remain at source_run; replacements are under this run's wave/attempt directories",
    }
    metrics = {
        "preparation_seconds": original["metrics"]["preparation_seconds"] + preparation,
        "eval_wall_seconds": original["metrics"]["eval_wall_seconds"] + elapsed,
        "recovery_preparation_seconds": preparation,
        "recovery_eval_wall_seconds": elapsed,
    }
    metrics["emulator_fps"] = sum(row.total_frames for row in rows) / metrics["eval_wall_seconds"]
    save_json(output / "protocol.json", protocol)
    save_json(
        output / "results.json",
        {"schema_version": 1, "complete": complete, "metrics": metrics, "rows": [row.as_dict() for row in rows]},
    )
    save_json(
        output / "artifact_hashes.json",
        {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
    )
    with closing(BackgroundUploader(args.run_name)) as uploader:
        uploader.upload_tree(output, base=output.parent)
    if not complete:
        raise RuntimeError("wave recovery exhausted its attempts; cell remains rejected")


def remote_recover(
    args: Args,
    git_sha: str,
    manifest_hash: str,
    recovery_hash: str,
    source_wall_seconds: float,
    source_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    root = Path("/opt/hal")
    started = time.perf_counter()
    if (
        sha256(root / "source_manifest.json") != manifest_hash
        or sha256(root / "scripts/recover_control_timing.py") != recovery_hash
    ):
        raise ValueError("recovery source identity differs from launch")
    overrides = {} if source_overrides is None else source_overrides
    verify_source(root, manifest_hash, overrides)
    if shutil.disk_usage("/dev/shm").free < 1024**3:
        raise RuntimeError("insufficient shared memory")
    env = os.environ.copy()
    env.update(
        HAL_GIT_SHA=git_sha,
        HAL_SOURCE_MANIFEST_SHA256=manifest_hash,
        HAL_RECOVERY_SOURCE_SHA256=recovery_hash,
        HAL_SOURCE_OVERRIDES=json.dumps(overrides, sort_keys=True),
        DISPLAY=":99",
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    for key, directory in (
        ("TMPDIR", "/tmp/control-timing"),
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/control-timing-inductor"),
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
        env[key] = directory
    subprocess.run(
        [
            "/opt/venv/bin/python",
            "-m",
            "scripts.recover_control_timing",
            "--delay",
            str(args.delay),
            "--replan",
            str(args.replan),
            "--source-run",
            args.source_run,
            "--run-name",
            args.run_name,
        ],
        cwd=root,
        env=env,
        check=True,
    )
    output = root / "runs" / args.run_name / Cell(args.delay, args.replan).name
    recovery_seconds = time.perf_counter() - started
    result = {name: (output / name).read_text() for name in ("protocol.json", "results.json", "artifact_hashes.json")}
    result["modal_runtime.json"] = json.dumps(
        {
            "schema_version": 2,
            "wall_seconds": source_wall_seconds + recovery_seconds,
            "source_wall_seconds": source_wall_seconds,
            "recovery_wall_seconds": recovery_seconds,
            "gpu_job_count": 2,
            "wall_seconds_kind": "sum of GPU function runtimes contributing to this complete cell",
            "python": sys.version,
        }
    )
    return result


if __name__ == "__main__":
    recover(tyro.cli(Args))
