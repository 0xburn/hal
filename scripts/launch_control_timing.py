"""Launch or collect isolated control-timing evaluations."""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import modal
import tyro

from hal.fixtures import ALL
from scripts.control_timing import EVALUATOR_GPU
from scripts.control_timing import Cell
from scripts.control_timing import grid
from scripts.control_timing import save_json
from scripts.control_timing import sha256
from scripts.control_timing import verify_source
from scripts.launch_modal import IMAGE
from scripts.launch_modal import PYPI_INDEX
from scripts.launch_modal import preflight_modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/opt/hal")
PILOT_CELLS = (Cell(0, 1), Cell(3, 4), Cell(6, 6))


@dataclass(frozen=True, slots=True)
class Args:
    action: Literal["pilot", "full", "collect"]
    output: Path = Path("runs/o59-control-timing")
    parallel: int = 8
    containers: int = 6
    phase: Literal["pilot", "full"] = "pilot"


def snapshot(output: Path) -> tuple[Path, str, str]:
    source = output / "source"
    if source.exists():
        raise FileExistsError(f"source snapshot already exists: {source}")
    source.mkdir(parents=True)
    for directory in ("hal", "scripts", "tests", "experiments"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".json", ".md", ".tsv"):
                target = source / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copyfile(ROOT / name, source / name)
    hashes = {str(path.relative_to(source)): sha256(path) for path in sorted(source.rglob("*")) if path.is_file()}
    save_json(output / "source_manifest.json", hashes)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_json(
        output / "source_identity.json",
        {"base_git_sha": git_sha, "source_manifest_sha256": sha256(output / "source_manifest.json")},
    )
    return source, git_sha, sha256(output / "source_manifest.json")


def evaluation_image(source: Path, secret: modal.Secret) -> modal.Image:
    fixtures = json.dumps(
        [
            {
                "name": f.name,
                "sha256": f.sha256,
                "size_bytes": f.size_bytes,
                "dest": str(f.dest),
                "r2_key": f.r2_key,
                "url": f.url,
                "extract": f.extract,
            }
            for f in ALL
        ],
        sort_keys=True,
    )
    return (
        modal.Image.from_registry(IMAGE)
        .add_local_file(source / "pyproject.toml", str(REMOTE / "pyproject.toml"), copy=True)
        .add_local_file(source / "uv.lock", str(REMOTE / "uv.lock"), copy=True)
        .add_local_file(
            ROOT / "vendor/melee-0.47.0+hal.realtime.1.tar.gz",
            str(REMOTE / "vendor/melee-0.47.0+hal.realtime.1.tar.gz"),
            copy=True,
        )
        .workdir(str(REMOTE))
        .run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --no-install-project")
        .add_local_file(source / "scripts/cache_modal_fixtures.py", "/opt/cache_modal_fixtures.py", copy=True)
        .run_commands(
            "/opt/venv/bin/python /opt/cache_modal_fixtures.py",
            env={"HAL_FIXTURE_MANIFEST": fixtures, "HAL_FIXTURE_ROOT": str(REMOTE)},
            secrets=[secret],
        )
        .add_local_dir(source, str(REMOTE), copy=True)
        .add_local_file(source.parent / "source_manifest.json", str(REMOTE / "source_manifest.json"), copy=True)
        .run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --offline --no-build-isolation")
    )


def remote_evaluate(
    delay: int,
    replan: int,
    run_name: str,
    boots: int,
    frames: int,
    parallel: int,
    git_sha: str,
    manifest_hash: str,
    source_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    started = time.perf_counter()
    overrides = {} if source_overrides is None else source_overrides
    verify_source(REMOTE, manifest_hash, overrides)
    environment = os.environ.copy()
    environment.update(
        HAL_GIT_SHA=git_sha,
        HAL_SOURCE_MANIFEST_SHA256=manifest_hash,
        HAL_SOURCE_OVERRIDES=json.dumps(overrides, sort_keys=True),
        UV_INDEX_URL=PYPI_INDEX,
        DISPLAY=":99",
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    for key, directory in (
        ("TMPDIR", "/tmp/control-timing"),
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/control-timing-inductor"),
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
        environment[key] = directory
    shared_memory = shutil.disk_usage("/dev/shm")
    print(f"[timing] shared memory free={shared_memory.free / 2**30:.2f} GiB", flush=True)
    if shared_memory.free < 1024**3:
        raise RuntimeError("evaluation requires at least 1 GiB of shared memory")
    with Path("/tmp/control-timing-xvfb.log").open("w") as display_log:
        display = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x720x24"], stdout=display_log, stderr=subprocess.STDOUT
        )
        try:
            subprocess.run(
                [
                    "/opt/venv/bin/python",
                    "-m",
                    "scripts.control_timing",
                    "--delay",
                    str(delay),
                    "--replan",
                    str(replan),
                    "--run-name",
                    run_name,
                    "--boots",
                    str(boots),
                    "--frames",
                    str(frames),
                    "--parallel",
                    str(parallel),
                ],
                cwd=REMOTE,
                env=environment,
                check=True,
            )
        finally:
            display.terminate()
            display.wait(timeout=10)
    output = REMOTE / "runs" / run_name / Cell(delay, replan).name
    return {
        "protocol.json": (output / "protocol.json").read_text(),
        "results.json": (output / "results.json").read_text(),
        "artifact_hashes.json": (output / "artifact_hashes.json").read_text(),
        "modal_runtime.json": json.dumps({"wall_seconds": time.perf_counter() - started, "python": sys.version}),
    }


def collect(output: Path, phase: str) -> None:
    launch = json.loads((output / f"{phase}-launch.json").read_text())
    states = {}
    for name, call_id in launch["calls"].items():
        target = output / phase / name
        if (target / "modal_runtime.json").exists():
            states[name] = "collected"
            continue
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError, modal.exception.TimeoutError:
            states[name] = "running"
            continue
        except subprocess.CalledProcessError as error:
            states[name] = f"failed: {error}"
            continue
        target.mkdir(parents=True, exist_ok=True)
        for filename, content in result.items():
            (target / filename).write_text(content)
        states[name] = "collected"
    save_json(output / f"{phase}-status.json", states)
    print(json.dumps(states, sort_keys=True), flush=True)


def completed_pilot_calls(output: Path, parallel: int) -> dict[str, str]:
    """Reuse full pilot cells only when their launch and frame budgets match."""
    status = json.loads((output / "pilot-status.json").read_text())
    expected = {cell.name for cell in PILOT_CELLS}
    if set(status) != expected or set(status.values()) != {"collected"}:
        raise ValueError("all three pilot cells must complete before the full wave")
    launch = json.loads((output / "pilot-launch.json").read_text())
    identity = json.loads((output / "source_identity.json").read_text())
    if (
        launch["gpu"] != EVALUATOR_GPU
        or launch["parallel"] != parallel
        or launch["source_manifest_sha256"] != identity["source_manifest_sha256"]
        or set(launch["calls"]) != expected
    ):
        raise ValueError("pilot launch differs from the full-wave configuration")
    for cell in PILOT_CELLS:
        root = output / "pilot" / cell.name
        protocol = json.loads((root / "protocol.json").read_text())
        result = json.loads((root / "results.json").read_text())
        if not result["complete"] or (
            protocol["args"]["boots"],
            protocol["args"]["frames"],
            protocol["args"]["parallel"],
        ) != (96, 7200, parallel):
            raise ValueError("pilot must contain a complete full-budget data point")
        totals = [sum(row["total_frames"] for row in result["rows"] if row["boot_index"] == i) for i in range(96)]
        if totals != [7200] * 96:
            raise ValueError("pilot has incomplete boot budgets")
    return launch["calls"]


def main(args: Args) -> None:
    if args.action == "collect":
        collect(args.output, args.phase)
        return
    phase = args.action
    if (args.output / f"{phase}-launch.json").exists():
        raise FileExistsError("this phase has already been launched")
    if phase == "pilot":
        source, git_sha, manifest_hash = snapshot(args.output)
        cells = PILOT_CELLS
        previous_calls = {}
    else:
        previous_calls = completed_pilot_calls(args.output, args.parallel)
        if not (args.output / "budget.json").exists():
            raise ValueError("save the measured evaluation budget before launching")
        identity = json.loads((args.output / "source_identity.json").read_text())
        git_sha, manifest_hash = identity["base_git_sha"], identity["source_manifest_sha256"]
        source = args.output / "source"
        cells = tuple(cell for cell in grid() if cell.name not in previous_calls)
    client, secret = preflight_modal("hal")
    run_name = f"{args.output.name}-{phase}-{manifest_hash[:8]}"
    app = modal.App(run_name)
    evaluator = app.function(
        image=evaluation_image(source, secret),
        secrets=[secret],
        gpu=EVALUATOR_GPU,
        cpu=16.0,
        memory=64 * 1024,
        ephemeral_disk=512 * 1024,
        timeout=4 * 3600,
        max_containers=args.containers,
        single_use_containers=True,
        serialized=True,
        include_source=False,
        name="evaluate-control-timing",
    )(remote_evaluate)
    with modal.enable_output(), app.run(client=client, detach=True):
        record = {
            "app_id": app.app_id,
            "run_name": run_name,
            "base_git_sha": git_sha,
            "source_manifest_sha256": manifest_hash,
            "gpu": EVALUATOR_GPU,
            "parallel": args.parallel,
            "boots": 96,
            "frames": 7200,
            "reused_pilot_cells": sorted(previous_calls),
            "calls": previous_calls.copy(),
        }
        for cell in cells:
            call = evaluator.spawn(cell.delay, cell.replan, run_name, 96, 7200, args.parallel, git_sha, manifest_hash)
            record["calls"][cell.name] = call.object_id
            save_json(args.output / f"{phase}-launch.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
