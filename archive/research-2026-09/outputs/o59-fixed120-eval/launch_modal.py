"""Launch the final O59 checkpoint's fixed-120 CPU evaluation on Modal L40S."""

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path

import modal
import torch

from hal.fixtures import ALL
from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from scripts.launch_modal import IMAGE
from scripts.launch_modal import PYPI_INDEX
from scripts.launch_modal import REMOTE_ROOT
from scripts.launch_modal import _prepare_remote
from scripts.launch_modal import closed_loop_retry_policy
from scripts.launch_modal import preflight_modal

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("outputs/o59-fixed120-eval/059_muon_history_decoder.py")
RUN_NAME = "260921-092548_059_muon_history_decoder_o59v5-cooldown32768"
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
OUTPUT_NAME = "eval96-step-0131072-fixed120"
TARGET = 120.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_image(secret: modal.Secret) -> modal.Image:
    def ignore(path: Path) -> bool:
        return "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo")

    fixture_manifest = json.dumps(
        [
            {
                "name": fixture.name,
                "sha256": fixture.sha256,
                "size_bytes": fixture.size_bytes,
                "dest": str(fixture.dest),
                "r2_key": fixture.r2_key,
                "url": fixture.url,
                "extract": fixture.extract,
            }
            for fixture in ALL
        ],
        sort_keys=True,
    )
    dependency = (
        modal.Image.from_registry(IMAGE)
        .add_local_file(ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
        .add_local_file(ROOT / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
        .add_local_file(
            ROOT / "vendor/melee-0.47.0+hal.realtime.1.tar.gz",
            str(REMOTE_ROOT / "vendor/melee-0.47.0+hal.realtime.1.tar.gz"),
            copy=True,
        )
        .workdir(str(REMOTE_ROOT))
        .run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --no-install-project")
    )
    fixtures = dependency.add_local_file(
        ROOT / "scripts/cache_modal_fixtures.py", "/opt/cache_modal_fixtures.py", copy=True
    ).run_commands(
        "/opt/venv/bin/python /opt/cache_modal_fixtures.py",
        env={"HAL_FIXTURE_MANIFEST": fixture_manifest, "HAL_FIXTURE_ROOT": str(REMOTE_ROOT)},
        secrets=[secret],
    )
    source = (
        fixtures.add_local_dir(ROOT / "hal", str(REMOTE_ROOT / "hal"), copy=True, ignore=ignore)
        .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True, ignore=ignore)
        .add_local_file(ROOT / "README.md", str(REMOTE_ROOT / "README.md"), copy=True)
        .add_local_file(ROOT / SOURCE, str(REMOTE_ROOT / SOURCE), copy=True)
        .add_local_file(
            ROOT / "outputs/o59-fixed120-eval/launch_modal.py",
            str(REMOTE_ROOT / "outputs/o59-fixed120-eval/launch_modal.py"),
            copy=True,
        )
    )
    return source.run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --offline --no-build-isolation")


def _run_eval(git_sha: str, source_sha: str) -> None:
    source = REMOTE_ROOT / SOURCE
    if _sha256(source) != source_sha:
        raise ValueError("Modal evaluation source differs from launch source")
    output = REMOTE_ROOT / "runs" / RUN_NAME / "checkpoints" / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=False)
    environment = _prepare_remote(skip_sm120_probe=False)
    provenance = {
        "schema_version": 1,
        "launched_at_utc": datetime.now(UTC).isoformat(),
        "base_git_sha": git_sha,
        "eval_source_sha256": source_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_run": RUN_NAME,
        "checkpoint_update": 131072,
        "desired_return": TARGET,
        "n_matchups": 96,
        "max_parallel": 32,
        "eval_seed": 0,
        "cpu_level": 9,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "melee": importlib.metadata.version("melee"),
        "iso_sha256": ISO.sha256,
        "dolphin_sha256": DOLPHIN_EXIAI.sha256,
        "uv_lock_sha256": _sha256(REMOTE_ROOT / "uv.lock"),
    }
    (output / "source_provenance.json").write_text(json.dumps(provenance, sort_keys=True))
    shutil.copyfile(source, output / "eval_source.py")
    command = [
        "uv",
        "run",
        str(SOURCE),
        "eval",
        "--checkpoint",
        "checkpoints/step-0131072.pt",
        "--run",
        RUN_NAME,
        "--n-matchups",
        "96",
        "--max-parallel",
        "32",
        "--output-name",
        OUTPUT_NAME,
        "--shared-wandb",
        "--wandb-namespace",
        "eval_fixed120",
        "--fixed-return",
        "120",
        "--expected-checkpoint-sha256",
        CHECKPOINT_SHA256,
    ]
    print(json.dumps({"command": command, "provenance": provenance}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, env=environment, check=True)


def main() -> None:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_sha = _sha256(ROOT / SOURCE)
    client, secret = preflight_modal("hal")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = f"hal-o59-fixed120-{stamp}-{source_sha[:8]}"
    app = modal.App(name=name, tags={"provider": "modal", "git_sha": git_sha, "eval": "fixed120"})
    image = _eval_image(secret)
    evaluator = app.function(
        image=image,
        secrets=[secret],
        gpu="L40S",
        cpu=32.0,
        memory=64 * 1024,
        ephemeral_disk=512 * 1024,
        timeout=2 * 60 * 60,
        retries=closed_loop_retry_policy(),
        max_containers=1,
        single_use_containers=True,
        serialized=True,
        include_source=False,
        name="fixed120-eval",
    )(_run_eval)
    with modal.enable_output(), app.run(name=name, client=client, detach=True):
        call = evaluator.spawn(git_sha, source_sha)
        print(json.dumps({"app_id": app.app_id, "function_call_id": call.object_id, "app_name": name}))


if __name__ == "__main__":
    main()
