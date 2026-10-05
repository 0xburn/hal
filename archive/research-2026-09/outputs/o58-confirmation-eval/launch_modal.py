"""Resume the preregistered O58 seed-1 return-conditioning comparison."""

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

from hal import r2
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
SOURCE = Path("experiments/058_return_conditioning.py")
TREATMENT_RUN = (
    "260920-042517_058_return_conditioning_attention055-d256-L16-h4-Lc256-t128x4-"
    "o1-2-3-4-5-6-9-12-16-20-d2r2-add-history-final-prefix-all-adamw-"
    "alr0.0017-g0.99855_ranked-anon-1_o58-return60"
)
CONTROL_RUN = (
    "260914-014038_055_history_cross_attention_attention055-d256-L16-h4-Lc256-t128x4-"
    "o1-2-3-4-5-6-9-12-16-20-d2r2-add-history-final-prefix-all-adamw-"
    "alr0.0017-g0.99855_ranked-anon-1_o55-b-add-history"
)
TREATMENT_SHA256 = "9a88a3401f59e43dd9db717f3d8c1275027458f0fba73748c3c9c2e8e6a5e78c"
CONTROL_SHA256 = "e8ef7228d0fb0197e31e328fee6b0667b768878b7f0cb3d803da0abdb35c5136"
PRIOR_EVIDENCE = (
    "frozen-targets.json",
    "selection.json",
    "selection-control-seed0/match_rows.json",
    "selection-target0-seed0/match_rows.json",
    "selection-target1-seed0/match_rows.json",
    "selection-target2-seed0/match_rows.json",
    "confirmation-control-seed1/match_rows.json",
)


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
            ROOT / "outputs/o58-confirmation-eval/launch_modal.py",
            str(REMOTE_ROOT / "outputs/o58-confirmation-eval/launch_modal.py"),
            copy=True,
        )
    )
    return source.run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --offline --no-build-isolation")


def _sync_prior_evidence(root: Path) -> dict[str, str]:
    client = r2.client()
    hashes = {}
    for name in PRIOR_EVIDENCE:
        key = f"runs/{TREATMENT_RUN}/return-evaluation-v1/{name}"
        content = client.get_object(Bucket=r2.bucket(), Key=key)["Body"].read()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    registration = json.loads((root / "frozen-targets.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    if registration["treatment_checkpoint_sha256"] != TREATMENT_SHA256:
        raise ValueError("saved O58 checkpoint identity changed")
    if registration["control_checkpoint_sha256"] != CONTROL_SHA256:
        raise ValueError("saved O55 checkpoint identity changed")
    if selection["selected_index"] != 2 or selection["selected_target"] != registration["targets"][2]:
        raise ValueError("saved O58 p90 target selection changed")
    return hashes


def _run_eval(git_sha: str, source_sha: str) -> None:
    source = REMOTE_ROOT / SOURCE
    if _sha256(source) != source_sha:
        raise ValueError("Modal evaluation source differs from launch source")
    environment = _prepare_remote(skip_sm120_probe=False)
    output = REMOTE_ROOT / "runs" / TREATMENT_RUN / "return-evaluation-v1"
    output.mkdir(parents=True, exist_ok=False)
    prior_hashes = _sync_prior_evidence(output)
    provenance = {
        "schema_version": 1,
        "launched_at_utc": datetime.now(UTC).isoformat(),
        "base_git_sha": git_sha,
        "eval_source_sha256": source_sha,
        "treatment_run": TREATMENT_RUN,
        "treatment_checkpoint_sha256": TREATMENT_SHA256,
        "control_run": CONTROL_RUN,
        "control_checkpoint_sha256": CONTROL_SHA256,
        "target_index": 2,
        "target_value": 20.905278205871582,
        "eval_seed": 1,
        "n_matchups": 96,
        "cpu_level": 9,
        "prior_evidence_sha256": prior_hashes,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "melee": importlib.metadata.version("melee"),
        "iso_sha256": ISO.sha256,
        "dolphin_sha256": DOLPHIN_EXIAI.sha256,
        "uv_lock_sha256": _sha256(REMOTE_ROOT / "uv.lock"),
    }
    (output / "confirmation_source_provenance.json").write_text(json.dumps(provenance, sort_keys=True))
    shutil.copyfile(source, output / "confirmation_eval_source.py")
    command = [
        "uv",
        "run",
        str(SOURCE),
        "suite",
        "--checkpoint",
        "final.pt",
        "--run",
        TREATMENT_RUN,
        "--control-checkpoint",
        "final.pt",
        "--control-run",
        CONTROL_RUN,
    ]
    print(json.dumps({"command": command, "provenance": provenance}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, env=environment, check=True)


def main() -> None:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_sha = _sha256(ROOT / SOURCE)
    client, secret = preflight_modal("hal")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = f"hal-o58-confirmation-{stamp}-{source_sha[:8]}"
    app = modal.App(name=name, tags={"provider": "modal", "git_sha": git_sha, "eval": "o58-confirmation"})
    evaluator = app.function(
        image=_eval_image(secret),
        secrets=[secret],
        gpu="L40S",
        cpu=32.0,
        memory=64 * 1024,
        ephemeral_disk=512 * 1024,
        timeout=3 * 60 * 60,
        retries=closed_loop_retry_policy(),
        max_containers=1,
        single_use_containers=True,
        serialized=True,
        include_source=False,
        name="confirmation-eval",
    )(_run_eval)
    with modal.enable_output(), app.run(name=name, client=client, detach=True):
        call = evaluator.spawn(git_sha, source_sha)
        print(json.dumps({"app_id": app.app_id, "function_call_id": call.object_id, "app_name": name}))


if __name__ == "__main__":
    main()
