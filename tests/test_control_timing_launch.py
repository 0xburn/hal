import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.control_timing import EVALUATOR_GPU
    from scripts.control_timing import save_json
    from scripts.control_timing import sha256
    from scripts.control_timing import verify_source
    from scripts.launch_control_timing import PILOT_CELLS
    from scripts.launch_control_timing import completed_pilot_calls
finally:
    sys.path.pop(0)


def test_source_extension_requires_exact_hashes_for_changed_and_unchanged_files(tmp_path: Path) -> None:
    (tmp_path / "stable.py").write_text("stable")
    (tmp_path / "guard.py").write_text("old guard")
    save_json(tmp_path / "source_manifest.json", {name: sha256(tmp_path / name) for name in ("stable.py", "guard.py")})
    manifest_hash = sha256(tmp_path / "source_manifest.json")
    verify_source(tmp_path, manifest_hash, {})
    (tmp_path / "guard.py").write_text("corrected guard")
    with pytest.raises(ValueError, match="source changed: guard.py"):
        verify_source(tmp_path, manifest_hash, {})
    overrides = {"guard.py": sha256(tmp_path / "guard.py")}
    verify_source(tmp_path, manifest_hash, overrides)
    (tmp_path / "stable.py").write_text("unexpected change")
    with pytest.raises(ValueError, match="source changed: stable.py"):
        verify_source(tmp_path, manifest_hash, overrides)


@pytest.mark.parametrize("name", ("../escape.py", "/absolute.py", "./relative.py"))
def test_source_extension_rejects_noncanonical_paths(tmp_path: Path, name: str) -> None:
    save_json(tmp_path / "source_manifest.json", {})
    with pytest.raises(ValueError, match="invalid source path"):
        verify_source(tmp_path, sha256(tmp_path / "source_manifest.json"), {name: "hash"})


@pytest.fixture
def completed_pilots(tmp_path: Path) -> Path:
    save_json(tmp_path / "pilot-status.json", {cell.name: "collected" for cell in PILOT_CELLS})
    save_json(tmp_path / "source_identity.json", {"source_manifest_sha256": "source-hash"})
    save_json(
        tmp_path / "pilot-launch.json",
        {
            "gpu": EVALUATOR_GPU,
            "parallel": 8,
            "source_manifest_sha256": "source-hash",
            "calls": {cell.name: f"call-{cell.name}" for cell in PILOT_CELLS},
        },
    )
    for cell in PILOT_CELLS:
        root = tmp_path / "pilot" / cell.name
        save_json(root / "protocol.json", {"args": {"boots": 96, "frames": 7200, "parallel": 8}})
        save_json(
            root / "results.json",
            {"complete": True, "rows": [{"boot_index": i, "total_frames": 7200} for i in range(96)]},
        )
    return tmp_path


def test_full_wave_reuses_all_three_pilot_calls(completed_pilots: Path) -> None:
    assert completed_pilot_calls(completed_pilots, 8) == {
        "d0-r1": "call-d0-r1",
        "d3-r4": "call-d3-r4",
        "d6-r6": "call-d6-r6",
    }


def test_pilot_reuse_rejects_changed_sampling_partition(completed_pilots: Path) -> None:
    with pytest.raises(ValueError, match="configuration"):
        completed_pilot_calls(completed_pilots, 4)


@pytest.mark.parametrize("boots,frames", ((8, 1200), (96, 1200), (95, 7200)))
def test_pilot_reuse_rejects_shortened_budgets(completed_pilots: Path, boots: int, frames: int) -> None:
    save_json(
        completed_pilots / "pilot/d0-r1/protocol.json",
        {"args": {"boots": boots, "frames": frames, "parallel": 8}},
    )
    with pytest.raises(ValueError, match="full-budget"):
        completed_pilot_calls(completed_pilots, 8)


def test_pilot_reuse_rejects_incomplete_rows_even_if_marked_complete(completed_pilots: Path) -> None:
    save_json(
        completed_pilots / "pilot/d0-r1/results.json",
        {"complete": True, "rows": [{"boot_index": i, "total_frames": 7200} for i in range(95)]},
    )
    with pytest.raises(ValueError, match="incomplete boot"):
        completed_pilot_calls(completed_pilots, 8)
