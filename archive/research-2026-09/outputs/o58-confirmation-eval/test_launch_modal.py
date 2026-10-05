"""Check that O58 confirmation resumes only the recorded selection."""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("launch_modal.py")
SPEC = importlib.util.spec_from_file_location("o58_confirmation_launch", SOURCE)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_sync_prior_evidence_preserves_bytes_and_validates_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registration = {
        "treatment_checkpoint_sha256": launcher.TREATMENT_SHA256,
        "control_checkpoint_sha256": launcher.CONTROL_SHA256,
        "targets": [0.0, 9.0, 20.905278205871582],
    }
    selection = {"selected_index": 2, "selected_target": registration["targets"][2]}
    payloads = {name: b"{}\n" for name in launcher.PRIOR_EVIDENCE}
    payloads["frozen-targets.json"] = json.dumps(registration).encode()
    payloads["selection.json"] = json.dumps(selection).encode()

    class FakeClient:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
            assert Bucket == "hal"
            prefix = f"runs/{launcher.TREATMENT_RUN}/return-evaluation-v1/"
            assert Key.startswith(prefix)
            return {"Body": io.BytesIO(payloads[Key.removeprefix(prefix)])}

    monkeypatch.setattr(launcher.r2, "client", FakeClient)
    monkeypatch.setattr(launcher.r2, "bucket", lambda: "hal")
    hashes = launcher._sync_prior_evidence(tmp_path)
    for name, content in payloads.items():
        assert (tmp_path / name).read_bytes() == content
        assert hashes[name] == hashlib.sha256(content).hexdigest()

    payloads["selection.json"] = b'{"selected_index": 0, "selected_target": 0.0}'
    with pytest.raises(ValueError, match="target selection changed"):
        launcher._sync_prior_evidence(tmp_path)
