"""Contracts for fitting one player's identity row onto a released 059 policy."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

from hal import streams
from hal.inference.action_sequence_artifact import ActionSequenceArtifact
from hal.inference.action_sequence_artifact import checkpoint_contract
from hal.inference.action_sequence_artifact import validate_checkpoint_provenance
from hal.inference.api import PolicySpec
from hal.inference.warmup import synthetic_context
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.representation.features import Context
from hal.representation.player_identity import FIRST_CONNECT_CODE_ID
from hal.representation.player_identity import PlayerVocabulary
from hal.training.batches import TrainBatch
from hal.wire import ACTION_DIM

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _ROOT / "experiments" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


finetune = _load("player_identity_finetune_062", "062_player_identity_finetune.py")


def _table(vocabulary: PlayerVocabulary, width: int = 3) -> torch.Tensor:
    return torch.arange(vocabulary.size * width, dtype=torch.float32).reshape(vocabulary.size, width) + 1.0


def test_inserting_an_identity_keeps_every_other_vector_on_its_code() -> None:
    base = PlayerVocabulary(("A#1", "C#3"))
    table = _table(base)
    plan = finetune.plan_identity(base, table, "B#2", None)

    assert plan.inserted
    assert plan.vocabulary.codes == ("A#1", "B#2", "C#3")
    assert plan.player_id == FIRST_CONNECT_CODE_ID + 1
    for code in base.codes:
        assert torch.equal(plan.embedding[plan.vocabulary.id_for_code(code)], table[base.id_for_code(code)])
    assert torch.equal(plan.embedding[:FIRST_CONNECT_CODE_ID], table[:FIRST_CONNECT_CODE_ID])
    assert torch.equal(plan.embedding[plan.player_id], torch.zeros(3))
    assert plan.initialized_from == "unconditioned"


def test_initializers_copy_a_rank_or_code_row() -> None:
    base = PlayerVocabulary(("A#1", "C#3"))
    table = _table(base)
    from_rank = finetune.plan_identity(base, table, "B#2", "master")
    assert torch.equal(from_rank.embedding[from_rank.player_id], table[3])
    from_code = finetune.plan_identity(base, table, "B#2", "C#3")
    assert torch.equal(from_code.embedding[from_code.player_id], table[base.id_for_code("C#3")])


def test_an_existing_identity_is_refit_in_place_from_its_own_row() -> None:
    base = PlayerVocabulary(("A#1", "C#3"))
    table = _table(base)
    plan = finetune.plan_identity(base, table, "C#3", None)
    assert not plan.inserted
    assert plan.vocabulary == base
    assert torch.equal(plan.embedding, table)
    assert plan.embedding.data_ptr() != table.data_ptr()


@pytest.mark.parametrize("identity", ["MASTER", "unknown", " "])
def test_identities_that_collide_with_ranks_or_are_empty_are_rejected(identity: str) -> None:
    base = PlayerVocabulary(("A#1",))
    with pytest.raises(ValueError):
        finetune.plan_identity(base, _table(base), identity, None)


def test_unknown_initializer_is_rejected() -> None:
    base = PlayerVocabulary(("A#1",))
    with pytest.raises(ValueError, match="init source"):
        finetune.plan_identity(base, _table(base), "B#2", "Z#9")


@pytest.mark.parametrize(
    ("ports", "target", "expected"),
    [((1, 2), 1, "p1"), ((1, 2), 2, "p2"), ((3, 4), 4, "p2"), ((4, 1), 1, "p1"), ((2, 3), 3, "p2")],
)
def test_ports_map_to_the_extractor_sides(ports: tuple[int, int], target: int, expected: str) -> None:
    assert finetune.ego_prefix_for(ports, target) == expected


def test_an_unoccupied_port_is_rejected() -> None:
    with pytest.raises(ValueError, match="not occupied"):
        finetune.ego_prefix_for((1, 2), 3)


def test_side_rules_resolve_relative_paths_and_reject_bad_rows(tmp_path: Path) -> None:
    sides = tmp_path / "sides.jsonl"
    sides.write_text(json.dumps({"path": "games/a.slp", "port": 3}) + "\n\n")
    (rule,) = finetune.read_side_rules(sides)
    assert rule.path == str((tmp_path / "games" / "a.slp").resolve())
    assert rule.port == 3 and rule.code is None

    sides.write_text(json.dumps({"path": "a.slp", "port": 5}) + "\n")
    with pytest.raises(ValueError, match="port"):
        finetune.read_side_rules(sides)
    sides.write_text("".join(json.dumps({"path": "a.slp", "port": 1}) + "\n" for _ in range(2)))
    with pytest.raises(ValueError, match="more than once"):
        finetune.read_side_rules(sides)


def test_validation_split_is_stable_and_respects_the_fraction() -> None:
    ids = [f"{index:08x}" + "0" * 24 for index in range(0, 2**32, 2**24)]
    held_out = [finetune.is_validation_replay(replay_id, 0.25) for replay_id in ids]
    assert sum(held_out) == len(ids) // 4
    assert held_out == [finetune.is_validation_replay(replay_id, 0.25) for replay_id in ids]


def _checkpoint(vocabulary: PlayerVocabulary) -> dict[str, object]:
    cfg = json.loads((_ROOT / "tests" / "fixtures" / "o59" / "checkpoint_config.json").read_text())
    cfg["source_names"] = tuple(cfg["source_names"])
    cfg["architecture"]["head_offsets"] = tuple(cfg["architecture"]["head_offsets"])
    cfg["player_vocab_size"] = vocabulary.size
    cfg["player_vocab_sha256"] = vocabulary.sha256
    provenance = {
        "schema_version": 1,
        "experiment_id": "059_muon_history_decoder_v5",
        "source_selection_sha256": "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477",
        "source_manifest_sha256": streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256,
        "player_sidecar_sha256": cfg["player_sidecar_sha256"],
        "player_vocab_sha256": vocabulary.sha256,
        "git_sha": "0" * 40,
    }
    return {
        "cfg": cfg,
        "provenance": provenance,
        "model": {"player_embedding.weight": _table(vocabulary), "trunk.weight": torch.ones(2)},
        "step": 7,
    }


def test_checkpoint_update_changes_only_identity_fields_and_passes_059_validation() -> None:
    base = PlayerVocabulary(("A#1", "C#3"))
    raw = _checkpoint(base)
    plan = finetune.plan_identity(base, _table(base), "B#2", None)
    updated = finetune.updated_checkpoint(raw, vocabulary=plan.vocabulary, embedding=plan.embedding, record={"x": 1})

    contract = checkpoint_contract(updated["cfg"])
    assert contract.model.player_vocab_size == plan.vocabulary.size
    assert contract.model.player_vocab_sha256 == plan.vocabulary.sha256
    validate_checkpoint_provenance(updated["provenance"], contract)

    model = cast(dict[str, torch.Tensor], updated["model"])
    assert torch.equal(model["player_embedding.weight"], plan.embedding)
    assert (
        bytes(model["player_code_bytes"].numpy()) == json.dumps(plan.vocabulary.codes, separators=(",", ":")).encode()
    )
    assert model["trunk.weight"] is cast(dict[str, torch.Tensor], raw["model"])["trunk.weight"]
    assert updated["step"] == 7
    assert cast(dict[str, object], updated["provenance"])["player_finetune"] == {"x": 1}
    with pytest.raises(ValueError, match="already a player fine-tune"):
        finetune.updated_checkpoint(updated, vocabulary=plan.vocabulary, embedding=plan.embedding, record={})


def test_checkpoint_update_rejects_a_mismatched_table() -> None:
    base = PlayerVocabulary(("A#1",))
    with pytest.raises(ValueError, match="does not match"):
        finetune.updated_checkpoint(_checkpoint(base), vocabulary=base, embedding=torch.zeros(2, 3), record={})


def _tiny_artifact(vocabulary: PlayerVocabulary) -> ActionSequenceArtifact:
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=64,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    torch.manual_seed(0)
    reference = ActionSequenceTransformer(config, vocabulary)
    with torch.no_grad():
        reference.player_projection.weight.normal_()
        reference.player_embedding.weight[1:].normal_()
    return ActionSequenceArtifact(
        config,
        {name: value.detach().clone() for name, value in reference.state_dict().items()},
        {},
        vocabulary,
        "test-checkpoint",
        20.0,
        PolicySpec("test", "hal.action_sequence.test", (), (0,)),
        2,
    )


def test_one_training_step_moves_only_the_target_identity_row() -> None:
    base = PlayerVocabulary(("A#1", "C#3"))
    artifact = _tiny_artifact(base)
    plan = finetune.plan_identity(base, artifact.model_tensors["player_embedding.weight"], "B#2", "MASTER")
    device = torch.device("cpu")
    model = finetune.build_trainable_model(artifact, plan, device)
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == [
        "player_embedding.weight"
    ]
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    batch_size, length = 2, artifact.model_config.L_ctx
    context = synthetic_context(length, batch_size, device, items=True)
    features = {**context.features, "ego_player_id": torch.full((batch_size, length), plan.player_id)}
    target = torch.rand(batch_size, artifact.model_config.sample_chunk_length, ACTION_DIM)
    batch = finetune.PlayerBatch(
        TrainBatch(Context(features, context.ctx_pad), target),
        torch.zeros(batch_size, length),
        torch.ones(batch_size, length, dtype=torch.bool),
    )
    generator = torch.Generator().manual_seed(0)
    positions = finetune.sample_prefix_positions(batch_size, length, generator)
    terms = finetune.behavior_cloning_objective(model, batch, positions, autocast_dtype=None)
    assert torch.isfinite(terms.objective) and torch.isfinite(terms.next_frame_bits)

    optimizer = torch.optim.Adam([model.player_embedding.weight], lr=0.1, weight_decay=0.0)
    terms.objective.backward()
    optimizer.step()

    after = model.state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed == ["player_embedding.weight"]
    rows = (before["player_embedding.weight"] != after["player_embedding.weight"]).any(dim=1).nonzero().flatten()
    assert rows.tolist() == [plan.player_id]


def test_prefix_positions_cover_only_the_supervised_suffix() -> None:
    positions = finetune.sample_prefix_positions(3, 256, torch.Generator().manual_seed(1))
    assert positions.shape == (3, finetune.POLICY_PREFIXES_PER_WINDOW)
    assert int(positions.min()) >= 128 and int(positions.max()) < 256
    assert all(len(set(row.tolist())) == finetune.POLICY_PREFIXES_PER_WINDOW for row in positions)
    assert torch.equal(positions, positions.sort(dim=1).values)


def test_return_labels_match_059() -> None:
    o59 = _load("muon_action_sequence_059_for_062", "059_muon_action_sequence.py")
    rng = np.random.default_rng(0)
    frames = 400
    sample = {
        "p1_percent": np.cumsum(rng.uniform(0, 1.5, frames)).astype(np.float32) % 150,
        "p2_percent": np.cumsum(rng.uniform(0, 1.5, frames)).astype(np.float32) % 150,
        "p1_stock": np.repeat(np.array([4, 3, 2], dtype=np.uint8), [150, 150, 100]),
        "p2_stock": np.repeat(np.array([4, 3, 2, 1, 0], dtype=np.uint8), [100, 100, 100, 60, 40]),
    }
    expected = o59.future_return_labels(dict(sample))
    actual = finetune.future_return_labels(dict(sample))
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_scan_and_train_reject_ambiguous_sources(tmp_path: Path) -> None:
    both = finetune.PlayerSource(replays=tmp_path, code="A#1", sides=tmp_path / "sides.jsonl")
    neither = finetune.PlayerSource(replays=tmp_path)
    for source in (both, neither):
        with pytest.raises(ValueError, match="exactly one"):
            source.rules()
    with pytest.raises(ValueError, match="--identity"):
        finetune._identity(None, replace(neither, sides=tmp_path / "sides.jsonl"))
    with pytest.raises(FileNotFoundError, match="--source.replays"):
        finetune.PlayerSource(code="A#1").rules()
