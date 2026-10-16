"""Experiment 062: fit one player's identity embedding from their games.

The released 059 policy conditions every context token on a 32-wide identity
row (``player_embedding``). This experiment adds one row for a new player, or
refits an existing one, and trains only that row on the player's own games.
Every other parameter stays frozen, so the base policy cannot drift and every
other identity keeps its exact behavior.

The objective is plain behavior cloning: the 059 offset-weighted action NLL
with all advantage weights equal to one. A clone should reproduce the
player's habits, including their mistakes, so it does not reweight actions by
outcome. Return conditioning keeps the 059 protocol (60-frame return, dropout
per context position) so the row is fit under both the conditioned and the
unconditioned decoder.

The output is an ordinary 059 policy bundle. The vocabulary gains the new
identity, the checkpoint config and provenance carry the new vocabulary hash,
and ``read_action_sequence_artifact`` validates it unchanged. Play it with
``hal-policy play OUTPUT --imitate IDENTITY``.

This file imports only ``hal``. It duplicates the few 059 objective pieces it
needs rather than importing that experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Annotated
from typing import Final
from typing import Literal
from typing import cast

import melee
import numpy as np
import torch
import tyro
from torch import Tensor

from hal.data.extract import extract_policy_world_replay
from hal.data.feature_stats import FeatureStats
from hal.data.index import extract_index_entry
from hal.data.policy_schema import policy_replay_identity
from hal.data.policy_schema import unpack_player_stock
from hal.data.policy_world_schema import decode_policy_world_replay_slice
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.data.schema import check_schema_version
from hal.inference.action_sequence_artifact import ActionSequenceArtifact
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import extract_policy_bundle
from hal.inference.bundle import write_policy_bundle
from hal.models.action_sequence import ActionSequenceTransformer
from hal.representation.features import ITEM_PLAYER_COLUMNS
from hal.representation.features import ITEM_PLAYER_PROJECTION
from hal.representation.features import FeatureProjection
from hal.representation.features import stack_actions
from hal.representation.player_identity import FIRST_CONNECT_CODE_ID
from hal.representation.player_identity import MASKED_PLAYER_ID
from hal.representation.player_identity import PlayerVocabulary
from hal.representation.player_identity import vocabulary_buffer
from hal.training import returns as returns_lib
from hal.training.batches import TrainBatch
from hal.training.checkpoints import checkpoint_sha256
from hal.training.replay_windows import Window
from hal.training.replay_windows import choose_chunk_starts
from hal.training.replay_windows import collate_windows
from hal.training.replay_windows import make_window
from hal.training.replay_windows import stable_window_rng
from hal.training.replay_windows import train_batch_from_columns

_EXPERIMENT_ID: Final[str] = "062_player_identity_finetune_v1"
# The 059 artifacts were materialized at MDS schema 7; the in-process
# extraction below must produce the same columns.
_BASE_MDS_SCHEMA_VERSION: Final[int] = 7
# 059 objective constants, duplicated by policy so this experiment stays standalone.
POLICY_PREFIXES_PER_WINDOW: Final[int] = 32
OFFSET_LOSS_WEIGHTS: Final[tuple[float, ...]] = (
    0.50,
    0.07,
    0.07,
    0.07,
    0.07,
    0.07,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
)
RETURN_HORIZON: Final[int] = 60
RETURN_GAMMA: Final[float] = 0.99855
BASE_RETURN_DROPOUT: Final[float] = 0.2
PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=ITEM_PLAYER_PROJECTION.columns | {"ego_return60", "ego_return60_valid"},
)
_RANK_INITIALIZERS: Final[frozenset[str]] = frozenset({"PLATINUM", "DIAMOND", "MASTER"})


# ---------------------------------------------------------------------------
# Selecting the player's side in each game


@dataclass(frozen=True, slots=True)
class SideRule:
    """How one game names the player: by connect code, or by an explicit port."""

    path: str
    code: str | None = None
    port: int | None = None

    def __post_init__(self) -> None:
        if (self.code is None) == (self.port is None):
            raise ValueError("a side rule needs exactly one of a connect code or a port")


@dataclass(frozen=True, slots=True)
class PlayerReplay:
    """One game in compact policy-world form plus the player's side and returns."""

    path: str
    replay_id: str
    ego_prefix: Literal["p1", "p2"]
    frames: int
    character: int
    compact: dict[str, object]
    returns: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SkippedReplay:
    path: str
    reason: str


def read_side_rules(sides_file: Path) -> tuple[SideRule, ...]:
    """Read ``{"path": ..., "port": 1-4}`` lines naming the player's port per game."""
    rules: list[SideRule] = []
    for line_number, line in enumerate(sides_file.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"path", "port"}:
            raise ValueError(f"{sides_file}:{line_number}: expected exactly path and port")
        port = row["port"]
        if not isinstance(port, int) or isinstance(port, bool) or port not in (1, 2, 3, 4):
            raise ValueError(f"{sides_file}:{line_number}: port must be 1-4 (libmelee numbering)")
        path = Path(str(row["path"])).expanduser()
        if not path.is_absolute():
            path = sides_file.parent / path
        rules.append(SideRule(str(path.resolve()), port=port))
    paths = [rule.path for rule in rules]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{sides_file} names a game more than once")
    return tuple(rules)


def code_side_rules(replays: Path, code: str) -> tuple[SideRule, ...]:
    code = code.strip()
    if not code:
        raise ValueError("connect code must be non-empty")
    paths = sorted(str(path.resolve()) for path in replays.rglob("*.slp"))
    return tuple(SideRule(path, code=code) for path in paths)


def ego_prefix_for(ports: Sequence[int], target_port: int) -> Literal["p1", "p2"]:
    """Map a libmelee port to the extractor's p1/p2, which are the two lowest occupied ports."""
    occupied = sorted(ports)
    if len(occupied) != 2:
        raise ValueError(f"expected a 1v1 game, got ports {occupied}")
    if target_port not in occupied:
        raise ValueError(f"port {target_port} is not occupied; occupied ports are {occupied}")
    return "p1" if target_port == occupied[0] else "p2"


def future_return_labels(sample: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The 059 60-frame return-conditioning target for both ports.

    This must stay identical to 059's ``future_return_labels``: the base
    decoder learned its return modulation on exactly these values.
    """
    reward = returns_lib.frame_reward(
        dict(sample),
        ego="p1",
        opp="p2",
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
    )
    terminal_events = returns_lib.match_point_events(sample["p1_stock"]) + returns_lib.match_point_events(
        sample["p2_stock"]
    )
    terminal_indices = np.flatnonzero(terminal_events)
    terminal = bool(terminal_indices.size) or returns_lib.infer_terminal_replay(dict(sample))
    if terminal_indices.size:
        reward[int(terminal_indices[0]) + 1 :] = 0.0
    frames = len(reward)
    values = np.zeros(frames, dtype=np.float64)
    for offset in range(1, min(RETURN_HORIZON, frames - 1) + 1):
        values[:-offset] += RETURN_GAMMA ** (offset - 1) * reward[offset:].astype(np.float64)
    available = (np.arange(frames) + RETURN_HORIZON < frames) | terminal
    values = np.where(available, values, np.nan).astype(np.float32)
    return {
        "p1_return60": values,
        "p2_return60": -values,
        "p1_return60_valid": available,
        "p2_return60_valid": available.copy(),
    }


def compact_return_labels(compact: Mapping[str, object]) -> dict[str, np.ndarray]:
    sample: dict[str, np.ndarray] = {}
    for port in ("p1", "p2"):
        sample[f"{port}_percent"] = np.asarray(compact[f"{port}_percent"], dtype=np.float32)
        sample[f"{port}_stock"] = unpack_player_stock(np.asarray(compact[f"{port}_state"]))
    if "mc_terminated" in compact:
        sample["mc_terminated"] = np.asarray(compact["mc_terminated"])
    return future_return_labels(sample)


def load_player_replay(rule: SideRule, minimum_frames: int) -> PlayerReplay | SkippedReplay:
    """Parse one game and keep it only if it is a human 1v1 that names the player once."""
    path = Path(rule.path)
    entry = extract_index_entry(path, compute_sha1=False, with_stats=False)
    if entry is None:
        return SkippedReplay(rule.path, "unreadable")
    if len(entry.players) != 2:
        return SkippedReplay(rule.path, "not 1v1")
    if any(player.player_type != "HUMAN" for player in entry.players):
        return SkippedReplay(rule.path, "non-human player")
    if rule.code is not None:
        matches = [player.port for player in entry.players if (player.code or "").strip() == rule.code]
        if not matches:
            return SkippedReplay(rule.path, "connect code absent")
        if len(matches) > 1:
            return SkippedReplay(rule.path, "connect code on both ports")
        target_port = matches[0]
    else:
        target_port = cast(int, rule.port)
    ports = [player.port for player in entry.players]
    try:
        ego_prefix = ego_prefix_for(ports, target_port)
    except ValueError as error:
        return SkippedReplay(rule.path, str(error))
    character = next(player.character for player in entry.players if player.port == target_port)

    sample = extract_policy_world_replay(str(path))
    if sample is None:
        return SkippedReplay(rule.path, "extraction failed")
    frames = int(np.asarray(sample["frame"]).shape[0])
    if frames < minimum_frames:
        return SkippedReplay(rule.path, f"{frames} frames < {minimum_frames}")
    # Rank is stored per replay but is not a model input; identity comes from the row we fit.
    columns = cast(dict[str, object], sample)
    columns["schema_version"] = SCHEMA_VERSION
    columns["p1_rank"] = np.full(frames, int(Rank.UNKNOWN), dtype=np.uint8)
    columns["p2_rank"] = np.full(frames, int(Rank.UNKNOWN), dtype=np.uint8)
    replay_id = policy_replay_identity(rule.path)
    compact = encode_policy_world_replay(columns, replay_id)
    return PlayerReplay(
        path=rule.path,
        replay_id=replay_id,
        ego_prefix=ego_prefix,
        frames=frames,
        character=character,
        compact=compact,
        returns=compact_return_labels(compact),
    )


def _load_player_replay_task(task: tuple[SideRule, int]) -> PlayerReplay | SkippedReplay:
    rule, minimum_frames = task
    try:
        return load_player_replay(rule, minimum_frames)
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException as error:  # peppi panics subclass BaseException; one bad file must not stop the run.
        return SkippedReplay(rule.path, repr(error))


def load_player_replays(
    rules: Sequence[SideRule], *, minimum_frames: int, workers: int
) -> tuple[list[PlayerReplay], list[SkippedReplay]]:
    tasks = [(rule, minimum_frames) for rule in rules]
    if workers <= 1:
        results = [_load_player_replay_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_load_player_replay_task, tasks, chunksize=4))
    kept = sorted((result for result in results if isinstance(result, PlayerReplay)), key=lambda r: r.replay_id)
    skipped = [result for result in results if isinstance(result, SkippedReplay)]
    return kept, skipped


def is_validation_replay(replay_id: str, fraction: float) -> bool:
    """Split by replay identity so a game never lands in both splits across reruns."""
    return int(replay_id[:8], 16) / 2**32 < fraction


# ---------------------------------------------------------------------------
# Identity vocabulary


@dataclass(frozen=True, slots=True)
class IdentityPlan:
    vocabulary: PlayerVocabulary
    player_id: int
    inserted: bool
    embedding: Tensor
    initialized_from: str


def plan_identity(
    base_vocabulary: PlayerVocabulary,
    base_embedding: Tensor,
    identity: str,
    init_from: str | None,
) -> IdentityPlan:
    """Insert or reuse one identity row without moving any other identity's vector.

    Connect-code IDs are positions in the sorted vocabulary, so inserting a
    code shifts every later ID by one. The embedding rows shift with them, so
    each code still maps to the same learned vector.
    """
    identity = identity.strip()
    if not identity:
        raise ValueError("identity must be non-empty")
    if identity.upper() in _RANK_INITIALIZERS or identity.upper() in Rank.__members__:
        raise ValueError(f"identity {identity!r} collides with a rank name")
    if base_embedding.shape[0] != base_vocabulary.size:
        raise ValueError("embedding rows do not match the vocabulary")
    present = identity in base_vocabulary.codes
    if present:
        vocabulary = base_vocabulary
        embedding = base_embedding.detach().clone()
    else:
        vocabulary = PlayerVocabulary(tuple(sorted((*base_vocabulary.codes, identity))))
        position = FIRST_CONNECT_CODE_ID + vocabulary.codes.index(identity)
        row = torch.zeros(1, base_embedding.shape[1], dtype=base_embedding.dtype)
        embedding = torch.cat((base_embedding[:position], row, base_embedding[position:])).detach().clone()
    player_id = vocabulary.id_for_code(identity)

    source = init_from
    if source is None:
        source = identity if present else "unconditioned"
    if source == "unconditioned":
        # The masked row is zero and contributes nothing; starting there starts
        # the fit from the base policy's identity-free behavior.
        embedding[player_id] = 0.0
    elif source.upper() in _RANK_INITIALIZERS:
        embedding[player_id] = embedding[int(Rank[source.upper()])]
    elif source in vocabulary.codes:
        embedding[player_id] = embedding[vocabulary.id_for_code(source)]
    else:
        raise ValueError(f"init source {source!r} is not 'unconditioned', a rank, or a vocabulary code")
    return IdentityPlan(vocabulary, player_id, not present, embedding, source)


# ---------------------------------------------------------------------------
# Windows and batches


@dataclass(frozen=True, slots=True)
class PlayerBatch:
    train: TrainBatch
    future_return: Tensor
    condition_present: Tensor

    def to(self, device: torch.device) -> PlayerBatch:
        return PlayerBatch(
            self.train.to(device),
            self.future_return.to(device, non_blocking=True),
            self.condition_present.to(device, non_blocking=True),
        )


def player_window(replay: PlayerReplay, *, start: int, length: int, player_id: int) -> Window:
    """Decode one full-context window with the player as ego."""
    stop = start + length
    if start < 0 or stop > replay.frames:
        raise ValueError(f"window [{start}, {stop}) is outside {replay.frames} frames")
    sample = cast(dict[str, np.ndarray], decode_policy_world_replay_slice(replay.compact, start, stop))
    check_schema_version(sample, expected=_BASE_MDS_SCHEMA_VERSION)
    for name, values in replay.returns.items():
        sample[name] = values[start:stop]
    window = make_window(sample, ego_prefix=replay.ego_prefix, start=0, pad=0, length=length, projection=PROJECTION)
    window["ego_player_id"] = np.full(length, player_id, dtype=np.int32)
    window["ctx_pad"] = np.int64(0)
    return window


def collate_player_batch(
    windows: list[Window],
    *,
    stats: dict[str, FeatureStats],
    context_length: int,
    return_dropout: float,
    generator: torch.Generator,
) -> PlayerBatch:
    columns = collate_windows(windows)
    train = train_batch_from_columns(
        columns, stats=stats, L_ctx=context_length, extra=ITEM_PLAYER_COLUMNS, projection=PROJECTION
    )
    future_return = torch.from_numpy(np.ascontiguousarray(columns["ego_return60"][:, :context_length]))
    available = torch.from_numpy(np.ascontiguousarray(columns["ego_return60_valid"][:, :context_length])).bool()
    keep = torch.rand(available.shape, generator=generator) >= return_dropout
    return PlayerBatch(train, future_return, available & keep)


def window_starts(replay: PlayerReplay, *, context_length: int, chunk_length: int, count: int, seed: int) -> list[int]:
    """Deterministic, non-overlapping full-context window starts for evaluation."""
    rng = stable_window_rng(seed, 0, replay.replay_id)
    starts = choose_chunk_starts(replay.frames, context_length, chunk_length, count, rng, require_full_context=True)
    return [int(start) - context_length for start in starts]


def sample_training_windows(
    replays: Sequence[PlayerReplay],
    *,
    batch_size: int,
    context_length: int,
    chunk_length: int,
    player_id: int,
    rng: np.random.Generator,
) -> list[Window]:
    """Draw windows uniformly over the player's frames across all training games."""
    usable = np.array([replay.frames - context_length - chunk_length + 1 for replay in replays], dtype=np.float64)
    if (usable < 1).any():
        raise ValueError("every training game must fit one full-context window")
    picks = rng.choice(len(replays), size=batch_size, p=usable / usable.sum())
    windows = []
    for index in picks:
        replay = replays[int(index)]
        start = int(rng.integers(0, replay.frames - context_length - chunk_length + 1))
        windows.append(player_window(replay, start=start, length=context_length + chunk_length, player_id=player_id))
    return windows


# ---------------------------------------------------------------------------
# Objective


def prepared_targets(model: ActionSequenceTransformer, batch: TrainBatch) -> tuple[Tensor, Tensor]:
    """Quantize history and future once, then align every head offset (as in 059)."""
    history = stack_actions(batch.context.features)
    full = model.codec.quantize(torch.cat((history, batch.target[:, : model.L_chunk]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in model.head_offsets], dim=2)
    suffix = slice(length // 2, None)
    return full[:, :length][:, suffix], targets[:, suffix]


def sample_prefix_positions(batch_size: int, context_length: int, generator: torch.Generator) -> Tensor:
    suffix_start = context_length // 2
    if context_length - suffix_start < POLICY_PREFIXES_PER_WINDOW:
        raise ValueError("the context suffix must expose at least 32 prefixes")
    draws = torch.rand(batch_size, context_length - suffix_start, generator=generator)
    chosen = draws.topk(POLICY_PREFIXES_PER_WINDOW, dim=1, largest=False).indices + suffix_start
    return chosen.sort(dim=1).values


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    objective: Tensor
    """059 offset-weighted joint NLL in nats, averaged over prefixes."""
    next_frame_bits: Tensor
    """Joint NLL of the next frame's controller state, in bits."""


def behavior_cloning_objective(
    model: ActionSequenceTransformer,
    batch: PlayerBatch,
    prefix_positions: Tensor,
    *,
    autocast_dtype: torch.dtype | None,
) -> ObjectiveTerms:
    if len(model.head_offsets) != len(OFFSET_LOSS_WEIGHTS):
        raise ValueError(f"expected {len(OFFSET_LOSS_WEIGHTS)} head offsets, got {len(model.head_offsets)}")
    context = batch.train.context
    if bool(context.ctx_pad.any()):
        raise ValueError("player windows must have full context")
    history, targets = prepared_targets(model, batch.train)
    device = context.ctx_pad.device
    with torch.autocast(
        device_type=device.type, dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None
    ):
        hidden = model(context.features, context.ctx_pad)
        rows = torch.arange(hidden.shape[0], device=device)[:, None]
        local = prefix_positions - hidden.shape[1] // 2
        nll = model.temporal.teacher_forced_nll(
            hidden,
            context.ctx_pad,
            prefix_positions,
            history[rows, local],
            targets[rows, local],
            batch.future_return[rows, prefix_positions],
            batch.condition_present[rows, prefix_positions],
        )
    joint = nll.float().sum(dim=-1)
    weights = torch.tensor(OFFSET_LOSS_WEIGHTS, device=device)
    return ObjectiveTerms(
        objective=(joint * weights).sum(dim=-1).mean(),
        next_frame_bits=joint[..., 0].mean() / math.log(2.0),
    )


# ---------------------------------------------------------------------------
# Model and export


def build_trainable_model(
    artifact: ActionSequenceArtifact, plan: IdentityPlan, device: torch.device
) -> ActionSequenceTransformer:
    """Build the base policy with the planned vocabulary; only the identity table trains."""
    config = replace(
        artifact.model_config,
        player_vocab_size=plan.vocabulary.size,
        player_vocab_sha256=plan.vocabulary.sha256,
    )
    state = dict(artifact.model_tensors)
    state["player_embedding.weight"] = plan.embedding
    state["player_code_bytes"] = torch.from_numpy(vocabulary_buffer(plan.vocabulary))
    with torch.device("meta"):
        model = ActionSequenceTransformer(config, plan.vocabulary)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.player_embedding.weight.requires_grad_(True)
    return model


def updated_checkpoint(
    raw: Mapping[str, object],
    *,
    vocabulary: PlayerVocabulary,
    embedding: Tensor,
    record: Mapping[str, object],
) -> dict[str, object]:
    """Return the base checkpoint with only the identity table and its hashes changed."""
    if not all(isinstance(raw.get(name), dict) for name in ("model", "cfg", "provenance")):
        raise ValueError("base checkpoint is missing model, cfg, or provenance")
    model = cast(dict[str, object], raw["model"])
    cfg = cast(dict[str, object], raw["cfg"])
    provenance = cast(dict[str, object], raw["provenance"])
    if "player_finetune" in provenance:
        raise ValueError("base checkpoint is already a player fine-tune; start from the released policy")
    weight = model["player_embedding.weight"]
    if not isinstance(weight, Tensor) or embedding.shape != (vocabulary.size, weight.shape[1]):
        raise ValueError("fitted embedding does not match the vocabulary")
    return {
        **raw,
        "model": {
            **model,
            "player_embedding.weight": embedding.detach().to(device="cpu", dtype=weight.dtype).contiguous(),
            "player_code_bytes": torch.from_numpy(vocabulary_buffer(vocabulary)),
        },
        "cfg": {**cfg, "player_vocab_size": vocabulary.size, "player_vocab_sha256": vocabulary.sha256},
        "provenance": {**provenance, "player_vocab_sha256": vocabulary.sha256, "player_finetune": dict(record)},
    }


def export_player_bundle(
    base: Path,
    output: Path,
    *,
    vocabulary: PlayerVocabulary,
    embedding: Tensor,
    record: Mapping[str, object],
    identity: str,
) -> str:
    """Write a 059 bundle that differs from ``base`` only in its identity table."""
    with extract_policy_bundle(base) as (manifest, root), tempfile.TemporaryDirectory(prefix="hal-062-") as scratch:
        raw = torch.load(root / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError("base checkpoint root must be an object")
        checkpoint = Path(scratch) / "checkpoint.pt"
        torch.save(
            updated_checkpoint(raw, vocabulary=vocabulary, embedding=embedding, record=record),
            checkpoint,
        )
        digest = checkpoint_sha256(checkpoint)
        backend = json.loads((root / manifest.backend_config).read_text())
        backend_path = Path(scratch) / "backend.json"
        backend_path.write_text(
            json.dumps({**backend, "checkpoint_sha256": digest}, sort_keys=True, separators=(",", ":"))
        )
        write_policy_bundle(
            output,
            BundleDescription(
                policy_name=f"{manifest.policy_name} + {identity}",
                backend=manifest.backend,
                backend_version=manifest.backend_version,
                required_observation_fields=manifest.required_observation_fields,
                supported_transport_delays=manifest.supported_transport_delays,
                requires_player_code=manifest.requires_player_code,
                source_sha256=digest,
            ),
            {"backend.json": backend_path, "stats.json": root / "stats.json", "checkpoint.pt": checkpoint},
        )
    artifact = read_action_sequence_artifact(output)
    if identity not in artifact.vocabulary.codes:
        raise RuntimeError("exported bundle does not contain the fitted identity")
    return digest


# ---------------------------------------------------------------------------
# Commands


@dataclass(frozen=True, slots=True)
class PlayerSource:
    replays: Path | None = None
    """Directory searched recursively for .slp files; required with --code."""
    code: str | None = None
    """Netplay connect code that marks the player's port, e.g. ABCD#123."""
    sides: Path | None = None
    """JSON lines of {"path", "port"} naming the player's port per game; for offline games without codes."""
    workers: int = max(1, (os.cpu_count() or 2) - 1)

    def rules(self) -> tuple[SideRule, ...]:
        if (self.code is None) == (self.sides is None):
            raise ValueError("pass exactly one of --source.code or --source.sides")
        if self.sides is not None:
            return read_side_rules(self.sides)
        if self.replays is None or not self.replays.is_dir():
            raise FileNotFoundError(f"--source.replays must be a directory with --source.code, got {self.replays}")
        return code_side_rules(self.replays, cast(str, self.code))


@dataclass(frozen=True)
class ScanArgs:
    base: Path
    source: PlayerSource
    identity: str | None = None
    """Vocabulary key for the player; defaults to the connect code."""


@dataclass(frozen=True)
class TrainArgs:
    base: Path
    output: Path
    source: PlayerSource
    identity: str | None = None
    """Vocabulary key for the player; defaults to the connect code. Use a stable label for offline-only players."""
    init_from: str | None = None
    """'unconditioned' (zero row), PLATINUM/DIAMOND/MASTER, or a vocabulary code. Defaults to the player's existing
    row when present, else 'unconditioned'."""
    steps: int = 1500
    batch_size: int = 16
    learning_rate: float = 1e-2
    validation_fraction: float = 0.1
    validation_windows_per_game: int = 4
    validation_max_windows: int = 512
    validate_every: int = 100
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"


def _identity(identity: str | None, source: PlayerSource) -> str:
    value = identity if identity is not None else source.code
    if value is None:
        raise ValueError("--identity is required with --source.sides")
    return value.strip()


def _git_sha() -> str:
    value = os.environ.get("HAL_GIT_SHA")
    if value is None:
        root = Path(__file__).resolve().parents[1]
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError(f"HAL_GIT_SHA must be a full lowercase Git SHA, got {value!r}")
    return value


def _summarize(kept: Sequence[PlayerReplay], skipped: Sequence[SkippedReplay]) -> dict[str, object]:
    return {
        "games": len(kept),
        "frames": sum(replay.frames for replay in kept),
        "hours": round(sum(replay.frames for replay in kept) / 60 / 3600, 2),
        "characters": dict(Counter(melee.Character(replay.character).name for replay in kept).most_common()),
        "skipped": dict(Counter(item.reason for item in skipped).most_common()),
    }


def scan(args: ScanArgs) -> dict[str, object]:
    artifact = read_action_sequence_artifact(args.base)
    identity = _identity(args.identity, args.source)
    config = artifact.model_config
    kept, skipped = load_player_replays(
        args.source.rules(),
        minimum_frames=config.L_ctx + config.sample_chunk_length,
        workers=args.source.workers,
    )
    report = {
        "identity": identity,
        "identity_in_base_vocabulary": identity in artifact.vocabulary.codes,
        **_summarize(kept, skipped),
    }
    print(json.dumps(report, indent=2))
    return report


def _validation_batches(
    replays: Sequence[PlayerReplay],
    *,
    args: TrainArgs,
    stats: dict[str, FeatureStats],
    context_length: int,
    chunk_length: int,
    player_id: int,
) -> list[tuple[PlayerBatch, Tensor]]:
    windows: list[Window] = []
    for replay in replays:
        for start in window_starts(
            replay,
            context_length=context_length,
            chunk_length=chunk_length,
            count=args.validation_windows_per_game,
            seed=args.seed,
        ):
            windows.append(
                player_window(replay, start=start, length=context_length + chunk_length, player_id=player_id)
            )
            if len(windows) == args.validation_max_windows:
                break
        if len(windows) == args.validation_max_windows:
            break
    generator = torch.Generator().manual_seed(args.seed + 1)
    batches = []
    for first in range(0, len(windows), args.batch_size):
        chunk = windows[first : first + args.batch_size]
        batch = collate_player_batch(
            chunk,
            stats=stats,
            context_length=context_length,
            return_dropout=BASE_RETURN_DROPOUT,
            generator=generator,
        )
        batches.append((batch, sample_prefix_positions(len(chunk), context_length, generator)))
    return batches


def with_player_id(batch: PlayerBatch, player_id: int) -> PlayerBatch:
    features = dict(batch.train.context.features)
    features["ego_player_id"] = torch.full_like(features["ego_player_id"], player_id)
    context = replace(batch.train.context, features=features)
    return replace(batch, train=replace(batch.train, context=context))


@torch.no_grad()
def evaluate(
    model: ActionSequenceTransformer,
    batches: Sequence[tuple[PlayerBatch, Tensor]],
    *,
    player_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    total = 0.0
    bits = 0.0
    rows = 0
    for batch, positions in batches:
        terms = behavior_cloning_objective(
            model,
            with_player_id(batch, player_id).to(device),
            positions.to(device),
            autocast_dtype=autocast_dtype,
        )
        count = positions.shape[0]
        total += float(terms.objective) * count
        bits += float(terms.next_frame_bits) * count
        rows += count
    return {"objective": total / rows, "next_frame_bits": bits / rows}


def train(args: TrainArgs) -> dict[str, object]:
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if args.output.resolve() == args.base.resolve():
        raise ValueError("output must differ from the base bundle")
    if SCHEMA_VERSION != _BASE_MDS_SCHEMA_VERSION:
        raise RuntimeError(f"extraction schema {SCHEMA_VERSION} differs from the base data schema 7")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    torch.manual_seed(args.seed)

    artifact = read_action_sequence_artifact(args.base)
    identity = _identity(args.identity, args.source)
    config = artifact.model_config
    context_length, chunk_length = config.L_ctx, config.sample_chunk_length
    plan = plan_identity(
        artifact.vocabulary, artifact.model_tensors["player_embedding.weight"], identity, args.init_from
    )
    print(
        f"[identity] {identity!r} -> id {plan.player_id} "
        f"({'inserted' if plan.inserted else 'existing row'}, init from {plan.initialized_from})",
        flush=True,
    )

    kept, skipped = load_player_replays(
        args.source.rules(), minimum_frames=context_length + chunk_length, workers=args.source.workers
    )
    summary = _summarize(kept, skipped)
    print(f"[data] {json.dumps(summary)}", flush=True)
    validation_games = [replay for replay in kept if is_validation_replay(replay.replay_id, args.validation_fraction)]
    training_games = [
        replay for replay in kept if not is_validation_replay(replay.replay_id, args.validation_fraction)
    ]
    if not training_games or not validation_games:
        raise ValueError(
            f"need at least one training and one validation game; got {len(training_games)} and {len(validation_games)}"
        )

    model = build_trainable_model(artifact, plan, device)
    initial_table = model.player_embedding.weight.detach().clone()
    stats = artifact.statistics
    validation = _validation_batches(
        validation_games,
        args=args,
        stats=stats,
        context_length=context_length,
        chunk_length=chunk_length,
        player_id=plan.player_id,
    )
    baseline = {
        "unconditioned": evaluate(
            model, validation, player_id=MASKED_PLAYER_ID, device=device, autocast_dtype=autocast_dtype
        ),
        "initial": evaluate(model, validation, player_id=plan.player_id, device=device, autocast_dtype=autocast_dtype),
    }
    print(f"[val] before: {json.dumps(baseline)}", flush=True)

    optimizer = torch.optim.Adam([model.player_embedding.weight], lr=args.learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        windows = sample_training_windows(
            training_games,
            batch_size=args.batch_size,
            context_length=context_length,
            chunk_length=chunk_length,
            player_id=plan.player_id,
            rng=rng,
        )
        batch = collate_player_batch(
            windows,
            stats=stats,
            context_length=context_length,
            return_dropout=BASE_RETURN_DROPOUT,
            generator=generator,
        ).to(device)
        positions = sample_prefix_positions(args.batch_size, context_length, generator).to(device)
        terms = behavior_cloning_objective(model, batch, positions, autocast_dtype=autocast_dtype)
        optimizer.zero_grad(set_to_none=True)
        terms.objective.backward()
        optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            metrics = evaluate(
                model, validation, player_id=plan.player_id, device=device, autocast_dtype=autocast_dtype
            )
            train_objective = float(terms.objective.detach())
            history.append({"step": float(step), "train_objective": train_objective, **metrics})
            print(f"[step {step}] train={train_objective:.4f} val={json.dumps(metrics)}", flush=True)

    fitted = model.player_embedding.weight.detach().cpu()
    others = torch.ones(fitted.shape[0], dtype=torch.bool)
    others[plan.player_id] = False
    if not torch.equal(fitted[others], initial_table.cpu()[others]):
        raise RuntimeError("an identity row other than the target changed during training")

    final = history[-1]
    record: dict[str, object] = {
        "experiment_id": _EXPERIMENT_ID,
        "git_sha": _git_sha(),
        "base_bundle_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": artifact.checkpoint_sha256,
        "identity": identity,
        "player_id": plan.player_id,
        "inserted": plan.inserted,
        "initialized_from": plan.initialized_from,
        "objective": "059 offset-weighted behavior cloning, unit advantage weights",
        "trainable": "player_embedding row only",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "return_dropout": BASE_RETURN_DROPOUT,
        "data": {**summary, "training_games": len(training_games), "validation_games": len(validation_games)},
        "validation": {
            **baseline,
            "fitted": {"objective": final["objective"], "next_frame_bits": final["next_frame_bits"]},
        },
    }
    digest = export_player_bundle(
        args.base, args.output, vocabulary=plan.vocabulary, embedding=fitted, record=record, identity=identity
    )
    report = {**record, "checkpoint_sha256": digest, "history": history, "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {args.output} (play with --imitate {identity!r})", flush=True)
    return report


type Command = (
    Annotated[ScanArgs, tyro.conf.subcommand(name="scan")] | Annotated[TrainArgs, tyro.conf.subcommand(name="train")]
)


def main(args: Command) -> None:
    if isinstance(args, ScanArgs):
        scan(args)
        return
    train(args)


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
