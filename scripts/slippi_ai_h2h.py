"""Play slippi-ai against 059 through HAL's Dolphin Session and policy adapter."""

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import multiprocessing.connection
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

import melee
import numpy as np
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.controller import validate_controller_action
from hal.eval.matchups import MatchConfig
from hal.eval.matchups import mirrored_configs
from hal.eval.observations import applied_action_from_frame
from hal.eval.observations import flatten_live_frame
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH
from hal.representation.observations import project_observation_columns
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.wire import BUTTON_BITS

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
SLIPPI_AI_COMMIT = "977b7a67d52d77f6183641ce7d86b462f29e18f2"
TIMING = FrameTiming(2, 0, 2, 2, 4)
RUNTIME = RuntimeConfig(1, (2,), replan_interval_frames=2)


def game_config(schedule: str, index: int) -> MatchConfig:
    if index < 0:
        raise ValueError("game index must be non-negative")
    if schedule == "standard":
        return mirrored_configs(index // 2 + 1)[-1]
    if schedule == "fox-fd":
        return MatchConfig(index // 2, melee.Stage.FINAL_DESTINATION, melee.Character.FOX, melee.Character.FOX)
    raise ValueError(f"unknown matchup schedule: {schedule}")


def scheduled_games(schedule: str, offset: int, games: int, seed: int) -> list[dict[str, object]]:
    if offset < 0 or offset % 2 or games < 2 or games % 2:
        raise ValueError("game count must be positive and even; offset must be non-negative and even")
    result = []
    for index in range(offset, offset + games):
        config = game_config(schedule, index)
        result.append(
            {
                "game": index,
                "pair": config.config_id,
                "hal_port": 1 + index % 2,
                "seed": seed + index // 2,
                "stage": config.stage.name,
                "characters_by_port": [config.character_port_1.name, config.character_port_2.name],
            }
        )
    return result


class ObservedSession(Session):
    """Retain the exact libmelee observation for upstream's own parser.

    HAL has no public raw-state hook. This override is tested against the Session
    source hash recorded in each run; remove it if Session adds an observer API.
    """

    latest_gamestate: melee.GameState

    def _step_blocking(self) -> melee.GameState:
        self.latest_gamestate = super()._step_blocking()
        return self.latest_gamestate


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def upstream_observation(value: Any) -> Any:
    """Copy raw state without HAL-only libmelee dataclass identities.

    HAL libmelee 0.47.0+hal.realtime.1 adds CanonicalFrame, which upstream
    libmelee 0.47.3 cannot unpickle. Keep field values and enums unchanged;
    upstream's Parser needs attribute access, not those dataclass identities.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return SimpleNamespace(
            **{
                field.name: upstream_observation(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
            }
        )
    if isinstance(value, dict):
        return {key: upstream_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [upstream_observation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(upstream_observation(item) for item in value)
    return value


def decode_action(value: object) -> ControllerAction:
    """Convert upstream's normalized pad coordinates to HAL's logical values."""
    if not isinstance(value, tuple) or len(value) != 6:
        raise ValueError("invalid slippi-ai controller response")
    if any(not isinstance(number, (int, float)) or isinstance(number, bool) for number in value[:5]):
        raise ValueError("slippi-ai analog values must be numbers")
    x, y, cx, cy, shoulder = cast(tuple[float, float, float, float, float], value[:5])
    buttons = value[5]
    if not isinstance(buttons, tuple) or any(not isinstance(name, str) for name in buttons):
        raise ValueError("invalid slippi-ai buttons")
    if len(set(buttons)) != len(buttons):
        raise ValueError("invalid slippi-ai buttons")
    mask = 0
    for name in cast(tuple[str, ...], buttons):
        if name.lower() not in BUTTON_BITS:
            raise ValueError(f"unsupported slippi-ai button: {name!r}")
        mask |= BUTTON_BITS[name.lower()]
    action = ControllerAction(2 * x - 1, 2 * y - 1, 2 * cx - 1, 2 * cy - 1, shoulder, 0.0, mask)
    validate_controller_action(action)
    return action


def receive(connection: multiprocessing.connection.Connection, timeout: float = 120) -> Any:
    if not connection.poll(timeout):
        raise TimeoutError("slippi-ai worker did not respond")
    return connection.recv()


def finish_game(session: Session) -> int:
    """Let Dolphin write the terminal replay frames and its Game End event."""
    for count in range(1, 181):
        _, live = session.step({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})
        if not live:
            return count
    raise RuntimeError("Dolphin did not leave the finished game within 180 frames")


@contextlib.contextmanager
def worker(args: argparse.Namespace) -> Iterator[multiprocessing.connection.Connection]:
    parent, child = socket.socketpair()
    with parent, child, (args.output / "worker.log").open("w") as log:
        command = [
            str(args.slippi_python),
            str(Path(__file__).with_name("slippi_ai_worker.py")),
            "--checkpoint",
            str(args.slippi_checkpoint),
            "--name",
            args.slippi_name,
            "--fd",
            str(child.fileno()),
        ]
        environment = dict(os.environ, PYTHONPATH=str(args.slippi_source), CUDA_VISIBLE_DEVICES="")
        with subprocess.Popen(command, pass_fds=(child.fileno(),), env=environment, stdout=log, stderr=log) as process:
            child.close()
            with multiprocessing.connection.Connection(parent.detach()) as connection:
                try:
                    yield connection
                finally:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def play_game(
    args: argparse.Namespace,
    policy: ActionSequencePolicy,
    connection: multiprocessing.connection.Connection,
    index: int,
) -> dict[str, object]:
    hal_port = 1 + index % 2
    opponent_port = 3 - hal_port
    seed = args.seed + index // 2
    config = game_config(args.schedule, index)
    connection.send((opponent_port, seed))
    if receive(connection) != "ready":
        raise ValueError("slippi-ai worker did not initialize")
    adapter = PolicyBatchAdapter(
        policy,
        RUNTIME,
        TIMING,
        player_identity=args.hal_identity,
        desired_return=policy.return_p90,
        temperature=1.0,
        observed_actions=True,
    )
    policy.reset_prediction(seed=seed)
    slot = Slot(0, hal_port)
    game_dir = args.output / f"game-{index:02d}"
    game_dir.mkdir()
    characters = {1: config.character_port_1.value, 2: config.character_port_2.value}
    matchup = Matchup(
        stage=config.stage,
        players=(
            PlayerSetup(port=1, character=config.character_port_1, cpu_level=0),
            PlayerSetup(port=2, character=config.character_port_2, cpu_level=0),
        ),
    )
    frames: list[int] = []
    actions: list[np.ndarray] = []
    stocks: list[tuple[int, int]] = []
    percents: list[tuple[float, float]] = []
    pending_rows: list[ObservationRow] = []
    pending_actions: deque[ControllerAction] = deque()
    hal_seconds = slippi_seconds = 0.0
    stopped = "frame_limit"
    started = time.perf_counter()
    with ObservedSession(
        iso_path=ISO_PATH,
        dolphin_path=EMULATOR_PATH,
        slippi_port=args.slippi_port,
        replay_dir=game_dir,
        blocking_input=True,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        disable_audio=True,
        polling_mode=True,
        step_timeout_seconds=30,
        start_timeout_seconds=60,
    ) as session:
        frame = session.start_match(matchup)
        for step in range(args.max_frames):
            raw = session.latest_gamestate
            if int(frame["id"]) != raw.frame or (frames and raw.frame != frames[-1] + 1):
                raise ValueError("HAL and slippi-ai observations lost frame alignment")
            connection.send(upstream_observation(raw))
            at = time.perf_counter()
            opponent_action = decode_action(receive(connection))
            slippi_seconds += time.perf_counter() - at
            at = time.perf_counter()
            flat = project_observation_columns(
                flatten_live_frame(frame, characters), policy.spec.required_observation_fields
            )
            pending_rows.append(
                ObservationRow(
                    raw.frame,
                    flat,
                    controller_to_action_vec(applied_action_from_frame(frame, hal_port)),
                    step == 0,
                )
            )
            if not pending_actions:
                plan = adapter.plan_rows({slot: pending_rows})[slot]
                pending_actions.extend(action_vec_to_controller(action) for action in plan[:2])
                pending_rows.clear()
            hal_action = pending_actions.popleft()
            hal_seconds += time.perf_counter() - at
            frames.append(raw.frame)
            actions.append(np.stack((controller_to_action_vec(hal_action), controller_to_action_vec(opponent_action))))
            stocks.append((raw.players[hal_port].stock, raw.players[opponent_port].stock))
            percents.append((raw.players[hal_port].percent, raw.players[opponent_port].percent))
            frame, in_game = session.step({hal_port: hal_action, opponent_port: opponent_action})
            last = session.latest_gamestate
            if in_game and any(last.players[port].stock == 0 for port in (1, 2)):
                stopped = "stocks"
                break
            if not in_game:
                stopped = "game_end"
                break
            if step and step % 1800 == 0:
                print(
                    f"game {index}: frame {raw.frame}, stocks {stocks[-1]}, {len(frames) / (time.perf_counter() - started):.1f} FPS",
                    flush=True,
                )
        final_state = last if in_game else raw
        final_stocks = tuple(final_state.players[port].stock for port in (hal_port, opponent_port))
        final_percents = tuple(final_state.players[port].percent for port in (hal_port, opponent_port))
        final_frame = final_state.frame
        finish_frames = finish_game(session) if stopped == "stocks" else 0
    connection.send(None)
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        game_dir / "trace.npz",
        frames=frames,
        actions=actions,
        stocks=stocks,
        percents=percents,
        final_frame=final_frame,
        final_stocks=final_stocks,
        final_percents=final_percents,
    )
    winner = None
    if stopped == "stocks" and final_stocks[0] != final_stocks[1]:
        winner = "hal" if final_stocks[0] > final_stocks[1] else "slippi-ai"
    result: dict[str, object] = {
        "schema_version": 1,
        "game": index,
        "hal_port": hal_port,
        "seed": seed,
        "pair": config.config_id,
        "stage": config.stage.name,
        "characters_by_port": [config.character_port_1.name, config.character_port_2.name],
        "stop_reason": stopped,
        "winner": winner,
        "stocks_hal_slippi": final_stocks,
        "percents_hal_slippi": final_percents,
        "frames": len(frames),
        "finish_frames": finish_frames,
        "wall_seconds": elapsed,
        "fps": len(frames) / elapsed,
        "hal_seconds": hal_seconds,
        "slippi_seconds": slippi_seconds,
        "artifacts": {path.name: sha256(path) for path in game_dir.iterdir() if path.is_file()},
    }
    (game_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slippi-python", type=Path, default=ROOT / "outputs/slippi-ai-h2h/venv/bin/python")
    parser.add_argument("--slippi-checkpoint", type=Path, default=ROOT / "outputs/slippi-ai-h2h/all_d21_imitation_v3")
    parser.add_argument("--slippi-source", type=Path, default=Path.home() / "src/slippi-ai")
    parser.add_argument("--slippi-name", default="Master Player")
    parser.add_argument("--hal-bundle", type=Path, default=ROOT / "runs/netplay/o59-vywk3cih.hal")
    parser.add_argument("--hal-identity", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--game-offset", type=int, default=0)
    parser.add_argument("--schedule", choices=("fox-fd", "standard"), default="fox-fd")
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--max-frames", type=int, default=28800)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--slippi-port", type=int, default=51941)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    schedule = scheduled_games(args.schedule, args.game_offset, args.games, args.seed)
    if args.max_frames < 1:
        raise ValueError("frame limit must be positive")
    if args.schedule_only:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
        return
    if importlib.metadata.version("melee") != "0.47.0+hal.realtime.1":
        raise ValueError("the observation bridge requires HAL's pinned libmelee")
    commit = subprocess.check_output(["git", "-C", str(args.slippi_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SLIPPI_AI_COMMIT:
        raise ValueError(f"untested slippi-ai revision: {commit}")
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output / "sources"
    source_dir.mkdir()
    for source in (Path(__file__), Path(__file__).with_name("slippi_ai_worker.py")):
        shutil.copy2(source, source_dir / source.name)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    policy = load_action_sequence_policy(
        args.hal_bundle, device=args.device, seed=args.seed, compiled=args.compiled, kv_cuda_graphs=args.compiled
    )
    if policy.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("expected the pinned 059 cooldown checkpoint")
    policy.prepare_prediction(RUNTIME, 4, 2)
    with worker(args) as connection:
        upstream = receive(connection)
        for index in range(args.game_offset, args.game_offset + args.games):
            config = game_config(args.schedule, index)
            for character in (config.character_port_1, config.character_port_2):
                if character.name not in upstream["allowed_characters"]:
                    raise ValueError(f"slippi-ai does not support {character.name}")
        metadata = {
            "schema_version": 1,
            "hal_git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "slippi_ai_git": commit,
            "arguments": vars(args),
            "upstream": upstream,
            "hal_checkpoint_sha256": policy.checkpoint_sha256,
            "hal_return": policy.return_p90,
            "hal_timing": {"transport": 2, "replan": 2, "prefix": 2, "horizon": 4},
            "slippi_delay": upstream["config"]["policy"]["delay"],
            "schedule": schedule,
            "treatment": "059 cooldown, checkpoint p90 return, KV history",
            "control": f"{upstream['checkpoint_name']}, upstream delayed agent",
            "invariants": "Recorded character/stage schedule, four stocks, temperature 1, mirrored ports; native per-policy delays",
            "emulator_rng": "not seeded; policy seeds do not fix Melee RNG",
            "python": sys.version,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "artifacts": {
                str(p): sha256(p)
                for p in (
                    args.hal_bundle,
                    args.slippi_checkpoint,
                    source_dir / Path(__file__).name,
                    source_dir / "slippi_ai_worker.py",
                    ROOT / "hal/sim/session.py",
                    Path(ISO_PATH),
                    Path(EMULATOR_PATH),
                )
            },
        }
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        results = []
        for index in range(args.game_offset, args.game_offset + args.games):
            results.append(play_game(args, policy, connection, index))
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        connection.send(None)


if __name__ == "__main__":
    main()
