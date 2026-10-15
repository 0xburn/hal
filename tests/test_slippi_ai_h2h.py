import importlib.util
import pickle
import sys
from pathlib import Path
from unittest.mock import Mock

import melee
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import controller_action_wire_values
from hal.eval.matchups import mirrored_configs
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session

_SPEC = importlib.util.spec_from_file_location(
    "hal_slippi_ai_h2h", Path(__file__).parents[1] / "scripts/slippi_ai_h2h.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_WORKER_SPEC = importlib.util.spec_from_file_location(
    "hal_slippi_ai_worker", Path(__file__).parents[1] / "scripts/slippi_ai_worker.py"
)
assert _WORKER_SPEC is not None and _WORKER_SPEC.loader is not None
_WORKER = importlib.util.module_from_spec(_WORKER_SPEC)
_WORKER_SPEC.loader.exec_module(_WORKER)


@pytest.mark.parametrize(
    ("digest", "name"),
    [
        ("4cb632296a5387f4a1a05f0a18cba5528ad50f3e8a343b7ed3d7f192b0ece4b7", "all_d21_imitation_v3"),
        ("194d95f774c8e902fcb7ba4249d1a54207d884cdee0b1a27e25a1fd803120870", "gm-v1"),
    ],
)
def test_worker_identifies_verified_artifacts(digest: str, name: str) -> None:
    assert _WORKER.checkpoint_name(digest) == name


def test_worker_rejects_unknown_artifact() -> None:
    with pytest.raises(ValueError, match="verified"):
        _WORKER.checkpoint_name("0" * 64)


def test_standard_96_games_preserve_hal_schedule_and_swap_ports() -> None:
    schedule = _MODULE.scheduled_games("standard", 0, 96, 20261007)
    configs = mirrored_configs(48)
    assert len(schedule) == 96
    for index, config in enumerate(configs):
        first, second = schedule[2 * index : 2 * index + 2]
        assert (first["hal_port"], second["hal_port"]) == (1, 2)
        assert first["seed"] == second["seed"] == 20261007 + index
        assert first["stage"] == second["stage"] == config.stage.name
        assert (
            first["characters_by_port"]
            == second["characters_by_port"]
            == [
                config.character_port_1.name,
                config.character_port_2.name,
            ]
        )


def test_parallel_schedule_shards_have_no_missing_or_duplicate_games() -> None:
    whole = _MODULE.scheduled_games("standard", 0, 96, 20261007)
    sharded = []
    for offset in range(0, 96, 12):
        sharded.extend(_MODULE.scheduled_games("standard", offset, 12, 20261007))
    assert sharded == whole


@pytest.mark.parametrize(("offset", "games"), [(-2, 2), (1, 2), (0, 1), (0, 0), (0, -2)])
def test_schedule_rejects_partial_pairs(offset: int, games: int) -> None:
    with pytest.raises(ValueError):
        _MODULE.scheduled_games("standard", offset, games, 0)


def test_fox_schedule_keeps_the_original_protocol() -> None:
    config = _MODULE.game_config("fox-fd", 5)
    assert config.config_id == 2
    assert config.stage is melee.Stage.FINAL_DESTINATION
    assert config.character_port_1 is config.character_port_2 is melee.Character.FOX


def test_upstream_controller_preserves_gamecube_wire_values() -> None:
    action = _MODULE.decode_action((1.0, 0.0, 0.5, 0.75, 0.5, ("A", "L")))
    assert controller_action_wire_values(action) == (80, -80, 0, 40, 70, 0, 0x140)
    assert _MODULE.decode_action((0.5, 0.5, 0.5, 0.5, 0.0, ())) == NEUTRAL_CONTROLLER_ACTION


@pytest.mark.parametrize(
    "value",
    [
        (0.5, 0.5, 0.5, 0.5, 0.0, ("START",)),
        (0.5, 0.5, 0.5, 0.5, 0.0, ("A", "A")),
        (float("nan"), 0.5, 0.5, 0.5, 0.0, ()),
        (1.1, 0.5, 0.5, 0.5, 0.0, ()),
        ("0.5", 0.5, 0.5, 0.5, 0.0, ()),
        (0.5,),
    ],
)
def test_upstream_controller_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        _MODULE.decode_action(value)


def test_upstream_snapshot_preserves_parser_fields_without_fork_types() -> None:
    raw = melee.GameState()
    player = melee.PlayerState()
    player.character = melee.Character.FOX
    player.position.x = 12.5
    player.controller_state.main_stick = (0.75, 0.5)
    player.controller_state.processed_button[melee.Button.BUTTON_A] = True
    raw.players = {1: player}
    snapshot = pickle.loads(pickle.dumps(_MODULE.upstream_observation(raw)))
    assert not hasattr(snapshot, "_canonical")
    assert snapshot.stage is raw.stage
    assert snapshot.players[1].character is melee.Character.FOX
    assert snapshot.players[1].position.x == 12.5
    assert snapshot.players[1].controller_state.main_stick == (0.75, 0.5)
    assert snapshot.players[1].controller_state.processed_button[melee.Button.BUTTON_A]


def test_finish_game_waits_for_game_end() -> None:
    session = Mock(spec=Session)
    session.step.side_effect = [({}, True), ({}, True), ({}, False)]
    assert _MODULE.finish_game(session) == 3
    assert session.step.call_count == 3
    session.step.assert_called_with({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})


def test_finish_game_rejects_a_stuck_terminal_scene() -> None:
    session = Mock(spec=Session)
    session.step.return_value = ({}, True)
    with pytest.raises(RuntimeError, match="180 frames"):
        _MODULE.finish_game(session)
    assert session.step.call_count == 180


@pytest.mark.integration
def test_raw_observation_hook_matches_hal_frames(tmp_path: Path) -> None:
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX, cpu_level=0),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=0),
        ),
    )
    with _MODULE.ObservedSession(
        ISO_PATH,
        dolphin_path=EMULATOR_PATH,
        replay_dir=tmp_path,
        blocking_input=True,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        disable_audio=True,
        polling_mode=True,
    ) as session:
        frame = session.start_match(matchup)
        for _ in range(8):
            raw = pickle.loads(pickle.dumps(session.latest_gamestate))
            assert raw.frame == frame["id"]
            assert raw.stage.value == frame["stage"]
            assert raw.to_canonical_dict()["ports"] == frame["ports"]
            frame, live = session.step({1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})
            assert live
