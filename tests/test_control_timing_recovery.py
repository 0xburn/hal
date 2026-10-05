import sys
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hal.eval.cross_stage import MatchRow
from hal.eval.harness import SessionConfig
from hal.eval.matchups import matchups_for_vs_cpu
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig
from hal.sim.rollout import ChunkPolicy
from hal.sim.rollout import VecMatch
from hal.sim.trajectory import Trajectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.control_timing import Cell
    from scripts.control_timing import PolicyFactory
    from scripts.recover_control_timing import evaluate_wave
    from scripts.recover_control_timing import merge_wave
    from scripts.recover_control_timing import missing_waves
finally:
    sys.path.pop(0)


def row(boot: int, taken: int = 1) -> MatchRow:
    return MatchRow(1, 2, 31, boot, 0, 7077, 7200, 120, 60, taken, 1)


def test_recovery_replaces_the_whole_wave_and_preserves_other_waves() -> None:
    schedule = [(1, 2)] * 16
    original = [row(i) for i in range(16) if i != 11]
    assert missing_waves(original, schedule) == (1,)
    repaired = merge_wave(original, [row(i, taken=2) for i in range(8)], 1, schedule)
    assert missing_waves(repaired, schedule) == ()
    assert repaired[:8] == original[:8]
    assert [item.boot_index for item in repaired] == list(range(16))
    assert all(item.stocks_taken == 2 for item in repaired[8:])


@pytest.mark.parametrize("replacement", ([row(i) for i in range(7)], [row(0)] * 8, [row(i + 8) for i in range(8)]))
def test_recovery_rejects_missing_duplicate_or_renumbered_slots(replacement: list[MatchRow]) -> None:
    with pytest.raises(ValueError):
        merge_wave([], replacement, 1, [(1, 2)] * 16)


@pytest.mark.parametrize("bad", (replace(row(0), total_frames=7100), replace(row(0), opp_character=3)))
def test_recovery_rejects_corruption_instead_of_treating_it_as_an_empty_boot(bad: MatchRow) -> None:
    with pytest.raises(ValueError):
        missing_waves([bad], [(1, 2)] * 8)


class SeedPolicy:
    return_p90 = 20.0

    def __init__(self) -> None:
        self.seed = 0
        self.rng = np.random.default_rng(0)

    def reset_prediction(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)


def test_repeated_recovery_keeps_original_wave_seed_matchups_and_slots(tmp_path: Path) -> None:
    policy = SeedPolicy()
    typed_policy = cast(ActionSequencePolicy, policy)
    cell = Cell(6, 6)
    runtime = RuntimeConfig(8, (0,), replan_interval_frames=6)
    original_factory = PolicyFactory(typed_policy, runtime, cell)
    for _ in range(6):
        original_factory()
    expected = policy.rng.random((8, 12))
    expected_pairs = matchups_for_vs_cpu(96)[40:48]
    seen: list[np.ndarray] = []

    def driver(
        session_cfg: SessionConfig,
        matches: Sequence[VecMatch],
        policy_factory: Callable[[], ChunkPolicy],
        *,
        max_frames: int,
        max_parallel: int,
        start_retries: int,
    ) -> list[list[Trajectory]]:
        assert (max_frames, max_parallel, start_retries) == (7200, 8, 0)
        assert session_cfg.instant_match_restart
        assert [
            (match.matchup.players[0].character, match.matchup.players[1].character) for match in matches
        ] == expected_pairs
        assert all(match.model_ports == (1,) and match.matchup.players[1].cpu_level == 9 for match in matches)
        policy_factory()
        seen.append(policy.rng.random((8, 12)))
        return [[] for _ in matches]

    for attempt in range(2):
        evaluate_wave(typed_policy, runtime, cell, 5, tmp_path / str(attempt), driver=driver)
    for values in seen:
        np.testing.assert_array_equal(values, expected)
