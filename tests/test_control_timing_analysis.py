import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hal.eval.cross_stage import MatchRow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.analyze_control_timing import boot_totals
    from scripts.analyze_control_timing import paired_rates
finally:
    sys.path.pop(0)


def row(boot: int = 0) -> MatchRow:
    return MatchRow(1, 2, 31, boot, 0, 3600, 3723, 120, 60, 2, 1)


def test_boot_clustering_preserves_all_games_and_pooled_denominator() -> None:
    rows = [row(), replace(row(), match_ordinal=1), replace(row(1), active_frames=7200, total_frames=7446)]
    totals = boot_totals(rows, ((1, 2), (1, 2)), 7446)
    points, draws = paired_rates(np.stack((totals, totals)))
    assert points[0, -1] == pytest.approx(0.75)
    np.testing.assert_array_equal(draws[0], draws[1])
    assert draws.shape == (2, 2000, 5)
    # An individual game's resampling would produce more than three outcomes.
    assert set(draws[0, :, -1]) == {0.5, 0.75, 1.0}


def test_paired_boot_draws_cancel_a_known_constant_rate_effect() -> None:
    totals = np.array([[[2, 1, 100, 50, 3600, 3723], [4, 1, 120, 20, 7200, 7446]]], dtype=float)
    treatment = totals.copy()
    treatment[:, :, 0] += treatment[:, :, 4] / 3600
    points, draws = paired_rates(np.concatenate((totals, treatment)))
    assert points[1, -1] - points[0, -1] == pytest.approx(1)
    np.testing.assert_allclose(draws[1, :, -1] - draws[0, :, -1], 1)
    np.testing.assert_array_equal(draws, paired_rates(np.concatenate((totals, treatment)))[1])


@pytest.mark.parametrize(
    "rows",
    (
        [row(), row()],
        [replace(row(), total_frames=1)],
        [replace(row(), opp_character=3)],
        [replace(row(), active_frames=0)],
        [replace(row(), match_ordinal=1)],
    ),
)
def test_reject_invalid_or_incomplete_boots(rows: list[MatchRow]) -> None:
    with pytest.raises(ValueError):
        boot_totals(rows, ((1, 2),), 3723)
