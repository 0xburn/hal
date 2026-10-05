"""Paired boot-cluster estimates and publication figures for the O59 timing grid."""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import tyro

from hal.eval.cross_stage import MatchRow
from hal.eval.matchups import matchups_for_vs_cpu
from scripts.control_timing import CHECKPOINT_SHA256
from scripts.control_timing import DELAYS
from scripts.control_timing import INTERVALS
from scripts.control_timing import Cell
from scripts.control_timing import grid
from scripts.control_timing import save_json

METRICS = (
    "stocks_taken_per_min",
    "stocks_lost_per_min",
    "damage_dealt_per_min",
    "damage_taken_per_min",
    "net_stock_per_min",
)


@dataclass(frozen=True, slots=True)
class Args:
    output: Path = Path("runs/o59-control-timing")


def boot_totals(rows: Sequence[MatchRow], schedule: Sequence[tuple[int, int]], frames: int) -> np.ndarray:
    totals = np.zeros((len(schedule), 6), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if not 0 <= row.boot_index < len(schedule):
            raise ValueError("unexpected boot index")
        if (row.ego_character, row.opp_character) != schedule[row.boot_index]:
            raise ValueError("matchup blocks differ")
        key = (row.boot_index, row.match_ordinal)
        if key in seen:
            raise ValueError("duplicate match row")
        seen.add(key)
        values = (
            row.stocks_taken,
            row.stocks_lost,
            row.damage_dealt,
            row.damage_taken,
            row.active_frames,
            row.total_frames,
        )
        if any(not np.isfinite(value) or value < 0 for value in values) or row.active_frames > row.total_frames:
            raise ValueError("invalid match counts")
        totals[row.boot_index, 5] += row.total_frames
        if row.active_frames:
            totals[row.boot_index, :5] += values[:5]
    if not np.all(totals[:, 5] == frames) or not np.all(totals[:, 4] > 0):
        raise ValueError("each boot must have its complete frame budget and active play")
    for boot in range(len(schedule)):
        ordinals = sorted(ordinal for index, ordinal in seen if index == boot)
        if ordinals != list(range(len(ordinals))):
            raise ValueError("match ordinals must be consecutive")
    return totals


def paired_rates(totals: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pool counts within resampled boots; reuse each draw across all cells."""
    if totals.ndim != 3 or totals.shape[2] != 6 or resamples < 1:
        raise ValueError("expected cells x boots x six counts and positive resamples")
    if not np.isfinite(totals).all() or np.any(totals < 0) or np.any(totals[:, :, 4] <= 0):
        raise ValueError("boot counts and active denominators must be valid")
    indices = np.random.default_rng(seed).integers(0, totals.shape[1], size=(resamples, totals.shape[1]))
    pooled = totals.sum(axis=1)
    sampled = totals[:, indices, :].sum(axis=2)
    point = pooled[:, :4] * 3600 / pooled[:, 4, None]
    draws = sampled[:, :, :4] * 3600 / sampled[:, :, 4, None]
    return np.concatenate((point, (point[:, 0] - point[:, 1])[:, None]), axis=-1), np.concatenate(
        (draws, (draws[:, :, 0] - draws[:, :, 1])[:, :, None]), axis=-1
    )


def interval(draws: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(lower), float(upper)


def neighbors(cell: Cell) -> tuple[Cell, ...]:
    d, r = DELAYS.index(cell.delay), INTERVALS.index(cell.replan)
    return tuple(
        Cell(DELAYS[i], INTERVALS[j])
        for i, j in ((d - 1, r), (d + 1, r), (d, r - 1), (d, r + 1))
        if 0 <= i < len(DELAYS) and 0 <= j < len(INTERVALS)
    )


def analyze(output: Path) -> None:
    cells = grid()
    schedule = tuple((int(a.value), int(b.value)) for a, b in matchups_for_vs_cpu(96))
    totals = []
    raw_rows: list[dict[str, int | float]] = []
    invariant: dict[str, object] | None = None
    for cell in cells:
        root = output / "full" / cell.name
        protocol = json.loads((root / "protocol.json").read_text())
        result = json.loads((root / "results.json").read_text())
        if protocol["schema_version"] != 1 or result["schema_version"] != 1 or not result["complete"]:
            raise ValueError("invalid or incomplete cell artifact")
        if protocol["checkpoint_sha256"] != CHECKPOINT_SHA256 or protocol["cell"] != {
            "delay": cell.delay,
            "replan": cell.replan,
        }:
            raise ValueError("checkpoint or timing treatment differs")
        if (protocol["args"]["boots"], protocol["args"]["frames"]) != (96, 7200):
            raise ValueError("not the full 96-boot protocol")
        selected = {
            key: protocol[key]
            for key in (
                "base_git_sha",
                "source_manifest_sha256",
                "checkpoint_sha256",
                "model_config",
                "backend",
                "history_mode",
                "kv_update_frames",
                "observed_actions",
                "player_identity",
                "desired_return",
                "temperature",
                "decoded_horizon",
                "matchup_schedule",
                "sampling_seed",
                "bootstrap",
                "environment",
            )
        }
        selected["parallel"] = protocol["args"]["parallel"]
        if invariant is not None and invariant != selected:
            raise ValueError(f"invariant inputs changed at {cell.name}")
        invariant = selected
        rows = [MatchRow.from_dict(row) for row in result["rows"]]
        totals.append(boot_totals(rows, schedule, 7200))
        raw_rows.extend({"delay": cell.delay, "replan": cell.replan, **row.as_dict()} for row in rows)
    points, draws = paired_rates(np.stack(totals))
    baseline = cells.index(Cell(0, 1))
    records: list[dict[str, object]] = []
    for i, cell in enumerate(cells):
        record: dict[str, object] = {"delay": cell.delay, "replan": cell.replan, "production": cell == Cell(3, 4)}
        for j, name in enumerate(METRICS):
            lower, upper = interval(draws[i, :, j])
            record.update({name: float(points[i, j]), name + "_lo": lower, name + "_hi": upper})
        delta = draws[i, :, -1] - draws[baseline, :, -1]
        lower, upper = interval(delta)
        record.update(delta_nsm=float(points[i, -1] - points[baseline, -1]), delta_nsm_lo=lower, delta_nsm_hi=upper)
        overlapping = []
        for neighbor in neighbors(cell):
            k = cells.index(neighbor)
            a, b = interval(draws[i, :, -1]), interval(draws[k, :, -1])
            if max(a[0], b[0]) <= min(a[1], b[1]):
                overlapping.append(neighbor.name)
        record["overlapping_neighbors"] = ";".join(overlapping)
        records.append(record)
    costs: list[dict[str, object]] = []
    for name, curve, values in (
        ("delay_at_R2", [Cell(d, 2) for d in DELAYS], DELAYS),
        ("open_loop_at_d2", [Cell(2, r) for r in INTERVALS], INTERVALS),
        ("delay_at_production_R4", [Cell(d, 4) for d in DELAYS], DELAYS),
    ):
        pairs = [(j, j + 1) for j in range(len(curve) - 1)] + [(0, len(curve) - 1)]
        for start, stop in pairs:
            i, k = cells.index(curve[start]), cells.index(curve[stop])
            width = values[stop] - values[start]
            lower, upper = interval((draws[i, :, -1] - draws[k, :, -1]) / width)
            costs.append(
                {
                    "curve": name,
                    "from": values[start],
                    "to": values[stop],
                    "nsm_cost_per_frame": float((points[i, -1] - points[k, -1]) / width),
                    "lo": lower,
                    "hi": upper,
                }
            )
    save_json(
        output / "summary.json", {"schema_version": 1, "cells": records, "costs": costs, "invariants": invariant}
    )
    for filename, rows in (("summary.csv", records), ("cost_per_frame.csv", costs), ("matches.csv", raw_rows)):
        with (output / filename).open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot(output, cells, points, draws)
    report(output, records, costs)


def plot(output: Path, cells: tuple[Cell, ...], points: np.ndarray, draws: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = points[:, -1].reshape(len(DELAYS), len(INTERVALS))
    figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
    bound = max(abs(float(values.min())), abs(float(values.max())))
    heatmap = axis.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    for i, d in enumerate(DELAYS):
        for j, r in enumerate(INTERVALS):
            k = cells.index(Cell(d, r))
            lo, hi = interval(draws[k, :, -1])
            production = "\nproduction" if (d, r) == (3, 4) else ""
            axis.text(
                j,
                i,
                f"{values[i, j]:.2f}\n[{lo:.2f}, {hi:.2f}]{production}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(values[i, j]) > bound * 0.65 else "black",
            )
    axis.set(
        xticks=range(len(INTERVALS)),
        xticklabels=INTERVALS,
        yticks=range(len(DELAYS)),
        yticklabels=DELAYS,
        xlabel="Replan interval R (frames)",
        ylabel="Control delay d (frames)",
        title="O59: net stocks per active minute (95% boot intervals)",
    )
    figure.colorbar(heatmap, ax=axis, label="Net stocks per minute")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"heatmap.{extension}", dpi=200)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, curve, x, title, xlabel in (
        (axes[0], [Cell(d, 2) for d in DELAYS], DELAYS, "Delay at R = 2", "Delay d (frames)"),
        (
            axes[1],
            [Cell(2, r) for r in INTERVALS],
            INTERVALS,
            "Open-loop interval at d = 2",
            "Replan interval R (frames)",
        ),
    ):
        indices = [cells.index(cell) for cell in curve]
        y = points[indices, -1]
        bounds = np.array([interval(draws[i, :, -1]) for i in indices])
        axis.plot(x, y, "o-")
        axis.fill_between(x, bounds[:, 0], bounds[:, 1], alpha=0.2)
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set(title=title, xlabel=xlabel, ylabel="Net stocks per active minute", xticks=x)
        axis.grid(alpha=0.2)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"marginals.{extension}", dpi=200)
    plt.close(figure)


def report(output: Path, records: list[dict[str, object]], costs: list[dict[str, object]]) -> None:
    lines = [
        "# O59 control timing",
        "",
        "Fixed checkpoint vywk3cih, final update 131072. All 30 cells use the same 96 prior-weighted matchup boots, 7,200 emulator frames each, versus level-9 CPU. The treatment varies delay d and replan interval R independently. Weights, decoder, return target, temperature, identity masking, matchup schedule, sampling seeds, and evaluator settings are fixed. Committed-action forcing is on.",
        "",
        "Checkpoint SHA-256: `" + CHECKPOINT_SHA256 + "`.",
        "",
        "The training chunk length is 28; trained offsets are 1–12, 16, 20, 24, 28. Every cell decodes the same contiguous 12 offsets. The named production timing cell is d=3, R=4 (two Slippi frames plus one inference frame). Its current deployed horizon is 8; the grid keeps horizon 12 fixed. Histories use the current KV-cache policy, observed actions, masked player identity, temperature 1, and the checkpoint p90 return target.",
        "",
        "Rates pool event counts and active frame counts (frame_id >= 0). Intervals are pointwise 95% percentile intervals from 2,000 boot-cluster resamples, seed 0. Each resample uses the same 96 boot indices in every cell. Differences subtract the two pooled rates within each paired resample. This pairs character-matchup blocks; Dolphin game RNG and restart stages are not controlled. There is no multiple-comparison adjustment.",
        "",
        "| d | R | NSM [95% interval] | Difference from (0,1) [95% interval] | Overlapping neighbor intervals |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in records:

        def value(key: str, current: dict[str, object] = row) -> float:
            return cast(float, current[key])

        name = f"{row['replan']}" + (" (production)" if row["production"] else "")
        lines.append(
            f"| {row['delay']} | {name} | {value('net_stock_per_min'):.3f} [{value('net_stock_per_min_lo'):.3f}, {value('net_stock_per_min_hi'):.3f}] | {value('delta_nsm'):+.3f} [{value('delta_nsm_lo'):+.3f}, {value('delta_nsm_hi'):+.3f}] | {row['overlapping_neighbors']} |"
        )
    lines += [
        "",
        "Neighbors are adjacent entries on either grid axis. Marginal interval overlap is descriptive; it is not a test of their paired difference.",
        "",
        "A positive cost means worse play. These endpoint averages do not assume a constant cost at every frame:",
        "",
    ]
    for row in costs:
        if row["from"] == (1 if row["curve"] == "open_loop_at_d2" else 0) and row["to"] == 6:
            lines.append(
                f"- {row['curve']}: {cast(float, row['nsm_cost_per_frame']):.3f} NSM per added frame, 95% interval [{cast(float, row['lo']):.3f}, {cast(float, row['hi']):.3f}]."
            )
    lines += [
        "",
        "`cost_per_frame.csv` gives every adjacent finite difference, divided by the actual frame increment. `summary.csv` also gives stocks taken/lost and damage dealt/taken per active minute. `matches.csv` contains all raw match rows; its stage field is the boot seed stage. Per-cell protocol, results, replay hashes, and Modal runtime are under `full/`. Replays are preserved at each cell's protocol.args.run_name R2 prefix. The three pilot cells use the complete protocol and are reused in the grid.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    analyze(tyro.cli(Args).output)
