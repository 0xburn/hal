"""Plot the paired Cody/Fox CPU evaluation at the pooled and opponent levels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DISPLAY_NAMES = {
    "CPTFALCON": "Captain Falcon",
    "DOC": "Dr. Mario",
    "GAMEANDWATCH": "Mr. Game & Watch",
    "POPO": "Ice Climbers",
    "YLINK": "Young Link",
}


def main() -> None:
    report = json.loads((ROOT / "comparison.json").read_text())
    characters = sorted(report["per_character"], key=lambda row: row["delta_net_stock_per_min"])
    names = [DISPLAY_NAMES.get(row["opponent"], row["opponent"].title()) for row in characters]
    values = np.array([row["delta_net_stock_per_min"] for row in characters], dtype=float)
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.13, top=0.91)
    colors = np.where(values >= 0, "#2a6f97", "#c27034")
    ax.barh(y, values, color=colors, height=0.67)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, names)
    ax.set_xlabel("Fixed 120 − p90 net stocks per active minute")
    ax.set_xlim(-1.18, 1.31)
    ax.set_ylim(-0.8, len(names) + 1.6)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    for index, value in enumerate(values):
        ax.text(
            value + (0.035 if value >= 0 else -0.035),
            index,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    point = report["pooled_net_stock_delta_from_rows"]
    low, high = report["paired_character_stratified_bootstrap_95_ci"]
    pooled_y = len(names) + 0.6
    ax.errorbar(
        point,
        pooled_y,
        xerr=np.array([[point - low], [high - point]]),
        fmt="D",
        markersize=5,
        capsize=4,
        color="#202020",
    )
    ax.text(-1.16, pooled_y, "All 96 boots", va="center", fontweight="bold")
    ax.text(high + 0.04, pooled_y, f"{point:+.2f}  [{low:+.2f}, {high:+.2f}]", va="center", fontsize=9)
    ax.axhline(len(names) - 0.4, color="#bbbbbb", linewidth=0.8)
    ax.set_title("Cody-conditioned Fox vs level-9 CPU: fixed 120 compared with p90", loc="left")
    fig.text(
        0.5,
        0.035,
        "Four boots per opponent; individual bars are exploratory. Pooled interval resamples matched boots within each opponent.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(ROOT / "comparison.png", dpi=180)
    fig.savefig(ROOT / "comparison.pdf")


if __name__ == "__main__":
    main()
