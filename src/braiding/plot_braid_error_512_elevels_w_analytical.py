from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator


ERROR_KEY = "ideal_target_gate_error_normalized"
REPO_ROOT = Path(__file__).resolve().parents[2]
REMAKE4_PATH = Path(__file__).with_name(
    "retry2_remake4_diagonal_minus_dims_8_32_56_80_256_512.txt"
)
ANALYTICAL_PATH = Path(__file__).with_name(
    "retry2_analytical_sweetspot_block_complete_dims_8_56_176_336_456_512.txt"
)
OUTPUT_PATH = REPO_ROOT / "texmex" / "figs" / "braid_error_512_elevels_w_analytical"

U_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    0.1: {"color": "#ff7f0e", "marker": "s"},
    2.0: {"color": "#d62728", "marker": "D"},
}
U_ORDER = (0.0, 0.1, 2.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def grouped_by_u(rows: list[dict[str, str]]) -> dict[float, list[dict[str, str]]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["interaction_u"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["projection_level"]))
    return grouped


def plot_u_series(axis, rows: list[dict[str, str]], u_value: float) -> None:
    style = U_STYLES[u_value]
    axis.plot(
        [int(row["projection_level"]) for row in rows],
        [float(row[ERROR_KEY]) for row in rows],
        color=style["color"],
        marker=style["marker"],
        linestyle="-",
        linewidth=2.0,
        markersize=6.0,
        label=fr"state-made, $U={u_value:g}$",
    )


def plot_analytical_series(axis, rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda row: int(row["projection_level"]))
    axis.plot(
        [int(row["projection_level"]) for row in rows],
        [float(row[ERROR_KEY]) for row in rows],
        color="#222222",
        marker="^",
        linestyle="--",
        linewidth=2.0,
        markersize=6.5,
        label="analytical sweet spot",
    )


def main() -> None:
    remake4_rows = read_rows(REMAKE4_PATH)
    analytical_rows = read_rows(ANALYTICAL_PATH)
    grouped = grouped_by_u(remake4_rows)

    fig, axis = plt.subplots(figsize=(6.4, 4.3))
    for u_value in U_ORDER:
        if u_value in grouped:
            plot_u_series(axis, grouped[u_value], u_value)
    plot_analytical_series(axis, analytical_rows)

    levels = sorted(
        {
            int(row["projection_level"])
            for rows in (*grouped.values(), analytical_rows)
            for row in rows
        }
    )
    axis.set_title(r"State-made braid error vs. state target")
    axis.set_xlabel(r"Projection dimension $\dim P$")
    axis.set_ylabel("Normalized Error")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(FixedLocator(levels))
    axis.xaxis.set_major_formatter(FixedFormatter([str(level) for level in levels]))
    axis.tick_params(axis="x", rotation=30)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_PATH.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08, dpi=220)
    plt.close(fig)

    print("dim\tanalytical")
    for row in sorted(analytical_rows, key=lambda item: int(item["projection_level"])):
        print(f"{int(row['projection_level'])}\t{float(row[ERROR_KEY]):.8e}")


if __name__ == "__main__":
    main()
