from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "texmex" / "figs"

U_VALUES = (0.0, 0.1, 0.5, 1.0, 2.0)
U_COLORS = {
    0.0: "#0b3c5d",
    0.1: "#5c2a9d",
    0.5: "#b34700",
    1.0: "#2e7d32",
    2.0: "#d17a00",
}

METHODS = (
    (
        "local",
        "Local target",
        "braiding_results_step_projected_braiding_local_U={u}.txt",
    ),
    (
        "matched",
        "Matched local operators",
        "braiding_results_matched_ops_U={u}.txt",
    ),
    (
        "ideal",
        "Ideal-operator baseline",
        "braiding_results_step_projected_braiding_U={u}.txt",
    ),
)

# The result tables print overlaps to ten decimal places. Exact zeros in
# |1 - overlap| therefore mean the deviation is unresolved below this scale.
DISPLAY_FLOOR = 5.0e-11


def read_overlap_deviations(filename_pattern: str) -> dict[float, np.ndarray]:
    deviations = {}
    for u_value in U_VALUES:
        path = DATA_DIR / filename_pattern.format(u=u_value)
        with path.open(newline="") as handle:
            rows = csv.reader(handle, delimiter="\t")
            next(rows)
            overlaps = np.array([float(row[-1]) for row in rows if row], dtype=float)
        deviations[u_value] = np.abs(1.0 - overlaps)
    return deviations


def load_results() -> dict[str, dict[float, np.ndarray]]:
    return {
        method_key: read_overlap_deviations(filename_pattern)
        for method_key, _, filename_pattern in METHODS
    }


def displayed(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, DISPLAY_FLOOR)


def save_figure(fig: plt.Figure, output_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08, dpi=220)
    plt.close(fig)


def format_log_axis(axis: plt.Axes) -> None:
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis="y", which="major", alpha=0.22)


def plot_sector_maps(results: dict[str, dict[float, np.ndarray]]) -> None:
    all_values = np.concatenate(
        [displayed(values) for method_results in results.values() for values in method_results.values()]
    )
    norm = LogNorm(vmin=DISPLAY_FLOOR, vmax=max(2.0, all_values.max()))

    fig, axes = plt.subplots(
        len(METHODS),
        1,
        figsize=(8.2, 8.4),
        sharex=True,
        constrained_layout=True,
    )

    image = None
    for axis, (method_key, method_label, _) in zip(axes, METHODS):
        for row_index, u_value in enumerate(U_VALUES):
            values = displayed(results[method_key][u_value])
            x_edges = np.linspace(0.0, 1.0, len(values) + 1)
            image = axis.pcolormesh(
                x_edges,
                [row_index - 0.42, row_index + 0.42],
                values[np.newaxis, :],
                norm=norm,
                cmap="viridis",
                shading="flat",
            )

        axis.set_yticks(
            range(len(U_VALUES)),
            [fr"$U={u_value:g}$  ($n={len(results[method_key][u_value])}$)" for u_value in U_VALUES],
        )
        axis.set_ylabel(method_label)
        axis.tick_params(axis="both", labelsize=10)
        axis.set_ylim(len(U_VALUES) - 0.5, -0.5)

    axes[-1].set_xlabel("Normalized sector-group position")
    fig.suptitle("Sector-resolved absolute overlap deviation", fontsize=15)
    colorbar = fig.colorbar(image, ax=axes, pad=0.02, aspect=35)
    colorbar.set_label(r"Absolute overlap deviation $|1-\mathcal{O}|$")
    colorbar.set_ticks([1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    save_figure(fig, "step_projection_sector_comparison")


def plot_distribution_summary(results: dict[str, dict[float, np.ndarray]]) -> None:
    fig, axes = plt.subplots(
        len(METHODS),
        1,
        figsize=(7.4, 8.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    rng = np.random.default_rng(7)

    for axis, (method_key, method_label, _) in zip(axes, METHODS):
        for position, u_value in enumerate(U_VALUES):
            raw_values = results[method_key][u_value]
            values = displayed(raw_values)
            jitter = rng.uniform(-0.12, 0.12, len(values))
            unresolved = raw_values < DISPLAY_FLOOR
            resolved = ~unresolved

            axis.scatter(
                np.full(resolved.sum(), position) + jitter[resolved],
                values[resolved],
                color=U_COLORS[u_value],
                alpha=0.55,
                s=25,
                linewidths=0,
                zorder=2,
            )
            if unresolved.any():
                axis.scatter(
                    np.full(unresolved.sum(), position) + jitter[unresolved],
                    values[unresolved],
                    marker="v",
                    facecolors="none",
                    edgecolors=U_COLORS[u_value],
                    s=38,
                    linewidths=1.0,
                    zorder=2,
                )

            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            axis.vlines(position, q1, q3, color="#222222", linewidth=4.0, zorder=3)
            axis.hlines(
                median,
                position - 0.18,
                position + 0.18,
                color="#222222",
                linewidth=2.0,
                zorder=4,
            )

        format_log_axis(axis)
        axis.set_ylabel(method_label)
        axis.set_xlim(-0.45, len(U_VALUES) - 0.55)

    axes[-1].set_xticks(range(len(U_VALUES)), [fr"$U={u_value:g}$" for u_value in U_VALUES])
    axes[-1].set_xlabel("Interaction strength")
    fig.supylabel(r"Absolute overlap deviation $|1-\mathcal{O}|$")
    fig.suptitle("Sector-to-sector deviation distributions", fontsize=15)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#222222", linewidth=4.0, label="Interquartile range"),
            Line2D([0], [0], color="#222222", linewidth=2.0, label="Median"),
            Line2D(
                [0],
                [0],
                marker="v",
                color="none",
                markeredgecolor="#555555",
                label=r"Unresolved; shown at $5\times10^{-11}$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    # save_figure(fig, "step_projection_distribution_comparison")


def main() -> None:
    results = load_results()
    plot_sector_maps(results)
    plot_distribution_summary(results)
    print(f"Saved comparison figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
