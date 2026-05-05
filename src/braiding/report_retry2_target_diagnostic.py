from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

INT_FIELDS = {"projection_level", "transport_dim", "component_levels"}
U_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    0.1: {"color": "#ff7f0e", "marker": "s"},
    2.0: {"color": "#d62728", "marker": "D"},
}
U_ORDER = (0.0, 0.1, 2.0)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate thesis plots for the retry2 target diagnostic."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("retry2_hypothesis_diagnostic_dims_8_32_56_80.txt"),
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=REPO_ROOT / "texmex" / "figs",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=REPO_ROOT / "texmex" / "generated",
    )
    parser.add_argument(
        "--stem-suffix",
        default="_retry2_target_diagnostic",
    )
    return parser


def parse_results(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                if key in INT_FIELDS:
                    row[key] = int(value)
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def grouped_by_u(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["interaction_u"], []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row["projection_level"])
    return grouped


def ordered_u_values(grouped):
    known = [u for u in U_ORDER if u in grouped]
    extras = sorted(u for u in grouped if u not in U_ORDER)
    return known + extras


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def apply_suffix(path: Path, suffix: str):
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def plot_series(axis, rows, key, style, label, linestyle="-", markerface=None):
    axis.plot(
        [row["projection_level"] for row in rows],
        [row[key] for row in rows],
        color=style["color"],
        marker=style["marker"],
        linestyle=linestyle,
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor=style["color"] if markerface is None else markerface,
        markeredgecolor=style["color"],
        markeredgewidth=1.1,
        label=label,
    )


def plot_derived_series(axis, rows, value_getter, style, label, linestyle="-", markerface=None):
    axis.plot(
        [row["projection_level"] for row in rows],
        [value_getter(row) for row in rows],
        color=style["color"],
        marker=style["marker"],
        linestyle=linestyle,
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor=style["color"] if markerface is None else markerface,
        markeredgecolor=style["color"],
        markeredgewidth=1.1,
        label=label,
    )


def format_projection_axis(axis, levels):
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(levels))
    axis.xaxis.set_major_formatter(FixedFormatter([str(level) for level in levels]))
    axis.minorticks_off()


def style_for_u(u_value):
    return U_STYLES.get(u_value, {"color": "#222222", "marker": "o"})


def add_u_legend(fig, u_values, extra_handles=(), ncol=None):
    handles = [
        Line2D(
            [0],
            [0],
            color=style_for_u(u_value)["color"],
            marker=style_for_u(u_value)["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=fr"$U={u_value:g}$",
        )
        for u_value in u_values
    ]
    handles.extend(extra_handles)
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        ncol=ncol or max(3, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )


def format_metric(value):
    return f"{value:.2e}"


def format_u(value):
    return f"{value:g}"


def write_summary_table(rows, output_path: Path):
    lines = [
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        (
            r"$U$ & $\dim P$ & $\dim T$ & Phys.\ vs.\ $\gamma$ target & "
            r"Phys.\ vs.\ $\Gamma^P$ target & Target mismatch & $\Gamma^P$ algebra \\"
        ),
        r"\hline",
    ]

    previous_u = None
    for row in sorted(rows, key=lambda item: (item["interaction_u"], item["projection_level"])):
        current_u = row["interaction_u"]
        if previous_u is not None and current_u != previous_u:
            lines.append(r"\hline")
        lines.append(
            " & ".join(
                [
                    format_u(current_u),
                    str(row["projection_level"]),
                    str(row["transport_dim"]),
                    format_metric(row["physical_target_gate_error_in_ideal_basis_normalized"]),
                    format_metric(row["physical_target_gate_error_against_physical_target_in_ideal_basis_normalized"]),
                    format_metric(row["ideal_vs_physical_target_gate_error_in_ideal_basis_normalized"]),
                    format_metric(row["physical_majorana_algebra_max_square_error_normalized"]),
                ]
            )
            + r" \\"
        )
        previous_u = current_u

    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_target_comparison(grouped, output_path: Path):
    u_values = ordered_u_values(grouped)
    levels = [row["projection_level"] for row in grouped[u_values[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), sharex=True)

    left_specs = (
        (
            "physical_target_gate_error_in_ideal_basis_normalized",
            r"Physical braid vs. state target $\gamma_2\gamma_3$",
            "-",
            None,
        ),
        (
            "physical_target_gate_error_against_physical_target_in_ideal_basis_normalized",
            r"Physical braid vs. projected analytical target $\Gamma_2^P\Gamma_3^P$",
            "--",
            "white",
        ),
    )
    for u_value in u_values:
        rows = grouped[u_value]
        style = style_for_u(u_value)
        for key, label, linestyle, markerface in left_specs:
            plot_series(
                axes[0],
                rows,
                key,
                style,
                f"_nolegend_{u_value}_{key}",
                linestyle=linestyle,
                markerface=markerface,
            )

    for u_value in u_values:
        rows = grouped[u_value]
        style = style_for_u(u_value)
        plot_series(
            axes[1],
            rows,
            "ideal_vs_physical_target_gate_error_in_ideal_basis_normalized",
            style,
            f"_nolegend_{u_value}_target_mismatch",
        )

    axes[0].set_title("Target used to score the physical braid")
    axes[1].set_title(r"State target vs. analytical $\Gamma^P$ target")
    for axis in axes:
        axis.set_xlabel(r"Projection dimension $\dim P$")
        format_projection_axis(axis, levels)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized Error")

    target_handles = [
        Line2D([0], [0], color="#444444", linestyle="-", linewidth=2.0, label=r"State target $\gamma$"),
        Line2D([0], [0], color="#444444", linestyle="--", linewidth=2.0, label=r"Analytical target $\Gamma^P$"),
    ]
    add_u_legend(fig, u_values, target_handles, ncol=5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    save_figure(fig, output_path)


def plot_majorana_mismatch(grouped, output_path: Path):
    u_values = ordered_u_values(grouped)
    levels = [row["projection_level"] for row in grouped[u_values[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), sharex=True)

    for u_value in u_values:
        rows = grouped[u_value]
        style = style_for_u(u_value)
        plot_series(
            axes[0],
            rows,
            "gamma2_ideal_vs_physical_error_in_ideal_basis_normalized",
            style,
            f"_nolegend_{u_value}_gamma23",
            linestyle="-",
        )
        plot_series(
            axes[1],
            rows,
            "physical_majorana_algebra_max_square_error_normalized",
            style,
            f"_nolegend_{u_value}_algebra",
        )

    axes[0].set_title(r"State Majoranas vs. projected analytical Majoranas")
    axes[1].set_title(r"Projected analytical Majorana square error")
    for axis in axes:
        axis.set_xlabel(r"Projection dimension $\dim P$")
        format_projection_axis(axis, levels)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized Error")

    gamma_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="-",
            linewidth=2.0,
            label=r"$\gamma_{2,3}-\Gamma_{2,3}^P$",
        ),
    ]
    add_u_legend(fig, u_values, gamma_handles, ncol=5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    save_figure(fig, output_path)


def plot_interaction_hierarchy(grouped, output_path: Path):
    u_values = ordered_u_values(grouped)
    levels = [row["projection_level"] for row in grouped[u_values[0]]]
    fig, axis = plt.subplots(1, 1, figsize=(6.1, 4.3))
    key = "ideal_vs_physical_target_gate_error_in_ideal_basis_normalized"

    for u_value in u_values:
        rows = grouped[u_value]
        plot_series(axis, rows, key, style_for_u(u_value), f"_nolegend_{u_value}")

    axis.set_title(r"Mismatch between state and analytical braid targets")
    axis.set_xlabel(r"Projection dimension $\dim P$")
    axis.set_ylabel("Normalized Error")
    format_projection_axis(axis, levels)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    add_u_legend(fig, u_values, ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    save_figure(fig, output_path)


def rows_up_to_projection(grouped, max_projection):
    filtered = {}
    for u_value, rows in grouped.items():
        kept_rows = [row for row in rows if row["projection_level"] <= max_projection]
        if kept_rows:
            filtered[u_value] = kept_rows
    if not filtered:
        raise ValueError(f"No rows found with projection_level <= {max_projection}.")
    return filtered


def projection_levels_for_grouped(grouped):
    return sorted({row["projection_level"] for rows in grouped.values() for row in rows})


def plot_single_target_error(grouped, key, title, output_path: Path):
    u_values = ordered_u_values(grouped)
    levels = projection_levels_for_grouped(grouped)
    fig, axis = plt.subplots(1, 1, figsize=(6.4, 4.3))

    for u_value in u_values:
        plot_series(axis, grouped[u_value], key, style_for_u(u_value), f"_nolegend_{u_value}")

    axis.set_title(title)
    axis.set_xlabel(r"Projection dimension $\dim P$")
    axis.set_ylabel("Normalized Error")
    format_projection_axis(axis, levels)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    add_u_legend(fig, u_values, ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    save_figure(fig, output_path)


def plot_target_error_difference(grouped, output_path: Path):
    u_values = ordered_u_values(grouped)
    levels = projection_levels_for_grouped(grouped)
    state_key = "physical_target_gate_error_in_ideal_basis_normalized"
    projected_key = "physical_target_gate_error_against_physical_target_in_ideal_basis_normalized"
    fig, axis = plt.subplots(1, 1, figsize=(6.4, 4.3))

    for u_value in u_values:
        plot_derived_series(
            axis,
            grouped[u_value],
            lambda row: row[state_key] - row[projected_key],
            style_for_u(u_value),
            f"_nolegend_{u_value}",
        )

    axis.set_title(r"State-target error minus $\Gamma^P$-target error")
    axis.set_xlabel(r"Projection dimension $\dim P$")
    axis.set_ylabel("Normalized Error Difference")
    format_projection_axis(axis, levels)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    add_u_legend(fig, u_values, ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    save_figure(fig, output_path)


def plot_requested_target_error_breakouts(grouped, fig_dir: Path, stem_suffix: str):
    specs = (
        (80, "dim80"),
        (512, "dim512"),
    )
    for max_projection, label in specs:
        subset = rows_up_to_projection(grouped, max_projection)
        plot_single_target_error(
            subset,
            "ideal_target_gate_error_normalized",
            r"State-made braid error vs. state target $e^{-\pi\gamma_2\gamma_3/4}$",
            apply_suffix(fig_dir / f"retry2_state_braid_state_target_error_{label}.pdf", stem_suffix),
        )
        plot_single_target_error(
            subset,
            "physical_target_gate_error_in_ideal_basis_normalized",
            r"Physical braid error vs. state target $e^{-\pi\gamma_2\gamma_3/4}$",
            apply_suffix(fig_dir / f"retry2_state_target_error_{label}.pdf", stem_suffix),
        )
        plot_single_target_error(
            subset,
            "physical_target_gate_error_against_physical_target_in_ideal_basis_normalized",
            r"Physical braid error vs. projected target $e^{-\pi\Gamma_2^P\Gamma_3^P/4}$",
            apply_suffix(fig_dir / f"retry2_projected_target_error_{label}.pdf", stem_suffix),
        )
        plot_target_error_difference(
            subset,
            apply_suffix(fig_dir / f"retry2_target_error_difference_{label}.pdf", stem_suffix),
        )


def main():
    args = build_argument_parser().parse_args()
    rows = parse_results(args.input)
    grouped = grouped_by_u(rows)

    plot_target_comparison(
        grouped,
        apply_suffix(args.fig_dir / "retry2_target_comparison.pdf", args.stem_suffix),
    )
    plot_majorana_mismatch(
        grouped,
        apply_suffix(args.fig_dir / "retry2_majorana_mismatch.pdf", args.stem_suffix),
    )
    plot_interaction_hierarchy(
        grouped,
        apply_suffix(args.fig_dir / "retry2_target_mismatch_hierarchy.pdf", args.stem_suffix),
    )
    write_summary_table(
        rows,
        apply_suffix(args.generated_dir / "retry2_target_diagnostic_summary.tex", args.stem_suffix),
    )
    plot_requested_target_error_breakouts(grouped, args.fig_dir, args.stem_suffix)


if __name__ == "__main__":
    main()
