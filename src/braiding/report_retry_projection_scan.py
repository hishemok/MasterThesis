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


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 200,
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


PREFERRED_U_ORDER = (0.0, 0.1, 2.0)
INT_FIELDS = {"projection_level", "transport_dim"}
U_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    0.1: {"color": "#ff7f0e", "marker": "s"},
    2.0: {"color": "#d62728", "marker": "D"},
}
IDEAL_REFERENCE_STYLE = {
    "color": "#222222",
    "marker": "o",
    "linestyle": "--",
    "markerfacecolor": "white",
}
BASIS_STYLES = {
    "physical_basis": {"linestyle": "-", "label": "Scored in physical basis"},
    "ideal_basis": {"linestyle": "--", "label": "Scored in ideal basis"},
}
CHANNEL_PANELS = (
    (
        "gamma2_to_minus_gamma3",
        r"$\gamma_2 \rightarrow -\gamma_3$",
    ),
    (
        "gamma3_to_gamma2",
        r"$\gamma_3 \rightarrow \gamma_2$",
    ),
    (
        "gamma1_to_gamma1",
        r"$\gamma_1 \rightarrow \gamma_1$",
    ),
    (
        "gamma0_to_gamma0",
        r"$\gamma_0 \rightarrow \gamma_0$",
    ),
)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready plots from retry_projection_scan_results.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("retry_projection_scan_results.txt"),
        help="Path to the tab-separated retry projection scan results table.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=REPO_ROOT / "texmex" / "figs",
        help="Directory where the generated figures should be written.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=REPO_ROOT / "texmex" / "generated",
        help="Directory where generated LaTeX tables should be written.",
    )
    return parser


def parse_results(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            if not raw_row or not any(raw_row.values()):
                continue
            row = {}
            for key, value in raw_row.items():
                if key in INT_FIELDS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return rows


def group_rows_by_u(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["interaction_u"], []).append(row)
    for group in grouped.values():
        group.sort(key=lambda item: item["projection_level"])
    return grouped


def ordered_u_values(grouped_rows):
    known = [u for u in PREFERRED_U_ORDER if u in grouped_rows]
    extras = sorted(u for u in grouped_rows if u not in PREFERRED_U_ORDER)
    return known + extras


def ideal_reference_rows(rows):
    fields_to_average = [
        "ideal_target_gate_error",
        "ideal_single_exchange_max_error",
    ]
    for channel_key, _ in CHANNEL_PANELS:
        fields_to_average.append(f"ideal_single_exchange_{channel_key}_error")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["projection_level"], []).append(row)

    averaged_rows = []
    for projection_level in sorted(grouped):
        level_rows = grouped[projection_level]
        averaged_row = {"projection_level": projection_level}
        for field in fields_to_average:
            averaged_row[field] = sum(row[field] for row in level_rows) / len(level_rows)
        averaged_rows.append(averaged_row)
    return averaged_rows


def projection_levels_from_rows(rows):
    return [row["projection_level"] for row in rows]


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def format_metric(value):
    return f"{value:.2e}"


def format_u_value(value):
    return f"{value:g}"


def write_summary_table_tex(rows, output_path: Path):
    sorted_rows = sorted(rows, key=lambda row: (row["interaction_u"], row["projection_level"]))
    lines = [
        r"\begin{tabular}{cccccccc}",
        r"\hline",
        (
            r"$U$ & $\dim P$ & Ideal gate & Phys.\ gate & Phys.\ gate vs.\ ideal & "
            r"Ideal exch. & Phys.\ exch. & Phys.\ exch.\ vs.\ ideal \\"
        ),
        r"\hline",
    ]

    previous_u = None
    for row in sorted_rows:
        current_u = row["interaction_u"]
        if previous_u is not None and current_u != previous_u:
            lines.append(r"\hline")
        lines.append(
            " & ".join(
                [
                    format_u_value(current_u),
                    str(row["projection_level"]),
                    format_metric(row["ideal_target_gate_error"]),
                    format_metric(row["physical_target_gate_error_in_physical_basis"]),
                    format_metric(row["physical_target_gate_error_in_ideal_basis"]),
                    format_metric(row["ideal_single_exchange_max_error"]),
                    format_metric(row["physical_single_exchange_in_physical_basis_max_error"]),
                    format_metric(row["physical_single_exchange_in_ideal_basis_max_error"]),
                ]
            )
            + r" \\"
        )
        previous_u = current_u

    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_series(axis, x_values, y_values, style, label, zorder=2):
    axis.plot(
        x_values,
        y_values,
        color=style["color"],
        marker=style["marker"],
        linestyle=style.get("linestyle", "-"),
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor=style.get("markerfacecolor", style["color"]),
        markeredgecolor=style["color"],
        markeredgewidth=1.2,
        label=label,
        zorder=zorder,
    )


def plot_interaction_comparison(grouped_rows, reference_rows, output_path: Path):
    u_values = ordered_u_values(grouped_rows)
    projection_levels = projection_levels_from_rows(reference_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    metric_specs = (
        (
            "ideal_target_gate_error",
            "physical_target_gate_error_in_physical_basis",
            "Phase-aligned target-gate error",
        ),
        (
            "ideal_single_exchange_max_error",
            "physical_single_exchange_in_physical_basis_max_error",
            "Max single-exchange error",
        ),
    )

    for axis, (reference_key, physical_key, title) in zip(axes, metric_specs):
        for u_value in u_values:
            rows = grouped_rows[u_value]
            style = U_STYLES.get(u_value, {"color": "black", "marker": "o"})
            plot_series(
                axis,
                projection_levels_from_rows(rows),
                [row[physical_key] for row in rows],
                style,
                fr"$U={u_value:g}$ physical",
            )
        plot_series(
            axis,
            projection_levels,
            [row[reference_key] for row in reference_rows],
            IDEAL_REFERENCE_STYLE,
            "Ideal reference",
            zorder=4,
        )
        axis.set_title(title)
        axis.set_xlabel(r"Projection dimension $\dim P$")
        axis.set_xticks(projection_levels)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_path)


def plot_basis_diagnostic(grouped_rows, output_path: Path):
    u_values = ordered_u_values(grouped_rows)
    projection_levels = projection_levels_from_rows(grouped_rows[u_values[0]])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    metric_specs = (
        (
            "physical_target_gate_error_in_physical_basis",
            "physical_target_gate_error_in_ideal_basis",
            "Physical target-gate error",
        ),
        (
            "physical_single_exchange_in_physical_basis_max_error",
            "physical_single_exchange_in_ideal_basis_max_error",
            "Physical max single-exchange error",
        ),
    )

    for axis, (physical_key, ideal_key, title) in zip(axes, metric_specs):
        for u_value in u_values:
            rows = grouped_rows[u_value]
            style = U_STYLES.get(u_value, {"color": "black", "marker": "o"})
            plot_series(
                axis,
                projection_levels_from_rows(rows),
                [row[physical_key] for row in rows],
                {
                    "color": style["color"],
                    "marker": style["marker"],
                    "linestyle": BASIS_STYLES["physical_basis"]["linestyle"],
                    "markerfacecolor": style["color"],
                },
                f"_nolegend_{u_value}_physical",
            )
            plot_series(
                axis,
                projection_levels_from_rows(rows),
                [row[ideal_key] for row in rows],
                {
                    "color": style["color"],
                    "marker": style["marker"],
                    "linestyle": BASIS_STYLES["ideal_basis"]["linestyle"],
                    "markerfacecolor": "white",
                },
                f"_nolegend_{u_value}_ideal",
            )
        axis.set_title(title)
        axis.set_xlabel(r"Projection dimension $\dim P$")
        axis.set_xticks(projection_levels)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Error")

    color_handles = [
        Line2D(
            [0],
            [0],
            color=U_STYLES.get(u_value, {"color": "black"})["color"],
            marker=U_STYLES.get(u_value, {"marker": "o"})["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=fr"$U={u_value:g}$",
        )
        for u_value in u_values
    ]
    basis_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=basis_style["linestyle"],
            linewidth=2.0,
            label=basis_style["label"],
        )
        for basis_style in BASIS_STYLES.values()
    ]
    fig.legend(
        color_handles + basis_handles,
        [handle.get_label() for handle in color_handles + basis_handles],
        loc="upper center",
        ncol=max(3, len(color_handles) + len(basis_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.10),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    save_figure(fig, output_path)


def plot_channel_breakdown(grouped_rows, reference_rows, output_path: Path):
    u_values = ordered_u_values(grouped_rows)
    projection_levels = projection_levels_from_rows(reference_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), sharex=True)
    axes = axes.ravel()

    for axis, (channel_key, channel_label) in zip(axes, CHANNEL_PANELS):
        physical_key = f"physical_single_exchange_in_physical_basis_{channel_key}_error"
        for u_value in u_values:
            rows = grouped_rows[u_value]
            style = U_STYLES.get(u_value, {"color": "black", "marker": "o"})
            plot_series(
                axis,
                projection_levels_from_rows(rows),
                [row[physical_key] for row in rows],
                style,
                fr"$U={u_value:g}$ physical",
            )
        plot_series(
            axis,
            projection_levels,
            [row[f"ideal_single_exchange_{channel_key}_error"] for row in reference_rows],
            IDEAL_REFERENCE_STYLE,
            "Ideal reference",
            zorder=4,
        )
        axis.set_title(channel_label)
        axis.set_xticks(projection_levels)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Single-exchange error")
    axes[2].set_ylabel("Single-exchange error")
    axes[2].set_xlabel(r"Projection dimension $\dim P$")
    axes[3].set_xlabel(r"Projection dimension $\dim P$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(fig, output_path)


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    rows = parse_results(args.input)
    grouped_rows = group_rows_by_u(rows)
    reference_rows = ideal_reference_rows(rows)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        args.fig_dir / "retry_projection_interaction_comparison.pdf",
        args.fig_dir / "retry_projection_basis_diagnostic.pdf",
        args.fig_dir / "retry_projection_channel_breakdown.pdf",
    ]
    summary_table_path = args.generated_dir / "retry_projection_scan_summary.tex"

    plot_interaction_comparison(grouped_rows, reference_rows, output_paths[0])
    plot_basis_diagnostic(grouped_rows, output_paths[1])
    plot_channel_breakdown(grouped_rows, reference_rows, output_paths[2])
    write_summary_table_tex(rows, summary_table_path)

    print(f"Loaded {len(rows)} scan rows from {args.input}")
    print("Generated figures:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print("Generated tables:")
    print(f"  {summary_table_path}")


if __name__ == "__main__":
    main()
