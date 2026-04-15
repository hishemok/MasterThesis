from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


METHOD_ORDER = ("ideal", "bilinear_fit", "local_projection", "physical")
METHOD_ALIASES = {
    "ideal": "ideal",
    "bilinear_fit": "bilinear_fit",
    "ad2": "local_projection",
    "local_projection": "local_projection",
    "physical": "physical",
}
METHOD_LABELS = {
    "ideal": "Ideal",
    "bilinear_fit": "Bilinear Fit",
    "local_projection": "Local Projection",
    "physical": "Physical",
}
METHOD_TABLE_LABELS = {
    "ideal": "Ideal",
    "bilinear_fit": r"Bilin.\ fit",
    "local_projection": r"Local proj.",
    "physical": "Physical",
}
METHOD_STYLES = {
    "ideal": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "#1f77b4",
    },
    "bilinear_fit": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
    "local_projection": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
        "markerfacecolor": "#2ca02c",
    },
    "physical": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": ":",
        "markerfacecolor": "white",
    },
}
METHOD_X_OFFSETS = {
    "ideal": -1.2,
    "bilinear_fit": -0.4,
    "local_projection": 0.4,
    "physical": 1.2,
}
U_PANEL_ORDER = (0.0, 0.1, 2.0)
U_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    0.1: {"color": "#ff7f0e", "marker": "s"},
    2.0: {"color": "#d62728", "marker": "D"},
}


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate figures and summary tables from savebraidvals4.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("savebraidvals4.txt"),
        help="Path to the four-method braiding summary text file.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=REPO_ROOT / "texmex" / "figs",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=REPO_ROOT / "texmex" / "generated",
        help="Directory for generated tables and data summaries.",
    )
    return parser


def parse_active_majoranas(text):
    matches = re.findall(r"γ([0-3])=([^,]+)(?:,|$)", text)
    ordered = [None, None, None, None]
    for idx, label in matches:
        ordered[int(idx)] = label.strip()
    return ordered


def parse_results(path: Path):
    blocks = []
    current_u = None
    current_block = None
    current_method = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.startswith("U ="):
            current_u = float(stripped.split("=", 1)[1].strip())
            current_block = None
            current_method = None
            continue

        if stripped.startswith("P_"):
            match = re.match(r"(?P<name>\S+)\s+cum_dim=(?P<dim>\d+)", stripped)
            if match is None:
                raise ValueError(f"Could not parse block header: {stripped}")
            current_block = {
                "u": current_u,
                "block_name": match.group("name"),
                "cum_dim": int(match.group("dim")),
                "methods": {},
            }
            blocks.append(current_block)
            current_method = None
            continue

        if current_block is None:
            continue

        if stripped.startswith("active_majoranas:"):
            current_block["active_majoranas"] = parse_active_majoranas(
                stripped.split(":", 1)[1].strip()
            )
            continue

        if stripped.startswith("physical_bilinear_residuals:"):
            match = re.search(r"AB=([0-9.eE+-]+),\s*AC=([0-9.eE+-]+)", stripped)
            if match is None:
                raise ValueError(f"Could not parse residuals line: {stripped}")
            current_block["physical_bilinear_residual_ab"] = float(match.group(1))
            current_block["physical_bilinear_residual_ac"] = float(match.group(2))
            continue

        method_match = re.match(
            r"(?P<method>ideal|bilinear_fit|ad2|local_projection|physical):\s+max_braid_error="
            r"(?P<max>[0-9.eE+-]+)\s+\|\s+normalized=(?P<norm>[0-9.eE+-]+)",
            stripped,
        )
        if method_match is not None:
            current_method = METHOD_ALIASES[method_match.group("method")]
            current_block["methods"][current_method] = {
                "max_braid_error": float(method_match.group("max")),
                "normalized_braid_error": float(method_match.group("norm")),
            }
            continue

        if stripped.startswith("braid_errors:"):
            if current_method is None:
                raise ValueError("Encountered braid_errors line before method header.")
            current_block["methods"][current_method]["braid_errors"] = stripped.split(
                ":", 1
            )[1].strip()
            continue

        if stripped.startswith("off_block_leakage="):
            if current_method is None:
                raise ValueError("Encountered parity metrics before method header.")
            match = re.match(
                r"off_block_leakage=([0-9.eE+-]+)\s+\|\s+odd_target_error=([0-9.eE+-]+)\s+\|\s+even_target_error=([0-9.eE+-]+)",
                stripped,
            )
            if match is None:
                raise ValueError(f"Could not parse parity metrics line: {stripped}")
            current_block["methods"][current_method]["off_block_leakage"] = float(
                match.group(1)
            )
            current_block["methods"][current_method]["odd_target_error"] = float(
                match.group(2)
            )
            current_block["methods"][current_method]["even_target_error"] = float(
                match.group(3)
            )

    return blocks


def group_blocks_by_u(blocks):
    grouped = {}
    for block in blocks:
        grouped.setdefault(block["u"], []).append(block)
    for group in grouped.values():
        group.sort(key=lambda item: item["cum_dim"])
    return grouped


def format_metric(value):
    return f"{value:.2e}"


def save_figure(fig, output_path: Path):
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_normalized_braid_errors(grouped_blocks, output_path: Path):
    u_values = [u for u in U_PANEL_ORDER if u in grouped_blocks]
    fig, axes = plt.subplots(1, len(u_values), figsize=(4.8 * len(u_values), 4.2), sharey=True)
    if len(u_values) == 1:
        axes = [axes]

    for axis, u_value in zip(axes, u_values):
        blocks = grouped_blocks[u_value]
        dims = [block["cum_dim"] for block in blocks]
        for method in METHOD_ORDER:
            values = [block["methods"][method]["normalized_braid_error"] for block in blocks]
            style = METHOD_STYLES[method]
            shifted_dims = [dim + METHOD_X_OFFSETS[method] for dim in dims]
            axis.plot(
                shifted_dims,
                values,
                label=METHOD_LABELS[method],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=6.0,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=1.4,
            )
        axis.set_title(f"$U = {u_value:g}$")
        axis.set_xlabel(r"Cumulative projection dimension $\dim P$")
        axis.set_xticks(dims)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Normalized braid error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_path)


def plot_physical_bilinear_residuals(grouped_blocks, output_path: Path):
    u_values = [u for u in U_PANEL_ORDER if u in grouped_blocks]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    residual_keys = (
        ("physical_bilinear_residual_ab", "AB residual"),
        ("physical_bilinear_residual_ac", "AC residual"),
    )

    for axis, (residual_key, title) in zip(axes, residual_keys):
        for u_value in u_values:
            blocks = grouped_blocks[u_value]
            dims = [block["cum_dim"] for block in blocks]
            values = [block[residual_key] for block in blocks]
            style = U_STYLES.get(u_value, {"color": "black", "marker": "o"})
            axis.plot(
                dims,
                values,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.0,
                label=fr"$U={u_value:g}$",
            )
        axis.set_title(title)
        axis.set_xlabel(r"Cumulative projection dimension $\dim P$")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Physical bilinear residual")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_path)


def plot_focused_physical_bilinear_residuals(grouped_blocks, output_path: Path):
    u_values = [u for u in U_PANEL_ORDER if u in grouped_blocks]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    residual_keys = (
        ("physical_bilinear_residual_ab", "AB residual"),
        ("physical_bilinear_residual_ac", "AC residual"),
    )

    excited_residual_values = []
    for u_value in u_values:
        for block in grouped_blocks[u_value][1:]:
            excited_residual_values.append(block["physical_bilinear_residual_ab"])
            excited_residual_values.append(block["physical_bilinear_residual_ac"])

    if excited_residual_values:
        y_min = min(excited_residual_values)
        y_max = max(excited_residual_values)
        y_padding = max(0.03, 0.1 * (y_max - y_min))
    else:
        y_min = 0.0
        y_max = 1.0
        y_padding = 0.05

    for axis, (residual_key, title) in zip(axes, residual_keys):
        for u_value in u_values:
            blocks = grouped_blocks[u_value][1:]
            dims = [block["cum_dim"] for block in blocks]
            values = [block[residual_key] for block in blocks]
            style = U_STYLES.get(u_value, {"color": "black", "marker": "o"})
            axis.plot(
                dims,
                values,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.0,
                label=fr"$U={u_value:g}$",
            )
        axis.set_title(f"{title} (excited-space focus)")
        axis.set_xlabel(r"Cumulative projection dimension $\dim P$")
        axis.set_ylim(y_min - y_padding, y_max + y_padding)
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Physical bilinear residual")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_path)


def plot_interaction_sensitive_errors(grouped_blocks, output_path: Path):
    u_values = [u for u in U_PANEL_ORDER if u in grouped_blocks]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    method_titles = (
        ("local_projection", f"{METHOD_LABELS['local_projection']} normalized braid error"),
        ("physical", f"{METHOD_LABELS['physical']} normalized braid error"),
    )

    for axis, (method, title) in zip(axes, method_titles):
        for u_value in u_values:
            blocks = grouped_blocks[u_value]
            dims = [block["cum_dim"] for block in blocks]
            values = [block["methods"][method]["normalized_braid_error"] for block in blocks]
            style = U_STYLES.get(u_value, {"color": "black", "marker": "o"})
            axis.plot(
                dims,
                values,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.0,
                label=fr"$U={u_value:g}$",
            )
        axis.set_title(title)
        axis.set_xlabel(r"Cumulative projection dimension $\dim P$")
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Normalized braid error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_path)


def write_summary_table_tex(blocks, output_path: Path):
    lines = [
        r"\begin{tabular}{cccccccc}",
        r"\hline",
        (
            r"$U$ & $\dim P$ & Res.\ AB & Res.\ AC & "
            + " & ".join(METHOD_TABLE_LABELS[method] for method in METHOD_ORDER)
            + r" \\"
        ),
        r"\hline",
    ]
    for block in blocks:
        lines.append(
            " & ".join(
                [
                    f"{block['u']:g}",
                    str(block["cum_dim"]),
                    format_metric(block["physical_bilinear_residual_ab"]),
                    format_metric(block["physical_bilinear_residual_ac"]),
                    format_metric(block["methods"]["ideal"]["normalized_braid_error"]),
                    format_metric(block["methods"]["bilinear_fit"]["normalized_braid_error"]),
                    format_metric(block["methods"]["local_projection"]["normalized_braid_error"]),
                    format_metric(block["methods"]["physical"]["normalized_braid_error"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(blocks, output_path: Path):
    fieldnames = [
        "u",
        "block_name",
        "cum_dim",
        "active_gamma0",
        "active_gamma1",
        "active_gamma2",
        "active_gamma3",
        "physical_bilinear_residual_ab",
        "physical_bilinear_residual_ac",
        "ideal_normalized_braid_error",
        "bilinear_fit_normalized_braid_error",
        "local_projection_normalized_braid_error",
        "physical_normalized_braid_error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for block in blocks:
            active = block.get("active_majoranas", [None, None, None, None])
            writer.writerow(
                {
                    "u": block["u"],
                    "block_name": block["block_name"],
                    "cum_dim": block["cum_dim"],
                    "active_gamma0": active[0],
                    "active_gamma1": active[1],
                    "active_gamma2": active[2],
                    "active_gamma3": active[3],
                    "physical_bilinear_residual_ab": block["physical_bilinear_residual_ab"],
                    "physical_bilinear_residual_ac": block["physical_bilinear_residual_ac"],
                    "ideal_normalized_braid_error": block["methods"]["ideal"]["normalized_braid_error"],
                    "bilinear_fit_normalized_braid_error": block["methods"]["bilinear_fit"]["normalized_braid_error"],
                    "local_projection_normalized_braid_error": block["methods"]["local_projection"]["normalized_braid_error"],
                    "physical_normalized_braid_error": block["methods"]["physical"]["normalized_braid_error"],
                }
            )


def write_summary_json(blocks, output_path: Path):
    output_path.write_text(json.dumps(blocks, indent=2), encoding="utf-8")


def main():
    args = build_argument_parser().parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_results(args.input)
    grouped = group_blocks_by_u(blocks)

    error_plot_path = args.fig_dir / "extended_projection_four_method_errors.pdf"
    residual_plot_path = args.fig_dir / "extended_projection_four_method_residuals.pdf"
    focused_residual_plot_path = args.fig_dir / "extended_projection_four_method_residuals_focus.pdf"
    interaction_plot_path = args.fig_dir / "extended_projection_four_method_interaction_errors.pdf"
    summary_table_path = args.generated_dir / "extended_projection_four_method_summary.tex"
    summary_csv_path = args.generated_dir / "extended_projection_four_method_summary.csv"
    summary_json_path = args.generated_dir / "extended_projection_four_method_summary.json"

    plot_normalized_braid_errors(grouped, error_plot_path)
    plot_physical_bilinear_residuals(grouped, residual_plot_path)
    plot_focused_physical_bilinear_residuals(grouped, focused_residual_plot_path)
    plot_interaction_sensitive_errors(grouped, interaction_plot_path)
    write_summary_table_tex(blocks, summary_table_path)
    write_summary_csv(blocks, summary_csv_path)
    write_summary_json(blocks, summary_json_path)

    print("Generated:")
    print(f"  {error_plot_path}")
    print(f"  {residual_plot_path}")
    print(f"  {focused_residual_plot_path}")
    print(f"  {interaction_plot_path}")
    print(f"  {summary_table_path}")
    print(f"  {summary_csv_path}")
    print(f"  {summary_json_path}")


if __name__ == "__main__":
    main()
