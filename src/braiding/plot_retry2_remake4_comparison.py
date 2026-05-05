from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ERROR_KEY = "physical_target_gate_error_in_ideal_basis_normalized"
DEFAULT_OLD = Path("src/braiding/retry2_target_diagnostic_dims_8_32_56_80_256_512.txt")
DEFAULT_NEW = Path("src/braiding/retry2_remake4_diagonal_minus_dims_8_32_56_80_256_512.txt")
DEFAULT_OUTPUT = Path("texmex/figs/retry2_remake4_state_target_error_comparison_extended")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def rows_by_u(rows: list[dict[str, str]]) -> dict[float, list[dict[str, str]]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["interaction_u"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["projection_level"]))
    return grouped


def plot_comparison(old_rows: list[dict[str, str]], new_rows: list[dict[str, str]], output: Path) -> None:
    old_grouped = rows_by_u(old_rows)
    new_grouped = rows_by_u(new_rows)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {0.0: "tab:blue", 0.1: "tab:orange", 2.0: "tab:green"}

    for interaction_u in sorted(set(old_grouped) & set(new_grouped)):
        color = colors.get(interaction_u)
        old = old_grouped[interaction_u]
        new = new_grouped[interaction_u]
        label_u = f"U={interaction_u:g}"

        ax.plot(
            [int(row["projection_level"]) for row in old],
            [float(row[ERROR_KEY]) for row in old],
            marker="o",
            linestyle="--",
            color=color,
            alpha=0.55,
            label=f"{label_u} old remake3/Both",
        )
        ax.plot(
            [int(row["projection_level"]) for row in new],
            [float(row[ERROR_KEY]) for row in new],
            marker="s",
            linestyle="-",
            color=color,
            label=f"{label_u} remake4 diagonal/minus",
        )

    ax.set_xlabel("projection dimension")
    ax.set_ylabel("state-made target error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 32, 56, 80, 256, 512])
    ax.set_xticklabels(["8", "32", "56", "80", "256", "512"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)


def print_summary(old_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> None:
    old = {(float(row["interaction_u"]), int(row["projection_level"])): float(row[ERROR_KEY]) for row in old_rows}
    new = {(float(row["interaction_u"]), int(row["projection_level"])): float(row[ERROR_KEY]) for row in new_rows}
    print("U\tdim\told_state_target\tnew_state_target\timprovement_factor")
    for key in sorted(set(old) & set(new)):
        improvement = old[key] / new[key] if new[key] and math.isfinite(new[key]) else math.nan
        print(f"{key[0]:g}\t{key[1]}\t{old[key]:.8e}\t{new[key]:.8e}\t{improvement:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    old_rows = read_rows(args.old)
    new_rows = read_rows(args.new)
    plot_comparison(old_rows, new_rows, args.output)
    print_summary(old_rows, new_rows)


if __name__ == "__main__":
    main()
