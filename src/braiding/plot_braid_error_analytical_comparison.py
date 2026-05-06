from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ERROR_KEY = "physical_target_gate_error_in_ideal_basis_normalized"
REPO_ROOT = Path(__file__).resolve().parents[2]
REMAKE4_PATH = Path(__file__).with_name("retry2_remake4_diagonal_minus_dims_8_32_56_80_256_512.txt")
ANALYTICAL_PATH = Path(__file__).with_name(
    "retry2_analytical_sweetspot_block_complete_dims_8_56_176_336_456_512.txt"
)
OUTPUT_PATH = REPO_ROOT / "texmex" / "figs" / "braid_error_analytical_comparison"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def select_u_zero(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if abs(float(row["interaction_u"])) < 1.0e-12]
    return sorted(selected, key=lambda row: int(row["projection_level"]))


def plot_series(ax, rows: list[dict[str, str]], label: str, marker: str, linestyle: str) -> None:
    ax.plot(
        [int(row["projection_level"]) for row in rows],
        [float(row[ERROR_KEY]) for row in rows],
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        label=label,
    )


def main() -> None:
    remake4_rows = select_u_zero(read_rows(REMAKE4_PATH))
    analytical_rows = sorted(read_rows(ANALYTICAL_PATH), key=lambda row: int(row["projection_level"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    plot_series(ax, remake4_rows, "remake4 diagonal/minus, U=0", "s", "-")
    plot_series(ax, analytical_rows, "analytical sweet spot, block-complete dims", "^", "-")

    ax.set_xlabel("projection dimension")
    ax.set_ylabel("target-gate braid error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    xticks = sorted(
        {
            int(row["projection_level"])
            for rows in (remake4_rows, analytical_rows)
            for row in rows
        }
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(dim) for dim in xticks], rotation=30, ha="right")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH.with_suffix(".pdf"))
    fig.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=200)
    plt.close(fig)

    print("dim\tremake4_u0\tanalytical")
    remake4_by_dim = {int(row["projection_level"]): float(row[ERROR_KEY]) for row in remake4_rows}
    analytical_by_dim = {int(row["projection_level"]): float(row[ERROR_KEY]) for row in analytical_rows}
    for dim in sorted(analytical_by_dim):
        print(
            f"{dim}\t{remake4_by_dim.get(dim, float('nan')):.8e}\t"
            f"{analytical_by_dim[dim]:.8e}"
        )


if __name__ == "__main__":
    main()
