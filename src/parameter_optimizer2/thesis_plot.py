import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from hamiltonian import HamiltonianModel
from measurements import calculate_parities
from operators import parity_operator_torch


LOSS_RE = re.compile(r"Loss:\s*([0-9.eE+-]+)")
N_RE = re.compile(r"configuration\s+(\d+)-\s*site system")


@dataclass
class SpectrumRecord:
    n: int
    u: float
    loss: float
    even: np.ndarray
    odd: np.ndarray
    delta_0: float
    low_energy_gap: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_configurations(filename: str | Path) -> list[dict]:
    path = Path(filename)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of configurations in {path}.")
    return data


def parse_loss(entry: dict) -> float:
    header = entry.get("header", "")
    match = LOSS_RE.search(header)
    if not match:
        return math.inf
    return float(match.group(1))


def infer_n(entry: dict) -> int:
    header = entry.get("header", "")
    match = N_RE.search(header)
    if match:
        return int(match.group(1))

    eps = entry.get("physical_parameters", {}).get("eps")
    if eps:
        return len(eps)

    raise ValueError(f"Could not infer system size from entry: {header!r}")


def infer_u(entry: dict) -> float:
    parameter_configs = entry.get("parameter_configs", {})
    fixed_u = parameter_configs.get("U", {}).get("fixed")
    if isinstance(fixed_u, (int, float)):
        return float(fixed_u)

    physical_u = entry.get("physical_parameters", {}).get("U")
    if physical_u:
        return float(physical_u[0])

    raw_u = entry.get("raw_parameter_values", {}).get("U")
    if raw_u:
        return float(raw_u[0])

    raise ValueError("Could not infer interaction strength U from configuration entry.")


def available_u_values(entries: Iterable[dict], n: int) -> list[float]:
    values = {round(infer_u(entry), 12) for entry in entries if infer_n(entry) == n}
    return sorted(values)


def find_best_entry(entries: Iterable[dict], n: int, u: float, tol: float = 1e-9) -> dict | None:
    matching = [
        entry
        for entry in entries
        if infer_n(entry) == n and abs(infer_u(entry) - u) <= tol
    ]
    if not matching:
        return None
    return min(matching, key=parse_loss)


def build_spectrum_record(entry: dict) -> SpectrumRecord:
    n = infer_n(entry)
    u = infer_u(entry)
    loss = parse_loss(entry)

    model = HamiltonianModel(n=n, param_configs=entry.get("parameter_configs"))
    theta_tensor = model.dict_to_tensor(entry["raw_parameter_values"])
    param_dict = model.tensor_to_dict(theta_tensor)
    phys_params = model.get_physical_parameters(param_dict)

    hamiltonian = model.build(phys_params)
    evals, evecs = torch.linalg.eigh(hamiltonian)
    parity = parity_operator_torch(n)
    even_states, odd_states, _, _ = calculate_parities(evals, evecs, parity)

    even = torch.real(even_states).detach().cpu().numpy()
    odd = torch.real(odd_states).detach().cpu().numpy()

    ground_shift = min(float(np.min(even)), float(np.min(odd)))
    even = even - ground_shift
    odd = odd - ground_shift

    if len(even) == 0 or len(odd) == 0:
        delta_0 = math.nan
        low_energy_gap = math.nan
    else:
        delta_0 = abs(float(even[0]) - float(odd[0]))
        if len(even) > 1 and len(odd) > 1:
            low_energy_gap = min(float(even[1]), float(odd[1])) - max(float(even[0]), float(odd[0]))
        else:
            low_energy_gap = math.nan

    return SpectrumRecord(
        n=n,
        u=u,
        loss=loss,
        even=even,
        odd=odd,
        delta_0=delta_0,
        low_energy_gap=low_energy_gap,
    )


def format_u_label(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_parity_summary(
    n_values: list[int],
    u_values: list[float],
    filename: str | Path,
    output: str | Path,
    max_levels: int | None = 4,
    degeneracy_tol: float = 1e-2,
) -> list[SpectrumRecord]:
    style_matplotlib()
    entries = load_configurations(filename)

    fig, axes = plt.subplots(
        len(n_values),
        1,
        figsize=(9.5, 2.8 * len(n_values) + 0.8),
        sharex=True,
        constrained_layout=True,
    )

    if len(n_values) == 1:
        axes = [axes]

    even_color = "#2F6690"
    odd_color = "#B23A48"
    connector_color = "#8E8E8E"
    selected_records: list[SpectrumRecord] = []

    for ax, n in zip(axes, n_values):
        ax.set_title(f"$n={n}$", loc="left", fontweight="bold")

        for group_index, u in enumerate(u_values):
            left = group_index - 0.43
            right = group_index + 0.43
            if group_index % 2 == 1:
                ax.axvspan(left, right, color="#F5F5F5", zorder=0)

            entry = find_best_entry(entries, n=n, u=u)
            if entry is None:
                ax.text(
                    group_index,
                    0.5,
                    "not saved",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#777777",
                )
                continue

            record = build_spectrum_record(entry)
            selected_records.append(record)

            even_levels = record.even[:max_levels] if max_levels is not None else record.even
            odd_levels = record.odd[:max_levels] if max_levels is not None else record.odd

            ax.hlines(
                even_levels,
                group_index - 0.33,
                group_index - 0.05,
                color=even_color,
                linewidth=2.8,
                zorder=3,
            )
            ax.hlines(
                odd_levels,
                group_index + 0.05,
                group_index + 0.33,
                color=odd_color,
                linewidth=2.8,
                zorder=3,
            )

            for even_level, odd_level in zip(even_levels, odd_levels):
                if abs(float(even_level) - float(odd_level)) <= degeneracy_tol:
                    ax.hlines(
                        0.5 * (float(even_level) + float(odd_level)),
                        group_index - 0.05,
                        group_index + 0.05,
                        color=connector_color,
                        linewidth=1.8,
                        linestyles="dashed",
                        zorder=2,
                    )

        ax.set_ylabel(r"$E - E_\mathrm{gs}$")
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.set_xlim(-0.6, len(u_values) - 0.4)

    axes[-1].set_xticks(range(len(u_values)))
    axes[-1].set_xticklabels([fr"$U={format_u_label(u)}$" for u in u_values])

    legend_handles = [
        Line2D([0], [0], color=even_color, lw=3, label="Even parity"),
        Line2D([0], [0], color=odd_color, lw=3, label="Odd parity"),
        Line2D([0], [0], color=connector_color, lw=2, ls="--", label="Near-degenerate pair"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.suptitle(
        "Parity-resolved spectra for selected optimized configurations",
        fontsize=15,
        fontweight="bold",
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return selected_records


def print_summary(records: list[SpectrumRecord]) -> None:
    if not records:
        print("No spectra were selected. Check the requested n and U values.")
        return

    print("\nSelected configurations")
    print("-" * 72)
    print(f"{'n':>3} {'U':>7} {'loss':>14} {'delta_0':>14} {'gap':>14}")
    print("-" * 72)
    for record in records:
        gap_str = f"{record.low_energy_gap:.6f}" if not math.isnan(record.low_energy_gap) else "nan"
        print(
            f"{record.n:>3d} "
            f"{record.u:>7.3f} "
            f"{record.loss:>14.6e} "
            f"{record.delta_0:>14.6f} "
            f"{gap_str:>14}"
        )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact thesis-ready parity-spectrum summary from saved "
            "configurations in configuration.json."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configuration.json",
        help="Path to configuration.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "texmex" / "figs" / "parity_summary.png",
        help="Where to save the generated summary figure.",
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="System sizes to include as separate panels.",
    )
    parser.add_argument(
        "--u-values",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 2.0],
        help="Interaction strengths to compare within each panel.",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=4,
        help="Maximum number of even and odd levels shown per configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = load_configurations(args.config)
    for n in args.n_values:
        values = ", ".join(format_u_label(u) for u in available_u_values(entries, n))
        print(f"Available U values for n={n}: {values}")

    selected = plot_parity_summary(
        n_values=args.n_values,
        u_values=args.u_values,
        filename=args.config,
        output=args.output,
        max_levels=args.max_levels,
    )

    print_summary(selected)
    print(f"\nSaved figure to: {args.output}")


if __name__ == "__main__":
    main()
