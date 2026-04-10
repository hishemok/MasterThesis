import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from measurements import Majorana_polarization_torch, calculate_parities
from operators import parity_operator_torch
from thesis_plot import (
    find_best_entry,
    format_u_label,
    load_configurations,
    parse_loss,
    repo_root,
    style_matplotlib,
)
from hamiltonian import HamiltonianModel


@dataclass(frozen=True)
class CaseSpec:
    n: int
    u: float


@dataclass
class DiagnosticsRecord:
    n: int
    u: float
    loss: float
    delta_0: float
    low_energy_gap: float
    charge_difference_0: float
    edge_mp_mean_abs: float
    bulk_mp_max_abs: float
    mp_profile: list[float]
    even_occupancy: list[float]
    odd_occupancy: list[float]
    site_charge_difference: list[float]
    even_hopping: list[float]
    odd_hopping: list[float]
    even_pairing: list[float]
    odd_pairing: list[float]


def parse_case(case_text: str) -> CaseSpec:
    try:
        n_text, u_text = case_text.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid case specification {case_text!r}. Use the form n:U, for example 3:0.1."
        ) from exc
    return CaseSpec(n=int(n_text), u=float(u_text))


def load_case_specs(case_texts: list[str]) -> list[CaseSpec]:
    return [parse_case(case) for case in case_texts]


def get_case_title(record: DiagnosticsRecord) -> str:
    return fr"$n={record.n},\ U={format_u_label(record.u)}$"


def build_diagnostics_record(entry: dict) -> DiagnosticsRecord:
    n = len(entry["physical_parameters"]["eps"])
    u = float(entry["parameter_configs"]["U"]["fixed"])
    loss = parse_loss(entry)

    model = HamiltonianModel(n=n, param_configs=entry.get("parameter_configs"))
    theta_tensor = model.dict_to_tensor(entry["raw_parameter_values"])
    param_dict = model.tensor_to_dict(theta_tensor)
    phys_params = model.get_physical_parameters(param_dict)

    hamiltonian = model.build(phys_params)
    evals, evecs = torch.linalg.eigh(hamiltonian)
    parity = parity_operator_torch(n)
    even_states, odd_states, even_vecs, odd_vecs = calculate_parities(evals, evecs, parity)

    even_energies = torch.real(even_states).detach().cpu().numpy()
    odd_energies = torch.real(odd_states).detach().cpu().numpy()
    ground_shift = min(float(np.min(even_energies)), float(np.min(odd_energies)))
    even_energies = even_energies - ground_shift
    odd_energies = odd_energies - ground_shift

    delta_0 = abs(float(even_energies[0]) - float(odd_energies[0]))
    low_energy_gap = min(float(even_energies[1]), float(odd_energies[1])) - max(
        float(even_energies[0]), float(odd_energies[0])
    )

    even_ground = even_vecs[:, 0]
    odd_ground = odd_vecs[:, 0]

    mp_all = Majorana_polarization_torch(even_vecs, odd_vecs, n)
    mp_profile = torch.real(mp_all[0]).detach().cpu().numpy()

    even_occupancy: list[float] = []
    odd_occupancy: list[float] = []
    site_charge_difference: list[float] = []
    for site in range(n):
        n_i = model.num_t[site]
        occ_even = torch.real(torch.vdot(even_ground, n_i @ even_ground)).item()
        occ_odd = torch.real(torch.vdot(odd_ground, n_i @ odd_ground)).item()
        even_occupancy.append(float(occ_even))
        odd_occupancy.append(float(occ_odd))
        site_charge_difference.append(abs(float(occ_even - occ_odd)))

    even_hopping: list[float] = []
    odd_hopping: list[float] = []
    even_pairing: list[float] = []
    odd_pairing: list[float] = []
    for bond in range(n - 1):
        hop_op = model.cre_t[bond] @ model.ann_t[bond + 1]
        pair_op = model.ann_t[bond] @ model.ann_t[bond + 1]

        hop_even = torch.abs(torch.vdot(even_ground, hop_op @ even_ground)).item()
        hop_odd = torch.abs(torch.vdot(odd_ground, hop_op @ odd_ground)).item()
        pair_even = torch.abs(torch.vdot(even_ground, pair_op @ even_ground)).item()
        pair_odd = torch.abs(torch.vdot(odd_ground, pair_op @ odd_ground)).item()

        even_hopping.append(float(hop_even))
        odd_hopping.append(float(hop_odd))
        even_pairing.append(float(pair_even))
        odd_pairing.append(float(pair_odd))

    charge_difference_0 = float(sum(site_charge_difference))
    edge_mp_mean_abs = 0.5 * (abs(float(mp_profile[0])) + abs(float(mp_profile[-1])))
    if n > 2:
        bulk_mp_max_abs = float(np.max(np.abs(mp_profile[1:-1])))
    else:
        bulk_mp_max_abs = 0.0

    return DiagnosticsRecord(
        n=n,
        u=u,
        loss=loss,
        delta_0=delta_0,
        low_energy_gap=low_energy_gap,
        charge_difference_0=charge_difference_0,
        edge_mp_mean_abs=edge_mp_mean_abs,
        bulk_mp_max_abs=bulk_mp_max_abs,
        mp_profile=[float(value) for value in mp_profile],
        even_occupancy=even_occupancy,
        odd_occupancy=odd_occupancy,
        site_charge_difference=site_charge_difference,
        even_hopping=even_hopping,
        odd_hopping=odd_hopping,
        even_pairing=even_pairing,
        odd_pairing=odd_pairing,
    )


def collect_records(config_file: Path, cases: list[CaseSpec]) -> list[DiagnosticsRecord]:
    entries = load_configurations(config_file)
    records: list[DiagnosticsRecord] = []

    for case in cases:
        entry = find_best_entry(entries, n=case.n, u=case.u)
        if entry is None:
            raise ValueError(
                f"No saved configuration found for n={case.n}, U={case.u:g} in {config_file}."
            )
        records.append(build_diagnostics_record(entry))

    return records


def plot_majorana_profiles(records: list[DiagnosticsRecord], output: Path) -> None:
    style_matplotlib()

    ncols = 2 if len(records) > 1 else 1
    nrows = math.ceil(len(records) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    mp_limit = max(max(abs(value) for value in record.mp_profile) for record in records)
    mp_limit = max(1.0, 1.1 * mp_limit)

    line_color = "#146C94"
    fill_color = "#8ECAE6"
    point_color = "#023047"
    charge_color = "#BC6C25"

    for ax, record in zip(axes.flat, records):
        x = np.arange(1, record.n + 1)
        mp = np.array(record.mp_profile)
        charge = np.array(record.site_charge_difference)

        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=1)
        ax.fill_between(x, 0, mp, color=fill_color, alpha=0.35, zorder=2)
        ax.plot(x, mp, color=line_color, linewidth=2.4, zorder=3)
        ax.scatter(x, mp, color=point_color, s=36, zorder=4)

        ax2 = ax.twinx()
        ax2.bar(x, charge, width=0.28, color=charge_color, alpha=0.22, zorder=0)
        ax2.set_ylim(0, max(0.02, 1.15 * max(charge.max(), 0.01)))
        ax2.set_yticks([])
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_color("#BBBBBB")

        ax.set_title(get_case_title(record), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xlabel("Site index")
        ax.set_ylabel("Majorana polarization")
        ax.set_ylim(-mp_limit, mp_limit)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)

        stats_text = (
            fr"$\delta_0={record.delta_0:.3g}$" "\n"
            fr"$\Delta_{{\mathrm{{gap}}}}={record.low_energy_gap:.3g}$" "\n"
            fr"$Q_0={record.charge_difference_0:.3g}$"
        )
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.3"},
        )

    for ax in axes.flat[len(records) :]:
        ax.axis("off")

    legend_lines = [
        plt.Line2D([0], [0], color=line_color, lw=2.4, label="Lowest-pair Majorana polarization"),
        plt.Rectangle((0, 0), 1, 1, color=charge_color, alpha=0.22, label="Site charge difference"),
    ]
    fig.legend(handles=legend_lines, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Local diagnostics of representative optimized configurations",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(records: list[DiagnosticsRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n",
                "U",
                "loss",
                "delta_0",
                "low_energy_gap",
                "charge_difference_0",
                "edge_mp_mean_abs",
                "bulk_mp_max_abs",
                "mp_profile",
                "site_charge_difference",
                "even_occupancy",
                "odd_occupancy",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.n,
                    record.u,
                    record.loss,
                    record.delta_0,
                    record.low_energy_gap,
                    record.charge_difference_0,
                    record.edge_mp_mean_abs,
                    record.bulk_mp_max_abs,
                    " ".join(f"{value:.6f}" for value in record.mp_profile),
                    " ".join(f"{value:.6f}" for value in record.site_charge_difference),
                    " ".join(f"{value:.6f}" for value in record.even_occupancy),
                    " ".join(f"{value:.6f}" for value in record.odd_occupancy),
                ]
            )


def write_summary_json(records: list[DiagnosticsRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_summary_tex(records: list[DiagnosticsRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{c c c c c c c}",
        r"\hline",
        r"$n$ & $U$ & $\delta_0$ & $\Delta_{\mathrm{gap}}$ & $Q_0$ & $\overline{|M_{\mathrm{edge}}|}$ & $\max |M_{\mathrm{bulk}}|$ \\",
        r"\hline",
    ]

    for record in records:
        lines.append(
            " & ".join(
                [
                    str(record.n),
                    format_u_label(record.u),
                    f"{record.delta_0:.3g}",
                    f"{record.low_energy_gap:.3g}",
                    f"{record.charge_difference_0:.3g}",
                    f"{record.edge_mp_mean_abs:.3g}",
                    f"{record.bulk_mp_max_abs:.3g}",
                ]
            )
            + r" \\"
        )

    lines.extend([r"\hline", r"\end{tabular}"])
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def print_summary(records: list[DiagnosticsRecord]) -> None:
    print("\nSelected diagnostic cases")
    print("-" * 94)
    print(
        f"{'n':>3} {'U':>7} {'loss':>12} {'delta_0':>12} {'gap':>12} "
        f"{'Q0':>12} {'edge|M|':>12} {'bulk|M|':>12}"
    )
    print("-" * 94)
    for record in records:
        print(
            f"{record.n:>3d} {record.u:>7.3f} {record.loss:>12.4e} "
            f"{record.delta_0:>12.4e} {record.low_energy_gap:>12.4e} "
            f"{record.charge_difference_0:>12.4e} {record.edge_mp_mean_abs:>12.4e} "
            f"{record.bulk_mp_max_abs:>12.4e}"
        )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis-ready local diagnostics for selected optimized "
            "configurations, including a Majorana-polarization figure and summary tables."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configuration.json",
        help="Path to configuration.json.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["2:0", "2:0.1", "3:0.1", "4:0.1"],
        help="Representative cases written as n:U, for example 3:0.1.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=root / "texmex" / "figs" / "majorana_character_summary.pdf",
        help="Output path for the thesis-ready figure.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "texmex" / "generated" / "majorana_character_summary.csv",
        help="Output path for the summary CSV file.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=root / "texmex" / "generated" / "majorana_character_summary.json",
        help="Output path for the detailed JSON file.",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=root / "texmex" / "generated" / "majorana_character_summary.tex",
        help="Output path for the LaTeX table snippet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_specs = load_case_specs(args.cases)
    records = collect_records(args.config, case_specs)

    plot_majorana_profiles(records, args.figure)
    write_summary_csv(records, args.csv)
    write_summary_json(records, args.json)
    write_summary_tex(records, args.tex)
    print_summary(records)

    print(f"\nSaved figure to: {args.figure}")
    print(f"Saved CSV to: {args.csv}")
    print(f"Saved JSON to: {args.json}")
    print(f"Saved LaTeX table to: {args.tex}")


if __name__ == "__main__":
    main()
