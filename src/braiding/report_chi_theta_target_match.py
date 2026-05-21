from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

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

import retry2
from get_mzm_JW import get_full_gammas as get_basic_full_gammas
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from remake_majoranas4 import make_majoranas_for_B_and_C_with_projection_dim
from extended_projection_braiding import normalize_projected_majorana


DEFAULT_BASELINE = Path(__file__).with_name("retry2_remake4_diagonal_minus_dims_8_32_56_80.txt")
DEFAULT_OUTPUT = Path(__file__).with_name("chi_theta_target_match_dims_8_32_56_80.txt")
DEFAULT_FIG = REPO_ROOT / "texmex" / "figs" / "chi_theta_target_match_error_80.pdf"

U_VALUES = [0.0, 0.1, 2.0]
PROJECTION_LEVELS = [8, 32, 56, 80]
U_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    0.1: {"color": "#ff7f0e", "marker": "s"},
    2.0: {"color": "#d62728", "marker": "D"},
}

FIELDNAMES = [
    "interaction_u",
    "projection_level",
    "transport_dim",
    "theta_b",
    "theta_c",
    "state_error_normalized",
    "local_y_state_target_error_normalized",
    "local_y_projected_target_error_normalized",
    "best_chi_target_mismatch_normalized",
    "local_y_target_mismatch_normalized",
    "gamma2_state_vs_best_chi_normalized",
    "gamma3_state_vs_best_chi_normalized",
]


def read_baseline(path: Path) -> dict[tuple[float, int], dict]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            key = (float(raw["interaction_u"]), int(raw["projection_level"]))
            rows[key] = raw
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def chi_from_theta(gamma_plus: np.ndarray, gamma_minus: np.ndarray, theta: float, label: str) -> np.ndarray:
    return normalize_projected_majorana(np.cos(theta) * gamma_plus + np.sin(theta) * gamma_minus, label)


def build_u_context(u_value: float) -> dict:
    print(f"preparing cached data for U={u_value:g}", flush=True)
    specified_vals = {"U": [u_value]}

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )
    h_full = builder.full_system_hamiltonian()
    _, eigvecs = np.linalg.eigh(h_full)
    (gamma_a1_full, gamma_a2_full), _, _ = get_basic_full_gammas(
        levels_to_include=retry2.COMPONENT_LEVELS,
        verbose=False,
        specified_vals=specified_vals,
    )
    h_static_bc_full = builder.subsystem_hamiltonian("B") + builder.subsystem_hamiltonian("C")
    return {
        "builder": builder,
        "eigvecs": eigvecs,
        "h_static_bc_full": h_static_bc_full,
        "gamma_a1_full": gamma_a1_full,
        "gamma_a2_full": gamma_a2_full,
        "specified_vals": specified_vals,
    }


def compute_row(u_value: float, projection_level: int, theta_values: np.ndarray, baseline: dict, context: dict) -> dict:
    print(f"target-match scan: U={u_value:g}, dim P={projection_level}", flush=True)
    builder = context["builder"]
    specified_vals = context["specified_vals"]
    projection_basis = retry2.get_projection_basis(context["eigvecs"], projection_level)
    h_static_bc_projected = retry2.project_operator(context["h_static_bc_full"], projection_basis)

    gamma_a1_full = context["gamma_a1_full"]
    gamma_a2_full = context["gamma_a2_full"]
    gamma0 = normalize_projected_majorana(retry2.project_operator(gamma_a1_full, projection_basis), "gamma0")
    gamma1 = normalize_projected_majorana(retry2.project_operator(gamma_a2_full, projection_basis), "gamma1")

    b_result, c_result = make_majoranas_for_B_and_C_with_projection_dim(
        projection_dim=projection_level,
        specified_vals=specified_vals,
        projection_basis=projection_basis,
        component_levels=retry2.COMPONENT_LEVELS,
        verbose=False,
        tocheck="Minus",
        transition_mode="diagonal",
    )
    gamma2_state = normalize_projected_majorana(b_result["gamma_projected"], "gamma2_state")
    gamma3_state = normalize_projected_majorana(c_result["gamma_projected"], "gamma3_state")

    term_a = 1j * gamma0 @ gamma1
    term_b_state = 1j * gamma0 @ gamma2_state
    term_c_state = 1j * gamma0 @ gamma3_state
    transport_dim = retry2.choose_transport_dim(
        term_a,
        term_b_state,
        term_c_state,
        static_term=h_static_bc_projected,
    )
    state_basis = retry2.get_initial_transport_basis(
        term_a,
        term_b_state,
        term_c_state,
        h_static_bc_projected,
        transport_dim,
    )

    operators = builder.get_operators()
    gamma_b_plus, gamma_b_minus = retry2.projected_majoranas(
        operators["cre"][retry2.B_INNER_SITE],
        operators["ann"][retry2.B_INNER_SITE],
        projection_basis,
    )
    gamma_c_plus, gamma_c_minus = retry2.projected_majoranas(
        operators["cre"][retry2.C_INNER_SITE],
        operators["ann"][retry2.C_INNER_SITE],
        projection_basis,
    )

    best = None
    for theta in theta_values:
        gamma2_chi = chi_from_theta(gamma_b_plus, gamma_b_minus, theta, "gamma2_chi")
        gamma3_chi = chi_from_theta(gamma_c_plus, gamma_c_minus, theta, "gamma3_chi")
        mismatch = retry2.compare_target_gates(
            state_basis,
            gamma2_state,
            gamma3_state,
            gamma2_chi,
            gamma3_chi,
        ) / np.sqrt(transport_dim)
        if best is None or mismatch < best["best_chi_target_mismatch_normalized"]:
            best = {
                "theta_b": float(theta),
                "theta_c": float(theta),
                "best_chi_target_mismatch_normalized": float(mismatch),
                "gamma2_state_vs_best_chi_normalized": float(
                    retry2.operator_mismatch_in_basis(state_basis, gamma2_state, gamma2_chi)
                ),
                "gamma3_state_vs_best_chi_normalized": float(
                    retry2.operator_mismatch_in_basis(state_basis, gamma3_state, gamma3_chi)
                ),
            }

    gamma2_y = chi_from_theta(gamma_b_plus, gamma_b_minus, np.pi / 2.0, "gamma2_y")
    gamma3_y = chi_from_theta(gamma_c_plus, gamma_c_minus, np.pi / 2.0, "gamma3_y")
    local_y_mismatch = retry2.compare_target_gates(
        state_basis,
        gamma2_state,
        gamma3_state,
        gamma2_y,
        gamma3_y,
    ) / np.sqrt(transport_dim)

    baseline_row = baseline[(float(u_value), int(projection_level))]
    row = {
        "interaction_u": float(u_value),
        "projection_level": int(projection_level),
        "transport_dim": int(transport_dim),
        "state_error_normalized": float(baseline_row["ideal_target_gate_error_normalized"]),
        "local_y_state_target_error_normalized": float(
            baseline_row["physical_target_gate_error_in_ideal_basis_normalized"]
        ),
        "local_y_projected_target_error_normalized": float(
            baseline_row["physical_target_gate_error_against_physical_target_in_ideal_basis_normalized"]
        ),
        "local_y_target_mismatch_normalized": float(local_y_mismatch),
    }
    row.update(best)
    print(
        f"  theta={row['theta_b'] / np.pi:.3f} pi, "
        f"state={row['state_error_normalized']:.2e}, "
        f"localY projected target={row['local_y_projected_target_error_normalized']:.2e}, "
        f"target mismatch={row['best_chi_target_mismatch_normalized']:.2e}",
        flush=True,
    )
    return row


def grouped_by_u(rows: list[dict]) -> dict[float, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["interaction_u"], []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row["projection_level"])
    return grouped


def format_projection_axis(axis, levels):
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(levels))
    axis.xaxis.set_major_formatter(FixedFormatter([str(level) for level in levels]))
    axis.minorticks_off()


def plot_rows(rows: list[dict], output_path: Path) -> None:
    grouped = grouped_by_u(rows)
    levels = sorted({row["projection_level"] for row in rows})
    fig, axis = plt.subplots(figsize=(6.4, 4.3))
    for u_value in U_VALUES:
        if u_value not in grouped:
            continue
        style = U_STYLES[u_value]
        x = [row["projection_level"] for row in grouped[u_value]]
        axis.plot(
            x,
            [row["state_error_normalized"] for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
        )
        axis.plot(
            x,
            [row["local_y_projected_target_error_normalized"] for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            linestyle=":",
            linewidth=2.0,
            markersize=6.0,
        )
        axis.plot(
            x,
            [row["best_chi_target_mismatch_normalized"] for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=2.0,
            markersize=6.0,
        )

    axis.set_title(r"Local $\chi(\theta)$ target match")
    axis.set_xlabel(r"Projection dimension $\dim P$")
    axis.set_ylabel("Normalized Error")
    axis.set_yscale("log")
    format_projection_axis(axis, levels)
    axis.grid(True, which="both", alpha=0.25)
    handles = [
        Line2D([0], [0], color="#444444", linestyle="-", linewidth=2.0, label="state-made braid"),
        Line2D([0], [0], color="#444444", linestyle=":", linewidth=2.0, label=r"local $Y$ braid vs local target"),
        Line2D([0], [0], color="#444444", linestyle="--", linewidth=2.0, label=r"best $\chi(\theta)$ target mismatch"),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=U_STYLES[u]["color"],
            marker=U_STYLES[u]["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=fr"$U={u:g}$",
        )
        for u in U_VALUES
        if u in grouped
    )
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--theta-count", type=int, default=49)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    baseline = read_baseline(args.baseline)
    theta_values = np.linspace(0.0, np.pi, args.theta_count)
    rows = []
    for u_value in U_VALUES:
        context = build_u_context(u_value)
        for projection_level in PROJECTION_LEVELS:
            rows.append(compute_row(u_value, projection_level, theta_values, baseline, context))
            save_rows(rows, args.output)
    save_rows(rows, args.output)
    plot_rows(rows, args.fig)
    print(f"wrote {args.output}")
    print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
