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


OUTPUT_PATH = Path(__file__).with_name("retry2_chi_theta_scan_dims_8_32_56_80.txt")
FIG_PATH = REPO_ROOT / "texmex" / "figs" / "chi_theta_physical_braid_error_80.pdf"
DETAIL_FIG_PATH = REPO_ROOT / "texmex" / "figs" / "chi_theta_best_angles_80.pdf"

U_VALUES = [0.0, 0.1, 2.0]
PROJECTION_LEVELS = [8, 32, 56, 80]
THETA_VALUES = np.linspace(0.0, np.pi, 25)

FIELDNAMES = [
    "interaction_u",
    "projection_level",
    "transport_dim",
    "theta_b",
    "theta_c",
    "state_error",
    "local_y_error",
    "chi_theta_error",
    "state_error_normalized",
    "local_y_error_normalized",
    "chi_theta_error_normalized",
    "state_vs_chi_target_mismatch_normalized",
    "gamma2_state_vs_chi_normalized",
    "gamma3_state_vs_chi_normalized",
    "chi_majorana_max_square_error_normalized",
    "chi_majorana_max_anticommutator_error_normalized",
]

U_STYLES = {
    0.0: {"color": "#1f77b4", "marker": "o"},
    0.1: {"color": "#ff7f0e", "marker": "s"},
    2.0: {"color": "#d62728", "marker": "D"},
}


def chi_from_theta(gamma_plus: np.ndarray, gamma_minus: np.ndarray, theta: float, label: str) -> np.ndarray:
    gamma = np.cos(theta) * gamma_plus + np.sin(theta) * gamma_minus
    return normalize_projected_majorana(gamma, label)


def silent_majorana_algebra(gammas: list[np.ndarray]) -> dict:
    identity = np.eye(gammas[0].shape[0], dtype=complex)
    square_errors = {
        f"gamma{index}": retry2.normalized_error(gamma @ gamma, identity)
        for index, gamma in enumerate(gammas)
    }
    anticommutator_errors = {}
    for left in range(len(gammas)):
        for right in range(left + 1, len(gammas)):
            anticommutator = gammas[left] @ gammas[right] + gammas[right] @ gammas[left]
            anticommutator_errors[f"gamma{left}_gamma{right}"] = (
                np.linalg.norm(anticommutator) / np.sqrt(anticommutator.shape[0])
            )
    return {
        "max_square_error": max(square_errors.values()),
        "max_anticommutator_error": max(anticommutator_errors.values()),
    }


def theta_pairs(theta_values: np.ndarray, independent_angles: bool):
    if independent_angles:
        for theta_b in theta_values:
            for theta_c in theta_values:
                yield theta_b, theta_c
    else:
        for theta in theta_values:
            yield theta, theta


def evaluate_case(
    u_value: float,
    projection_level: int,
    theta_values: np.ndarray,
    independent_angles: bool,
) -> dict:
    print(f"\nchi(theta) scan: U={u_value:g}, dim P={projection_level}")
    specified_vals = {"U": [u_value]}

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )
    h_full = builder.full_system_hamiltonian()
    _, eigvecs = np.linalg.eigh(h_full)
    projection_basis = retry2.get_projection_basis(eigvecs, projection_level)

    h_static_bc_full = builder.subsystem_hamiltonian("B") + builder.subsystem_hamiltonian("C")
    h_static_bc_projected = retry2.project_operator(h_static_bc_full, projection_basis)

    (gamma_a1_full, gamma_a2_full), _, _ = get_basic_full_gammas(
        levels_to_include=retry2.COMPONENT_LEVELS,
        verbose=False,
        specified_vals=specified_vals,
    )
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

    print("  evolving state-made reference")
    _, _, _, u_state = retry2.evolve_system(
        term_a,
        term_b_state,
        term_c_state,
        static_term=h_static_bc_projected,
        transport_dim=transport_dim,
    )
    state_error = retry2.compare_to_target_gate(u_state, state_basis, gamma2_state, gamma3_state)

    best = None
    for theta_b, theta_c in theta_pairs(theta_values, independent_angles):
        gamma2_chi = chi_from_theta(gamma_b_plus, gamma_b_minus, theta_b, "gamma2_chi")
        gamma3_chi = chi_from_theta(gamma_c_plus, gamma_c_minus, theta_c, "gamma3_chi")
        target_mismatch = retry2.compare_target_gates(
            state_basis,
            gamma2_state,
            gamma3_state,
            gamma2_chi,
            gamma3_chi,
        )
        normalized_mismatch = target_mismatch / np.sqrt(transport_dim)
        if best is None or normalized_mismatch < best["state_vs_chi_target_mismatch_normalized"]:
            algebra = silent_majorana_algebra([gamma0, gamma1, gamma2_chi, gamma3_chi])
            best = {
                "theta_b": float(theta_b),
                "theta_c": float(theta_c),
                "state_vs_chi_target_mismatch_normalized": float(target_mismatch / np.sqrt(transport_dim)),
                "gamma2_state_vs_chi_normalized": float(
                    retry2.operator_mismatch_in_basis(state_basis, gamma2_state, gamma2_chi)
                ),
                "gamma3_state_vs_chi_normalized": float(
                    retry2.operator_mismatch_in_basis(state_basis, gamma3_state, gamma3_chi)
                ),
                "chi_majorana_max_square_error_normalized": float(algebra["max_square_error"]),
                "chi_majorana_max_anticommutator_error_normalized": float(algebra["max_anticommutator_error"]),
                "_gamma2_chi": gamma2_chi,
                "_gamma3_chi": gamma3_chi,
            }

    gamma2_best = best["_gamma2_chi"]
    gamma3_best = best["_gamma3_chi"]
    _, _, _, u_chi = retry2.evolve_system(
        term_a,
        1j * gamma0 @ gamma2_best,
        1j * gamma0 @ gamma3_best,
        static_term=h_static_bc_projected,
        transport_dim=transport_dim,
    )
    chi_error = retry2.compare_to_target_gate(u_chi, state_basis, gamma2_state, gamma3_state)
    best["chi_theta_error"] = float(chi_error)
    best["chi_theta_error_normalized"] = float(chi_error / np.sqrt(transport_dim))

    gamma2_y = chi_from_theta(gamma_b_plus, gamma_b_minus, np.pi / 2.0, "gamma2_y")
    gamma3_y = chi_from_theta(gamma_c_plus, gamma_c_minus, np.pi / 2.0, "gamma3_y")
    _, _, _, u_y = retry2.evolve_system(
        term_a,
        1j * gamma0 @ gamma2_y,
        1j * gamma0 @ gamma3_y,
        static_term=h_static_bc_projected,
        transport_dim=transport_dim,
    )
    local_y_error = retry2.compare_to_target_gate(u_y, state_basis, gamma2_state, gamma3_state)

    row = {
        "interaction_u": float(u_value),
        "projection_level": int(projection_level),
        "transport_dim": int(transport_dim),
        "state_error": float(state_error),
        "local_y_error": float(local_y_error),
        "state_error_normalized": float(state_error / np.sqrt(transport_dim)),
        "local_y_error_normalized": float(local_y_error / np.sqrt(transport_dim)),
    }
    row.update({key: value for key, value in best.items() if not key.startswith("_")})

    print(
        "  best: "
        f"theta_B={row['theta_b'] / np.pi:.3f} pi, "
        f"theta_C={row['theta_c'] / np.pi:.3f} pi, "
        f"state={row['state_error_normalized']:.3e}, "
        f"Y={row['local_y_error_normalized']:.3e}, "
        f"chi={row['chi_theta_error_normalized']:.3e}"
    )
    return row


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key in {"projection_level", "transport_dim"}:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def grouped_by_u(rows: list[dict]) -> dict[float, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["interaction_u"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["projection_level"])
    return grouped


def format_projection_axis(axis, levels):
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_locator(FixedLocator(levels))
    axis.xaxis.set_major_formatter(FixedFormatter([str(level) for level in levels]))
    axis.minorticks_off()


def plot_results(rows: list[dict], output_path: Path, detail_output_path: Path) -> None:
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
            [row["local_y_error_normalized"] for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            linestyle=":",
            linewidth=2.0,
            markersize=6.0,
        )
        axis.plot(
            x,
            [row["chi_theta_error_normalized"] for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            linestyle="--",
            linewidth=2.0,
            markersize=6.0,
        )

    axis.set_title(r"Physical braid with optimized local $\chi(\theta)$")
    axis.set_xlabel(r"Projection dimension $\dim P$")
    axis.set_ylabel("Normalized Error")
    axis.set_yscale("log")
    format_projection_axis(axis, levels)
    axis.grid(True, which="both", alpha=0.25)
    handles = [
        Line2D([0], [0], color="#444444", linestyle="-", linewidth=2.0, label="state-made"),
        Line2D([0], [0], color="#444444", linestyle=":", linewidth=2.0, label=r"local $Y$ only"),
        Line2D([0], [0], color="#444444", linestyle="--", linewidth=2.0, label=r"best local $\chi(\theta)$"),
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
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    for u_value in U_VALUES:
        if u_value not in grouped:
            continue
        style = U_STYLES[u_value]
        x = [row["projection_level"] for row in grouped[u_value]]
        axes[0].plot(
            x,
            [row["theta_b"] / np.pi for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=fr"$U={u_value:g}$",
        )
        axes[1].plot(
            x,
            [row["theta_c"] / np.pi for row in grouped[u_value]],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
        )
    axes[0].set_title(r"Best $\theta_B/\pi$")
    axes[1].set_title(r"Best $\theta_C/\pi$")
    for axis in axes:
        axis.set_xlabel(r"Projection dimension $\dim P$")
        axis.set_ylabel(r"$\theta/\pi$")
        axis.set_ylim(-0.04, 1.04)
        format_projection_axis(axis, levels)
        axis.grid(True, alpha=0.25)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    detail_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(detail_output_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(detail_output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan local chi(theta) physical braid errors up to dim P=80.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--fig", type=Path, default=FIG_PATH)
    parser.add_argument("--detail-fig", type=Path, default=DETAIL_FIG_PATH)
    parser.add_argument("--n-points", type=int, default=retry2.N_POINTS)
    parser.add_argument("--theta-count", type=int, default=len(THETA_VALUES))
    parser.add_argument("--independent-angles", action="store_true")
    parser.add_argument("--u-values", nargs="+", type=float, default=U_VALUES)
    parser.add_argument("--projection-levels", nargs="+", type=int, default=PROJECTION_LEVELS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    retry2.N_POINTS = args.n_points
    retry2.VERBOSE = False
    theta_values = np.linspace(0.0, np.pi, args.theta_count)

    rows = [] if args.force else load_rows(args.output)
    done = {(row["interaction_u"], row["projection_level"]) for row in rows}
    for u_value in args.u_values:
        for projection_level in args.projection_levels:
            key = (float(u_value), int(projection_level))
            if key in done:
                print(f"skipping existing chi(theta) row U={u_value:g}, dim P={projection_level}")
                continue
            rows.append(
                evaluate_case(
                    float(u_value),
                    int(projection_level),
                    theta_values,
                    independent_angles=args.independent_angles,
                )
            )
            rows.sort(key=lambda row: (row["interaction_u"], row["projection_level"]))
            save_rows(rows, args.output)

    plot_results(rows, args.fig, args.detail_fig)
    print(f"\nwrote {args.output}")
    print(f"wrote {args.fig}")
    print(f"wrote {args.detail_fig}")


if __name__ == "__main__":
    main()
