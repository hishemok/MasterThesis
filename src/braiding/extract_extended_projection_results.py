from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import expm

try:
    from .extended_projection_braiding import (
        build_hamiltonian,
        build_projection_stack,
        build_total_parity_full,
        build_total_parity_projected,
        delta_pulse,
        evolve_system,
        normalize_projected_majorana,
        phase_aligned_error,
    )
    from .get_mzm_JW import get_full_gammas
    from .hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
except ImportError:
    from extended_projection_braiding import (
        build_hamiltonian,
        build_projection_stack,
        build_total_parity_full,
        build_total_parity_projected,
        delta_pulse,
        evolve_system,
        normalize_projected_majorana,
        phase_aligned_error,
    )
    from get_mzm_JW import get_full_gammas
    from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATED_DIR = REPO_ROOT / "texmex" / "generated"


def flatten_site(subsystem, site, n_sites=3):
    return subsystem * n_sites + site


def build_junction_operator(builder, site_a, site_b):
    operators = builder.get_operators()
    left, right = min(site_a, site_b), max(site_a, site_b)
    key = (left, right)
    dim = operators["num"][0].shape[0]
    junction = np.zeros((dim, dim), dtype=complex)
    junction += -builder.t[0] * operators["hop"][key]
    junction += builder.Delta[0] * operators["pair"][key]
    return 0.5 * (junction + junction.conj().T)


def project_and_normalize(full_operator, basis, label):
    return normalize_projected_majorana(basis.conj().T @ full_operator @ basis, label)


def decompose_against(term, desired):
    dim = term.shape[0]
    coeff = np.trace(desired.conj().T @ term).real / dim
    residual = np.linalg.norm(term - coeff * desired) / np.linalg.norm(term)
    return {"desired_coeff": float(coeff), "relative_residual": float(residual)}


def exchange_metrics(u_kato, gamma_list, transport_dim):
    checks = {
        "gamma2_to_minus_gamma3": (gamma_list[2], -gamma_list[3]),
        "gamma3_to_gamma2": (gamma_list[3], gamma_list[2]),
        "gamma1_to_gamma1": (gamma_list[1], gamma_list[1]),
        "gamma0_to_gamma0": (gamma_list[0], gamma_list[0]),
    }
    errors = {}
    for label, (source, target) in checks.items():
        transformed = u_kato.conj().T @ source @ u_kato
        errors[label] = float(np.linalg.norm(transformed - target))

    max_error = max(errors.values())
    return {
        "single_exchange_errors": errors,
        "max_exchange_error_raw": float(max_error),
        "max_exchange_error_normalized": float(max_error / np.sqrt(transport_dim)),
    }


def parity_gate_metrics(u_kato, v0, parity_op, gamma2, gamma3):
    u_ground = v0.conj().T @ u_kato @ v0
    parity_ground = v0.conj().T @ parity_op @ v0

    parity_vals, parity_vecs = np.linalg.eigh(parity_ground)
    u_parity = parity_vecs.conj().T @ u_ground @ parity_vecs

    odd_indices = np.flatnonzero(parity_vals < -1e-8)
    even_indices = np.flatnonzero(parity_vals > 1e-8)
    mixed_indices = np.flatnonzero(np.abs(parity_vals) <= 1e-8)

    odd_block = u_parity[np.ix_(odd_indices, odd_indices)]
    even_block = u_parity[np.ix_(even_indices, even_indices)]
    off_block = (
        np.linalg.norm(u_parity[np.ix_(odd_indices, even_indices)])
        + np.linalg.norm(u_parity[np.ix_(even_indices, odd_indices)])
    )

    u_target = expm(-0.25 * np.pi * (gamma2 @ gamma3))
    u_target_ground = v0.conj().T @ u_target @ v0
    u_target_parity = parity_vecs.conj().T @ u_target_ground @ parity_vecs
    odd_target = u_target_parity[np.ix_(odd_indices, odd_indices)]
    even_target = u_target_parity[np.ix_(even_indices, even_indices)]

    odd_error_raw = phase_aligned_error(odd_block, odd_target)
    even_error_raw = phase_aligned_error(even_block, even_target)

    return {
        "parity_eigenvalues": [float(value) for value in np.real_if_close(parity_vals)],
        "odd_dim": int(len(odd_indices)),
        "even_dim": int(len(even_indices)),
        "mixed_dim": int(len(mixed_indices)),
        "off_block_leakage_raw": float(off_block),
        "off_block_leakage_normalized": float(off_block / np.sqrt(max(1, len(parity_vals)))),
        "odd_target_error_raw": float(odd_error_raw),
        "odd_target_error_normalized": float(odd_error_raw / np.sqrt(max(1, len(odd_indices)))),
        "even_target_error_raw": float(even_error_raw),
        "even_target_error_normalized": float(even_error_raw / np.sqrt(max(1, len(even_indices)))),
    }


def transported_endpoint_data(t_total, delta_max, delta_min, steepness, width, gamma_list, transport_dim):
    h0, _ = build_hamiltonian(0.0, t_total, delta_max, delta_min, steepness, width, *gamma_list)
    h1, _ = build_hamiltonian(t_total, t_total, delta_max, delta_min, steepness, width, *gamma_list)

    evals_0, evecs_0 = np.linalg.eigh(h0)
    evals_1, evecs_1 = np.linalg.eigh(h1)
    v0 = evecs_0[:, :transport_dim]
    v1 = evecs_1[:, :transport_dim]
    p0 = v0 @ v0.conj().T
    p1 = v1 @ v1.conj().T
    return {
        "evals_0": evals_0,
        "evals_1": evals_1,
        "V0": v0,
        "VT": v1,
        "P0": p0,
        "PT": p1,
    }


def pulse_couplings(t_value, args, steepness, width):
    delta_1 = (
        delta_pulse(t_value, 0.0, width, steepness, args.delta_max, args.delta_min)
        + delta_pulse(t_value, args.t_total, width, steepness, args.delta_max, args.delta_min)
        - args.delta_min
    )
    delta_2 = delta_pulse(t_value, args.t_total / 3.0, width, steepness, args.delta_max, args.delta_min)
    delta_3 = delta_pulse(t_value, 2.0 * args.t_total / 3.0, width, steepness, args.delta_max, args.delta_min)
    return delta_1, delta_2, delta_3


def build_term_hamiltonian(t_value, args, steepness, width, term_1, term_2, term_3):
    delta_1, delta_2, delta_3 = pulse_couplings(t_value, args, steepness, width)
    hamiltonian = delta_1 * term_1 + delta_2 * term_2 + delta_3 * term_3
    return hamiltonian, (delta_1, delta_2, delta_3)


def evolve_term_protocol(term_1, term_2, term_3, args, steepness, width, transport_dim):
    times = np.linspace(0.0, args.t_total, args.n_points)
    dt = times[1] - times[0] if args.n_points > 1 else args.t_total
    dim = term_1.shape[0]
    energies = np.zeros((args.n_points, dim))
    couplings = np.zeros((args.n_points, 3))
    u_kato = np.eye(dim, dtype=complex)

    hamiltonian, couplings[0] = build_term_hamiltonian(times[0], args, steepness, width, term_1, term_2, term_3)
    evals, evecs = np.linalg.eigh(hamiltonian)
    energies[0] = evals
    basis = evecs[:, :transport_dim]

    for idx in range(1, len(times)):
        hamiltonian, couplings[idx] = build_term_hamiltonian(
            times[idx],
            args,
            steepness,
            width,
            term_1,
            term_2,
            term_3,
        )
        evals, evecs = np.linalg.eigh(hamiltonian)
        energies[idx] = evals
        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        u_kato = expm(-dt * kato_generator) @ u_kato
        basis = next_basis

    return times, energies, couplings, u_kato


def endpoint_data_from_terms(term_1, term_2, term_3, args, steepness, width, transport_dim):
    h0, _ = build_term_hamiltonian(0.0, args, steepness, width, term_1, term_2, term_3)
    h1, _ = build_term_hamiltonian(args.t_total, args, steepness, width, term_1, term_2, term_3)
    evals_0, evecs_0 = np.linalg.eigh(h0)
    evals_1, evecs_1 = np.linalg.eigh(h1)
    v0 = evecs_0[:, :transport_dim]
    v1 = evecs_1[:, :transport_dim]
    p0 = v0 @ v0.conj().T
    p1 = v1 @ v1.conj().T
    return {
        "evals_0": evals_0,
        "evals_1": evals_1,
        "V0": v0,
        "VT": v1,
        "P0": p0,
        "PT": p1,
    }


def build_result_row(
    model,
    u_value,
    args,
    projection_blocks,
    eigvals,
    block_index,
    projection_dim,
    transport_dim,
    times,
    energies,
    u_kato,
    ground_data,
    gamma_list,
    parity_projected,
    ab_decomposition,
    ac_decomposition,
):
    stop = projection_blocks[block_index]["stop"]
    next_gap = float(eigvals[stop] - eigvals[stop - 1]) if stop < len(eigvals) else None
    max_transport_splitting = float(np.max(energies[:, transport_dim - 1] - energies[:, 0]))
    min_gap_above_transport = float(np.min(energies[:, transport_dim] - energies[:, transport_dim - 1]))
    adiabatic_lower_time = float(1.0 / min_gap_above_transport)
    degeneracy_upper_time = float(1.0 / max_transport_splitting) if max_transport_splitting > 0 else float("inf")
    exchange = exchange_metrics(u_kato, gamma_list, transport_dim)
    gate = parity_gate_metrics(u_kato, ground_data["V0"], parity_projected, gamma_list[2], gamma_list[3])
    group_dims = [projection_blocks[idx]["dim"] for idx in range(block_index + 1)]

    return {
        "model": model,
        "U": float(u_value),
        "included_groups": f"0-{block_index}",
        "group_dims": "+".join(str(dim) for dim in group_dims),
        "projection_dim": int(projection_dim),
        "transport_dim": int(transport_dim),
        "hfull_spread_included": float(eigvals[stop - 1] - eigvals[0]),
        "hfull_gap_to_next": next_gap,
        "max_transport_splitting": max_transport_splitting,
        "min_gap_above_transport": min_gap_above_transport,
        "adiabatic_lower_time": adiabatic_lower_time,
        "degeneracy_upper_time": degeneracy_upper_time,
        "adiabatic_window_ratio": float(degeneracy_upper_time / adiabatic_lower_time),
        "suggested_geometric_mean_time": float(np.sqrt(adiabatic_lower_time * degeneracy_upper_time)),
        "requested_T": float(args.t_total),
        "T_in_simple_window": bool(adiabatic_lower_time < args.t_total < degeneracy_upper_time),
        "holonomy_unitarity_error": float(
            np.linalg.norm(u_kato.conj().T @ u_kato - np.eye(projection_dim, dtype=complex))
        ),
        "loop_closure_error": float(np.linalg.norm(ground_data["PT"] - ground_data["P0"])),
        "ab_decomposition": ab_decomposition,
        "ac_decomposition": ac_decomposition,
        **exchange,
        "parity_gate": gate,
    }


def collect_case(u_value, args):
    specified_vals = {"U": [u_value]}
    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )
    h_full = builder.full_system_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(h_full)
    parity_full = build_total_parity_full(builder)
    projection_blocks = build_projection_stack(
        eigvals,
        eigvecs,
        parity_full,
        energy_tol=args.degeneracy_tol,
    )
    basis_stack = [block["basis"] for block in projection_blocks]
    cumulative_stack = [np.hstack(basis_stack[: idx + 1]) for idx in range(len(basis_stack))]

    (gamma_a1_full, gamma_a2_full), (_, gamma_b2_full), (_, gamma_c2_full) = get_full_gammas(
        levels_to_include=args.levels,
        verbose=False,
        specified_vals=specified_vals,
    )

    junction_ab_full = build_junction_operator(
        builder,
        flatten_site(0, 2),
        flatten_site(1, 0),
    )
    junction_ac_full = build_junction_operator(
        builder,
        flatten_site(0, 2),
        flatten_site(2, 0),
    )

    width = args.t_total / 3.0
    steepness = 20.0 / width
    ideal_results = []
    physical_results = []

    for block_index, basis in enumerate(cumulative_stack):
        projection_dim = basis.shape[1]
        if projection_dim > args.max_dim:
            break

        transport_dim = projection_dim // 2
        if projection_dim % 2 != 0:
            continue

        gamma_a1 = project_and_normalize(gamma_a1_full, basis, "gamma_a1")
        gamma_a2 = project_and_normalize(gamma_a2_full, basis, "gamma_a2")
        gamma_b2 = project_and_normalize(gamma_b2_full, basis, "gamma_b2")
        gamma_c2 = project_and_normalize(gamma_c2_full, basis, "gamma_c2")
        gamma_list = [gamma_a1, gamma_a2, gamma_b2, gamma_c2]

        parity_projected = build_total_parity_projected(builder, basis)

        junction_ab = basis.conj().T @ junction_ab_full @ basis
        junction_ac = basis.conj().T @ junction_ac_full @ basis
        desired_ab = 1j * gamma_a1 @ gamma_b2
        desired_ac = 1j * gamma_a1 @ gamma_c2
        ab_decomposition = decompose_against(junction_ab, desired_ab)
        ac_decomposition = decompose_against(junction_ac, desired_ac)

        ideal_row = None
        if args.models in {"ideal", "both"}:
            ideal_times, ideal_energies, _, ideal_u_kato = evolve_system(
                args.t_total,
                args.delta_max,
                args.delta_min,
                steepness,
                width,
                gamma_a1,
                gamma_a2,
                gamma_b2,
                gamma_c2,
                n_points=args.n_points,
                transport_dim=transport_dim,
            )
            ideal_ground_data = transported_endpoint_data(
                args.t_total,
                args.delta_max,
                args.delta_min,
                steepness,
                width,
                gamma_list,
                transport_dim,
            )
            ideal_row = build_result_row(
                "ideal_projected_majorana",
                u_value,
                args,
                projection_blocks,
                eigvals,
                block_index,
                projection_dim,
                transport_dim,
                ideal_times,
                ideal_energies,
                ideal_u_kato,
                ideal_ground_data,
                gamma_list,
                parity_projected,
                ab_decomposition,
                ac_decomposition,
            )
            ideal_results.append(ideal_row)

        physical_row = None
        if args.models in {"physical", "both"}:
            term_a = 1j * gamma_a1 @ gamma_a2
            physical_times, physical_energies, _, physical_u_kato = evolve_term_protocol(
                term_a,
                junction_ab,
                junction_ac,
                args,
                steepness,
                width,
                transport_dim,
            )
            physical_ground_data = endpoint_data_from_terms(
                term_a,
                junction_ab,
                junction_ac,
                args,
                steepness,
                width,
                transport_dim,
            )
            physical_row = build_result_row(
                "physical_projected_junction",
                u_value,
                args,
                projection_blocks,
                eigvals,
                block_index,
                projection_dim,
                transport_dim,
                physical_times,
                physical_energies,
                physical_u_kato,
                physical_ground_data,
                gamma_list,
                parity_projected,
                ab_decomposition,
                ac_decomposition,
            )
            physical_results.append(physical_row)

        status = [f"U={u_value:g}", f"dim={projection_dim:>3}"]
        if ideal_row is not None:
            status.append(
                "ideal_gate="
                f"{max(ideal_row['parity_gate']['odd_target_error_normalized'], ideal_row['parity_gate']['even_target_error_normalized']):.3e}"
            )
        if physical_row is not None:
            status.append(
                "physical_gate="
                f"{max(physical_row['parity_gate']['odd_target_error_normalized'], physical_row['parity_gate']['even_target_error_normalized']):.3e}"
            )
        status.append(f"residual={max(ab_decomposition['relative_residual'], ac_decomposition['relative_residual']):.3e}")
        print(" ".join(status))

    return {
        "U": float(u_value),
        "selection": builder.selection,
        "groups": [
            {
                "group": block["group_index"],
                "dim": block["dim"],
                "even_dim": block["even_dim"],
                "odd_dim": block["odd_dim"],
                "mixed_dim": block["mixed_dim"],
                "energy_min": block["energy_min"],
                "energy_max": block["energy_max"],
                "energy_spread": block["energy_spread"],
                "gap_to_next": block["gap_to_next"],
            }
            for block in projection_blocks
        ],
        "results": physical_results,
        "physical_results": physical_results,
        "ideal_results": ideal_results,
    }


def metric(value):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(value)
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        return f"{value:.3e}"
    return f"{value:.3f}"


def flatten_rows(cases, result_key="results"):
    rows = []
    for case in cases:
        for result in case[result_key]:
            rows.append(
                {
                    "model": result["model"],
                    "U": result["U"],
                    "included_groups": result["included_groups"],
                    "group_dims": result["group_dims"],
                    "projection_dim": result["projection_dim"],
                    "transport_dim": result["transport_dim"],
                    "hfull_spread_included": result["hfull_spread_included"],
                    "hfull_gap_to_next": result["hfull_gap_to_next"],
                    "max_transport_splitting": result["max_transport_splitting"],
                    "min_gap_above_transport": result["min_gap_above_transport"],
                    "adiabatic_lower_time": result["adiabatic_lower_time"],
                    "degeneracy_upper_time": result["degeneracy_upper_time"],
                    "adiabatic_window_ratio": result["adiabatic_window_ratio"],
                    "suggested_geometric_mean_time": result["suggested_geometric_mean_time"],
                    "requested_T": result["requested_T"],
                    "T_in_simple_window": result["T_in_simple_window"],
                    "holonomy_unitarity_error": result["holonomy_unitarity_error"],
                    "loop_closure_error": result["loop_closure_error"],
                    "ab_desired_coeff": result["ab_decomposition"]["desired_coeff"],
                    "ab_relative_residual": result["ab_decomposition"]["relative_residual"],
                    "ac_desired_coeff": result["ac_decomposition"]["desired_coeff"],
                    "ac_relative_residual": result["ac_decomposition"]["relative_residual"],
                    "max_exchange_error_raw": result["max_exchange_error_raw"],
                    "max_exchange_error_normalized": result["max_exchange_error_normalized"],
                    "off_block_leakage_raw": result["parity_gate"]["off_block_leakage_raw"],
                    "off_block_leakage_normalized": result["parity_gate"]["off_block_leakage_normalized"],
                    "odd_target_error_raw": result["parity_gate"]["odd_target_error_raw"],
                    "odd_target_error_normalized": result["parity_gate"]["odd_target_error_normalized"],
                    "even_target_error_raw": result["parity_gate"]["even_target_error_raw"],
                    "even_target_error_normalized": result["parity_gate"]["even_target_error_normalized"],
                }
            )
    return rows


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path, rows):
    lines = [
        r"\begin{tabular}{ccccccccc}",
        r"\hline",
        r"$U$ & Groups & $\dim P$ & $E_\delta$ & $E_{\mathrm{gap}}$ & $1/E_\delta$ & Resid. & Braid err.$/\sqrt{d}$ & Gate err.$/\sqrt{d}$ \\",
        r"\hline",
    ]
    for row in rows:
        residual = max(row["ab_relative_residual"], row["ac_relative_residual"])
        gate_error = max(row["odd_target_error_normalized"], row["even_target_error_normalized"])
        lines.append(
            " & ".join(
                [
                    f"{row['U']:.1f}",
                    f"{row['included_groups']} ({row['group_dims']})",
                    str(row["projection_dim"]),
                    metric(row["max_transport_splitting"]),
                    metric(row["min_gap_above_transport"]),
                    metric(row["degeneracy_upper_time"]),
                    metric(residual),
                    metric(row["max_exchange_error_normalized"]),
                    metric(gate_error),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compact_tex(path, rows):
    lines = [
        r"\begin{tabular}{cccccccccc}",
        r"\hline",
        r"$U$ & Levels & $\dim P$ & $d_p$ & Braid raw & Braid norm. & Odd raw & Odd norm. & Even raw & Even norm. \\",
        r"\hline",
    ]
    for row in rows:
        parity_dim = int(round(float(row["transport_dim"]) / 2.0))
        lines.append(
            " & ".join(
                [
                    f"{float(row['U']):.1f}",
                    row["included_groups"],
                    str(row["projection_dim"]),
                    str(parity_dim),
                    metric(row["max_exchange_error_raw"]),
                    metric(row["max_exchange_error_normalized"]),
                    metric(row["odd_target_error_raw"]),
                    metric(row["odd_target_error_normalized"]),
                    metric(row["even_target_error_raw"]),
                    metric(row["even_target_error_normalized"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Extract extended-projection braiding errors for the thesis.")
    parser.add_argument("--u-values", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--degeneracy-tol", type=float, default=1e-2)
    parser.add_argument("--max-dim", type=int, default=80)
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--models", choices=["ideal", "physical", "both"], default="both")
    parser.add_argument("--t-total", type=float, default=1000.0)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    return parser


def main():
    args = build_argument_parser().parse_args()
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    cases = [collect_case(u_value, args) for u_value in args.u_values]
    ideal_rows = flatten_rows(cases, "ideal_results")
    physical_rows = flatten_rows(cases, "physical_results")

    payload = {
        "args": {
            "u_values": args.u_values,
            "levels": args.levels,
            "degeneracy_tol": args.degeneracy_tol,
            "max_dim": args.max_dim,
            "n_points": args.n_points,
            "models": args.models,
            "t_total": args.t_total,
            "delta_max": args.delta_max,
            "delta_min": args.delta_min,
            "generated_dir": str(args.generated_dir),
        },
        "active_majoranas": ["gammaA1", "gammaA2", "gammaB2", "gammaC2"],
        "results": cases,
    }

    output_base = args.generated_dir / "extended_projection_braiding"
    if ideal_rows:
        write_csv(output_base.with_suffix(".csv"), ideal_rows)
        write_tex(output_base.with_suffix(".tex"), ideal_rows)
        write_compact_tex(output_base.with_name(f"{output_base.name}_compact.tex"), ideal_rows)
    output_base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    physical_output_base = args.generated_dir / "extended_projection_physical_junction_braiding"
    if physical_rows:
        write_csv(physical_output_base.with_suffix(".csv"), physical_rows)
        write_tex(physical_output_base.with_suffix(".tex"), physical_rows)
        write_compact_tex(
            physical_output_base.with_name(f"{physical_output_base.name}_compact.tex"),
            physical_rows,
        )

    if ideal_rows:
        print(f"Saved {output_base.with_suffix('.csv')}")
        print(f"Saved {output_base.with_suffix('.tex')}")
        print(f"Saved {output_base.with_name(f'{output_base.name}_compact.tex')}")
    print(f"Saved {output_base.with_suffix('.json')}")
    if physical_rows:
        print(f"Saved {physical_output_base.with_suffix('.csv')}")
        print(f"Saved {physical_output_base.with_suffix('.tex')}")
        print(f"Saved {physical_output_base.with_name(f'{physical_output_base.name}_compact.tex')}")


if __name__ == "__main__":
    main()
