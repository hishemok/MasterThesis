import argparse
import csv
from pathlib import Path

from remake_majoranas import candidate_metadata, get_full_gammas
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse, plot_results
from extended_projection_braiding import normalize_projected_majorana


OPERATOR_ACTION_COLUMNS = [
    ("γ2 -> -γ3", "gamma2_to_minus_gamma3"),
    ("γ3 ->  γ2", "gamma3_to_gamma2"),
    ("γ1 ->  γ1", "gamma1_to_gamma1"),
    ("γ0 ->  γ0", "gamma0_to_gamma0"),
]



def build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC):
    Δ1 = delta_pulse(t, 0, width, s, Δ_max, Δ_min) + delta_pulse(t, T_total, width, s, Δ_max, Δ_min) - Δ_min
    Δ2 = delta_pulse(t, T_total / 3, width, s, Δ_max, Δ_min)
    Δ3 = delta_pulse(t, 2 * T_total / 3, width, s, Δ_max, Δ_min)

    H = Δ1 * T_A + Δ2 * T_AB + Δ3 * T_AC
    return H, (Δ1, Δ2, Δ3)

def get_projection_basis(eigenvectors, levels_to_include):
    basis = eigenvectors[:, :levels_to_include]
    overlap = basis.conj().T @ basis

    # The sliced eigenvectors form an orthonormal basis for the projected space.
    if not np.allclose(overlap, np.eye(levels_to_include, dtype=complex)):
        raise ValueError("Projection basis is not orthonormal: V†V != I")
    return basis

def projected_majoranas(cdag, c, P):
    gamma_1 = cdag + c
    gamma_2 = 1j * (cdag - c)

    A1 = P.conj().T @ gamma_1 @ P
    A2 = P.conj().T @ gamma_2 @ P

    # Majoranas should be Hermitian. After projection they generally do not remain
    # idempotent, so checking A^2 = A is not the right condition here.
    if not np.allclose(A1, A1.conj().T):
        print("Warning: Projected Majorana A1 is not Hermitian: A1 != A1†")
    if not np.allclose(A2, A2.conj().T):
        print("Warning: Projected Majorana A2 is not Hermitian: A2 != A2†")

    return A1, A2

def project_ideal_majoranas(gamma, P):
    projected_gamma = P.conj().T @ gamma @ P
    return projected_gamma




def evolve_system(
    t_total,
    delta_max,
    delta_min,
    steepness,
    width,
    term_a,
    term_b,
    term_c,
    n_points=1000,
    transport_dim=None,
    verbose=False,
):
    times = np.linspace(0, t_total, n_points)
    dt = times[1] - times[0] if n_points > 1 else t_total

    dim = term_a.shape[0]
    if transport_dim is None:
        transport_dim = dim // 2
    if not 0 < transport_dim < dim:
        raise ValueError(f"transport_dim must be between 1 and {dim - 1}, got {transport_dim}.")

    energies = np.zeros((n_points, dim))
    couplings = np.zeros((n_points, 3))
    u_kato = np.eye(dim, dtype=complex)

    hamiltonian, couplings[0] = build_projected_hamiltonian(times[0], t_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c)
    evals, evecs = np.linalg.eigh(hamiltonian)
    energies[0] = evals
    basis = evecs[:, :transport_dim]

    if verbose:
        print("Analyzing ad2 projected braid...")
    for idx in tqdm(range(1, len(times)), total=len(times) - 1, disable=not verbose):
        hamiltonian, couplings[idx] = build_projected_hamiltonian(
            times[idx],
            t_total,
            delta_max,
            delta_min,
            steepness,
            width,
            term_a,
            term_b,
            term_c,
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


def phase_aligned_error(U, target):
    overlap = np.trace(target.conj().T @ U)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(U - np.exp(1j * phase) * target)


def get_initial_transport_basis(
    t_total,
    delta_max,
    delta_min,
    steepness,
    width,
    term_a,
    term_b,
    term_c,
    transport_dim,
):
    hamiltonian_0, _ = build_projected_hamiltonian(
        0.0, t_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c
    )
    _, evecs_0 = np.linalg.eigh(hamiltonian_0)
    return evecs_0[:, :transport_dim]


def check_operator_action(u_kato, gamma_list):
    expected_maps = [
        ("γ2 -> -γ3", gamma_list[2], -gamma_list[3]),
        ("γ3 ->  γ2", gamma_list[3], gamma_list[2]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    errors = {}
    for label, source, target in expected_maps:
        transformed = u_kato.conj().T @ source @ u_kato
        errors[label] = np.linalg.norm(transformed - target)

    return {
        "errors": errors,
        "max_error": max(errors.values()) if errors else 0.0,
    }


def format_operator_action(action_check):
    ordered_labels = [
        "γ2 -> -γ3",
        "γ3 ->  γ2",
        "γ1 ->  γ1",
        "γ0 ->  γ0",
    ]
    return ", ".join(
        f"{label}={action_check['errors'][label]:.2e}"
        for label in ordered_labels
    )


def compare_to_target_gate(u_kato, transport_basis, gamma2_target, gamma3_target):
    u_subspace = transport_basis.conj().T @ u_kato @ transport_basis
    target_full = expm(-0.25 * np.pi * (gamma2_target @ gamma3_target))
    target_subspace = transport_basis.conj().T @ target_full @ transport_basis
    return phase_aligned_error(u_subspace, target_subspace)


def flatten_operator_action(prefix, action_check):
    row = {}
    for label, column_suffix in OPERATOR_ACTION_COLUMNS:
        row[f"{prefix}_{column_suffix}_error"] = float(action_check["errors"][label])
    row[f"{prefix}_max_error"] = float(action_check["max_error"])
    return row


def flatten_candidate(prefix, candidate):
    return {
        f"{prefix}_target_axis": candidate["target_axis"],
        f"{prefix}_overlap_x": float(candidate["overlap_x"]),
        f"{prefix}_overlap_y": float(candidate["overlap_y"]),
        f"{prefix}_score": float(candidate["score"]),
        f"{prefix}_coefficients_plus": ",".join(f"{value:.6g}" for value in candidate["coefficients_plus"]),
        f"{prefix}_coefficients_minus": ",".join(f"{value:.6g}" for value in candidate["coefficients_minus"]),
    }


def build_result_row(
    u_value,
    projection_level,
    transport_dim,
    best_b_candidate,
    best_c_candidate,
    ideal_operator_action,
    physical_operator_action_ideal_basis,
    physical_operator_action_physical_basis,
    ideal_target_error,
    physical_target_error_ideal_basis,
    physical_target_error_physical_basis,
):
    row = {
        "interaction_u": float(u_value),
        "projection_level": int(projection_level),
        "transport_dim": int(transport_dim),
        "ideal_target_gate_error": float(ideal_target_error),
        "physical_target_gate_error_in_ideal_basis": float(physical_target_error_ideal_basis),
        "physical_target_gate_error_in_physical_basis": float(physical_target_error_physical_basis),
    }
    row.update(flatten_operator_action("ideal_single_exchange", ideal_operator_action))
    row.update(
        flatten_operator_action(
            "physical_single_exchange_in_ideal_basis",
            physical_operator_action_ideal_basis,
        )
    )
    row.update(
        flatten_operator_action(
            "physical_single_exchange_in_physical_basis",
            physical_operator_action_physical_basis,
        )
    )
    row.update(flatten_candidate("best_b_candidate", best_b_candidate))
    row.update(flatten_candidate("best_c_candidate", best_c_candidate))
    return row


def result_fieldnames():
    fieldnames = [
        "interaction_u",
        "projection_level",
        "transport_dim",
        "ideal_target_gate_error",
        "physical_target_gate_error_in_ideal_basis",
        "physical_target_gate_error_in_physical_basis",
        "best_b_candidate_target_axis",
        "best_b_candidate_overlap_x",
        "best_b_candidate_overlap_y",
        "best_b_candidate_score",
        "best_b_candidate_coefficients_plus",
        "best_b_candidate_coefficients_minus",
        "best_c_candidate_target_axis",
        "best_c_candidate_overlap_x",
        "best_c_candidate_overlap_y",
        "best_c_candidate_score",
        "best_c_candidate_coefficients_plus",
        "best_c_candidate_coefficients_minus",
    ]
    for prefix in (
        "ideal_single_exchange",
        "physical_single_exchange_in_ideal_basis",
        "physical_single_exchange_in_physical_basis",
    ):
        for _, column_suffix in OPERATOR_ACTION_COLUMNS:
            fieldnames.append(f"{prefix}_{column_suffix}_error")
        fieldnames.append(f"{prefix}_max_error")
    return fieldnames


def save_results_table(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run the retry projection scan with reconstructed ideal Majoranas.",
    )
    parser.add_argument(
        "--u-values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 2.0],
        help="Interaction strengths U to scan.",
    )
    parser.add_argument(
        "--projection-levels",
        type=int,
        nargs="+",
        default=[8, 32, 56, 80, 512],
        help="Projection dimensions to include in the cumulative projector.",
    )
    parser.add_argument(
        "--levels-to-include",
        type=int,
        default=4,
        help="Number of even/odd subsystem pairs used in the Majorana fit.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=300,
        help="Number of time steps in each Kato evolution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("retry_projection_scan_results.txt"),
        help="Path to the tab-separated results table to write.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose progress output from the Majorana fit and braid evolution.",
    )
    return parser





def main():
    args = build_argument_parser().parse_args()

    n_points = args.n_points
    T_total = 1.0
    width = T_total / 3
    steepness = 20 / width

    verbose = not args.quiet
    interaction_u_values = args.u_values
    projection_level_list = args.projection_levels
    levels_to_include = args.levels_to_include
    results_output_path = args.output
    all_results = []
    candidate_rankings = {}

    print("Retry projection scan configuration:")
    print(f"  U values: {interaction_u_values}")
    print(f"  Projection dimensions: {projection_level_list}")
    print(f"  Majorana fit levels: {levels_to_include}")
    print(f"  Time steps: {n_points}")
    print(f"  Output path: {results_output_path}")

    for u_value in interaction_u_values:
        print(f"\nRunning scan for U={u_value}")
        specified_vals = {"U": [u_value]}

        builder = BraidingHamiltonianBuilder(
            n_sites=3,
            dupes=3,
            specified_vals=specified_vals,
            config_path=default_config_path(),
        )

        H_full = builder.full_system_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(H_full)

        # These full-space junction operators do not depend on the cumulative
        # projection block, so build them once and only re-project inside the loop.
        operators = builder.get_operators()

        B_inner = 3
        C_inner = 6
        creB2 = operators["cre"][B_inner]
        annB2 = operators["ann"][B_inner]
        creC2 = operators["cre"][C_inner]
        annC2 = operators["ann"][C_inner]

        for levels in projection_level_list:
            print("Working with projection level:", levels)
            P = get_projection_basis(eigvecs, levels)
            (
                (gamma_A1_full, gamma_A2_full),
                (gamma_B1_full, gamma_B2_full),
                (gamma_C1_full, gamma_C2_full),
            ), candidate_data = get_full_gammas(
                levels_to_include=levels_to_include,
                verbose=verbose,
                specified_vals=specified_vals,
                projection_basis=P,
                return_candidates=True,
            )
            candidate_rankings[(u_value, levels)] = {
                "A": {
                    "x": candidate_metadata(candidate_data["A"]["x"]),
                    "y": candidate_metadata(candidate_data["A"]["y"]),
                },
                "B": {
                    "x": candidate_metadata(candidate_data["B"]["x"]),
                    "y": candidate_metadata(candidate_data["B"]["y"]),
                    "selected": candidate_metadata(candidate_data["B"]["selected"]),
                },
                "C": {
                    "x": candidate_metadata(candidate_data["C"]["x"]),
                    "y": candidate_metadata(candidate_data["C"]["y"]),
                    "selected": candidate_metadata(candidate_data["C"]["selected"]),
                },
            }
            best_b_candidate = candidate_rankings[(u_value, levels)]["B"]["selected"]
            best_c_candidate = candidate_rankings[(u_value, levels)]["C"]["selected"]

            gamma0 = normalize_projected_majorana(project_ideal_majoranas(gamma_A1_full, P), "gamma0")
            gamma1 = normalize_projected_majorana(project_ideal_majoranas(gamma_A2_full, P), "gamma1")
            gamma2_ideal = normalize_projected_majorana(project_ideal_majoranas(gamma_B2_full, P), "gamma2_ideal")
            gamma3_ideal = normalize_projected_majorana(project_ideal_majoranas(gamma_C2_full, P), "gamma3_ideal")

            gamma_b_phys_components = projected_majoranas(creB2, annB2, P)
            gamma_c_phys_components = projected_majoranas(creC2, annC2, P)
            b_phys_index = 0 if best_b_candidate["target_axis"] == "x" else 1
            c_phys_index = 0 if best_c_candidate["target_axis"] == "x" else 1

            gamma_2_phys = normalize_projected_majorana(
                gamma_b_phys_components[b_phys_index],
                "gamma_2_phys",
            )
            gamma_3_phys = normalize_projected_majorana(
                gamma_c_phys_components[c_phys_index],
                "gamma_3_phys",
            )

            # Term A
            TA = 1j * (gamma0 @ gamma1)

            # Braid in ideal system
            TB_ideal = 1j * (gamma0 @ gamma2_ideal)
            TC_ideal = 1j * (gamma0 @ gamma3_ideal)
            transport_dim = TA.shape[0] // 2

            times, energies, couplings, u_kato_ideal = evolve_system(
                t_total=T_total,
                delta_max=1.0,
                delta_min=0.0,
                steepness=steepness,
                width=width,
                term_a=TA,
                term_b=TB_ideal,
                term_c=TC_ideal,
                n_points=n_points,
                transport_dim=transport_dim,
                verbose=verbose,
            )

            # Braid in physical system
            TB_phys = 1j * (gamma0 @ gamma_2_phys)
            TC_phys = 1j * (gamma0 @ gamma_3_phys)

            times, energies, couplings, u_kato_phys = evolve_system(
                t_total=T_total,
                delta_max=1.0,
                delta_min=0.0,
                steepness=steepness,
                width=width,
                term_a=TA,
                term_b=TB_phys,
                term_c=TC_phys,
                n_points=n_points,
                transport_dim=transport_dim,
                verbose=verbose,
            )

            ideal_reference_gammas = [gamma0, gamma1, gamma2_ideal, gamma3_ideal]
            physical_reference_gammas = [gamma0, gamma1, gamma_2_phys, gamma_3_phys]

            ideal_basis = get_initial_transport_basis(
                t_total=T_total,
                delta_max=1.0,
                delta_min=0.0,
                steepness=steepness,
                width=width,
                term_a=TA,
                term_b=TB_ideal,
                term_c=TC_ideal,
                transport_dim=transport_dim,
            )
            physical_basis = get_initial_transport_basis(
                t_total=T_total,
                delta_max=1.0,
                delta_min=0.0,
                steepness=steepness,
                width=width,
                term_a=TA,
                term_b=TB_phys,
                term_c=TC_phys,
                transport_dim=transport_dim,
            )

            ideal_operator_action = check_operator_action(u_kato_ideal, ideal_reference_gammas)
            physical_operator_action_ideal_basis = check_operator_action(
                u_kato_phys, ideal_reference_gammas
            )
            physical_operator_action_physical_basis = check_operator_action(
                u_kato_phys, physical_reference_gammas
            )

            ideal_target_error = compare_to_target_gate(
                u_kato_ideal,
                ideal_basis,
                gamma2_ideal,
                gamma3_ideal,
            )
            physical_target_error_ideal_basis = compare_to_target_gate(
                u_kato_phys,
                ideal_basis,
                gamma2_ideal,
                gamma3_ideal,
            )
            physical_target_error_physical_basis = compare_to_target_gate(
                u_kato_phys,
                physical_basis,
                gamma_2_phys,
                gamma_3_phys,
            )

            print(f"Projection level: {levels}")
            print(f"  best B candidate: {best_b_candidate}")
            print(f"  best C candidate: {best_c_candidate}")
            print(f"  ideal operator action:    {format_operator_action(ideal_operator_action)}")
            print(
                "  physical operator action in ideal basis:    "
                f"{format_operator_action(physical_operator_action_ideal_basis)}"
            )
            print(
                "  physical operator action in physical basis: "
                f"{format_operator_action(physical_operator_action_physical_basis)}"
            )
            print(f"  ideal phase-aligned target-gate error:    {ideal_target_error:.4e}")
            print(
                "  physical phase-aligned target-gate error in ideal basis:    "
                f"{physical_target_error_ideal_basis:.4e}"
            )
            print(
                "  physical phase-aligned target-gate error in physical basis: "
                f"{physical_target_error_physical_basis:.4e}"
            )

            result_row = build_result_row(
                u_value=u_value,
                projection_level=levels,
                transport_dim=transport_dim,
                best_b_candidate=best_b_candidate,
                best_c_candidate=best_c_candidate,
                ideal_operator_action=ideal_operator_action,
                physical_operator_action_ideal_basis=physical_operator_action_ideal_basis,
                physical_operator_action_physical_basis=physical_operator_action_physical_basis,
                ideal_target_error=ideal_target_error,
                physical_target_error_ideal_basis=physical_target_error_ideal_basis,
                physical_target_error_physical_basis=physical_target_error_physical_basis,
            )
            all_results.append(result_row)
            save_results_table(all_results, results_output_path)
            print(f"  Results table updated: {results_output_path}")

    print(f"\nSaved {len(all_results)} result rows to {results_output_path}")


if __name__ == "__main__":
    main()
