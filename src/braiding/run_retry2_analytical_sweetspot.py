from __future__ import annotations

from pathlib import Path

import numpy as np

import retry2
from hamiltonian_builder import BraidingHamiltonianBuilder
from extended_projection_braiding import normalize_projected_majorana


OUTPUT_PATH = Path(__file__).with_name("retry2_analytical_sweetspot_block_complete_dims_8_56_176_336_456_512.txt")
PROJECTION_LEVELS = [8, 56, 176, 336, 456, 512]
SWEETSPOT_LABEL = "AnalyticalSweetSpot"
GAP_TIE_TOLERANCE = 1.0e-9


def projected_site_majoranas(operators, site: int, projection_basis: np.ndarray):
    gamma_plus = retry2.project_operator(operators["cre"][site] + operators["ann"][site], projection_basis)
    gamma_minus = retry2.project_operator(
        1j * (operators["cre"][site] - operators["ann"][site]),
        projection_basis,
    )
    return gamma_plus, gamma_minus


def flatten_zero_state_made_errors(row: dict, projection_level: int) -> None:
    row["b_matrix_error"] = 0.0
    row["b_matrix_error_normalized"] = 0.0
    row["c_matrix_error"] = 0.0
    row["c_matrix_error_normalized"] = 0.0


def choose_transport_dim_preferring_small_ties(term_a, term_b, term_c, static_term=None):
    candidates = retry2.candidate_transport_dims(term_a.shape[0])
    min_gaps = retry2.scan_transport_gaps(term_a, term_b, term_c, static_term=static_term)
    candidate_gaps = [(candidate, float(min_gaps[candidate - 1])) for candidate in candidates]
    best_gap = max(gap for _, gap in candidate_gaps)
    tied = [
        candidate
        for candidate, gap in candidate_gaps
        if abs(gap - best_gap) <= GAP_TIE_TOLERANCE * max(1.0, abs(best_gap))
    ]
    transport_dim = min(tied)
    ranked = sorted(candidate_gaps, key=lambda item: item[1], reverse=True)

    print("transport dimension chosen by spectral gap:")
    print(
        f"  selected transport_dim={transport_dim}, "
        f"min gap above band={min_gaps[transport_dim - 1]:.4e}"
    )
    preview = ", ".join(f"{candidate}:{gap:.2e}" for candidate, gap in ranked[:5])
    print(f"  best candidate gaps dim:gap = {preview}")
    return transport_dim


def run_one_case(projection_level: int):
    print(f"\nanalytical sweet spot, projection level={projection_level}")
    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=None,
        t=[1.0, 1.0],
        U=[0.0, 0.0],
        eps=[0.0, 0.0, 0.0],
        Delta=[1.0, 1.0],
    )

    h_full = builder.full_system_hamiltonian()
    _, eigvecs = np.linalg.eigh(h_full)
    projection_basis = retry2.get_projection_basis(eigvecs, projection_level)

    h_static_bc_full = builder.subsystem_hamiltonian("B") + builder.subsystem_hamiltonian("C")
    h_static_bc_projected = retry2.project_operator(h_static_bc_full, projection_basis)

    operators = builder.get_operators()
    gamma_a0_plus, gamma_a0_minus = projected_site_majoranas(operators, 0, projection_basis)
    gamma_a2_plus, gamma_a2_minus = projected_site_majoranas(operators, 2, projection_basis)
    gamma_b0_plus, gamma_b0_minus = projected_site_majoranas(operators, 3, projection_basis)
    gamma_c0_plus, gamma_c0_minus = projected_site_majoranas(operators, 6, projection_basis)

    # For H = -t hop + Delta pair, t=Delta leaves minus(site 0) and plus(site 2)
    # as the exact end Majoranas of an isolated three-dot chain.
    gamma0 = normalize_projected_majorana(gamma_a2_plus, "gamma0_A_plus2")
    gamma1 = normalize_projected_majorana(gamma_a0_minus, "gamma1_A_minus0")
    gamma2 = normalize_projected_majorana(gamma_b0_minus, "gamma2_B_minus0")
    gamma3 = normalize_projected_majorana(gamma_c0_minus, "gamma3_C_minus0")

    gamma2_phys = gamma2
    gamma3_phys = gamma3

    term_a = 1j * (gamma0 @ gamma1)
    term_b_ideal = 1j * (gamma0 @ gamma2)
    term_c_ideal = 1j * (gamma0 @ gamma3)
    term_b_phys = term_b_ideal
    term_c_phys = term_c_ideal

    transport_dim = choose_transport_dim_preferring_small_ties(
        term_a,
        term_b_ideal,
        term_c_ideal,
        static_term=h_static_bc_projected,
    )

    print("analytical Majorana algebra:")
    ideal_algebra_check = retry2.check_majorana_algebra([gamma0, gamma1, gamma2, gamma3])
    physical_algebra_check = ideal_algebra_check

    print("evolving analytical system...")
    _, _, _, u_kato_ideal = retry2.evolve_system(
        term_a,
        term_b_ideal,
        term_c_ideal,
        static_term=h_static_bc_projected,
        transport_dim=transport_dim,
    )
    u_kato_phys = u_kato_ideal

    print("analytical braid action:")
    ideal_operator_action = retry2.check_operator_action(u_kato_ideal, gamma0, gamma1, gamma2, gamma3)
    physical_operator_action_ideal_basis = ideal_operator_action
    physical_operator_action_physical_basis = ideal_operator_action

    ideal_basis = retry2.get_initial_transport_basis(
        term_a,
        term_b_ideal,
        term_c_ideal,
        h_static_bc_projected,
        transport_dim,
    )
    physical_basis = ideal_basis

    ideal_target_error = retry2.compare_to_target_gate(u_kato_ideal, ideal_basis, gamma2, gamma3)
    physical_target_error_ideal_basis = ideal_target_error
    physical_target_error_physical_basis = ideal_target_error
    physical_target_error_physical_target_ideal_basis = ideal_target_error
    physical_target_error_ideal_target_physical_basis = ideal_target_error

    ideal_vs_physical_target_error_ideal_basis = 0.0
    ideal_vs_physical_target_error_physical_basis = 0.0
    gamma2_ideal_vs_physical_error_ideal_basis = 0.0
    gamma3_ideal_vs_physical_error_ideal_basis = 0.0
    gamma2_ideal_vs_physical_error_physical_basis = 0.0
    gamma3_ideal_vs_physical_error_physical_basis = 0.0
    physical_vs_ideal_target_error_ideal_basis = 0.0

    print("analytical braid action in transported band:")
    ideal_transport_operator_action = retry2.check_operator_action_in_basis(
        u_kato_ideal,
        ideal_basis,
        gamma0,
        gamma1,
        gamma2,
        gamma3,
    )
    physical_transport_operator_action_ideal_basis = ideal_transport_operator_action
    physical_transport_operator_action_physical_basis = ideal_transport_operator_action

    row = {
        "interaction_u": 0.0,
        "projection_level": int(projection_level),
        "transport_dim": int(transport_dim),
        "component_levels": 0,
        "tocheck": SWEETSPOT_LABEL,
        "ideal_target_gate_error": float(ideal_target_error),
        "ideal_target_gate_error_normalized": float(ideal_target_error / np.sqrt(transport_dim)),
        "physical_target_gate_error_in_ideal_basis": float(physical_target_error_ideal_basis),
        "physical_target_gate_error_in_ideal_basis_normalized": float(
            physical_target_error_ideal_basis / np.sqrt(transport_dim)
        ),
        "physical_target_gate_error_in_physical_basis": float(physical_target_error_physical_basis),
        "physical_target_gate_error_in_physical_basis_normalized": float(
            physical_target_error_physical_basis / np.sqrt(transport_dim)
        ),
        "physical_vs_ideal_target_gate_error_in_ideal_basis": float(physical_vs_ideal_target_error_ideal_basis),
        "physical_vs_ideal_target_gate_error_in_ideal_basis_normalized": float(
            physical_vs_ideal_target_error_ideal_basis / np.sqrt(transport_dim)
        ),
        "physical_target_gate_error_against_physical_target_in_ideal_basis": float(
            physical_target_error_physical_target_ideal_basis
        ),
        "physical_target_gate_error_against_physical_target_in_ideal_basis_normalized": float(
            physical_target_error_physical_target_ideal_basis / np.sqrt(transport_dim)
        ),
        "physical_target_gate_error_against_ideal_target_in_physical_basis": float(
            physical_target_error_ideal_target_physical_basis
        ),
        "physical_target_gate_error_against_ideal_target_in_physical_basis_normalized": float(
            physical_target_error_ideal_target_physical_basis / np.sqrt(transport_dim)
        ),
        "ideal_vs_physical_target_gate_error_in_ideal_basis": float(ideal_vs_physical_target_error_ideal_basis),
        "ideal_vs_physical_target_gate_error_in_ideal_basis_normalized": float(
            ideal_vs_physical_target_error_ideal_basis
        ),
        "ideal_vs_physical_target_gate_error_in_physical_basis": float(ideal_vs_physical_target_error_physical_basis),
        "ideal_vs_physical_target_gate_error_in_physical_basis_normalized": float(
            ideal_vs_physical_target_error_physical_basis
        ),
        "gamma2_ideal_vs_physical_error_in_ideal_basis_normalized": float(
            gamma2_ideal_vs_physical_error_ideal_basis
        ),
        "gamma3_ideal_vs_physical_error_in_ideal_basis_normalized": float(
            gamma3_ideal_vs_physical_error_ideal_basis
        ),
        "gamma2_ideal_vs_physical_error_in_physical_basis_normalized": float(
            gamma2_ideal_vs_physical_error_physical_basis
        ),
        "gamma3_ideal_vs_physical_error_in_physical_basis_normalized": float(
            gamma3_ideal_vs_physical_error_physical_basis
        ),
    }
    flatten_zero_state_made_errors(row, projection_level)
    row.update(retry2.flatten_operator_action("ideal_single_exchange", ideal_operator_action, projection_level))
    row.update(
        retry2.flatten_operator_action(
            "physical_single_exchange_in_ideal_basis",
            physical_operator_action_ideal_basis,
            projection_level,
        )
    )
    row.update(
        retry2.flatten_operator_action(
            "physical_single_exchange_in_physical_basis",
            physical_operator_action_physical_basis,
            projection_level,
        )
    )
    row.update(
        retry2.flatten_operator_action(
            "ideal_transport_single_exchange",
            ideal_transport_operator_action,
            transport_dim,
        )
    )
    row.update(
        retry2.flatten_operator_action(
            "physical_transport_single_exchange_in_ideal_basis",
            physical_transport_operator_action_ideal_basis,
            transport_dim,
        )
    )
    row.update(
        retry2.flatten_operator_action(
            "physical_transport_single_exchange_in_physical_basis",
            physical_transport_operator_action_physical_basis,
            transport_dim,
        )
    )
    row.update(retry2.flatten_algebra_check("ideal_majorana_algebra", ideal_algebra_check))
    row.update(retry2.flatten_algebra_check("physical_majorana_algebra", physical_algebra_check))
    return row


def main():
    results = []
    for projection_level in PROJECTION_LEVELS:
        row = run_one_case(projection_level)
        results.append(row)
        results.sort(key=lambda item: int(item["projection_level"]))
        retry2.save_results_table(results, OUTPUT_PATH)
        print(f"saved {len(results)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
