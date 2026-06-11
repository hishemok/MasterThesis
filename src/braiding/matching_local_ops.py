from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from block_match_tools2 import (
    find_blocks,
    fit_blockwise_to_majorana_pair,
    format_blocks,
    hermitian_part,
)
from braiding_model import delta_pulse
from get_mzm_JW import get_full_gammas as get_majoranas_JW
from get_mzm_JW import precompute_operators
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from step_projected_braiding_local import even_odd_splitter,find_close_groups,find_group_bases,parity_op,project_majoranas

from extended_projection_braiding import normalize_projected_majorana

N_SITES = 3
DUPES = 3
LEVELS_TO_INCLUDE = 4
B_INNER_SITE = 3
C_INNER_SITE = 6

T_TOTAL = 1.0
DELTA_MAX = 1.0
DELTA_MIN = 0.0
WIDTH = T_TOTAL / 3
S = 20 / WIDTH


def project_operator(operator, basis):
    return basis.conj().T @ operator @ basis


def projected_local_majoranas(cdag, c, basis):
    gamma_plus = project_operator(cdag + c, basis)
    gamma_minus = project_operator(1j * (cdag - c), basis)
    return gamma_plus, gamma_minus


def select_physical_majorana(gamma_plus, gamma_minus, mode="minus_only"):
    if mode == "minus_only":
        return gamma_minus
    if mode == "plus_only":
        return gamma_plus
    if mode == "plus_minus":
        return gamma_plus + gamma_minus
    raise ValueError(f"Unknown mode={mode}.")


def project_local_majoranas(bases, operators, mode="minus_only"):
    local_majoranas = []

    for idx, basis in enumerate(bases):
        B_plus, B_minus = projected_local_majoranas(
            operators["cre"][B_INNER_SITE],
            operators["ann"][B_INNER_SITE],
            basis,
        )
        C_plus, C_minus = projected_local_majoranas(
            operators["cre"][C_INNER_SITE],
            operators["ann"][C_INNER_SITE],
            basis,
        )

        B_local = normalize_projected_majorana(
            select_physical_majorana(B_plus, B_minus, mode),
            f"B local group {idx}",
        )
        C_local = normalize_projected_majorana(
            select_physical_majorana(C_plus, C_minus, mode),
            f"C local group {idx}",
        )
        local_majoranas.append((B_local, C_local))

    return local_majoranas


def build_projected_hamiltonian(t, term_a, term_b, term_c, static_term=None):
    delta_1 = (
        delta_pulse(t, 0, WIDTH, S, DELTA_MAX, DELTA_MIN)
        + delta_pulse(t, T_TOTAL, WIDTH, S, DELTA_MAX, DELTA_MIN)
        - DELTA_MIN
    )
    delta_2 = delta_pulse(t, T_TOTAL / 3, WIDTH, S, DELTA_MAX, DELTA_MIN)
    delta_3 = delta_pulse(t, 2 * T_TOTAL / 3, WIDTH, S, DELTA_MAX, DELTA_MIN)

    hamiltonian = delta_1 * term_a + delta_2 * term_b + delta_3 * term_c
    if static_term is not None:
        hamiltonian = hamiltonian + static_term
    return hamiltonian


def get_braiding_terms(projected_majoranas, control_majoranas, idx):
    A1_proj, A2_proj, *_ = projected_majoranas[idx]
    B_control, C_control = control_majoranas[idx]

    TA = hermitian_part(1j * A1_proj @ A2_proj)
    TB = hermitian_part(1j * A1_proj @ B_control)
    TC = hermitian_part(1j * A1_proj @ C_control)
    return TA, TB, TC


def evolve_system(n_steps, projected_majoranas, control_majoranas, idx, bases, h_full):
    TA, TB, TC = get_braiding_terms(projected_majoranas, control_majoranas, idx)
    time_arr = np.linspace(0, T_TOTAL, n_steps)
    dt = time_arr[1] - time_arr[0]

    transport_dim = TA.shape[0] // 2
    group_basis = bases[idx]
    static_term = hermitian_part(group_basis.conj().T @ h_full @ group_basis)

    U_kato = np.eye(TA.shape[0], dtype=complex)
    H0 = build_projected_hamiltonian(0, TA, TB, TC, static_term=static_term)
    _, evecs = np.linalg.eigh(H0)

    initial_basis = evecs[:, :transport_dim]
    basis = initial_basis

    for t in tqdm(time_arr[1:], desc=f"Evolving group {idx}", leave=False):
        H_t = build_projected_hamiltonian(t, TA, TB, TC, static_term=static_term)
        _, evecs = np.linalg.eigh(H_t)

        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        U_kato = expm(-dt * kato_generator) @ U_kato
        basis = next_basis

    return U_kato, initial_basis, static_term


def target_unitary(B_operator, C_operator):
    return expm(-np.pi / 4 * B_operator @ C_operator)


def unitary_overlap(U, V):
    return abs(np.trace(V.conj().T @ U)) / U.shape[0]


def build_sector_data(U_value, mode="minus_only"):
    specified_vals = {"U": [U_value]}

    builder = BraidingHamiltonianBuilder(n_sites=N_SITES,dupes=DUPES,specified_vals=specified_vals,config_path=default_config_path(),)

    h_full = builder.full_system_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(h_full)
    operators = builder.get_operators()

    gamma_groups = get_majoranas_JW(
        levels_to_include=LEVELS_TO_INCLUDE,
        specified_vals=specified_vals,
    )
    (gamma_A1, gamma_A2), (gamma_B1, gamma_B2), (gamma_C1, gamma_C2) = gamma_groups

    ops = precompute_operators(n=N_SITES, dup=DUPES)
    full_parity = parity_op(ops, sites=N_SITES * DUPES)
    even_energies, odd_energies, _, _, even_idxs, odd_idxs = even_odd_splitter(eigvecs,eigvals,full_parity)

    groups = find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs)
    bases = find_group_bases(groups, eigvecs)

    projected_majorana_groups = project_majoranas(bases,gamma_A1,gamma_A2,gamma_B1,gamma_B2,gamma_C1,gamma_C2)
    local_majorana_groups = project_local_majoranas(bases, operators, mode=mode)

    return {"h_full": h_full,"bases": bases,"projected_majoranas": projected_majorana_groups,"local_majoranas": local_majorana_groups,"gamma_B": (gamma_B1, gamma_B2),"gamma_C": (gamma_C1, gamma_C2),"specified_vals": specified_vals,
    }


def fit_local_majoranas_to_ideal(data):
    bases = data["bases"]
    local_majoranas = data["local_majoranas"]
    gamma_B1, gamma_B2 = data["gamma_B"]
    gamma_C1, gamma_C2 = data["gamma_C"]

    fitted_majoranas = []
    reports = []

    for idx, basis in enumerate(bases):
        blocks = find_blocks(
            basis,
            specified_vals=data["specified_vals"],
            levels=LEVELS_TO_INCLUDE,
            n_subsystems=DUPES,
        )
        B_local, C_local = local_majoranas[idx]

        B1_proj = project_operator(gamma_B1, basis)
        B2_proj = project_operator(gamma_B2, basis)
        C1_proj = project_operator(gamma_C1, basis)
        C2_proj = project_operator(gamma_C2, basis)

        B_fit = fit_blockwise_to_majorana_pair(B_local, B1_proj, B2_proj, blocks)
        C_fit = fit_blockwise_to_majorana_pair(C_local, C1_proj, C2_proj, blocks)

        fitted_majoranas.append((B_fit.fit, C_fit.fit))
        reports.append((B_fit, C_fit))

        print(f"Group {idx:02d}: dim={basis.shape[1]:3d}, "f"blocks=[{format_blocks(blocks)}], "f"B error={B_fit.error:.3e}, C error={C_fit.error:.3e}, "f"B offblock={B_fit.offblock_error:.3e}, C offblock={C_fit.offblock_error:.3e}, B coeffs={np.round(B_fit.coeffs, 2)}, C coeffs={np.round(C_fit.coeffs, 2)}"
        )

    return fitted_majoranas, reports


def run_matched_operator_scan(U_values=(0.0,0.1, 0.5, 1.0, 2.0), n_steps=300, write_results=True):
    for U_value in U_values:
        print(f"\nProcessing U={U_value}")
        data = build_sector_data(U_value)
        fitted_majoranas, fit_reports = fit_local_majoranas_to_ideal(data)

        results = {}
        for idx, _ in enumerate(data["projected_majoranas"]):
            U_kato, initial_basis, static_term = evolve_system(n_steps,data["projected_majoranas"],fitted_majoranas,idx,data["bases"],data["h_full"],
            )
            B_fit, C_fit = fitted_majoranas[idx]
            U_target = target_unitary(B_fit, C_fit)
            B_report, C_report = fit_reports[idx]


            U_kato_sub = initial_basis.conj().T @ U_kato @ initial_basis
            U_target_sub = initial_basis.conj().T @ U_target @ initial_basis

            B_coeffs = B_report.coeffs
            C_coeffs = C_report.coeffs

            overlap = unitary_overlap(U_kato_sub, U_target_sub)
            print(f"Group {idx:02d}: matched target overlap={overlap:.10f}")
            results[idx] = {"basis_dimension": data["bases"][idx].shape[1],"static_term_norm": np.linalg.norm(static_term),"B_fit_error": B_report.error,"C_fit_error": C_report.error,"B_offblock_error": B_report.offblock_error,"C_offblock_error": C_report.offblock_error, "B_coeffs": np.round(B_coeffs, 2), "C_coeffs": np.round(C_coeffs, 2), "overlap": overlap,
            }

        if write_results:
            write_results_file(U_value, results)


def write_results_file(U_value, results):
    output_file = Path(__file__).parent / f"braiding_results_matched_ops_U={U_value}.txt"
    with open(output_file, "w") as file:
        file.write(
            "Group\tBasis Dimension\tStatic Term Norm\t"
            "B Fit Error\tC Fit Error\tB Offblock Error\tC Offblock Error\tB Coeffs\tC Coeffs\tUnitary Overlap\n"
        )
        for idx, result in results.items():
            file.write(f"{idx}\t"f"{result['basis_dimension']}\t"f"{result['static_term_norm']:.10f}\t"f"{result['B_fit_error']:.10e}\t"f"{result['C_fit_error']:.10e}\t"f"{result['B_offblock_error']:.10e}\t"f"{result['C_offblock_error']:.10e} \t"f"{result['B_coeffs']}\t"f"{result['C_coeffs']}\t"f"{result['overlap']:.10f}\n"
            )


if __name__ == "__main__":
    run_matched_operator_scan()
