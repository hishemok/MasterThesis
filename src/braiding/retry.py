from get_mzm_JW import get_full_gammas
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse, plot_results
from extended_projection_braiding import normalize_projected_majorana



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





if __name__ == "__main__":
    n_points = 300
    T_total = 1.0
    width = T_total / 3
    steepness = 20 / width

    verbose = True
    specified_vals = {"U": [0.0]}

    (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_full_gammas(
    levels_to_include=4,
    verbose=verbose,
    specified_vals=specified_vals,
    )


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

    print(operators.keys())
    
    A_outer = 0
    A_inner = 2
    B_inner = 3
    B_outer = 5
    C_inner = 6
    C_outer = 8

    creA1 = operators["cre"][A_outer]
    annA1 = operators["ann"][A_outer]
    creA2 = operators["cre"][A_inner]
    annA2 = operators["ann"][A_inner]
    creB1 = operators["cre"][B_outer]
    annB1 = operators["ann"][B_outer]
    creB2 = operators["cre"][B_inner]   
    annB2 = operators["ann"][B_inner]
    creC1 = operators["cre"][C_outer]
    annC1 = operators["ann"][C_outer]
    creC2 = operators["cre"][C_inner]
    annC2 = operators["ann"][C_inner]

    projection_level_list = [8, 32, 56, 80]


    for levels in projection_level_list:
        print("Working with projection level:", levels)
        P = get_projection_basis(eigvecs, levels)
        A1_proj, A2_proj = projected_majoranas(creA1, annA1, P)
        B1_proj, B2_proj = projected_majoranas(creB1, annB1, P)
        C1_proj, C2_proj = projected_majoranas(creC1, annC1, P)

        gamma0 = normalize_projected_majorana(project_ideal_majoranas(gamma_A1_full, P), "gamma0")
        gamma1 = normalize_projected_majorana(project_ideal_majoranas(gamma_A2_full, P), "gamma1")
        gamma2_ideal = normalize_projected_majorana(project_ideal_majoranas(gamma_B2_full, P), "gamma2_ideal")
        gamma3_ideal = normalize_projected_majorana(project_ideal_majoranas(gamma_C2_full, P), "gamma3_ideal")


        gamma_2_phys_real =  normalize_projected_majorana(projected_majoranas(creB2, annB2, P)[0], "gamma_2_phys_real")
        gamma_2_phys_imag = normalize_projected_majorana(projected_majoranas(creB2, annB2, P)[1], "gamma_2_phys_imag")
        gamma_3_phys_real =  normalize_projected_majorana(projected_majoranas(creC2, annC2, P)[0], "gamma_3_phys_real")
        gamma_3_phys_imag = normalize_projected_majorana(projected_majoranas(creC2, annC2, P)[1], "gamma_3_phys_imag")

        #Term A 
        TA = 1j * (gamma0 @ gamma1)
        
        #Braid in ideal system
        TB_ideal = 1j * (gamma0 @ gamma2_ideal) 
        TC_ideal = 1j * (gamma0 @ gamma3_ideal) 
        #Evolve ideal system
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


        #Braid in physical system
        TB_phys = 1j * (gamma0 @ gamma_2_phys_imag) 
        TC_phys =  1j * (gamma0 @ gamma_3_phys_imag)

        #Evolve physical system
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
        physical_operator_action = check_operator_action(u_kato_phys, ideal_reference_gammas)

        ideal_target_error = compare_to_target_gate(
            u_kato_ideal,
            ideal_basis,
            gamma2_ideal,
            gamma3_ideal,
        )
        physical_target_error = compare_to_target_gate(
            u_kato_phys,
            physical_basis,
            gamma2_ideal,
            gamma3_ideal,
        )

        print(f"Projection level: {levels}")
        print(f"  ideal operator action:    {format_operator_action(ideal_operator_action)}")
        print(f"  physical operator action: {format_operator_action(physical_operator_action)}")
        print(f"  ideal phase-aligned target-gate error:    {ideal_target_error:.4e}")
        print(f"  physical phase-aligned target-gate error: {physical_target_error:.4e}")
