""" More realistic braiding test using projected microscopic junction operators.

The old sweet-spot model used
    H_eff(t) = Δ1(t) i γ0 γ1 + Δ2(t) i γ0 γ2 + Δ3(t) i γ0 γ3
with hand-picked Majoranas inside the projected low-energy space.

Here I keep the same pulse order, but replace the external ideal terms by
projected microscopic junction terms:
    T_AB = P^† H_AB^junc P
    T_AC = P^† H_AC^junc P

so the effective Hamiltonian becomes
    H_eff(t) = Δ1(t) i γ0 γ1 + λ_AB(t) T_AB + λ_AC(t) T_AC

If the microscopic couplings are close to the ideal braid picture, then T_AB
and T_AC should mostly line up with the desired Majorana bilinears. If not,
other bilinears should appear in their decomposition, and the braid checks
should get worse.
"""


from get_mzm_JW import get_full_gammas
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse, plot_results
from extended_projection_braiding import normalize_projected_majorana, build_total_parity_full, build_projection_stack


def project_operator(operator_full, basis):
    return basis.conj().T @ operator_full @ basis


def project_and_normalize(operator_full, basis):
    operator_sub = project_operator(operator_full, basis)
    return normalize_projected_majorana(operator_sub, "projected_operator")


def flatten_site(subsystem, site, n_sites=3):
    return subsystem * n_sites + site


def build_junction_operator(operators, site_a, site_b, t_couple=0.0, delta_couple=0.0):
    left = min(site_a, site_b)
    right = max(site_a, site_b)
    key = (left, right)

    dim = operators["num"][0].shape[0]
    junction = np.zeros((dim, dim), dtype=complex)

    if t_couple != 0:
        junction += -float(t_couple) * operators["hop"][key]
    if delta_couple != 0:
        junction += float(delta_couple) * operators["pair"][key]

    return 0.5 * (junction + junction.conj().T)


def build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC):
    Δ1 = delta_pulse(t, 0, width, s, Δ_max, Δ_min) + delta_pulse(t, T_total, width, s, Δ_max, Δ_min) - Δ_min
    Δ2 = delta_pulse(t, T_total / 3, width, s, Δ_max, Δ_min)
    Δ3 = delta_pulse(t, 2 * T_total / 3, width, s, Δ_max, Δ_min)

    H = Δ1 * T_A + Δ2 * T_AB + Δ3 * T_AC
    return H, (Δ1, Δ2, Δ3)


def evolve_projected_system(
    T_total,
    Δ_max,
    Δ_min,
    s,
    width,
    T_A,
    T_AB,
    T_AC,
    n_points=1000,
    transport_dim=None,
    verbose=False,
):
    times = np.linspace(0, T_total, n_points)
    dt = times[1] - times[0] if n_points > 1 else T_total

    dim = T_A.shape[0]
    if transport_dim is None:
        transport_dim = dim // 2
    if not 0 < transport_dim < dim:
        raise ValueError(f"transport_dim must be between 1 and {dim - 1}, got {transport_dim}.")

    energies = np.zeros((n_points, dim))
    couplings = np.zeros((n_points, 3))
    U_kato = np.eye(dim, dtype=complex)

    H0, coupling0 = build_projected_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
    evals, evecs = np.linalg.eigh(H0)
    energies[0] = evals
    couplings[0] = coupling0

    V = evecs[:, :transport_dim]

    if verbose:
        print("Analyzing projected microscopic braid...")
    for i in tqdm(range(1, len(times)), total=len(times) - 1, disable=not verbose):
        t = times[i]
        H, couplings[i] = build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)

        evals, evecs = np.linalg.eigh(H)
        energies[i] = evals

        W = evecs[:, :transport_dim]
        P = V @ V.conj().T
        P_next = W @ W.conj().T
        K = P @ ((P_next - P) / dt) - ((P_next - P) / dt) @ P
        U_kato = expm(-dt * K) @ U_kato
        V = W

    return times, energies, couplings, U_kato


def build_total_parity_projected(builder, basis):
    operators = builder.get_operators()
    num_ops = operators["num"]
    dim_full = num_ops[0].shape[0]
    identity_full = np.eye(dim_full, dtype=complex)
    parity_full = identity_full.copy()

    for number_op in num_ops:
        parity_full = parity_full @ (identity_full - 2 * number_op)

    return basis.conj().T @ parity_full @ basis


def get_ground_manifold_data(T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC, transport_dim):
    H0, _ = build_projected_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
    HT, _ = build_projected_hamiltonian(T_total, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)

    evals_0, evecs_0 = np.linalg.eigh(H0)
    evals_T, evecs_T = np.linalg.eigh(HT)

    V0 = evecs_0[:, :transport_dim]
    VT = evecs_T[:, :transport_dim]

    P0 = V0 @ V0.conj().T
    PT = VT @ VT.conj().T

    return {
        "evals_0": evals_0,
        "evals_T": evals_T,
        "V0": V0,
        "VT": VT,
        "P0": P0,
        "PT": PT,
    }


def phase_aligned_error(U, target):
    overlap = np.trace(target.conj().T @ U)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(U - np.exp(1j * phase) * target)


def check_majorana_algebra(gamma_list, verbose=False):
    if not verbose:
        return

    print("\nMajorana algebra checks")
    dim = gamma_list[0].shape[0]
    identity = np.eye(dim, dtype=complex)

    for i, gamma in enumerate(gamma_list):
        hermitian_error = np.linalg.norm(gamma - gamma.conj().T)
        square_error = np.linalg.norm(gamma @ gamma - identity)
        print(
            f"γ{i}: ||γ - γ†|| = {hermitian_error:.2e}, "
            f"||γ² - I|| = {square_error:.2e}"
        )

    for i in range(len(gamma_list)):
        for j in range(i + 1, len(gamma_list)):
            anticommutator_error = np.linalg.norm(
                gamma_list[i] @ gamma_list[j] + gamma_list[j] @ gamma_list[i]
            )
            print(f"{{γ{i}, γ{j}}}: {anticommutator_error:.2e}")


def check_path_properties(times, energies, parity_op, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC, transport_dim, verbose=False):
    max_hermiticity_error = 0.0
    max_parity_commutator = 0.0

    for t in times:
        H, _ = build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
        max_hermiticity_error = max(max_hermiticity_error, np.linalg.norm(H - H.conj().T))
        max_parity_commutator = max(
            max_parity_commutator,
            np.linalg.norm(H @ parity_op - parity_op @ H),
        )

    ground_splitting = np.max(energies[:, transport_dim - 1] - energies[:, 0])
    min_gap = np.min(energies[:, transport_dim] - energies[:, transport_dim - 1])

    if verbose:
        print("\nPath checks")
        print(f"max_t ||H(t) - H(t)†|| = {max_hermiticity_error:.2e}")
        print(f"max_t ||[H(t), P_tot]|| = {max_parity_commutator:.2e}")
        print(f"max_t (E{transport_dim - 1} - E0) = {ground_splitting:.2e}")
        print(f"min_t (E{transport_dim} - E{transport_dim - 1}) = {min_gap:.2e}")

    return {
        "max_hermiticity_error": max_hermiticity_error,
        "max_parity_commutator": max_parity_commutator,
        "ground_splitting": ground_splitting,
        "min_gap": min_gap,
    }


def check_kato_transport(U_kato, P0, PT, verbose=False):
    dim = U_kato.shape[0]
    identity = np.eye(dim, dtype=complex)
    unitary_error = np.linalg.norm(U_kato.conj().T @ U_kato - identity)
    transport_error = np.linalg.norm(U_kato @ P0 @ U_kato.conj().T - PT)
    loop_closure_error = np.linalg.norm(PT - P0)

    if verbose:
        print("\nKato transport checks")
        print(f"||U†U - I|| = {unitary_error:.2e}")
        print(f"||U P0 U† - PT|| = {transport_error:.2e}")
        print(f"||PT - P0|| = {loop_closure_error:.2e}")

    return {
        "unitary_error": unitary_error,
        "transport_error": transport_error,
        "loop_closure_error": loop_closure_error,
    }


def check_single_exchange(U_kato, gamma_list, verbose=False):
    expected_maps = [
        ("γ2 -> -γ3", gamma_list[2], -gamma_list[3]),
        ("γ3 ->  γ2", gamma_list[3], gamma_list[2]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    errors = {}
    if verbose:
        print("\nSingle-exchange checks")
    for label, source, target in expected_maps:
        transformed = U_kato.conj().T @ source @ U_kato
        error = np.linalg.norm(transformed - target)
        errors[label] = error
        if verbose:
            print(f"{label}: {error:.2e}")

    return {
        "errors": errors,
        "max_error": max(errors.values()) if errors else 0.0,
    }


def format_single_exchange_errors(single_exchange):
    ordered_labels = [
        "γ2 -> -γ3",
        "γ3 ->  γ2",
        "γ1 ->  γ1",
        "γ0 ->  γ0",
    ]
    return ", ".join(
        f"{label}={single_exchange['errors'][label]:.2e}"
        for label in ordered_labels
    )


def check_double_exchange(U_kato, gamma_list, verbose=False):
    U_double = U_kato @ U_kato
    expected_maps = [
        ("γ2 -> -γ2", gamma_list[2], -gamma_list[2]),
        ("γ3 -> -γ3", gamma_list[3], -gamma_list[3]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    if not verbose:
        return

    print("\nDouble-exchange checks")
    for label, source, target in expected_maps:
        transformed = U_double.conj().T @ source @ U_double
        error = np.linalg.norm(transformed - target)
        print(f"{label}: {error:.2e}")


def check_four_exchanges(U_kato, gamma_list, verbose=False):
    U_four = U_kato @ U_kato @ U_kato @ U_kato

    if not verbose:
        return

    print("\nFour-exchange checks")
    for i, gamma in enumerate(gamma_list):
        transformed = U_four.conj().T @ gamma @ U_four
        print(f"γ{i} -> γ{i}: {np.linalg.norm(transformed - gamma):.2e}")


def check_parity_resolved_gate(U_kato, V0, parity_op, γ2, γ3, verbose=False):
    U_ground = V0.conj().T @ U_kato @ V0
    parity_ground = V0.conj().T @ parity_op @ V0

    parity_vals, parity_vecs = np.linalg.eigh(parity_ground)
    U_parity = parity_vecs.conj().T @ U_ground @ parity_vecs

    odd_indices = np.flatnonzero(parity_vals < -1e-8)
    even_indices = np.flatnonzero(parity_vals > 1e-8)
    mixed_indices = np.flatnonzero(np.abs(parity_vals) <= 1e-8)

    off_block = (
        np.linalg.norm(U_parity[np.ix_(odd_indices, even_indices)])
        + np.linalg.norm(U_parity[np.ix_(even_indices, odd_indices)])
    )
    odd_block = U_parity[np.ix_(odd_indices, odd_indices)]
    even_block = U_parity[np.ix_(even_indices, even_indices)]

    U_target = expm(-0.25 * np.pi * (γ2 @ γ3))
    U_target_ground = V0.conj().T @ U_target @ V0
    U_target_parity = parity_vecs.conj().T @ U_target_ground @ parity_vecs
    odd_target = U_target_parity[np.ix_(odd_indices, odd_indices)]
    even_target = U_target_parity[np.ix_(even_indices, even_indices)]

    odd_error = phase_aligned_error(odd_block, odd_target)
    even_error = phase_aligned_error(even_block, even_target)

    if verbose:
        print("\nParity-resolved gate checks")
        print(
            "parity counts in transported manifold: "
            f"odd={len(odd_indices)}, even={len(even_indices)}, mixed={len(mixed_indices)}"
        )
        print(f"off-block leakage in parity basis: {off_block:.2e}")
        print(f"odd-block eigenvalues:  {np.round(np.linalg.eigvals(odd_block), 8)}")
        print(f"even-block eigenvalues: {np.round(np.linalg.eigvals(even_block), 8)}")
        print(f"odd-block target error:  {odd_error:.2e}")
        print(f"even-block target error: {even_error:.2e}")

    return {
        "off_block_leakage": off_block,
        "odd_target_error": odd_error,
        "even_target_error": even_error,
        "odd_dim": len(odd_indices),
        "even_dim": len(even_indices),
        "mixed_dim": len(mixed_indices),
    }


def get_bilinear_components(term, gamma_labels, gamma_ops):
    dim = term.shape[0]
    rows = []

    for i in range(len(gamma_ops)):
        for j in range(i + 1, len(gamma_ops)):
            basis_op = 1j * gamma_ops[i] @ gamma_ops[j]
            coeff = np.trace(basis_op.conj().T @ term).real / dim
            rows.append((abs(coeff), coeff, i, j, f"i {gamma_labels[i]} {gamma_labels[j]}"))

    rows.sort(reverse=True, key=lambda item: item[0])
    return rows


def reconstruct_bilinear_term(rows, gamma_ops):
    term = np.zeros_like(gamma_ops[0], dtype=complex)

    for _, coeff, i, j, _ in rows:
        term += coeff * (1j * gamma_ops[i] @ gamma_ops[j])

    return 0.5 * (term + term.conj().T)


def print_bilinear_decomposition(rows, title, max_terms=None, verbose=False):
    if not verbose:
        return

    print(f"\n{title}")
    if max_terms is None:
        max_terms = len(rows)

    for _, coeff, _, _, label in rows[:max_terms]:
        print(f"{label}: {coeff:.3e}")


def get_partner_index(index):
    pair_map = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 5,
        5: 4,
    }
    return pair_map[index]


def choose_active_majoranas(rows_AB, rows_AC, gamma_labels, gamma_ops, verbose=False):
    _, coeff_AB, i_AB, j_AB, label_AB = rows_AB[0]
    _, coeff_AC, i_AC, j_AC, label_AC = rows_AC[0]

    pair_AB = {i_AB, j_AB}
    pair_AC = {i_AC, j_AC}
    common = pair_AB & pair_AC

    if len(common) != 1:
        raise ValueError(
            "Could not identify a unique shared Majorana between the dominant AB and AC junction terms."
        )

    γ0_idx = common.pop()
    γ2_idx = j_AB if i_AB == γ0_idx else i_AB
    γ3_idx = j_AC if i_AC == γ0_idx else i_AC
    γ1_idx = get_partner_index(γ0_idx)

    if verbose:
        print("\nActive Majoranas chosen from dominant projected junction terms")
        print(f"AB dominant term: {label_AB} = {coeff_AB:.3e}")
        print(f"AC dominant term: {label_AC} = {coeff_AC:.3e}")
        print(f"Shared Majorana γ0 = {gamma_labels[γ0_idx]}")
        print(f"Partner on same subsystem γ1 = {gamma_labels[γ1_idx]}")
        print(f"Exchange pair: γ2 = {gamma_labels[γ2_idx]}, γ3 = {gamma_labels[γ3_idx]}")

    return {
        "γ0_idx": γ0_idx,
        "γ1_idx": γ1_idx,
        "γ2_idx": γ2_idx,
        "γ3_idx": γ3_idx,
        "γ0": gamma_ops[γ0_idx],
        "γ1": gamma_ops[γ1_idx],
        "γ2": gamma_ops[γ2_idx],
        "γ3": gamma_ops[γ3_idx],
    }


def run_model_comparison(
    model_name,
    T_total,
    Δ_max,
    Δ_min,
    s,
    width,
    T_A,
    T_AB,
    T_AC,
    n_points,
    transport_dim,
    gamma_list,
    verbose=False,
):
    times, energies, couplings, U_kato = evolve_projected_system(
        T_total=T_total,
        Δ_max=Δ_max,
        Δ_min=Δ_min,
        s=s,
        width=width,
        T_A=T_A,
        T_AB=T_AB,
        T_AC=T_AC,
        n_points=n_points,
        transport_dim=transport_dim,
        verbose=verbose,
    )

    return {
        "name": model_name,
        "times": times,
        "energies": energies,
        "couplings": couplings,
        "U_kato": U_kato,
        "single_exchange": check_single_exchange(U_kato, gamma_list, verbose=verbose),
    }

if __name__ == "__main__":
    from get_mzm_JW import get_full_gammas
    from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
    specified_vals = {"U": [0.1]}
    verbose = False
    run_extended_diagnostics = True
    models_to_run = ("ideal", "bilinear_fit", "physical")

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
    A_edge = flatten_site(0, 2)
    B_edge = flatten_site(1, 0)
    C_edge = flatten_site(2, 0)

    t_AB = builder.t[0]
    delta_AB = builder.Delta[0]
    t_AC = builder.t[0]
    delta_AC = builder.Delta[0]

    junction_AB_full = build_junction_operator(
        operators, A_edge, B_edge, t_couple=t_AB, delta_couple=delta_AB
    )
    junction_AC_full = build_junction_operator(
        operators, A_edge, C_edge, t_couple=t_AC, delta_couple=delta_AC
    )
    
    energy_tol = 1e-2
    P_full = build_total_parity_full(builder)
    projection_blocks = build_projection_stack(eigvals, eigvecs, P_full, energy_tol=energy_tol)

    # P_stack contains the basis matrices V_gs, V_excited1, ...
    # Use these for projections: O_sub = V.conj().T @ O_full @ V.
    P_stack = [block["basis"] for block in projection_blocks]

    # P_projector_stack contains the literal full-space projectors V V†.
    P_projector_stack = [block["projector"] for block in projection_blocks]

    # Cumulative bases are useful when you want P_gs, then
    # P_gs + P_excited1, then P_gs + P_excited1 + P_excited2, ...
    P_cumulative_stack = [
        np.hstack(P_stack[: idx + 1])
        for idx in range(len(P_stack))
    ]

    even_energies = []
    odd_energies = []
    for block in projection_blocks:
        even_energies.extend([block["energy_center"]] * block["even_dim"])
        odd_energies.extend([block["energy_center"]] * block["odd_dim"])


    if verbose:
        print(f"Built {len(P_stack)} energy-manifold basis blocks with energy_tol={energy_tol:g}.")
        print(f"Total parity counts: even={sum(block['even_dim'] for block in projection_blocks)}, "
              f"odd={sum(block['odd_dim'] for block in projection_blocks)}, "
              f"mixed={sum(block['mixed_dim'] for block in projection_blocks)}")

        for block in projection_blocks:
            gap = "--" if block["gap_to_next"] is None else f"{block['gap_to_next']:.4e}"
            print(
                f"{block['name']}: columns {block['start']}:{block['stop']}, "
                f"dim={block['dim']}, even={block['even_dim']}, odd={block['odd_dim']}, "
                f"mixed={block['mixed_dim']}, E=[{block['energy_min']:.4f}, {block['energy_max']:.4f}], "
                f"spread={block['energy_spread']:.4e}, gap_next={gap}"
            )

        for idx, basis in enumerate(P_cumulative_stack):
            print(f"Cumulative P up to {projection_blocks[idx]['name']}: basis shape = {basis.shape}")

    # These checks get expensive quickly because the matrices grow as 8, 32, 56, ...
    # Increase this once the small cases are behaving.
    max_cumulative_checks = min(16, len(P_cumulative_stack))
    # Keep this modest for sweeps: each time step does a dense eigh + expm.
    n_points = 300
    make_plots = False

    all_stored_vals = []

    for block, P in zip(projection_blocks[:max_cumulative_checks], P_cumulative_stack[:max_cumulative_checks]):
        if verbose:
            print(f"\nRunning checks up to {block['name']} with basis shape {P.shape}")
        V_ref = P
        dim_sub = V_ref.shape[1]
        transport_dim = dim_sub // 2

        if dim_sub % 2 != 0:
            if verbose:
                print(f"Skipping odd-dimensional projection with dim={dim_sub}.")
            continue

        if verbose:
            orthonormality_error = np.linalg.norm(V_ref.conj().T @ V_ref - np.eye(dim_sub))
            print(f"projection basis orthonormality error: {orthonormality_error:.2e}")

        gamma_A1_sub = V_ref.conj().T @ gamma_A1_full @ V_ref
        gamma_A2_sub = V_ref.conj().T @ gamma_A2_full @ V_ref
        gamma_B1_sub = V_ref.conj().T @ gamma_B1_full @ V_ref
        gamma_B2_sub = V_ref.conj().T @ gamma_B2_full @ V_ref
        gamma_C1_sub = V_ref.conj().T @ gamma_C1_full @ V_ref
        gamma_C2_sub = V_ref.conj().T @ gamma_C2_full @ V_ref
        gamma_A1_sub = normalize_projected_majorana(gamma_A1_sub, "gamma_A1_sub")
        gamma_A2_sub = normalize_projected_majorana(gamma_A2_sub, "gamma_A2_sub")
        gamma_B1_sub = normalize_projected_majorana(gamma_B1_sub, "gamma_B1_sub")
        gamma_B2_sub = normalize_projected_majorana(gamma_B2_sub, "gamma_B2_sub")
        gamma_C1_sub = normalize_projected_majorana(gamma_C1_sub, "gamma_C1_sub")
        gamma_C2_sub = normalize_projected_majorana(gamma_C2_sub, "gamma_C2_sub")
        gamma_labels = ["γA1", "γA2", "γB1", "γB2", "γC1", "γC2"]
        gamma_ops = [gamma_A1_sub, gamma_A2_sub, gamma_B1_sub, gamma_B2_sub, gamma_C1_sub, gamma_C2_sub]

        T_AB = project_operator(junction_AB_full, V_ref)
        T_AC = project_operator(junction_AC_full, V_ref)

        rows_AB = get_bilinear_components(T_AB, gamma_labels, gamma_ops)
        rows_AC = get_bilinear_components(T_AC, gamma_labels, gamma_ops)

        print_bilinear_decomposition(rows_AB, "Projected AB junction decomposition", max_terms=6, verbose=verbose)
        print_bilinear_decomposition(rows_AC, "Projected AC junction decomposition", max_terms=6, verbose=verbose)

        active = choose_active_majoranas(rows_AB, rows_AC, gamma_labels, gamma_ops, verbose=verbose)
        γ0, γ1, γ2, γ3 = active["γ0"], active["γ1"], active["γ2"], active["γ3"]

        T_A = 1j * γ0 @ γ1
        T_AB_bilinear_fit = reconstruct_bilinear_term(rows_AB, gamma_ops)
        T_AC_bilinear_fit = reconstruct_bilinear_term(rows_AC, gamma_ops)
        T_AB_ideal = 1j * γ0 @ γ2
        T_AC_ideal = 1j * γ0 @ γ3
        bilinear_residual_AB = np.linalg.norm(T_AB - T_AB_bilinear_fit) / np.sqrt(dim_sub)
        bilinear_residual_AC = np.linalg.norm(T_AC - T_AC_bilinear_fit) / np.sqrt(dim_sub)


        T_total = 1.0
        Δ_max = 1.0
        Δ_min = 0.0
        width = T_total / 3
        s = 20 / width

        gamma_list = [γ0, γ1, γ2, γ3]
        model_terms = {
            "ideal": (T_A, T_AB_ideal, T_AC_ideal),
            "bilinear_fit": (T_A, T_AB_bilinear_fit, T_AC_bilinear_fit),
            "physical": (T_A, T_AB, T_AC),
        }
        model_results = {}

        for model_name in models_to_run:
            if model_name not in model_terms:
                raise ValueError(f"Unknown model '{model_name}'. Expected one of {tuple(model_terms)}.")

            model_T_A, model_T_AB, model_T_AC = model_terms[model_name]
            model_results[model_name] = run_model_comparison(
                model_name=model_name,
                T_total=T_total,
                Δ_max=Δ_max,
                Δ_min=Δ_min,
                s=s,
                width=width,
                T_A=model_T_A,
                T_AB=model_T_AB,
                T_AC=model_T_AC,
                n_points=n_points,
                transport_dim=transport_dim,
                gamma_list=gamma_list,
                verbose=verbose,
            )

        if make_plots:
            physical_result = model_results.get("physical")
            if physical_result is not None:
                plot_results(
                    physical_result["times"],
                    physical_result["energies"],
                    physical_result["couplings"],
                )

        parity_gate = None

        if run_extended_diagnostics:
            physical_result = model_results["physical"]
            parity_projected = build_total_parity_projected(builder, V_ref)
            ground_data = get_ground_manifold_data(
                T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC, transport_dim
            )

            check_majorana_algebra(gamma_list, verbose=verbose)
            check_path_properties(
                physical_result["times"],
                physical_result["energies"],
                parity_projected,
                T_total,
                Δ_max,
                Δ_min,
                s,
                width,
                T_A,
                T_AB,
                T_AC,
                transport_dim,
                verbose=verbose,
            )
            check_kato_transport(
                physical_result["U_kato"], ground_data["P0"], ground_data["PT"], verbose=verbose
            )
            check_double_exchange(physical_result["U_kato"], gamma_list, verbose=verbose)
            check_four_exchanges(physical_result["U_kato"], gamma_list, verbose=verbose)
            parity_gate = check_parity_resolved_gate(
                physical_result["U_kato"],
                ground_data["V0"],
                parity_projected,
                γ2,
                γ3,
                verbose=verbose,
            )

        summary_lines = [
            f"{block['name']} cum_dim={dim_sub}",
            (
                " bilinear_projection_residuals: "
                f"AB={bilinear_residual_AB:.2e}, AC={bilinear_residual_AC:.2e}"
            ),
        ]

        for model_name in models_to_run:
            single_exchange = model_results[model_name]["single_exchange"]
            summary_lines.append(
                f" {model_name}: max_braid_error={single_exchange['max_error']:.2e} "
                f"| normalized={single_exchange['max_error'] / np.sqrt(dim_sub):.2e}"
            )
            summary_lines.append(
                f"  braid_errors: {format_single_exchange_errors(single_exchange)}"
            )

        if parity_gate is not None:
            summary_lines.extend(
                [
                    f" off_block_leakage={parity_gate['off_block_leakage']:.2e}",
                    (
                        f" odd_target_error={parity_gate['odd_target_error']:.2e} "
                        f"| normalized {parity_gate['odd_target_error'] / np.sqrt(dim_sub):.2e}"
                    ),
                    (
                        f" even_target_error={parity_gate['even_target_error']:.2e} "
                        f"| normalized {parity_gate['even_target_error'] / np.sqrt(dim_sub):.2e}"
                    ),
                ]
            )

        print("\n".join(summary_lines))
