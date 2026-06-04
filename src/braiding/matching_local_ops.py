import numpy as np
import matplotlib.pyplot as plt
from get_mzm_JW import get_full_gammas as get_majoranas_JW
from get_mzm_JW import precompute_operators
from remake_majoranas3 import make_majoranas_for_B_and_C_with_projection_dim
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse
from extended_projection_braiding import normalize_projected_majorana
from pathlib import Path
from step_projected_braiding_local import split_evals_by_parity, even_odd_splitter, project_operator, normalize_projected_majorana, parity_op, find_close_groups, find_group_bases, project_majoranas



def project_majoranas(bases, gamma_A1_full, gamma_A2_full, gamma_B1_full, gamma_B2_full, gamma_C1_full, gamma_C2_full):
    projected_majoranas = []
    for i, basis in enumerate(bases):
        A1_proj = normalize_projected_majorana(basis.conj().T @ gamma_A1_full @ basis, f"A1 group {i}")
        A2_proj = normalize_projected_majorana(basis.conj().T @ gamma_A2_full @ basis, f"A2 group {i}")
        B1_proj = normalize_projected_majorana(basis.conj().T @ gamma_B1_full @ basis, f"B1 group {i}")
        B2_proj = normalize_projected_majorana(basis.conj().T @ gamma_B2_full @ basis, f"B2 group {i}")
        C1_proj = normalize_projected_majorana(basis.conj().T @ gamma_C1_full @ basis, f"C1 group {i}")
        C2_proj = normalize_projected_majorana(basis.conj().T @ gamma_C2_full @ basis, f"C2 group {i}")

        projected_majoranas.append((A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj))
    return projected_majoranas


def project_operator(operator, basis):
    return basis.conj().T @ operator @ basis

def projected_local_majoranas(cdag, c, basis):
    gamma_plus = project_operator(cdag + c, basis)
    gamma_minus = project_operator(1j * (cdag - c), basis)
    return gamma_plus, gamma_minus

def select_physical_majorana(gamma_plus, gamma_minus,mode = "minus_only" ):
    if mode == "minus_only":
        return gamma_minus
    if mode == "plus_only":
        return gamma_plus
    if mode == "plus_minus":
        return gamma_plus + gamma_minus
    raise ValueError(f"Unknown mode={mode}.")


def project_local_majoranas(bases, operators, mode = "minus_only", B_INNER_SITE = 3, C_INNER_SITE = 6):
    local_majoranas = []
    for i, basis in enumerate(bases):
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

        B_local = normalize_projected_majorana(select_physical_majorana(B_plus, B_minus, mode), f"B local group {i}")
        C_local = normalize_projected_majorana(select_physical_majorana(C_plus, C_minus, mode), f"C local group {i}")
        local_majoranas.append((B_local, C_local))
    return local_majoranas



def hermitian_part(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def tensor_product(*matrices):
    result = np.asarray(matrices[0], dtype=complex)
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def pair_even_odd_by_energy(even_energies, odd_energies):
    pairs = []
    used_odd = set()

    for even_idx, even_energy in enumerate(even_energies):
        available_odd = [idx for idx in range(len(odd_energies)) if idx not in used_odd]
        odd_idx = min(available_odd, key=lambda idx: abs(odd_energies[idx] - even_energy))
        used_odd.add(odd_idx)
        pairs.append((even_idx, odd_idx))

    return pairs


def build_single_subsystem_pair_label_operator(levels_to_include=4):
    builder_sub = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=1,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )
    h_sub = builder_sub.full_system_hamiltonian()
    sub_eigvals, sub_eigvecs = np.linalg.eigh(h_sub)
    sub_ops = precompute_operators(n=3, dup=1)
    sub_parity = parity_op(sub_ops, sites=3)
    even_energies, odd_energies, even_vecs, odd_vecs, even_idxs, odd_idxs = even_odd_splitter(
        sub_eigvecs,
        sub_eigvals,
        sub_parity,
    )

    dim = sub_eigvecs.shape[0]
    label_operator = np.zeros((dim, dim), dtype=complex)
    pairs = pair_even_odd_by_energy(even_energies, odd_energies)

    for label, (even_idx, odd_idx) in enumerate(pairs[:levels_to_include]):
        even_vec = even_vecs[:, even_idx]
        odd_vec = odd_vecs[:, odd_idx]
        pair_projector = np.outer(even_vec, even_vec.conj()) + np.outer(odd_vec, odd_vec.conj())
        label_operator += label * pair_projector

    return hermitian_part(label_operator), min(levels_to_include, len(pairs))


def build_full_pair_label_operator(levels_to_include=4):
    single_label_operator, n_labels = build_single_subsystem_pair_label_operator(
        levels_to_include=levels_to_include,
    )
    identity_sub = np.eye(single_label_operator.shape[0], dtype=complex)
    label_base = n_labels + 1

    full_label_operator = (
        tensor_product(single_label_operator, identity_sub, identity_sub)
        + label_base * tensor_product(identity_sub, single_label_operator, identity_sub)
        + label_base**2 * tensor_product(identity_sub, identity_sub, single_label_operator)
    )
    return hermitian_part(full_label_operator), label_base




def decode_pair_label(label_value):
    label_int = int(round(float(np.real(label_value))))
    a_label = label_int % pair_label_base
    b_label = (label_int // pair_label_base) % pair_label_base
    c_label = (label_int // pair_label_base**2) % pair_label_base
    return a_label, b_label, c_label


def split_group_by_pair_labels(base, label_tol=1e-6):
    label_subspace = hermitian_part(base.conj().T @ pair_label_operator @ base)
    label_eigvals, label_eigvecs = np.linalg.eigh(label_subspace)
    order = np.argsort(label_eigvals)
    label_eigvals = label_eigvals[order]
    label_eigvecs = label_eigvecs[:, order]

    labeled_blocks = []
    start = 0
    for stop in range(1, len(label_eigvals) + 1):
        finished = stop == len(label_eigvals)
        separated = not finished and abs(label_eigvals[stop] - label_eigvals[start]) > label_tol
        if not (finished or separated):
            continue

        block_vectors = label_eigvecs[:, start:stop]
        label = decode_pair_label(np.mean(label_eigvals[start:stop]))
        labeled_blocks.append((label, block_vectors))
        start = stop

    return labeled_blocks


def block_diag(matrices):
    total_dim = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((total_dim, total_dim), dtype=complex)
    start = 0
    for matrix in matrices:
        stop = start + matrix.shape[0]
        result[start:stop, start:stop] = matrix
        start = stop
    return result


def least_squares_fit_single_block(proj1, proj2, local):
    I = np.eye(local.shape[0], dtype=complex)

    A = np.column_stack([
        local.flatten(),
        I.flatten(),
    ])

    y = np.column_stack([
        proj1.flatten(),
        proj2.flatten(),
    ])


    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    B1_fit = coeffs[0, 0] * local + coeffs[1, 0] * I
    B2_fit = coeffs[0, 1] * local + coeffs[1, 1] * I

    fit_error_B1 = np.linalg.norm(B1_fit - proj1) / np.linalg.norm(proj1)
    fit_error_B2 = np.linalg.norm(B2_fit - proj2) / np.linalg.norm(proj2)

    if fit_error_B1 > fit_error_B2:
        fit = B2_fit
        fit_error = fit_error_B2
        match = proj2
    else:
        fit = B1_fit
        fit_error = fit_error_B1
        match = proj1

    return fit, fit_error, match


def least_squares_fit_blockwise(proj1, proj2, local, labeled_blocks):
    block_vectors = [vectors for label, vectors in labeled_blocks]
    block_dims = [vectors.shape[1] for vectors in block_vectors]
    if sum(block_dims) != local.shape[0]:
        return least_squares_fit_single_block(proj1, proj2, local)

    full_block_rotation = np.column_stack(block_vectors)
    coverage_error = np.linalg.norm(
        full_block_rotation @ full_block_rotation.conj().T - np.eye(local.shape[0], dtype=complex)
    )
    if coverage_error > 1e-6:
        return least_squares_fit_single_block(proj1, proj2, local)

    fit_blocks = []
    match_blocks = []

    for label, block_basis in labeled_blocks:
        local_block = block_basis.conj().T @ local @ block_basis
        proj1_block = block_basis.conj().T @ proj1 @ block_basis
        proj2_block = block_basis.conj().T @ proj2 @ block_basis
        fit_block, fit_error_block, match_block = least_squares_fit_single_block(
            proj1_block,
            proj2_block,
            local_block,
        )
        fit_blocks.append(fit_block)
        match_blocks.append(match_block)

    fit_block_basis = block_diag(fit_blocks)
    match_block_basis = block_diag(match_blocks)

    fit = full_block_rotation @ fit_block_basis @ full_block_rotation.conj().T
    match = full_block_rotation @ match_block_basis @ full_block_rotation.conj().T
    fit_error = np.linalg.norm(fit - match) / np.linalg.norm(match)

    return fit, fit_error, match


def least_squares_fit(proj1, proj2, local, labeled_blocks=None):
    fit, fit_error, match = least_squares_fit_single_block(proj1, proj2, local)

    if fit_error > 0.5:
        if labeled_blocks is not None and len(labeled_blocks) > 1:
            fit, fit_error, match = least_squares_fit_blockwise(proj1, proj2, local, labeled_blocks)

    return fit, fit_error, match

def fit_local_majoranas_to_ideal(local_majoranas,**anythang):
    gamma_B1 = anythang["gamma_B1"]
    gamma_B2 = anythang["gamma_B2"]
    gamma_C1 = anythang["gamma_C1"]
    gamma_C2 = anythang["gamma_C2"]

    bases = anythang["bases"]
    print(len(bases))

    B_operators = []
    C_operators = []
    B_matches = []
    C_matches = []


    for i, base in enumerate(bases):
        labeled_blocks = split_group_by_pair_labels(base)
        block_summary = ", ".join(f"{label}:{vectors.shape[1]}" for label, vectors in labeled_blocks)
        B_local, C_local = local_majoranas[i]
        B1_proj = base.conj().T @ gamma_B1 @ base
        B2_proj = base.conj().T @ gamma_B2 @ base
        C1_proj = base.conj().T @ gamma_C1 @ base
        C2_proj = base.conj().T @ gamma_C2 @ base


        B_fit, B_fit_error, B_match = least_squares_fit(B1_proj, B2_proj, B_local, labeled_blocks=labeled_blocks)
        C_fit, C_fit_error, C_match = least_squares_fit(C1_proj, C2_proj, C_local, labeled_blocks=labeled_blocks)
        print(f"Group {i}: Group shape: {base.shape[1]}, blocks [{block_summary}], B fit error = {B_fit_error:.4e}, C fit error = {C_fit_error:.4e}")

        B_operators.append(B_fit)
        C_operators.append(C_fit)
        B_matches.append(B_match)
        C_matches.append(C_match)

    return B_operators, C_operators, B_matches, C_matches

        


def get_braiding_terms(B_ops, C_ops, projected_majoranas, idx):
    """idx needs to range from 0 to len(projected_majoranas)"""
    A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj = projected_majoranas[idx]
    B_local, C_local = B_ops[idx], C_ops[idx]
    TA = hermitian_part(1j * A1_proj @ A2_proj)
    TB = hermitian_part(1j* A1_proj @ B_local)
    TC = hermitian_part(1j* A1_proj @ C_local)
    return TA, TB, TC


def build_projected_hamiltonian(t, term_a, term_b, term_c, static_term=None):

 
    delta_1 = (
        delta_pulse(t, 0, Width, S, Delta_max, Delta_min)
        + delta_pulse(t, T_total, Width, S, Delta_max, Delta_min)
        - Delta_min
    )
    delta_2 = delta_pulse(t, T_total / 3, Width, S, Delta_max, Delta_min)
    delta_3 = delta_pulse(t, 2 * T_total / 3, Width, S, Delta_max, Delta_min)

    hamiltonian = delta_1 * term_a + delta_2 * term_b + delta_3 * term_c
    if static_term is not None:
        hamiltonian = hamiltonian + static_term
    return hamiltonian, (delta_1, delta_2, delta_3)

def split_spectrum_by_parity_over_time(time_arr, energies, vecs, parity_projected):
    even_by_time = []
    odd_by_time = []

    for evals, evecs in zip(energies, vecs):
        even_energies, odd_energies = split_evals_by_parity(evals, evecs, parity_projected)
        even_by_time.append(even_energies)
        odd_by_time.append(odd_energies)

    return np.array(even_by_time), np.array(odd_by_time)


def plot_projected_hamiltonian_spectrum(
    TA,
    TB,
    TC,
    idx,
    static_term=None,
    n_points=300,
    current_time=None,
    show=True,
    save_path=None,
):
    time_arr = np.linspace(0, T_total, n_points)
    energies = []
    vecs = []

    for t in time_arr:
        H_t, deltas = build_projected_hamiltonian(t, TA, TB, TC, static_term=static_term)
        evals, evecs = np.linalg.eigh(H_t)
        energies.append(evals)
        vecs.append(evecs)

    energies = np.array(energies)
    vecs = np.array(vecs)
    parity_projected = bases[idx].conj().T @ full_parity @ bases[idx]
    even_energies, odd_energies = split_spectrum_by_parity_over_time(
        time_arr,
        energies,
        vecs,
        parity_projected,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_arr, even_energies, color="tab:blue", linewidth=1.2)
    ax.plot(time_arr, odd_energies, color="tab:orange", linewidth=1.2)
    ax.plot([], [], color="tab:blue", label="Even parity")
    ax.plot([], [], color="tab:orange", label="Odd parity")

    if current_time is not None:
        ax.axvline(current_time, color="black", linestyle="--", linewidth=1.0, label=f"t = {current_time:.3f}")

    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Energy", fontsize=18)
    ax.set_title(f"Projected Hamiltonian Spectrum During Braiding, group {idx}", fontsize=20)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()

    return fig, ax, time_arr, even_energies, odd_energies

def evolve_system(n_steps, projected_majoranas, B_ops, C_ops, idx):
    TA, TB, TC = get_braiding_terms(B_ops, C_ops, projected_majoranas, idx)
    time_arr = np.linspace(0, T_total, n_steps)
    dt = time_arr[1] - time_arr[0]

    transport_dim = TA.shape[0] // 2
    print(f"Transport dimension: {transport_dim}")
    group_basis = bases[idx]
    static_term = group_basis.conj().T @ h_full @ group_basis
    static_term = hermitian_part(static_term)
    
    U_kato = np.eye(TA.shape[0], dtype=complex)
    H_t, deltas = build_projected_hamiltonian(0, TA, TB, TC, static_term=static_term)
    evals, evecs = np.linalg.eigh(H_t)


    initial_basis = evecs[:, :transport_dim]
    basis = initial_basis  # Start in the subspace of the lowest transport_dim eigenstates

    for t in tqdm(time_arr[1:], desc="Evolving system"):
        H_t, deltas = build_projected_hamiltonian(t, TA, TB, TC, static_term=static_term)

        evals, evecs = np.linalg.eigh(H_t)

       
        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        # print(projector)
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        U_kato = expm(-dt * kato_generator) @ U_kato
        basis = next_basis


    return U_kato, initial_basis, static_term

def unitary_overlap(U, V):
    d = U.shape[0]
    return abs(np.trace(V.conj().T @ U)) / d

def target_unitary(B_match, C_match):
    # This is the ideal unitary we want to match, based on the ideal Majorana operators
    return np.array(expm(-np.pi/4 * B_match @ C_match))


if __name__ == "__main__":
    U_vals = [0.0, 0.1, 2.0]
    for Uval in U_vals:
        print(f"Processing U={Uval}")
        specified_vals = {"U": [Uval]}
        mode = "minus_only"  # Options: "minus_only", "plus_only", "plus_minus"
        B_INNER_SITE = 3
        C_INNER_SITE = 6

        builder = BraidingHamiltonianBuilder(
            n_sites=3,
            dupes=3,
            specified_vals=specified_vals,
            config_path=default_config_path(),
        )
        h_full = builder.full_system_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(h_full)
        operators = builder.get_operators()


        (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_majoranas_JW(levels_to_include=4, specified_vals=specified_vals)


        ops = precompute_operators(n=3, dup=3)
        full_parity = parity_op(ops, sites=9)

        even_energies, odd_energies, even_vecs, odd_vecs, even_idxs, odd_idxs = even_odd_splitter(eigvecs, eigvals, full_parity)

        groups = find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs)
        bases = find_group_bases(groups, eigvecs)

        local_majoranas = project_local_majoranas(bases, operators, mode=mode)


        pair_label_operator, pair_label_base = build_full_pair_label_operator(levels_to_include=4)


        B_ops, C_ops, B_matches, C_matches = fit_local_majoranas_to_ideal(local_majoranas, gamma_B1=gamma_B1_full, gamma_B2=gamma_B2_full, gamma_C1=gamma_C1_full, gamma_C2=gamma_C2_full, bases=bases)

        T_total = 1.0
        Delta_max = 10.0
        Delta_min = 0.0
        Width = T_total / 3
        S = 20 / Width


        projected_majoranas = project_majoranas(bases, gamma_A1_full, gamma_A2_full, gamma_B1_full, gamma_B2_full, gamma_C1_full, gamma_C2_full)
        # U_kato, initial_basis, static_term = evolve_system(n_steps=1000, projected_majoranas=projected_majoranas, B_ops=B_ops, C_ops=C_ops, idx=0)
        results = {}
        for idx in range(len(projected_majoranas)):
            print(f"Processing group {idx} with basis dimension {bases[idx].shape[1]}")
            U_kato, initial_basis, static_term = evolve_system(n_steps=301, projected_majoranas=projected_majoranas, B_ops=B_ops, C_ops=C_ops, idx=idx)
            B_match = B_matches[idx]
            C_match = C_matches[idx]
            U_target = target_unitary(B_match, C_match)


            overlap = unitary_overlap(U_kato, U_target)
            print(f"Group {idx}: Unitary overlap with target = {overlap:.4f}")
            results[idx] = {
                "overlap": overlap,
                "basis_dimension": bases[idx].shape[1],
                "static_term_norm": np.linalg.norm(static_term),
                "B_fit_error": np.linalg.norm(B_ops[idx] - B_match) / np.linalg.norm(B_match),
                "C_fit_error": np.linalg.norm(C_ops[idx] - C_match) / np.linalg.norm(C_match),
            }



        from pathlib import Path

        cwd = Path(__file__).parent
        output_file = cwd / f"braiding_results_matched_ops_U={Uval}.txt"
        with open(output_file, "w") as f:
            f.write("Group\tBasis Dimension\tStatic Term Norm\tB Fit Error\tC Fit Error\tUnitary Overlap\n")
            for idx, res in results.items():
                f.write(f"{idx}\t{res['basis_dimension']}\t{res['static_term_norm']:.10f}\t{res['B_fit_error']:.10f}\t{res['C_fit_error']:.10f}\t{res['overlap']:.10f}\n")
