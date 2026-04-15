import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt


def delta_pulse(t, T_peak, width, s, Δ_max, Δ_min):
    """
    Improved smooth delta pulse with controllable width and steepness
    """
    # Calculate rise and fall times
    T_start = T_peak - width/2
    T_end = T_peak + width/2
    
    # Smooth step functions
    rise = 1/(1 + np.exp(-s*(t - T_start)))
    fall = 1/(1 + np.exp(s*(t - T_end)))
    
    return Δ_min + (Δ_max - Δ_min) * rise * fall

# @njit
def build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3):
    """
    Constructs the time-dependent Hamiltonian H(t) = Σ Δ_j(t) iγ₀γ_j
    """
    
    # Time-dependent couplings
    Δ1 = delta_pulse(t, 0, width, s, Δ_max, Δ_min) + delta_pulse(t, T_total, width, s, Δ_max, Δ_min) - Δ_min
    Δ2 = delta_pulse(t, T_total/3, width, s, Δ_max, Δ_min)
    Δ3 =  delta_pulse(t, 2*T_total/3, width, s, Δ_max, Δ_min)


    # Construct Hamiltonian terms
    H = (Δ1 * 1j * γ0 @ γ1 + 
         Δ2 * 1j * γ0 @ γ2 + 
         Δ3 * 1j * γ0 @ γ3)

    
    return H, (Δ1, Δ2, Δ3)


# @njit
def analyze_spectrum(T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3, n_points=1000):
    times = np.linspace(0, T_total, n_points)
    dim = γ0.shape[0]
    energies = np.zeros((n_points, dim))
    couplings = np.zeros((n_points, 3))  # Store Δ1, Δ2, Δ3
    
    print("Analyzing spectrum over time...")
    for i, t in tqdm(enumerate(times), total=n_points):
        H, couplings[i] = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
        
        e_vals = np.linalg.eigvalsh(H)
        energies[i] = e_vals
    
    return times, energies, couplings

def plot_results(times, energies, couplings):
    plt.figure(figsize=(12, 8))
    
    # Plot energy spectrum
    plt.subplot(2, 1, 1)
    linestyles = ['-', '--', '-.', ':'] * max(1, int(np.ceil(energies.shape[1] / 4)))
    for i in range(energies.shape[1]):
        plt.plot(times, energies[:, i], label=f'E{i}', linestyle=linestyles[i])
    plt.ylabel('Energy')
    plt.title('Energy Spectrum Evolution')
    if energies.shape[1] <= 12:
        plt.legend()
    
    # Plot couplings
    plt.subplot(2, 1, 2)
    labels = ['Δ₁(t)', 'Δ₂(t)', 'Δ₃(t)']
    linestyles = ['--', '-', '-']
    for i in range(3):
        plt.plot(times, couplings[:, i], label=labels[i], linestyle=linestyles[i])
    # for i in range(3):
    #     plt.plot(times, couplings[:, i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Coupling Strength')
    plt.title('Time-Dependent Couplings')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    labels = ['Δ₁(t)', 'Δ₂(t)', 'Δ₃(t)']
    linestyles = ['--', '-', '-']
    for i in range(3):
        plt.plot(times, couplings[:, i], label=labels[i], linestyle=linestyles[i], linewidth=4)
    # for i in range(3):
    #     plt.plot(times, couplings[:, i], label=labels[i])
    plt.xlabel('Time', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Coupling Strength', fontsize=18)
    plt.yticks(fontsize=16)
    plt.title('Time-Dependent Couplings', fontsize=24)
    plt.legend(fontsize=16)
    
    plt.tight_layout()
    plt.show()









def evolve_system(
    T_total,
    Δ_max,
    Δ_min,
    s,
    width,
    γ0,
    γ1,
    γ2,
    γ3,
    n_points=1000,
    transport_dim=None,
):
    times = np.linspace(0, T_total, n_points)
    dt = times[1] - times[0] if n_points > 1 else T_total

    dim = γ0.shape[0]
    if transport_dim is None:
        transport_dim = dim // 2
    if not 0 < transport_dim < dim:
        raise ValueError(f"transport_dim must be between 1 and {dim - 1}, got {transport_dim}.")

    energies = np.zeros((n_points, dim))
    couplings = np.zeros((n_points, 3))  # Store Δ1, Δ2, Δ3

    U_kato = np.eye(dim, dtype=complex)


    H0, coupling0 = build_hamiltonian(0,T_total, Δ_max, Δ_min, s, width, γ0,γ1,γ2,γ3)
    evals, evecs = np.linalg.eigh(H0)
    energies[0] = evals
    couplings[0] = coupling0

    V = evecs[:, :transport_dim]  # Initial transported subspace basis
    
    print("Analyzing spectrum over time...")
    for i in tqdm(range(1, len(times)), total=len(times) - 1):
        t = times[i]
        H, couplings[i] = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
        
        evals, evecs = np.linalg.eigh(H)
        energies[i] = evals

        W = evecs[:, :transport_dim]  # Next transported subspace basis
        P = V @ V.conj().T
        P_next = W @ W.conj().T
        K = P @ ((P_next - P) / dt) - ((P_next - P) / dt) @ P
        U_kato = expm(-dt * K) @ U_kato
        V = W  # Move to next basis


    return times, energies, couplings, U_kato



def build_total_parity_projected(builder, V_ref):
    operators = builder.get_operators()
    num_ops = operators["num"]
    dim_full = num_ops[0].shape[0]
    identity_full = np.eye(dim_full, dtype=complex)
    parity_full = identity_full.copy()

    for number_op in num_ops:
        parity_full = parity_full @ (identity_full - 2 * number_op)

    return V_ref.conj().T @ parity_full @ V_ref


def build_total_parity_full(builder):
    operators = builder.get_operators()
    num_ops = operators["num"]
    dim_full = num_ops[0].shape[0]
    identity_full = np.eye(dim_full, dtype=complex)
    parity_full = identity_full.copy()

    for number_op in num_ops:
        parity_full = parity_full @ (identity_full - 2 * number_op)

    return parity_full


def group_energy_manifolds(eigvals, energy_tol=1e-2):
    """Group consecutive eigenvalues into approximately degenerate manifolds."""
    groups = []
    start = 0

    for stop in range(1, len(eigvals)):
        if eigvals[stop] - eigvals[start] > energy_tol:
            groups.append((start, stop))
            start = stop

    groups.append((start, len(eigvals)))
    return groups


def build_projection_stack(eigvals, eigvecs, parity_full, energy_tol=1e-2):
    """
    Build one subspace basis for each approximately degenerate energy manifold.

    The returned ``basis`` matrices have shape ``(full_dim, manifold_dim)`` and
    are the objects you should use as ``V`` in projections like ``V† O V``.
    The returned ``projector`` matrices are the literal full-space projectors
    ``V V†``.
    """
    projection_blocks = []

    for group_index, (start, stop) in enumerate(group_energy_manifolds(eigvals, energy_tol=energy_tol)):
        basis = eigvecs[:, start:stop]
        projector = basis @ basis.conj().T
        parity_block = basis.conj().T @ parity_full @ basis
        parity_eigs = np.linalg.eigvalsh(parity_block)
        even_dim = int(np.sum(parity_eigs > 1e-8))
        odd_dim = int(np.sum(parity_eigs < -1e-8))
        mixed_dim = int(np.sum(np.abs(parity_eigs) <= 1e-8))

        projection_blocks.append(
            {
                "name": "P_gs" if group_index == 0 else f"P_excited{group_index}",
                "group_index": group_index,
                "start": start,
                "stop": stop,
                "dim": stop - start,
                "energy_min": float(eigvals[start]),
                "energy_max": float(eigvals[stop - 1]),
                "energy_center": float(np.mean(eigvals[start:stop])),
                "energy_spread": float(eigvals[stop - 1] - eigvals[start]),
                "gap_to_next": float(eigvals[stop] - eigvals[stop - 1]) if stop < len(eigvals) else None,
                "even_dim": even_dim,
                "odd_dim": odd_dim,
                "mixed_dim": mixed_dim,
                "parity_eigenvalues": parity_eigs,
                "basis": basis,
                "projector": projector,
            }
        )

    return projection_blocks


def get_ground_manifold_data(
    γ0,
    γ1,
    γ2,
    γ3,
    *,
    T_total,
    Δ_max,
    Δ_min,
    s,
    width,
    transport_dim,
):
    H0, _ = build_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
    HT, _ = build_hamiltonian(T_total, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)

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


def check_majorana_algebra(gamma_list):
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


def check_path_properties(
    times,
    energies,
    parity_op,
    γ0,
    γ1,
    γ2,
    γ3,
    *,
    T_total,
    Δ_max,
    Δ_min,
    s,
    width,
    transport_dim,
):
    max_hermiticity_error = 0.0
    max_parity_commutator = 0.0

    for t in times:
        H, _ = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
        max_hermiticity_error = max(max_hermiticity_error, np.linalg.norm(H - H.conj().T))
        max_parity_commutator = max(
            max_parity_commutator,
            np.linalg.norm(H @ parity_op - parity_op @ H),
        )

    manifold_splitting = np.max(energies[:, transport_dim - 1] - energies[:, 0])
    min_gap = np.min(energies[:, transport_dim] - energies[:, transport_dim - 1])

    print("\nPath checks")
    print(f"max_t ||H(t) - H(t)†|| = {max_hermiticity_error:.2e}")
    print(f"max_t ||[H(t), P_tot]|| = {max_parity_commutator:.2e}")
    print(f"max_t (E{transport_dim - 1} - E0) = {manifold_splitting:.2e}")
    print(f"min_t (E{transport_dim} - E{transport_dim - 1}) = {min_gap:.2e}")


def check_kato_transport(U_kato, P0, PT):
    dim = U_kato.shape[0]
    identity = np.eye(dim, dtype=complex)
    unitary_error = np.linalg.norm(U_kato.conj().T @ U_kato - identity)
    transport_error = np.linalg.norm(U_kato @ P0 @ U_kato.conj().T - PT)
    loop_closure_error = np.linalg.norm(PT - P0)

    print("\nKato transport checks")
    print(f"||U†U - I|| = {unitary_error:.2e}")
    print(f"||U P0 U† - PT|| = {transport_error:.2e}")
    print(f"||PT - P0|| = {loop_closure_error:.2e}")


def compute_single_exchange_metrics(U_kato, gamma_list, transport_dim):
    checks = {
        "gamma2_to_minus_gamma3": (gamma_list[2], -gamma_list[3]),
        "gamma3_to_gamma2": (gamma_list[3], gamma_list[2]),
        "gamma1_to_gamma1": (gamma_list[1], gamma_list[1]),
        "gamma0_to_gamma0": (gamma_list[0], gamma_list[0]),
    }

    errors = {}
    for label, (source, target) in checks.items():
        transformed = U_kato.conj().T @ source @ U_kato
        errors[label] = float(np.linalg.norm(transformed - target))

    max_error = max(errors.values())
    return {
        "single_exchange_errors": errors,
        "max_exchange_error_raw": float(max_error),
        "max_exchange_error_normalized": float(max_error / np.sqrt(transport_dim)),
    }


def check_single_exchange(U_kato, gamma_list):
    metrics = compute_single_exchange_metrics(U_kato, gamma_list, gamma_list[0].shape[0] // 2)

    print("\nSingle-exchange checks")
    print(f"γ2 -> -γ3: {metrics['single_exchange_errors']['gamma2_to_minus_gamma3']:.2e}")
    print(f"γ3 ->  γ2: {metrics['single_exchange_errors']['gamma3_to_gamma2']:.2e}")
    print(f"γ1 ->  γ1: {metrics['single_exchange_errors']['gamma1_to_gamma1']:.2e}")
    print(f"γ0 ->  γ0: {metrics['single_exchange_errors']['gamma0_to_gamma0']:.2e}")
    print(f"max single-exchange error: {metrics['max_exchange_error_raw']:.2e}")
    return metrics


def check_double_exchange(U_kato, gamma_list):
    U_double = U_kato @ U_kato
    expected_maps = [
        ("γ2 -> -γ2", gamma_list[2], -gamma_list[2]),
        ("γ3 -> -γ3", gamma_list[3], -gamma_list[3]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    print("\nDouble-exchange checks")
    for label, source, target in expected_maps:
        transformed = U_double.conj().T @ source @ U_double
        error = np.linalg.norm(transformed - target)
        print(f"{label}: {error:.2e}")


def check_four_exchanges(U_kato, gamma_list):
    U_four = U_kato @ U_kato @ U_kato @ U_kato

    print("\nFour-exchange checks")
    for i, gamma in enumerate(gamma_list):
        transformed = U_four.conj().T @ gamma @ U_four
        print(f"γ{i} -> γ{i}: {np.linalg.norm(transformed - gamma):.2e}")


def check_parity_resolved_gate(U_kato, V0, parity_op, γ2, γ3):
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

    print("\nParity-resolved gate checks")
    print(
        "parity counts in transported manifold: "
        f"odd={len(odd_indices)}, even={len(even_indices)}, mixed={len(mixed_indices)}"
    )
    print(f"off-block leakage in parity basis: {off_block:.2e}")
    print(f"odd-block eigenvalues:  {np.round(np.linalg.eigvals(odd_block), 8)}")
    print(f"even-block eigenvalues: {np.round(np.linalg.eigvals(even_block), 8)}")
    print(f"odd-block target error:  {phase_aligned_error(odd_block, odd_target):.2e}")
    print(f"even-block target error: {phase_aligned_error(even_block, even_target):.2e}")
    value_dict = {
        "off_block_leakage": off_block,
        "odd_block_eigenvalues": np.round(np.linalg.eigvals(odd_block), 8),
        "even_block_eigenvalues": np.round(np.linalg.eigvals(even_block), 8),
        "odd_block_target_error": phase_aligned_error(odd_block, odd_target),
        "even_block_target_error": phase_aligned_error(even_block, even_target),
    }
    return value_dict


def normalize_projected_majorana(gamma, label):
    dim = gamma.shape[0]
    scale_squared = np.trace(gamma @ gamma).real / dim
    if scale_squared <= 0:
        raise ValueError(f"{label} has non-positive normalization scale {scale_squared:.3e}.")
    return gamma / np.sqrt(scale_squared)


if __name__ == "__main__":
    from get_mzm_JW import get_full_gammas
    from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path

    specified_vals = {"U": [2.0]}

    (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_full_gammas(
        levels_to_include=4,
        verbose=False,
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

    plt.hlines(even_energies, 0, 1, colors='blue', label='Even parity', linestyles='dashed')
    plt.hlines(odd_energies, 2, 3,  colors='red', label='Odd parity', linestyles='dashed')
    plt.legend()
    plt.title("Parity-resolved spectrum of the full system")
    plt.tight_layout()
    plt.show()

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
    n_points = 300
    make_plots = False

    all_stored_vals = []

    for block, P in zip(projection_blocks[:max_cumulative_checks], P_cumulative_stack[:max_cumulative_checks]):
        print(f"\nRunning checks up to {block['name']} with basis shape {P.shape}")
        V_ref = P
        dim_sub = V_ref.shape[1]
        transport_dim = dim_sub // 2

        if dim_sub % 2 != 0:
            print(f"Skipping odd-dimensional projection with dim={dim_sub}.")
            continue

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





        # The projected AB/AC junctions couple dominantly to B2/C2 for these
        # optimized configurations; B1/C1 has essentially zero overlap.
        γ0, γ1, γ2, γ3 = gamma_A1_sub, gamma_A2_sub, gamma_B2_sub, gamma_C2_sub

        T_total = 1.0
        Δ_max = 1.0
        Δ_min = 0
        width = T_total/3
        s = 20/width
        # Parameters
        params = {
            'T_total': T_total,
            'Δ_max': Δ_max,
            'Δ_min': Δ_min,
            's': s,
            'width': width,
            'γ0': γ0,
            'γ1': γ1,
            'γ2': γ2,
            'γ3': γ3,
            'n_points': n_points,
            'transport_dim': transport_dim,
        }


        times, energies, couplings, U_kato = evolve_system(**params)
        # if make_plots:
        #     plot_results(times, energies, couplings)




        gamma_list = [γ0, γ1, γ2, γ3]
        parity_projected = build_total_parity_projected(builder, V_ref)
        ground_data = get_ground_manifold_data(
            γ0,
            γ1,
            γ2,
            γ3,
            T_total=T_total,
            Δ_max=Δ_max,
            Δ_min=Δ_min,
            s=s,
            width=width,
            transport_dim=transport_dim,
        )

        check_majorana_algebra(gamma_list)
        check_path_properties(
            times,
            energies,
            parity_projected,
            γ0,
            γ1,
            γ2,
            γ3,
            T_total=T_total,
            Δ_max=Δ_max,
            Δ_min=Δ_min,
            s=s,
            width=width,
            transport_dim=transport_dim,
        )
        


        check_kato_transport(U_kato, ground_data["P0"], ground_data["PT"])
        exchange_vals = check_single_exchange(U_kato, gamma_list)
        check_double_exchange(U_kato, gamma_list)
        check_four_exchanges(U_kato, gamma_list)
        store_vals = check_parity_resolved_gate(U_kato, ground_data["V0"], parity_projected, γ2, γ3)
        #Add block info to stored values
        store_vals["block_name"] = block["name"]
        store_vals["projection_dim"] = dim_sub
        store_vals["manifold_dim"] = block["dim"]
        store_vals["max_exchange_error_raw"] = exchange_vals["max_exchange_error_raw"]
        store_vals["max_exchange_error_normalized"] = exchange_vals["max_exchange_error_normalized"]


        all_stored_vals.append(store_vals)
    
    print("\nSummary of parity-resolved gate checks for cumulative projections U=2.0:")
    for vals in all_stored_vals:
        print(
              f"Up to {vals['block_name']} "
              f"(cumulative dim={vals['projection_dim']}, manifold dim={vals['manifold_dim']}): "
              f"braid error={vals['max_exchange_error_raw']:.2e}, "
              f"braid error/sqrt(d)={vals['max_exchange_error_normalized']:.2e}, "
              f"off-block leakage={vals['off_block_leakage']:.2e}, "
              f"odd-block target error={vals['odd_block_target_error']:.2e}, "
              f"even-block target error={vals['even_block_target_error']:.2e}"
        )
