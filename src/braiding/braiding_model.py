from get_mzm_JW import get_full_gammas, subsys_parity_oper
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path


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
    energies = np.zeros((n_points, 8))  # Store all 8 eigenvalues
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
    linestyles = ['-', '--', '-.', ':']*2
    for i in range(8):
        plt.plot(times, energies[:, i], label=f'E{i}', linestyle=linestyles[i])
    plt.ylabel('Energy')
    plt.title('Energy Spectrum Evolution')
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









def evolve_system(T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3, n_points=1000):
    times = np.linspace(0, T_total, n_points)
    dt = T_total/n_points

    energies = np.zeros((n_points, 8))  # Store all 8 eigenvalues
    couplings = np.zeros((n_points, 3))  # Store Δ1, Δ2, Δ3

    U_kato = np.eye(8)


    H0, coupling0 = build_hamiltonian(0,T_total, Δ_max, Δ_min, s, width, γ0,γ1,γ2,γ3)
    evals, evecs = np.linalg.eigh(H0)
    energies[0] = evals
    couplings[0] = coupling0

    V = evecs[:, :4]  # Initial degenerate subspace basis
    
    print("Analyzing spectrum over time...")
    for i in tqdm(range(1,len(times)), total=n_points):
        t = times[i]
        H, couplings[i] = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
        
        evals, evecs = np.linalg.eigh(H)
        energies[i] = evals

        W = evecs[:, :4]  # Next degenerate subspace basis
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


def get_ground_manifold_data(γ0, γ1, γ2, γ3):
    H0, _ = build_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
    HT, _ = build_hamiltonian(T_total, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)

    evals_0, evecs_0 = np.linalg.eigh(H0)
    evals_T, evecs_T = np.linalg.eigh(HT)

    V0 = evecs_0[:, :4]
    VT = evecs_T[:, :4]

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


def check_path_properties(times, energies, parity_op, γ0, γ1, γ2, γ3):
    max_hermiticity_error = 0.0
    max_parity_commutator = 0.0

    for t in times:
        H, _ = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, γ0, γ1, γ2, γ3)
        max_hermiticity_error = max(max_hermiticity_error, np.linalg.norm(H - H.conj().T))
        max_parity_commutator = max(
            max_parity_commutator,
            np.linalg.norm(H @ parity_op - parity_op @ H),
        )

    ground_splitting = np.max(energies[:, 3] - energies[:, 0])
    min_gap = np.min(energies[:, 4] - energies[:, 3])

    print("\nPath checks")
    print(f"max_t ||H(t) - H(t)†|| = {max_hermiticity_error:.2e}")
    print(f"max_t ||[H(t), P_tot]|| = {max_parity_commutator:.2e}")
    print(f"max_t (E3 - E0) = {ground_splitting:.2e}")
    print(f"min_t (E4 - E3) = {min_gap:.2e}")


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


def check_single_exchange(U_kato, gamma_list):
    expected_maps = [
        ("γ2 -> -γ3", gamma_list[2], -gamma_list[3]),
        ("γ3 ->  γ2", gamma_list[3], gamma_list[2]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    print("\nSingle-exchange checks")
    for label, source, target in expected_maps:
        transformed = U_kato.conj().T @ source @ U_kato
        error = np.linalg.norm(transformed - target)
        print(f"{label}: {error:.2e}")


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

    off_block = np.linalg.norm(U_parity[:2, 2:]) + np.linalg.norm(U_parity[2:, :2])
    odd_block = U_parity[:2, :2]
    even_block = U_parity[2:, 2:]

    U_target = expm(-0.25 * np.pi * (γ2 @ γ3))
    U_target_ground = V0.conj().T @ U_target @ V0
    U_target_parity = parity_vecs.conj().T @ U_target_ground @ parity_vecs
    odd_target = U_target_parity[:2, :2]
    even_target = U_target_parity[2:, 2:]

    print("\nParity-resolved gate checks")
    print(f"parity eigenvalues in GS manifold: {np.round(parity_vals, 8)}")
    print(f"off-block leakage in parity basis: {off_block:.2e}")
    print(f"odd-block eigenvalues:  {np.round(np.linalg.eigvals(odd_block), 8)}")
    print(f"even-block eigenvalues: {np.round(np.linalg.eigvals(even_block), 8)}")
    print(f"odd-block target error:  {phase_aligned_error(odd_block, odd_target):.2e}")
    print(f"even-block target error: {phase_aligned_error(even_block, even_target):.2e}")


if __name__ == "__main__":

    (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_full_gammas(levels_to_include=4, verbose=False)


    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals={"U": [0.1]},
        config_path=default_config_path(),
    )


    H_full = builder.full_system_hamiltonian()


    _, eigvecs = np.linalg.eigh(H_full)

    V_ref = eigvecs[:,:8]  # Reference basis (full eigenbasis at t=0)
    gamma_A1_sub = V_ref.conj().T @ gamma_A1_full @ V_ref
    gamma_A2_sub = V_ref.conj().T @ gamma_A2_full @ V_ref
    gamma_B1_sub = V_ref.conj().T @ gamma_B1_full @ V_ref
    gamma_B2_sub = V_ref.conj().T @ gamma_B2_full @ V_ref
    gamma_C1_sub = V_ref.conj().T @ gamma_C1_full @ V_ref
    gamma_C2_sub = V_ref.conj().T @ gamma_C2_full @ V_ref
    gamma_A1_sub /= np.sqrt(np.trace(gamma_A1_sub @ gamma_A1_sub).real / 8)
    gamma_A2_sub /= np.sqrt(np.trace(gamma_A2_sub @ gamma_A2_sub).real / 8)
    gamma_B1_sub /= np.sqrt(np.trace(gamma_B1_sub @ gamma_B1_sub).real / 8)
    gamma_B2_sub /= np.sqrt(np.trace(gamma_B2_sub @ gamma_B2_sub).real / 8)
    gamma_C1_sub /= np.sqrt(np.trace(gamma_C1_sub @ gamma_C1_sub).real / 8)
    gamma_C2_sub /= np.sqrt(np.trace(gamma_C2_sub @ gamma_C2_sub).real / 8)





    γ0, γ1, γ2, γ3 = gamma_A1_sub, gamma_A2_sub, gamma_B1_sub, gamma_C1_sub

    T_total = 1000.0
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
        'n_points': 10000
    }


    times, energies, couplings, U_kato = evolve_system(**params)
    plot_results(times, energies, couplings)




    gamma_list = [γ0, γ1, γ2, γ3]
    parity_projected = build_total_parity_projected(builder, V_ref)
    ground_data = get_ground_manifold_data(γ0, γ1, γ2, γ3)

    check_majorana_algebra(gamma_list)
    check_path_properties(times, energies, parity_projected, γ0, γ1, γ2, γ3)
    check_kato_transport(U_kato, ground_data["P0"], ground_data["PT"])
    check_single_exchange(U_kato, gamma_list)
    check_double_exchange(U_kato, gamma_list)
    check_four_exchanges(U_kato, gamma_list)
    check_parity_resolved_gate(U_kato, ground_data["V0"], parity_projected, γ2, γ3)
