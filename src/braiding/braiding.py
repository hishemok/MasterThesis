from full_system_hamiltonian import *
from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
from numba import njit, objmode
from scipy.sparse.linalg import eigsh
import time


def majorana_operators(n):
    create, annihilate, number = precompute_ops(n)
    majorana_ops = []
    for j in range(n):
        f_dag = create[j]
        f = annihilate[j]
        gamma_1 = f + f_dag
        gamma_2 = -1j * (f - f_dag)

        majorana_ops.append((gamma_1, gamma_2))
    return np.array(majorana_ops)

def braid_operator(gamma_a, gamma_b):
    return expm(0.25 * np.pi * (gamma_a @ gamma_b))

def majorana_polarization(even_vecs, odd_vecs, n):
    """
    Calculate the Majorana polarization for each state.
    """
    majorana_ops = majorana_operators(n)
    MP = []
    # print(even_vecs.shape, odd_vecs.shape)

    for i in range(even_vecs.shape[1]):
        evec = even_vecs[:, i]
        ovec = odd_vecs[:, i]
        MP_j = np.zeros(n)
        for j in range(n):
            gamma_1, gamma_2 = majorana_ops[j]
            amp1 = np.vdot(evec, gamma_1 @ ovec)
            amp2 = np.vdot(evec, gamma_2 @ ovec)
            MP_j[j] = np.abs(amp1)**2 + np.abs(amp2)**2
        MP.append(MP_j)
    return np.array(MP)

def majorana_polarization_from_rho(rh0_j, gamma_ops):

    MP = np.zeros(len(gamma_ops))
    for j, (gamma_1, gamma_2) in enumerate(gamma_ops):
        amp1 = np.trace(rh0_j @ gamma_1).real
        amp2 = np.trace(rh0_j @ gamma_2).real
        MP[j] = amp1**2 + amp2**2
    return MP

def quick_MP_check(even_vecs, odd_vecs, n, threshold=1e-3, verbose = False):
    """
    Quickly check if states are Majorana zero modes by evaluating overlaps with Majorana operators.
    """
    MP = majorana_polarization(even_vecs, odd_vecs, n)
    if verbose: print("Checking Majorana character at edges:")
    for i in range(MP.shape[0]):
        MP_i = MP[i]
        left = MP_i[0]
        right = MP_i[-1]
        # mid = MP_i[n//2]
        if abs(1 - left) and abs(1 - right) < threshold:
            pass
        else:
            print(f"Warning: State {i} does not exhibit strong Majorana character at the edges\n MP_{i}: {MP_i}.")
    
    if verbose: print("Quick Majorana check complete.")


def divide_to_even_odd(eigenvalues, eigenvectors, num):
    """Divide eigenstates into even and odd parity using the total parity operator."""
    P = total_parity(num)
    
    n_states = len(eigenvalues)
    dim = eigenvectors.shape[0]
    
    # Preallocate arrays (worst case: all even or all odd)
    even_states_tmp = np.zeros(n_states, dtype=eigenvalues.dtype)
    odd_states_tmp = np.zeros(n_states, dtype=eigenvalues.dtype)
    even_vecs_tmp = np.zeros((dim, n_states), dtype=eigenvectors.dtype)
    odd_vecs_tmp = np.zeros((dim, n_states), dtype=eigenvectors.dtype)
    
    even_count = 0
    odd_count = 0

    for idx in range(n_states):
        vec = eigenvectors[:, idx]
        parity = np.vdot(vec, P @ vec).real
        
        if np.abs(parity - 1.0) < 1e-9:
            even_states_tmp[even_count] = eigenvalues[idx]
            even_vecs_tmp[:, even_count] = vec
            even_count += 1
        elif np.abs(parity + 1.0) < 1e-9:
            odd_states_tmp[odd_count] = eigenvalues[idx]
            odd_vecs_tmp[:, odd_count] = vec
            odd_count += 1
        else:
            raise ValueError("State has non-integer parity expectation value.")

    # Slice to actual counts
    even_states = even_states_tmp[:even_count]
    odd_states = odd_states_tmp[:odd_count]
    even_vecs = even_vecs_tmp[:, :even_count]
    odd_vecs = odd_vecs_tmp[:, :odd_count]
    
    return even_states, odd_states, even_vecs, odd_vecs


def global_density_matrix(evecs, N):
    """Compute the reduced density matrix for a set of eigenvectors."""
    dim = evecs.shape[0]
    rho = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N):
        vec = evecs[:, i]
        rho += np.outer(vec, vec.conj())
    
    rho /= N
    return rho


def partial_trace(rho, keep, dims):
    """
    Partial trace over subsystems NOT in keep.
    """
    dims = list(dims)
    N = len(dims)
    traced = [i for i in range(N) if i not in keep]

    # reshape to tensor
    reshaped = rho.reshape(dims + dims)

    # IMPORTANT: trace highest indices first
    for i in sorted(traced, reverse=True):
        reshaped = np.trace(
            reshaped,
            axis1=i,
            axis2=i + reshaped.ndim//2
        )

    dim_keep = int(np.prod([dims[i] for i in keep]))
    return reshaped.reshape(dim_keep, dim_keep)

def entanglement_entropy(rho, eps=1e-12):
    """
    Calculate the von Neumann entanglement entropy of a density matrix rho.
    S = 0 -- subsystem is disentangled
    S = ln(2) -- single majorana shared
    S = ln(4) -- full fermion delocalization
    """
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > eps]
    return -np.sum(evals * np.log(evals))

def local_occupations_from_rho(rho, num_ops):
    """
    rho      : reduced density matrix (2^n × 2^n)
    num_ops  : list of number operators n_k for the subsystem
    returns  : array of <n_k>
    """
    occs = np.zeros(len(num_ops))
    for k, nk in enumerate(num_ops):
        occs[k] = np.trace(rho @ nk).real
    return occs

def majorana_correlation_matrix(rho, gamma_ops):
    """
    rho        : reduced density matrix
    gamma_ops  : list of Majorana operators [γ1, γ2, ..., γ2n]
    returns    : antisymmetric correlation matrix C_ab
    """
    n = len(gamma_ops)
    C = np.zeros((n, n))
    for i in range(n):
        g1, g2 = gamma_ops[i]
        for j in range(n):
            g3, g4 = gamma_ops[j]

            C[i,j] = 0.5 * (np.trace(rho @ g1@g3) + np.trace(rho @ g2@g4))
            # print(C[i,j])
    return C

def check_state(state, majorana_ops, threshold=1e-3):
    """Check if a state is a Majorana zero mode by evaluating its overlap with Majorana operators."""
    overlaps = []
    for gamma_1, gamma_2 in majorana_ops:
        overlap_1 = np.abs(np.vdot(state, gamma_1 @ state))
        overlap_2 = np.abs(np.vdot(state, gamma_2 @ state))
        overlaps.append((overlap_1, overlap_2))
    max_overlap = max(max(pair) for pair in overlaps)
    return max_overlap < threshold


def get_gs_pairs(n, H):
    """   
    Get ground state energy pairs (even and odd) and their vectors for a given Hamiltonian.

    Args:
        n (int): Number of sites in the system.
        H (np.ndarray): Hamiltonian matrix.
    
    Returns:
        ((float, np.ndarray), (float, np.ndarray)): Tuple containing even and odd ground state energies and vectors. Order: ((even_energy, even_vector), (odd_energy, odd_vector))
    
    """

    evals, evecs = np.linalg.eigh(H)
    cre, ann, num = precompute_ops(n)
    even_states, odd_states, even_vecs, odd_vecs = divide_to_even_odd(evals, evecs, num)

    even_eigval_gs = even_states[0]
    odd_eigval_gs = odd_states[0]
    even_eigvec_gs = even_vecs[:, 0]
    odd_eigvec_gs = odd_vecs[:, 0]

    return (even_eigval_gs, even_eigvec_gs), (odd_eigval_gs, odd_eigvec_gs)


def orthonormalize(vs):
    """Orthonormalize columns of vs using QR decomposition."""
    q, _ = np.linalg.qr(vs)
    return q

# @njit
def berry_phase_kato(times, evecs):
    n = len(times)
    dt = times[1] - times[0]
    
    dim = evecs.shape[1]
    U = np.eye(dim, dtype=np.complex128)

    for i in range(n - 1):
        # Pull out the two eigenvectors at step i and i+1
        v1 = evecs[i, :, 0]
        v2 = evecs[i, :, 1]
        w1 = evecs[i+1, :, 0]
        w2 = evecs[i+1, :, 1]

        # Build orthonormal subspace basis
        V = np.column_stack((v1, v2))
        W = np.column_stack((w1, w2))
        P = V @ V.conj().T
        P_next = W @ W.conj().T

        dPdt = (P_next - P) / dt
        K = P @ dPdt - dPdt @ P

        # with objmode(tmp='complex128[:,:]'):
        tmp = expm(-dt * K)

        U = tmp @ U

    return U
@njit
def berry_phase_kato_low_energy(times, V_low):
    """
    Berry phase calculation in the low-energy K-dimensional subspace.
    
    Parameters
    ----------
    times : array_like
        Array of time steps, shape (n_steps,)
    V_low : array_like
        Low-energy eigenvectors, shape (n_steps, dim, K)
    
    Returns
    -------
    U_berry : complex ndarray
        Berry unitary in K-dimensional subspace, shape (K, K)
    """
    n_steps, dim, K = V_low.shape
    U_berry = np.eye(K, dtype=np.complex128)
    
    for i in range(n_steps - 1):
        dt = times[i+1] - times[i]
        V = V_low[i]       # (dim, K)
        W = V_low[i+1]     # (dim, K)
        
        # Projectors onto subspace
        P = V @ V.conj().T
        P_next = W @ W.conj().T
        
        dPdt = (P_next - P) / dt
        K_mat = P @ dPdt - dPdt @ P
        
        # Evolution in full space
        with objmode(U_step_full='complex128[:,:]'):
            U_step_full = expm(-dt * K_mat)
        
        # Restrict to subspace: V^dagger * U_step * V
        U_step_subspace = V.conj().T @ U_step_full @ V
        
        # Accumulate Berry unitary
        U_berry = U_step_subspace @ U_berry
    
    return U_berry

@njit
def find_number_of_degenerate_ground_states(evals, tol=1e-6):
    """
    Find the number of degenerate ground states given a list of eigenvalues.
    
    Parameters
    ----------
    evals : array_like
        Array of eigenvalues.
    tol : float, optional
        Tolerance for degeneracy check. Default is 1e-6.
    Returns
    -------
    int
        Number of degenerate ground states.
    """
    ground_energy = evals[0]
    degenerate_count = 1
    
    for energy in evals[1:]:
        if abs(energy - ground_energy) < tol:
            degenerate_count += 1
        else:
            break
    
    return degenerate_count

@njit
def simple_delta_pulse(t, T_peak, width, s, max_val, min_val):

    
    T_start = T_peak - width / 2
    T_end = T_peak + width / 2

    rise = 1/(1 + np.exp(-s*(t - T_start)))
    fall = 1/(1 + np.exp(s*(t - T_end)))

    return min_val + (max_val - min_val) * rise * fall

def build_Hamiltonian(current_time, n_sites, dupes, T_total, t, U, eps, Delta, width, s, couple_defs, eps_detune=None, operators=None):
    # Pulses
    t_AB = (
        simple_delta_pulse(current_time, 0, width, s, couple_defs['AB']['t'], 0)
        + simple_delta_pulse(current_time, T_total, width, s, couple_defs['AB']['t'], 0)
    )
    d_AB = (
        simple_delta_pulse(current_time, 0, width, s, couple_defs['AB']['delta'], 0)
        + simple_delta_pulse(current_time, T_total, width, s, couple_defs['AB']['delta'], 0)
    )

    t_BC = simple_delta_pulse(
        current_time, 2*T_total/3, width, s, couple_defs['BC']['t'], 0
    )
    d_BC = simple_delta_pulse(
        current_time, 2*T_total/3, width, s, couple_defs['BC']['delta'], 0
    )

    #  Time-dependent coupling list
    couplings = [
        (couple_defs['AB']['A'], couple_defs['AB']['B'], t_AB, d_AB),
        (couple_defs['BC']['A'], couple_defs['BC']['B'], t_BC, d_BC),
    ]
    

    #  Time-dependent detuning
    eps_now = None
    if eps_detune is not None:
        pulseB = simple_delta_pulse(
            current_time, T_total/3, width, s,
            eps_detune.get(1, 0.0), 0.0
        )
        eps_now = {1: pulseB}
    

    
    H_t = big_H(
        n_sites, dupes,
        t, U, eps, Delta,
        couplings=couplings,
        eps_detune=eps_now,
        operators=operators
    )

    pulses = {
        "t_AB": t_AB,
        "d_AB": d_AB,
        "t_BC": t_BC,
        "d_BC": d_BC,
        "eps_B": eps_now[1] if eps_now else 0.0
    }

    return H_t, pulses


def time_evolution2(T_total, n_steps, n_sites, dupes, t, U, eps, Delta, width, s, couple_defs, eps_detune=None, K=8):
    big_N = n_sites * dupes
    Time_array = np.linspace(0, T_total, n_steps)
    dim = 2**big_N

    # Storage arrays
    E_low = np.zeros((n_steps, K))
    V_low = np.zeros((n_steps, dim, K), dtype=complex)
    gap   = np.zeros(n_steps)

   
    # Single site information storage
    gamma_ops = majorana_operators(n_sites)

    # Storage for observables
    MP_storage = np.zeros((n_steps, dupes, len(gamma_ops)))
    parity_expectations = np.zeros((n_steps, dupes))
    entropy = np.zeros((n_steps, dupes))
    local_occupations = np.zeros((n_steps, dupes, n_sites))
    majorana_correlations = np.zeros((n_steps, dupes, len(gamma_ops), len(gamma_ops)))


    # Operator precomputation
    cre, ann, num = precompute_ops(big_N)
    hop_ops = {}
    pair_ops = {}
    dens_ops = {}

    for i in range(big_N):
        for j in range(i+1, big_N):
            hop_ops[(i,j)] = cre[i] @ ann[j] + ann[i] @ cre[j]
            pair_ops[(i,j)] = cre[i] @ cre[j] + ann[j] @ ann[i]
            dens_ops[(i,j)] = num[i] @ num[j]
    
    operators = {"cre": cre, "ann": ann, "num": num,
                 "hop": hop_ops, "pair": pair_ops, "dens": dens_ops}
    
    _, _, single_site_num = precompute_ops(n_sites)

    all_pulses = []
    P_j = np.eye(2**n_sites)
    for nk in single_site_num:
        P_j = P_j @ (np.eye(2**n_sites) - 2*nk)


    for i, current in enumerate(tqdm(Time_array)):
        H_t, pulses = build_Hamiltonian( current, n_sites, dupes, T_total, t, U, eps, Delta, width, s, couple_defs, eps_detune=eps_detune, operators=operators
        )
        all_pulses.append(pulses)
  
        eigvals, evecs = np.linalg.eigh(H_t)


        E_low[i] = eigvals[:K]
        V_low[i] = evecs[:, :K]
        gap[i]   = eigvals[K+1] - eigvals[K]


        
        rho = global_density_matrix(evecs, N=K) # Should equal 4 for n_sites=3 with the active couplings
        for j in range(dupes):
            keep_sites = list(range(j*n_sites, (j+1)*n_sites))
            rho_reduced = partial_trace(rho, keep=keep_sites, dims=[2]*big_N) 
            
            parity_expectation = np.trace(rho_reduced @ P_j).real
            parity_expectations[i, j] = parity_expectation

            MP_storage[i, j] = majorana_polarization_from_rho(rho_reduced, gamma_ops)

            entropy[i, j] = entanglement_entropy(rho_reduced)

            local_occupations[i, j] = local_occupations_from_rho(rho_reduced, single_site_num)


            majorana_correlations[i, j] = majorana_correlation_matrix(rho_reduced, gamma_ops)
        
    
        
    return (Time_array, E_low, V_low, gap, MP_storage, local_occupations, majorana_correlations, entropy, all_pulses)

if __name__ == "__main__":
    n_sites = 3
    dupes = 3

    cre, ann, num = precompute_ops(n_sites * dupes)
    P = total_parity(num)

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")

    t, U, eps, Delta = pars


    T_total = 10
    n_steps = 500
    width = T_total / 3
    s = T_total * 6
    t_couple = 1.0
    delta_couple = 1.0
    eps_detune = {1: 1.0}  # Detune PMM
    K = 4


    couple_defs = {
        'AB': {'A': (0,2), 'B': (1,0), 't': t_couple, 'delta': delta_couple},
        'BC': {'A': (1,0), 'B': (2,0), 't': t_couple, 'delta': delta_couple},
    }

    # Time evolution
    outputs = time_evolution2(T_total,n_steps ,n_sites, dupes, t, U, eps, Delta, width, s, couple_defs, eps_detune=eps_detune, K = K)

    (Time_array, E_low, V_low, gap, MP_storage, local_occupations, majorana_correlations, entropy, all_pulses) = outputs

    #Plot gap
    plt.figure(figsize=(7,4))
    plt.plot(Time_array, gap)
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Gap to excited states")
    plt.title("Adiabatic gap")
    plt.grid(True)
    plt.show()

    print("Minimum gap:", gap.min())


    #Plot energy spectrum
    plt.figure(figsize=(7,4))
    for j in range(K):
        plt.plot(Time_array, E_low[:, j], label=f"State {j}")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Low-energy spectrum")
    plt.legend()
    plt.grid(True)
    plt.show()


    # # Plot majorana correlation matrix elements for each PMM
    # for i in range(majorana_correlations.shape[1]): 
    #     plt.figure(figsize=(7,4))
    #     for a in range(majorana_correlations.shape[2]):
    #         for b in range(majorana_correlations.shape[3]):
    #             plt.plot(Time_array, majorana_correlations[:, i, a, b], label=f"C[{a},{b}]")
    #     plt.xlabel("Time")
    #     plt.ylabel("Majorana Correlation C_ab")
    #     plt.title(f"Majorana Correlation Matrix Elements – PMM {i}")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True)
    #     plt.show()
    

    # #plot local occupations for each PMM
    # for i in range(local_occupations.shape[1]):
    #     plt.figure(figsize=(7,4))
    #     for j in range(local_occupations.shape[2]):
    #         plt.plot(Time_array, local_occupations[:, i, j], label=f"Site {j}")
    #     plt.xlabel("Time")
    #     plt.ylabel("Local occupation ⟨n⟩")
    #     plt.title(f"Local occupations – PMM {i}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    
    # # Plot entanglement entropy for each PMM
    # for i in range(entropy.shape[1]):
    #     plt.figure(figsize=(7,4))
    #     plt.plot(Time_array, entropy[:, i])
    #     plt.xlabel("Time")
    #     plt.ylabel("Entanglement Entropy S")
    #     plt.title(f"Entanglement Entropy – PMM {i}")
    #     plt.grid(True)
    #     plt.show()


    # # Plot pulses
    # t_AB = np.array([p["t_AB"] for p in all_pulses])
    # d_AB = np.array([p["d_AB"] for p in all_pulses])
    # t_BC = np.array([p["t_BC"] for p in all_pulses])
    # d_BC = np.array([p["d_BC"] for p in all_pulses])
    # eps_B = np.array([p["eps_B"] for p in all_pulses])

    # plt.subplots(figsize=(7,6), nrows=2, ncols=1, sharex=True)
    # plt.subplot(2,1,1)
    # plt.plot(Time_array, t_AB, label="t A↔B")
    # plt.plot(Time_array, t_BC, label="t B↔C")
    # plt.plot(Time_array, eps_B, "--", label="ε B")
    # plt.ylabel("Tunneling amplitude")
    # plt.title("Tunneling pulses")
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(2,1,2)
    # plt.plot(Time_array, d_AB, label="Δ A↔B")
    # plt.plot(Time_array, d_BC, label="Δ B↔C")
    # plt.plot(Time_array, eps_B, "--", label="ε B")
    # plt.xlabel("Time")
    # plt.ylabel("Pairing / detuning amplitude")
    # plt.title("Pairing & detuning pulses")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    U_berry = berry_phase_kato_low_energy(Time_array, V_low)
    # print("V_low shape",V_low.shape)
    # print("Berry shape",U_berry.shape)

    # print("Berry unitary:")
    # print(U_berry)

    # # Eigenphases
    # phases = np.angle(np.linalg.eigvals(U_berry)) / np.pi
    # print("Berry phases / π:", phases)
    gamma_ops = majorana_operators(np.log2(K).astype(int))
    gammas = []
    for g1, g2 in gamma_ops:
        gammas.append(g1)
        gammas.append(g2)


    E_sub = np.array([E_low[:, k] for k in range(K)])  # shape (K, n_steps)
    phase_mat = np.diag(np.exp(-1j * np.trapz(E_sub, axis=1))) # type: ignore
    U_total = phase_mat @ U_berry
    print("Total unitary:")
    # print(U)
    det_phase = np.linalg.det(U_total)**(1/K)
    U_phys = U_total / det_phase
    print(U_phys)

    #Dont remember which is right
    U_target_full1 = braid_operator(gammas[1], gammas[2])
    U_target_full2 = braid_operator(gammas[2], gammas[3])
    print(gammas[1].shape)
    print(U_target_full1.shape)
    print(V_low.shape)
    U_target_sub1 = V_low[0].conj().T @ U_target_full1 @ V_low[0]
    U_target_sub2 = V_low[0].conj().T @ U_target_full2 @ V_low[0]


    F = abs(np.trace(U_phys.conj().T @ U_target_sub1))**2 / K**2
    print(f"Fidelity with target braiding gate: {F:.6f}")
    F = abs(np.trace(U_phys.conj().T @ U_target_sub2))**2 / K**2
    print(f"Fidelity with target braiding gate: {F:.6f}")

