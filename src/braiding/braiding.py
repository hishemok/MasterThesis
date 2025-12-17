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
        U_step_full = expm(-dt * K_mat)
        
        # Restrict to subspace: V^dagger * U_step * V
        U_step_subspace = V.conj().T @ U_step_full @ V
        
        # Accumulate Berry unitary
        U_berry = U_step_subspace @ U_berry
    
    return U_berry

@njit
def simple_delta_pulse(t, T_peak, width, s, max_val, min_val):

    
    T_start = T_peak - width / 2
    T_end = T_peak + width / 2

    rise = 1/(1 + np.exp(-s*(t - T_start)))
    fall = 1/(1 + np.exp(s*(t - T_end)))

    return min_val + (max_val - min_val) * rise * fall

def build_Hamiltonian(current_time,
                      n_sites, dupes,
                      T_total,
                      t, U, eps, Delta,
                      width, s,
                      couple_defs,
                      eps_detune=None,
                      operators=None):

    
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

# def build_Hamiltonian(current_time,n_sites, dupes, T_total, t, U, eps, Delta , width, s, couplings=(), eps_detune=None):

#     puls_dict = {}

#     pulseAB_t = simple_delta_pulse(current_time, T_peak=0, width=width, s=s, max_val=t_couple1, min_val=0) + simple_delta_pulse(current_time, T_peak=T_total, width=width, s=s, max_val=t_couple1, min_val=0)
#     pulseAB_delta = simple_delta_pulse(current_time, T_peak=0, width=width, s=s, max_val=delta_couple1, min_val=0) + simple_delta_pulse(current_time, T_peak=T_total, width=width, s=s, max_val=delta_couple1, min_val=0)

#     if eps_detune is not None:
#         pulseB = simple_delta_pulse(current_time, T_peak=T_total/3, width=width, s=s, max_val=eps_detune.get(1, 1), min_val=0)
#     else:
#         pulseB = 0

#     pulseBC_t = simple_delta_pulse(current_time, T_peak=2*T_total/3, width=width, s=s, max_val=t_couple2, min_val=0)
#     pulseBC_delta = simple_delta_pulse(current_time, T_peak=2*T_total/3, width=width, s=s, max_val=delta_couple2, min_val=0)

#     puls_dict['A_to_B'] = (pulseAB_t, pulseAB_delta)
#     puls_dict['B_self'] = pulseB
#     puls_dict['B_to_C'] = (pulseBC_t, pulseBC_delta)
#     H_t = big_H(n_sites, dupes, t, U, eps, Delta,
#                 couplings=couplings,
#                 eps_detune=None)
#     return H_t, puls_dict

# def time_evolution2(T_total,n_steps ,n_sites, dupes, t, U, eps, Delta, width, s, couplings = (), eps_detune = None):
    
#     ## TIme how long each calculation takes

#     n_steps = n_steps
#     Time_array = np.linspace(0, T_total, n_steps)

#     dims = 2**(n_sites * dupes)

#     eigvals = np.zeros((n_steps, dims))
#     eigvecs = np.zeros((n_steps, dims, dims), dtype=complex)

#     sets_eigvals_even = np.zeros((n_steps, dupes, 2**(n_sites) // 2 ))
#     sets_eigvals_odd = np.zeros_like(sets_eigvals_even)
#     sets_eigvecs_even = np.zeros((n_steps, dupes, 2**(n_sites), 2**(n_sites) // 2), dtype=complex)
#     sets_eigvecs_odd = np.zeros_like(sets_eigvecs_even)

#     MP_storage = np.zeros((n_steps, dupes, 2**(n_sites) // 2, n_sites))

#     all_pulses = []


#     for i in tqdm(range(n_steps)):
#         current = Time_array[i]
#         H_t, pulses = build_Hamiltonian(current,n_sites, dupes, T_total, t, U, eps, Delta, width, s, couplings=couplings, eps_detune=eps_detune)
        
#         all_pulses.append(pulses)
#         eigvals_t, eigvecs_t = np.linalg.eigh(H_t)

#         eigvals[i] = eigvals_t
#         eigvecs[i] = eigvecs_t

#         for j in range(dupes):
#             set_H_t = extract_effective_H(H_t, n_sites, dupes, target=j)
#             set_evals_t, set_evecs_t = np.linalg.eigh(set_H_t)

#             cre, ann, num = precompute_ops(n_sites)
#             set_even_states, set_odd_states, set_even_vecs, set_odd_vecs = divide_to_even_odd(set_evals_t, set_evecs_t, num)

#             MP = majorana_polarization(set_even_vecs, set_odd_vecs, n_sites)


#             MP_storage[i, j] = MP

#             sets_eigvals_even[i, j] = set_even_states
#             sets_eigvals_odd[i, j] = set_odd_states
#             sets_eigvecs_even[i, j] = set_even_vecs
#             sets_eigvecs_odd[i, j] = set_odd_vecs
#             exit()
#     return (eigvals,
#             eigvecs,
#             sets_eigvals_even,
#             sets_eigvals_odd,
#             sets_eigvecs_even,
#             sets_eigvecs_odd,
#             MP_storage,
#             Time_array,
#             all_pulses)
        


# def time_evolution2(T_total, n_steps, n_sites, dupes, t, U, eps, Delta,width, s, couple_defs, eps_detune=None):

#     Time_array = np.linspace(0, T_total, n_steps)
#     dims = 2**(n_sites * dupes)
#     eigvals = np.zeros((n_steps, dims))
#     eigvecs = np.zeros((n_steps, dims, dims), dtype=complex)    

#     sets_eigvals_even = np.zeros((n_steps, dupes, 2**(n_sites) // 2 ))
#     sets_eigvals_odd = np.zeros_like(sets_eigvals_even)
#     sets_eigvecs_even = np.zeros((n_steps, dupes, 2**(n_sites), 2**(n_sites) // 2), dtype=complex)
#     sets_eigvecs_odd = np.zeros_like(sets_eigvecs_even) 

#     MP_storage = np.zeros((n_steps, dupes, 2**(n_sites) // 2, n_sites))

#     all_pulses = []
#     for i in tqdm(range(n_steps)):
#         current = Time_array[i]
#         H_t, pulses = build_Hamiltonian(current,
#                                         n_sites, dupes,
#                                         T_total,
#                                         t, U, eps, Delta,
#                                         width, s,
#                                         couple_defs,
#                                         eps_detune=eps_detune)
        
#         all_pulses.append(pulses)
#         eigvals_t, eigvecs_t = np.linalg.eigh(H_t)

#         eigvals[i] = eigvals_t
#         eigvecs[i] = eigvecs_t

#         for j in range(dupes):
#             set_H_t = extract_effective_H(H_t, n_sites, dupes, target=j)
#             set_evals_t, set_evecs_t = np.linalg.eigh(set_H_t)

#             cre, ann, num = precompute_ops(n_sites)
#             set_even_states, set_odd_states, set_even_vecs, set_odd_vecs = divide_to_even_odd(set_evals_t, set_evecs_t, num)

#             MP = majorana_polarization(set_even_vecs, set_odd_vecs, n_sites)


#             MP_storage[i, j] = MP

#             sets_eigvals_even[i, j] = set_even_states
#             sets_eigvals_odd[i, j] = set_odd_states
#             sets_eigvecs_even[i, j] = set_even_vecs
#             sets_eigvecs_odd[i, j] = set_odd_vecs
#     return (eigvals,
#             eigvecs,
#             sets_eigvals_even,
#             sets_eigvals_odd,
#             sets_eigvecs_even,
#             sets_eigvecs_odd,
#             MP_storage,
#             Time_array,
#             all_pulses)
def time_evolution2(T_total, n_steps, n_sites, dupes, t, U, eps, Delta, width, s, couple_defs, eps_detune=None,K=4
):
    big_N = n_sites * dupes
    Time_array = np.linspace(0, T_total, n_steps)
    dim = 2**big_N

    # --- Low-energy storage (for Berry, gap, gates) ---
    E_low = np.zeros((n_steps, K))
    V_low = np.zeros((n_steps, dim, K), dtype=complex)
    gap   = np.zeros(n_steps)


    sets_eigvals_even = np.zeros((n_steps, dupes, 2**(n_sites) // 2 ))
    sets_eigvals_odd = np.zeros_like(sets_eigvals_even)

    # --- Majorana polarization ---
    MP_storage = np.zeros((n_steps, dupes, 2**n_sites//2, n_sites))

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

    for i, current in enumerate(tqdm(Time_array)):
        H_t, pulses = build_Hamiltonian(
            current,
            n_sites, dupes,
            T_total,
            t, U, eps, Delta,
            width, s,
            couple_defs,
            eps_detune=eps_detune,
            operators=operators
        )
        
        
        all_pulses.append(pulses)

        # low energy
        
        # evals, evecs = eigsh(H_t, k=K+1, which="SA")
        # order = np.argsort(evals)
        eigvals, evecs = np.linalg.eigh(H_t)
        E_low[i] = eigvals[:K]
        V_low[i] = evecs[:, :K]
        gap[i]   = eigvals[K] - eigvals[K-1]

        # E_low[i] = evals[order[:K]]
        # V_low[i] = evecs[:, order[:K]]
        # gap[i]   = evals[order[K]] - evals[order[K-1]]
        
        


        # --- MAJORANA POLARIZATION ---
        
        for j in range(dupes):
            set_H = extract_effective_H(H_t, n_sites, dupes, target=j)
            evals_s, evecs_s = np.linalg.eigh(set_H)
            # cre, ann, num = precompute_ops(n_sites)

            even_e, odd_e, even_v, odd_v = divide_to_even_odd(
                evals_s, evecs_s, single_site_num
            )

            sets_eigvals_even[i, j] = even_e
            sets_eigvals_odd[i, j] = odd_e

            MP_storage[i, j] = majorana_polarization(
                even_v, odd_v, n_sites
            )
        
        
    return (Time_array, E_low, V_low, gap, sets_eigvals_even, sets_eigvals_odd, MP_storage, all_pulses)

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



    couple_defs = {
        'AB': {'A': (0,2), 'B': (1,0), 't': t_couple, 'delta': delta_couple},
        'BC': {'A': (1,0), 'B': (2,0), 't': t_couple, 'delta': delta_couple},
    }

    # Time evolution
    outputs = time_evolution2(T_total,n_steps ,n_sites, dupes, t, U, eps, Delta, width, s, couple_defs, eps_detune=eps_detune)

    (Time_array, E_low, V_low, gap, sets_eigvals_even, sets_eigvals_odd, MP_storage, all_pulses) = outputs


    #plot even odd eigvals
    for j in range(dupes):
        plt.figure(figsize=(7,4))
        plt.title(f"Set {j} eigenvalues")
        plt.plot(sets_eigvals_even[:, j], label="Even")
        plt.plot(sets_eigvals_odd[:, j], label="Odd")
        plt.xlabel("Time step")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(True)
        plt.show()











    plt.figure(figsize=(7,4))
    plt.plot(Time_array, gap)
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Gap to excited states")
    plt.title("Adiabatic gap")
    plt.grid(True)
    plt.show()

    print("Minimum gap:", gap.min())

    plt.figure(figsize=(7,4))
    for j in range(2):
        plt.plot(Time_array, E_low[:, j], label=f"State {j}")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Low-energy spectrum")
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(MP_storage.shape[1]):
        plt.figure(figsize=(7,4))
        total_MP = MP_storage[:, i].sum(axis=(1,2))
        plt.plot(Time_array, total_MP)
        plt.xlabel("Time")
        plt.ylim(0, 10.1)
        plt.ylabel("Σ Majorana polarization")
        plt.title(f"Total Majorana polarization – PMM {i}")
        plt.grid(True)
        plt.show()


    t_AB = np.array([p["t_AB"] for p in all_pulses])
    d_AB = np.array([p["d_AB"] for p in all_pulses])
    t_BC = np.array([p["t_BC"] for p in all_pulses])
    d_BC = np.array([p["d_BC"] for p in all_pulses])
    eps_B = np.array([p["eps_B"] for p in all_pulses])

    plt.subplots(figsize=(7,6), nrows=2, ncols=1, sharex=True)
    plt.subplot(2,1,1)
    plt.plot(Time_array, t_AB, label="t A↔B")
    plt.plot(Time_array, t_BC, label="t B↔C")
    plt.plot(Time_array, eps_B, "--", label="ε B")
    plt.ylabel("Tunneling amplitude")
    plt.title("Tunneling pulses")
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(Time_array, d_AB, label="Δ A↔B")
    plt.plot(Time_array, d_BC, label="Δ B↔C")
    plt.plot(Time_array, eps_B, "--", label="ε B")
    plt.xlabel("Time")
    plt.ylabel("Pairing / detuning amplitude")
    plt.title("Pairing & detuning pulses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(7,4))
    # plt.plot(Time_array, t_AB, label="t A↔B")
    # plt.plot(Time_array, t_BC, label="t B↔C")
    # plt.plot(Time_array, eps_B, "--", label="ε B")
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.title("Control pulses")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    U_berry = berry_phase_kato_low_energy(Time_array, V_low)
    print("V_low shape",V_low.shape)
    print("Berry shape",U_berry.shape)

    print("Berry unitary:")
    print(U_berry)

    # Eigenphases
    phases = np.angle(np.linalg.eigvals(U_berry)) / np.pi
    print("Berry phases / π:", phases)


