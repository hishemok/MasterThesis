from full_system_hamiltonian import *
from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
from numba import njit, objmode


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

@njit
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

        with objmode(tmp='complex128[:,:]'):
            tmp = expm(-dt * K)

        U = tmp @ U

    return U


@njit
def simple_delta_pulse(t, T_peak, width, s, max_val, min_val):

    
    T_start = T_peak - width / 2
    T_end = T_peak + width / 2

    rise = 1/(1 + np.exp(-s*(t - T_start)))
    fall = 1/(1 + np.exp(s*(t - T_end)))

    return min_val + (max_val - min_val) * rise * fall


def build_Hamiltonian(current_time,n_sites, dupes, T_total, t, U, eps, Delta, t_couple1, delta_couple1, t_couple2, delta_couple2, eps_detune, width, s):

    puls_dict = {}

    pulseAB_t = simple_delta_pulse(current_time, T_peak=0, width=width, s=s, max_val=t_couple1, min_val=0) + simple_delta_pulse(current_time, T_peak=T_total, width=width, s=s, max_val=t_couple1, min_val=0)
    pulseAB_delta = simple_delta_pulse(current_time, T_peak=0, width=width, s=s, max_val=delta_couple1, min_val=0) + simple_delta_pulse(current_time, T_peak=T_total, width=width, s=s, max_val=delta_couple1, min_val=0)

    pulseB = simple_delta_pulse(current_time, T_peak=T_total/3, width=width, s=s, max_val=eps_detune.get(1, 1), min_val=0)

    pulseBC_t = simple_delta_pulse(current_time, T_peak=2*T_total/3, width=width, s=s, max_val=t_couple2, min_val=0)
    pulseBC_delta = simple_delta_pulse(current_time, T_peak=2*T_total/3, width=width, s=s, max_val=delta_couple2, min_val=0)

    puls_dict['A_to_B'] = (pulseAB_t, pulseAB_delta)
    puls_dict['B_self'] = pulseB
    puls_dict['B_to_C'] = (pulseBC_t, pulseBC_delta)
    H_t = big_H(n_sites, dupes, t, U, eps, Delta,
                couple_A=(0,2),
                couple_B=(1,0),
                t_couple1=pulseAB_t,
                delta_couple1=pulseAB_delta,
                couple_C=(1,0),
                couple_D=(2,0),
                t_couple2=pulseBC_t,
                delta_couple2=pulseBC_delta,
                eps_detune={1: pulseB})
    return H_t, puls_dict

def time_evolution2(T_total,n_steps ,n_sites, dupes, t, U, eps, Delta, t_couple1, delta_couple1, t_couple2, delta_couple2, eps_detune, width, s):
    
    n_steps = n_steps
    Time_array = np.linspace(0, T_total, n_steps)

    dims = 2**(n_sites * dupes)

    eigvals = np.zeros((n_steps, dims))
    eigvecs = np.zeros((n_steps, dims, dims), dtype=complex)

    sets_eigvals_even = np.zeros((n_steps, dupes, 2**(n_sites) // 2 ))
    sets_eigvals_odd = np.zeros_like(sets_eigvals_even)
    sets_eigvecs_even = np.zeros((n_steps, dupes, 2**(n_sites), 2**(n_sites) // 2), dtype=complex)
    sets_eigvecs_odd = np.zeros_like(sets_eigvecs_even)

    MP_storage = np.zeros((n_steps, dupes, 2**(n_sites) // 2, n_sites))

    all_pulses = []


    for i in tqdm(range(n_steps)):
        current = Time_array[i]
        H_t, pulses = build_Hamiltonian(current,n_sites, dupes, T_total, t, U, eps, Delta, t_couple1, delta_couple1, t_couple2, delta_couple2, eps_detune, width, s)
        all_pulses.append(pulses)
        eigvals_t, eigvecs_t = np.linalg.eigh(H_t)

        eigvals[i] = eigvals_t
        eigvecs[i] = eigvecs_t

        for j in range(dupes):
            set_H_t = extract_effective_H(H_t, n_sites, dupes, target=j)
            set_evals_t, set_evecs_t = np.linalg.eigh(set_H_t)
            cre, ann, num = precompute_ops(n_sites)
            set_even_states, set_odd_states, set_even_vecs, set_odd_vecs = divide_to_even_odd(set_evals_t, set_evecs_t, num)

            MP = majorana_polarization(set_even_vecs, set_odd_vecs, n_sites)
 
            MP_storage[i, j] = MP

            sets_eigvals_even[i, j] = set_even_states
            sets_eigvals_odd[i, j] = set_odd_states
            sets_eigvecs_even[i, j] = set_even_vecs
            sets_eigvecs_odd[i, j] = set_odd_vecs
    return (eigvals,
            eigvecs,
            sets_eigvals_even,
            sets_eigvals_odd,
            sets_eigvecs_even,
            sets_eigvecs_odd,
            MP_storage,
            Time_array,
            all_pulses)
        


if __name__ == "__main__":
    n_sites = 3
    dupes = 3

    cre, ann, num = precompute_ops(n_sites * dupes)
    P = total_parity(num)

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")

    t, U, eps, Delta = pars


    T_total = 10
    n_steps = 50
    width = T_total / 3
    s = T_total * 6
    t_couple = 1.0
    delta_couple = 1.0
    eps_detune = {1: 1.0}  # Detune PMM

    timearray = np.linspace(0, T_total, n_steps)
    pulseAB = [simple_delta_pulse(t, T_peak=0, width=width, s=s, max_val=t_couple, min_val=0) + simple_delta_pulse(t, T_peak=T_total, width=width, s=s, max_val=t_couple, min_val=0)  for t in timearray] 
    pulseB = [simple_delta_pulse(t, T_peak=T_total/3, width=width, s=s, max_val=1.0, min_val=0) for t in timearray]
    PulseBC = [simple_delta_pulse(t, T_peak=2*T_total/3, width=width, s=s, max_val=t_couple, min_val=0) for t in timearray]

    plt.plot(timearray, pulseAB, label='t A to B')
    plt.plot(timearray, pulseB, label='eps B')
    plt.plot(timearray, PulseBC, label='t B to C')
    plt.xlabel('Time')
    plt.ylabel('Pulse Amplitude')
    plt.title('Tuning Pulses Over Time')
    plt.legend()
    plt.grid()
    plt.show()



    # Time evolution
    outputs = time_evolution2(T_total,n_steps ,n_sites, dupes, t, U, eps, Delta, t_couple, delta_couple, t_couple, delta_couple, eps_detune, width, s)

    (eigvals, eigvecs, sets_eigvals_even, sets_eigvals_odd, sets_eigvecs_even, sets_eigvecs_odd, MP_storage, Time_array, all_pulses) = outputs

    # U_berry = berry_phase_kato(Time_array, eigvecs)
    

    

    cre, ann, num = precompute_ops(n_sites * dupes)

    even_states, odd_states, even_vecs, odd_vecs = divide_to_even_odd(eigvals, eigvecs,num)

    #Plot 12 lowest eigenvalues for each set over time
    for i in range(12):
        plt.hlines(even_states[i], xmin=0, xmax=0.5, colors='b', linestyles='dashed', label='Even States' if i == 0 else "")
        plt.hlines(odd_states[i], xmin=1, xmax=1.5, colors='r', linestyles='dotted', label='Odd States' if i == 0 else "")
    
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Lowest 12 Eigenvalues Over Time')
    plt.xticks([0.25, 1.25], ["Even", "Odd"])

    plt.legend()
    plt.show()




    for i in range(dupes):
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Spectrum Evolution for Set {i+1}', fontsize=16)

        plt.subplot(2, 1, 1)
        plt.title('Even Parity States')
        for j in range(sets_eigvals_even.shape[2]):
            plt.plot(Time_array, sets_eigvals_even[:, i, j], label=f'State {j+1}')
        plt.ylabel('Energy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title('Odd Parity States')
        for j in range(sets_eigvals_odd.shape[2]):
            plt.plot(Time_array, sets_eigvals_odd[:, i, j], label=f'State {j+1}')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()

        plt.tight_layout()
        plt.show()


    #Plot 8 lowest eigenvalues over time
    plt.figure(figsize=(10, 6))
    plt.title('Lowest 8 Eigenvalues Over Time')
    for j in range(4):
        plt.plot(Time_array, eigvals[:, j], label=f'State {j+1}')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()

    #Majorana Polarization plots
    for i in range(dupes):
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Majorana Polarization Evolution for Set {i+1}', fontsize=16)

        for j in range(MP_storage.shape[2]):
            for k in range(n_sites):
                plt.plot(Time_array, MP_storage[:, i, j, k], label=f'State {j+1}, Site {k+1}')
        
        plt.xlabel('Time')
        plt.ylabel('Majorana Polarization')
        plt.legend()
        plt.show()
    
    t_AB = np.array([p["A_to_B"][0] for p in all_pulses])
    delta_AB = np.array([p["A_to_B"][1] for p in all_pulses])

    t_BC = np.array([p["B_to_C"][0] for p in all_pulses])
    delta_BC = np.array([p["B_to_C"][1] for p in all_pulses])

    eps_B = np.array([p["B_self"] for p in all_pulses])
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True
    )

    # ---- Plot t pulses ----
    ax1.plot(Time_array, t_AB, label=r"$t_{A\leftrightarrow B}$")
    ax1.plot(Time_array, t_BC, label=r"$t_{B\leftrightarrow C}$")
    ax1.plot(Time_array, eps_B, "--", label=r"$\varepsilon_B$")

    ax1.set_ylabel("t / ε amplitude")
    ax1.set_title("Tunneling & detuning pulses")
    ax1.legend()
    ax1.grid(True)

    # ---- Plot delta pulses ----
    ax2.plot(Time_array, delta_AB, label=r"$\Delta_{A\leftrightarrow B}$")
    ax2.plot(Time_array, delta_BC, label=r"$\Delta_{B\leftrightarrow C}$")
    ax2.plot(Time_array, eps_B, "--", label=r"$\varepsilon_B$")

    ax2.set_ylabel("Δ / ε amplitude")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()