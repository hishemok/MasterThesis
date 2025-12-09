from full_system_hamiltonian import *
from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import matplotlib.pyplot as plt


def majorana_operators(n):
    create, annihilate, number = precompute_ops(n)
    majorana_ops = []
    for j in range(n):
        f_dag = create[j]
        f = annihilate[j]
        gamma_1 = f + f_dag
        gamma_2 = -1j * (f - f_dag)
        majorana_ops.append((gamma_1, gamma_2))
    return majorana_ops

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



def divide_to_even_odd(eigenvalues, eigenvectors, n_sites):
    """Divide eigenstates into even and odd parity using the total parity operator."""
    cre, ann, num = precompute_ops(n_sites)
    P = total_parity(num)

    even_states = []
    odd_states = []
    even_vecs = []
    odd_vecs = []

    for idx, ev in enumerate(eigenvalues):
        vec = eigenvectors[:, idx]
        parity = np.vdot(vec, P @ vec).real
        if np.isclose(parity, 1.0):
            even_states.append(ev)
            even_vecs.append(vec)
        elif np.isclose(parity, -1.0):
            odd_states.append(ev)
            odd_vecs.append(vec)
        else:
            raise ValueError("State has non-integer parity expectation value.")

    return np.array(even_states), np.array(odd_states), np.array(even_vecs).T, np.array(odd_vecs).T


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
    even_states, odd_states, even_vecs, odd_vecs = divide_to_even_odd(evals, evecs, n)

    even_eigval_gs = even_states[0]
    odd_eigval_gs = odd_states[0]
    even_eigvec_gs = even_vecs[:, 0]
    odd_eigvec_gs = odd_vecs[:, 0]

    return (even_eigval_gs, even_eigvec_gs), (odd_eigval_gs, odd_eigvec_gs)


def single_Hamiltonian(n_sites, t, U, eps, Delta):
    """Construct the Hamiltonian for a single n_sites system."""
    cre, ann, num = precompute_ops(n_sites)
    H = np.zeros((2**n_sites, 2**n_sites), dtype=complex)

    # Hopping terms
    for i in range(n_sites - 1):
        H += -t[i] * (cre[i] @ ann[i + 1] + ann[i] @ cre[i + 1])
        H += Delta[i] * (cre[i] @ cre[i + 1] + ann[i + 1] @ ann[i])

    # On-site interaction terms
    for i in range(n_sites-1):
        H += U[i] * num[i] @ num[i + 1]

    # On-site energy terms
    for i in range(n_sites):
        H += eps[i] * num[i]

    return H

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


if __name__ == "__main__":
    n_sites = 3
    dupes = 3

    cre, ann, num = precompute_ops(n_sites * dupes)
    P = total_parity(num)

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")

    t, U, eps, Delta = pars
#    H = big_H(n_sites, dupes,  t, U, eps, Delta)
    set1 = single_Hamiltonian(n_sites, t, U, eps, Delta)

    set2 = single_Hamiltonian(n_sites, t, U, eps, Delta)
     
    set3 = single_Hamiltonian(n_sites, t, U, eps, Delta)
    sets = [set1, set2, set3]
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Spectra for {dupes} Identical {n_sites}-Site Systems', fontsize=16)
    
    original_eps = eps 
    detuned_eps = [1.0] * n_sites
    all_eps = [original_eps, detuned_eps, original_eps]
    print(all_eps)
    for i in range(dupes):
        # eps = all_eps[i]
        H = single_Hamiltonian(n_sites, t, U, eps, Delta)
        evals, evecs = np.linalg.eigh(H)    
        even_states, odd_states, even_vecs, odd_vecs = divide_to_even_odd(evals, evecs, n_sites)

        MP = majorana_polarization(even_vecs, odd_vecs, n_sites)
        print(f"Majorana Polarizations for Set {i+1}:")
        for j in range(MP.shape[0]):
            quick_MP_check(even_vecs, odd_vecs, n_sites, threshold=1e-2)
            # print(f"State {j}: {MP[j]}")


        plt.subplot(1, dupes, i+1)
        plt.hlines(even_states, xmin=-0.2, xmax=0.2, colors='b', label='Even', linestyles='solid')
        plt.hlines(odd_states, xmin=0.8, xmax=1.2, colors='r', label='Odd', linestyles='dashed')
        plt.title(f'Spectrum for Set {i+1}')
        plt.xticks([0.0, 1.0], ["Even", "Odd"])
        plt.ylabel('Energy')
        plt.xlim(-0.3, 1.3)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()


    set1 = single_Hamiltonian(n_sites, t, U, eps, Delta)
    set2 = single_Hamiltonian(n_sites, t, U, eps, Delta)
    set3 = single_Hamiltonian(n_sites, t, U, eps, Delta)

    set1_even, set1_odd = get_gs_pairs(n_sites, set1)
    set2_even, set2_odd = get_gs_pairs(n_sites, set2)
    set3_even, set3_odd = get_gs_pairs(n_sites, set3)

    majoranas = majorana_operators(n_sites * dupes)
    
    # Rightmost Majorana operator of site 0
    γ1 = majoranas[0][2]  
    #leftmost majorana operator of site 1
    γ0 = majoranas[1][0]
    #rightmost majorana operator of site 1
    γ2 = majoranas[1][2]
    #leftmost majorana operator of site 2
    γ3 = majoranas[2][0]




