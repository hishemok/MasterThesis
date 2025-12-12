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


def delta_pulse(T_total, t, rise_time, min_val, max_val):
    """
    Sigmoid delta pulse function for smooth parameter tuning."""
    if t < rise_time:
        # Rising edge
        x = t / rise_time
        return min_val + (max_val - min_val) / (1 + np.exp(-10 * (x - 0.5)))
    elif t > T_total - rise_time:
        # Falling edge
        x = (T_total - t) / rise_time
        return min_val + (max_val - min_val) / (1 + np.exp(-10 * (x - 0.5)))
    else:
        # Constant region
        return max_val



def parameter_tuning(current_time,n_sites, dupes, T_total, t, U, eps, Delta, t_couple, delta_couple, eps_detune):
    """
    Tunes parameters in the Hamiltonian by the order A to B, then B by itself, then B to C.
    Function couples A to B through rising t_couple and delta_couple, then decouples B from A while coupling B to itself, then decouples B from itself while coupling B to C.
    The couplings happen through a smooth delta pulse (sigmoid + reverse sigmoid) over total time T_total.
    This function passes into the time evolution function to evolve the system under the time-dependent Hamiltonian.
    Args:
        current_time (float): Current time in the tuning process.
        n_sites (int): Number of sites in each system.
        dupes (int): Number of identical systems.
        T_total (float): Total time for the tuning process.
        t, U, eps, Delta: Hamiltonian parameters.
        t_couple (float): Coupling strength between systems.
        delta_couple (float): Coupling detuning parameter.
        eps_detune (dict): Detuning values for specific systems.    
    """
    if current_time < T_total / 3:
        # Coupling A to B
        t_c = delta_pulse(T_total / 3, current_time, rise_time=T_total / 9, min_val=0, max_val=t_couple)
        delta_c = delta_pulse(T_total / 3, current_time, rise_time=T_total / 9, min_val=0, max_val=delta_couple)
        H_t = big_H(n_sites, dupes, t, U, eps, Delta,
                    couple_A=(0,2),
                    couple_B=(1,0),
                    t_couple=t_c,
                    delta_couple=delta_c,
                    eps_detune=eps_detune)
    elif current_time < 2 * T_total / 3:
        # Couple B to itself by tuning epsilons
        eps_puls = delta_pulse(T_total / 3, current_time - T_total / 3, rise_time=T_total / 9, min_val=0, max_val=eps_detune.get(1, 1))
        H_t = big_H(n_sites, dupes, t, U, eps, Delta,
                    eps_detune={1: eps_puls})
    else:
        # Decoupling B from C
        t_c = delta_pulse(T_total / 3, current_time - 2 * T_total / 3, rise_time=T_total / 9, min_val=t_couple, max_val=0)
        delta_c = delta_pulse(T_total / 3, current_time - 2 * T_total / 3, rise_time=T_total / 9, min_val=delta_couple, max_val=0)
        H_t = big_H(n_sites, dupes, t, U, eps, Delta,
                    couple_A=(1,0),
                    couple_B=(2,0),
                    t_couple=t_c,
                    delta_couple=delta_c,
                    eps_detune=eps_detune)
    
    return H_t


def time_evolve_system():
    pass


if __name__ == "__main__":
    n_sites = 3
    dupes = 3

    cre, ann, num = precompute_ops(n_sites * dupes)
    P = total_parity(num)

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")

    t, U, eps, Delta = pars
#    H = big_H(n_sites, dupes,  t, U, eps, Delta)


    # fullH = big_H(n_sites, dupes, t, U, eps, Delta,
    #               couple_A=(0,2),   # PMM 0, site 2
    #               couple_B=(1,0),   # PMM 1, site 0
    #               t_couple=0.1,
    #               delta_couple=100)

    fullH = big_H(n_sites, dupes, t, U, eps, Delta,
                    eps_detune={1: 1.0})  # Detune PMM

    set1 = extract_effective_H(fullH, n_sites, dupes, target=0)
    set2 = extract_effective_H(fullH, n_sites, dupes, target=1)
    set3 = extract_effective_H(fullH, n_sites, dupes, target=2)
    sets = [set1, set2, set3]


    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Spectra for {dupes} Identical {n_sites}-Site Systems', fontsize=16)
    
    original_eps = eps 
    detuned_eps = [1.0] * n_sites
    all_eps = [original_eps, detuned_eps, original_eps]
    print(all_eps)
    for i in range(dupes):
        # eps = all_eps[i]
        H = sets[i]
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



    # H_uncoupled = big_H(n_sites, dupes, t, U, eps, Delta, t_couple=0, delta_couple=0)
    # H_coupled   = big_H(n_sites, dupes, t, U, eps, Delta,
    #                 couple_A=(0,2), couple_B=(1,0),
    #                 t_couple=0.1, delta_couple=100)

    # evals_u = np.linalg.eigvalsh(H_uncoupled)
    # evals_c = np.linalg.eigvalsh(H_coupled)
    # plt.plot(evals_u[:40], 'o-', label='uncoupled')
    # plt.plot(evals_c[:40], 'x-', label='coupled')
    # plt.legend()
    # plt.xlabel('level index')
    # plt.ylabel('energy')
    # plt.show()


  


