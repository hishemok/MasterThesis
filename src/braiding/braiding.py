from full_system_hamiltonian import *
from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import matplotlib.pyplot as plt


def find_majorana_operators(bigN):
    create, annihilate, number = precompute_ops(bigN)
    majorana_ops = []
    for j in range(bigN):
        f_dag = create[j]
        f = annihilate[j]
        gamma_1 = f + f_dag
        gamma_2 = -1j * (f - f_dag)
        majorana_ops.append((gamma_1, gamma_2))
    return majorana_ops




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






if __name__ == "__main__":
    n_sites = 3
    dupes = 3

    cre, ann, num = precompute_ops(n_sites * dupes)
    P = total_parity(num)

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")

    t, U, eps, Delta = pars
    H = big_H(n_sites, dupes,  t, U, eps, Delta)

    evals, evecs = np.linalg.eigh(H)    
    even_states, odd_states, even_vecs, odd_vecs = divide_to_even_odd(evals, evecs, n_sites * dupes)

    for i in range(3):
        print(f"Even state {i} energy: {even_states[i]}| Parity 1")
        print(f"Odd state {i} energy: {odd_states[i]}| Parity -1")






